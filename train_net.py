import numpy as np
import time
import torch
import torch.nn as nn
from lib.dataset.coco import COCODataset
from lib.transform import Resize, BatchCollator, Compose, FlipLeftRight, FlipTopBottom
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import lib
from lib.model.faster_rcnn import faster_rcnn_resnet18
from pycocotools.cocoeval import COCOeval
import argparse
import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from torch.utils.tensorboard import SummaryWriter
from lib.util import WarmingUpScheduler
from lib.util import SummaryWriterWrap
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import json


@torch.no_grad()
def evaluate(net, dataloader, device, result_file, cfg):
    results = []
    net.eval()
    dataset = dataloader.dataset
    class_transform = dataset.class_transform
    for images, labels, bboxes, samples in dataloader:

        # if cfg["train"]["multi_process"]:
        #     dist.barrier()

        scores, labels, bboxes = net(images, None, None)
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        bboxes = bboxes.detach().cpu().numpy()

        for i in range(len(samples)):
            results_per_image = []
            for score, label, bbox in zip(scores[i], labels[i], bboxes[i]):
                if score == -1:
                    break
                x1, y1, x2, y2 = bbox
                size = samples[i]["size"]
                resize = samples[i]["resize"]
                x1 = size[0] / resize[0] * x1
                x2 = size[0] / resize[0] * x2
                y1 = size[1] / resize[1] * y1
                y2 = size[1] / resize[1] * y2

                results_per_image.append({
                    "image_id": samples[i]["image_id"],
                    "category_id": class_transform.decode(int(label)),
                    "score": float(score),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })
            # print("image {} {} objs {} detected".format(
            #     samples[i]["image_id"], len(samples[i]["label"]), len(results_per_image)))
            # print(results_per_image)
            results.extend(results_per_image)

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    if not cfg["train"]["multi_process"]:
        if len(results) > 0:
            dataset = dataloader.dataset
            cocoGT = dataset.coco
            imgIds = dataset.ids
            cocoDT = cocoGT.loadRes(results)
            cocoEval = COCOeval(cocoGT, cocoDT, "bbox")
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        else:
            print("There is no object detected")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-works", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gpus", help="gpu index, like 0,1")
    parser.add_argument("--resume-from")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    return args


def update_cfg(cfg, args):
    def update(cfg, args, group, name):
        if args.__dict__[name] is not None:
            if args.__dict__[name] == "":
                return
            cfg[group][name] = args.__dict__[name]

    update(cfg, args, "dataset", "batch_size")
    update(cfg, args, "dataset", "num_works")
    update(cfg, args, "train", "epochs")
    update(cfg, args, "train", "lr")
    update(cfg, args, "train", "gpus")
    update(cfg, args, "train", "resume_from")


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, "r")as f:
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
        cfg = yaml.load(f, Loader)

    update_cfg(cfg, args)

    assert cfg["train"]["gpus"] is not None and cfg["train"]["gpus"] != "";
    if type(cfg["train"]["gpus"]) is list:
        gpus = cfg["train"]["gpus"]
    elif type(cfg["train"]["gpus"]) is int:
        gpus = [cfg["train"]["gpus"], ]
    else:
        gpus = cfg["train"]["gpus"].strip().split(",")
        gpus = [int(i) for i in gpus if i != ""]
    cfg["train"]["gpus"] = gpus
    assert len(gpus) >= 2

    if cfg["train"]["multi_process"]:
        if args.local_rank is None:
            print("""
            python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)
            """)
            exit()
        else:
            cfg["train"]["local_rank"] = args.local_rank
        assert len(cfg["train"]["gpus"]) // 2 > args.local_rank

    log_dir = cfg["train"]["log"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(os.getcwd(), log_dir)
    checkpoint_dir = cfg["train"]["checkpoint"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
    result_dir = cfg["train"]["result"]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_dir = os.path.join(os.getcwd(), result_dir)

    if cfg["debug"]:
        print("Running in debug mode.")

    if cfg["train"]["multi_process"] and cfg["train"]["local_rank"] != 0:
        pass
    else:
        print(json.dumps(cfg, indent=2))

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])
    train_transform = Compose([
        Resize(cfg["dataset"]["resize"]),
        FlipLeftRight(),
        # FlipTopBottom(),
    ])
    val_transform = Compose([
        Resize(cfg["dataset"]["resize"]),
    ])

    train_data = cfg["dataset"]["train_data"]
    train_dataset = COCODataset(train_data["root"], train_data["annFile"], train_transform, debug=cfg["debug"])
    num_classes = len(train_dataset.classes.keys())
    val_data = cfg["dataset"]["val_data"]
    val_dataset = COCODataset(val_data["root"], val_data["annFile"], val_transform, debug=cfg["debug"])

    batch_size = cfg["dataset"]["batch_size"]

    if cfg["train"]["multi_process"]:
        work_size = int(os.environ["WORLD_SIZE"])
        local_rank = cfg["train"]["local_rank"]
        train_sampler = DistributedSampler(train_dataset, num_replicas=work_size, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=work_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                  num_workers=cfg["dataset"]["num_works"], sampler=train_sampler,
                            collate_fn=BatchCollator(cfg["dataset"]["max_objs_per_image"], image_transform))
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["dataset"]["batch_size"], shuffle=False,
                                  num_workers=cfg["dataset"]["num_works"], sampler=val_sampler,
                                collate_fn=BatchCollator(cfg["dataset"]["max_objs_per_image"], image_transform))

    if cfg["train"]["multi_process"] and cfg["train"]["local_rank"] != 0:
        writer = None
    else:
        writer = SummaryWriterWrap(log_dir)
    if cfg["train"]["multi_process"]:
        device0 = torch.device(gpus[cfg["train"]["local_rank"]*2])
        device1 = torch.device(gpus[cfg["train"]["local_rank"]*2+1])
    else:
        device0 = torch.device(gpus[0])
        device1 = torch.device(gpus[1])
    faster_rcnn = lib.model.faster_rcnn.__dict__[cfg["model"]["name"]](
        device0=device0,
        device1=device1,
        image_size=cfg["dataset"]["resize"],
        num_classes=num_classes,
        max_objs_per_image=cfg["dataset"]["max_objs_per_image"],
        backbone_pretrained=cfg["model"]["backbone_pertrained"],
        logger=writer,
        obj_thresh=cfg["model"]["obj_thresh"],
    )

    if cfg["model"]["load_from"] is not None and cfg["model"]["load_from"] != "":
        state_dict = torch.load(cfg["model"]["load_from"], torch.device('cpu'))
        if "epoch" in state_dict:
            faster_rcnn.load_state_dict(state_dict["model"])
        else:
            faster_rcnn.load_state_dict(state_dict)

    # optimizer = optim.SGD(faster_rcnn.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    optimizer = optim.SGD([
        {"params": faster_rcnn.backbone.parameters()},
        {"params": faster_rcnn.rpn.conv.parameters()},
        {"params": faster_rcnn.rpn.cls.parameters()},
        {"params": faster_rcnn.rpn.reg.parameters(), "lr": 0.001},
        {"params": faster_rcnn.roi_head.parameters()},
        {"params": faster_rcnn.cls.parameters()},
        {"params": faster_rcnn.reg.parameters(), "lr": 0.001},
    ], lr=cfg["train"]["lr"], weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[8, 10])
    warmup_scheduler = WarmingUpScheduler(optimizer, init_factor=0.1, steps=50)
    epochs = cfg["train"]["epochs"]
    epoch = 0

    if cfg["train"]["resume_from"] is not None and cfg["train"]["resume_from"] != "":
        state_dict = torch.load(cfg["train"]["resume_from"])
        faster_rcnn.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        epoch = state_dict["epoch"] + 1

    if cfg["train"]["multi_process"]:
        work_size = int(os.environ["WORLD_SIZE"])
        assert len(cfg["train"]["gpus"]) // 2 == work_size
        torch.distributed.init_process_group(backend='nccl', world_size=work_size, rank=cfg["train"]["local_rank"])
        faster_rcnn.to_gpus()
        # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
        # This error indicates that your module has parameters that were not used in producing loss.
        # You can enable unused parameter detection by
        #   (1) passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`;
        #   (2) making sure all `forward` function outputs participate in calculating loss.
        # If you already have done the above two steps, then the distributed data parallel module wasn't able to
        # locate the output tensors in the return value of your module's `forward` function.
        # Please include the loss function and the structure of the return value of `forward` of your module
        # when reporting this issue (e.g. list, dict, iterable).
        faster_rcnn = DDP(faster_rcnn, find_unused_parameters=True)
    else:
        faster_rcnn.to_gpus()

    iters_per_epoch = len(train_dataloader)
    iter_width = len(str(iters_per_epoch))
    for epoch in range(epoch, epochs):
        faster_rcnn.train()
        iteration = 0
        for images, labels, bboxes, _ in train_dataloader:
            if writer is not None:
                writer.step(epoch * iters_per_epoch + iteration)

            # if cfg["train"]["multi_process"]:
            #     dist.barrier()

            optimizer.zero_grad()
            rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss = faster_rcnn(images, labels, bboxes)
            if isinstance(faster_rcnn, nn.DataParallel):
                # (num_gpus,) -> (1,)
                rpn_cls_loss = torch.mean(rpn_cls_loss)
                rpn_reg_loss = torch.mean(rpn_reg_loss)
                rcnn_cls_loss = torch.mean(rcnn_cls_loss)
                rcnn_reg_loss = torch.mean(rcnn_reg_loss)

            total_loss = rpn_cls_loss + rpn_reg_loss + rcnn_cls_loss + rcnn_reg_loss
            total_loss.backward()
            _lr = optimizer.state_dict()["param_groups"][0]["lr"]
            optimizer.step()
            warmup_scheduler.step()

            rpn_cls_loss = rpn_cls_loss.detach().cpu().item()
            rpn_reg_loss = rpn_reg_loss.detach().cpu().item()
            rcnn_cls_loss = rcnn_cls_loss.detach().cpu().item()
            rcnn_reg_loss = rcnn_reg_loss.detach().cpu().item()
            total_loss = total_loss.detach().cpu().item()

            if writer is not None:
                writer.add_scalar("rpn/cls_loss", rpn_cls_loss)
                writer.add_scalar("rpn/reg_loss", rpn_reg_loss)
                writer.add_scalar("rcnn/cls_loss", rcnn_cls_loss)
                writer.add_scalar("rcnn/reg_loss", rcnn_reg_loss)
                writer.add_scalar("total_loss", total_loss)

            if cfg["train"]["multi_process"] and cfg["train"]["local_rank"] != 0:
                pass
            elif iteration % cfg["train"]["log_every_step"] == 0:
                log_string = "{} "
                log_string += "epoch {:03} iter {:0" + str(iter_width) + "}/{} "
                log_string += "lr {:0.6f} "
                log_string += "rpn_cls_loss {:6.4f} rpn_reg_loss {:6.4f} "
                log_string += "rcnn_cls_loss {:6.4f} rcnn_reg_loss {:6.4f} "
                log_string += "total_loss {:6.4f}"
                print(log_string.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    epoch, iteration, len(train_dataloader),
                    _lr,
                    rpn_cls_loss,
                    rpn_reg_loss,
                    rcnn_cls_loss,
                    rcnn_reg_loss,
                    total_loss,
                ))
            iteration += 1

        if cfg["train"]["multi_process"] and cfg["train"]["local_rank"] != 0:
            pass
        else:
            if isinstance(faster_rcnn, (nn.DataParallel, DDP)):
                model_state_dict = faster_rcnn.module.state_dict()
            else:
                model_state_dict = faster_rcnn.state_dict()
            save_dict = {
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch
            }
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-{:04}.pth".format(epoch))
            torch.save(save_dict, checkpoint_path)
            sym_path = os.path.join(checkpoint_dir, "checkpoint-latest.pth")
            torch.save(save_dict, sym_path)  # 软连接方式出错
            if writer is not None:
                writer.flush()
        scheduler.step()

        if cfg["train"]["multi_process"]:
            result_file_pattern = "val-epoch{}" + "-{}.json".format(cfg["train"]["local_rank"])
        else:
            result_file_pattern = "val-epoch{}.json"
        evaluate(faster_rcnn, val_dataloader, None, os.path.join(result_dir, result_file_pattern.format(epoch)), cfg)

    if writer is not None:
        writer.close()
    if cfg["train"]["multi_process"]:
        dist.destroy_process_group()

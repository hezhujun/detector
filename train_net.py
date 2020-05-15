import numpy as np
import torch
import torch.nn as nn
from lib.dataset.coco import COCODataset
from lib.transform import Resize, BatchCollator, Compose, FlipLeftRight
from torch.utils.data import DataLoader
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


@torch.no_grad()
def evaluate(net, dataloader, device):
    results = []
    net.eval()
    dataset = dataloader.dataset
    class_transform = dataset.class_transform
    for images, labels, bboxes, samples in dataloader:
        images = images.to(device)

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

    log_dir = cfg["train"]["log"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(os.getcwd(), log_dir)
    checkpoint_dir = cfg["train"]["checkpoint"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)

    if cfg["debug"]:
        print("Running in debug mode.")

    print(cfg)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])
    train_transform = Compose([
        Resize(cfg["dataset"]["resize"]),
        FlipLeftRight(),
    ])
    val_transform = Compose([
        Resize(cfg["dataset"]["resize"]),
    ])
    train_data = cfg["dataset"]["train_data"]
    train_dataset = COCODataset(train_data["root"], train_data["annFile"], train_transform, debug=cfg["debug"])
    num_classes = len(train_dataset.classes.keys())
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["dataset"]["batch_size"], shuffle=True,
                                  num_workers=cfg["dataset"]["num_works"],
                            collate_fn=BatchCollator(cfg["dataset"]["max_objs_per_image"], image_transform))

    val_data = cfg["dataset"]["val_data"]
    val_dataset = COCODataset(val_data["root"], val_data["annFile"], val_transform, debug=cfg["debug"])
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["dataset"]["batch_size"], shuffle=False,
                                  num_workers=cfg["dataset"]["num_works"],
                                collate_fn=BatchCollator(cfg["dataset"]["max_objs_per_image"], image_transform))

    writer = SummaryWriterWrap(log_dir)
    faster_rcnn = lib.model.faster_rcnn.__dict__[cfg["model"]["name"]](
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

    if cfg["train"]["gpus"] is not None and cfg["train"]["gpus"] != "":
        if type(cfg["train"]["gpus"]) is list:
            gpus = cfg["train"]["gpus"]
        elif type(cfg["train"]["gpus"]) is int:
            gpus = [cfg["train"]["gpus"],]
        else:
            gpus = cfg["train"]["gpus"].strip().split(",")
            gpus = [int(i) for i in gpus if i != ""]
        cfg["train"]["gpus"] = gpus
        device = [torch.device("cuda", i) for i in gpus]
    else:
        cfg["train"]["gpus"] = None
        device = torch.device("cpu")

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
        # if isinstance(faster_rcnn, nn.DataParallel):
        #     # DataParallel在forward阶段分发模型和输入
        #     # 保证每次forward时每个device的模型都一样
        #     # 所以不用担心已经初始化的faster_rcnn再次加载模型参数导致不同devcie的模型参数不一样
        #     faster_rcnn.module.load_state_dict(state_dict["model"])
        # else:
        #     faster_rcnn.load_state_dict(state_dict["model"])
        faster_rcnn.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        epoch = state_dict["epoch"] + 1

    if cfg["train"]["gpus"] is None:
        faster_rcnn = faster_rcnn.to(device)
    if len(cfg["train"]["gpus"]) == 1:
        device = device[0]
        faster_rcnn = faster_rcnn.to(device)
    else:
        # 这里暂时不用DistributedDataParallel，因为测试阶段需要在同一进程收集所有的输出
        # 使用DataParallel方便操作
        faster_rcnn = faster_rcnn.to(device[0])
        faster_rcnn = nn.DataParallel(faster_rcnn, device_ids=device, output_device=device[0])
        device = device[0]   # 只需把输入传到device[0]，DataParallel会自动把输入分发到各个device中

    iters_per_epoch = len(train_dataloader)
    iter_width = len(str(iters_per_epoch))
    for epoch in range(epoch, epochs):
        faster_rcnn.train()
        iteration = 0
        for images, labels, bboxes, _ in train_dataloader:
            writer.step(epoch * iters_per_epoch + iteration)

            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

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
            warmup_scheduler.step(epoch*iters_per_epoch + iteration)

            rpn_cls_loss = rpn_cls_loss.detach().cpu().item()
            rpn_reg_loss = rpn_reg_loss.detach().cpu().item()
            rcnn_cls_loss = rcnn_cls_loss.detach().cpu().item()
            rcnn_reg_loss = rcnn_reg_loss.detach().cpu().item()
            total_loss = total_loss.detach().cpu().item()

            writer.add_scalar("rpn/cls_loss", rpn_cls_loss)
            writer.add_scalar("rpn/reg_loss", rpn_reg_loss)
            writer.add_scalar("rcnn/cls_loss", rcnn_cls_loss)
            writer.add_scalar("rcnn/reg_loss", rcnn_reg_loss)
            writer.add_scalar("total_loss", total_loss)

            if iteration % cfg["train"]["log_every_step"] == 0:
                log_string = "epoch {:03} iter {:0" + str(iter_width) + "}/{} "
                log_string += "lr {:0.6f} "
                log_string += "rpn_cls_loss {:6.4f} rpn_reg_loss {:6.4f} "
                log_string += "rcnn_cls_loss {:6.4f} rcnn_reg_loss {:6.4f} "
                log_string += "total_loss {:6.4f}"
                print(log_string.format(
                    epoch, iteration, len(train_dataloader),
                    _lr,
                    rpn_cls_loss,
                    rpn_reg_loss,
                    rcnn_cls_loss,
                    rcnn_reg_loss,
                    total_loss,
                ))
            iteration += 1

        save_dict = {
            "model": faster_rcnn.module.state_dict() if isinstance(faster_rcnn, nn.DataParallel) else faster_rcnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }
        scheduler.step()
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-{:04}.pth".format(epoch))
        torch.save(save_dict, checkpoint_path)
        sym_path = os.path.join(checkpoint_dir, "checkpoint-latest.pth")
        torch.save(save_dict, sym_path)  # 软连接方式出错
        writer.flush()

        evaluate(faster_rcnn, val_dataloader, device)

    writer.close()

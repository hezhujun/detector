import numpy as np
import torch
import torch.nn as nn
from lib.dataset.coco import COCODataset
from lib.transform import Resize, BatchCollator
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


@torch.no_grad()
def evaluate(net, dataloader, device):
    results = []
    net.eval()
    for images, labels, bboxes, samples in dataloader:
        images = images.to(device)

        scores, labels, bboxes = faster_rcnn(images, None, None)
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
                    "category_id": int(label),
                    "score": float(score),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })
            print("image {} {} objs {} detected".format(
                samples[i]["image_id"], len(samples[i]["label"]), len(results_per_image)))
            print(results_per_image)
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

    resize = Resize(cfg["dataset"]["resize"])
    dataset = COCODataset(cfg["dataset"]["root"], cfg["dataset"]["annFile"], resize, debug=cfg["debug"])
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])
    dataloader = DataLoader(dataset, batch_size=cfg["dataset"]["batch_size"], shuffle=False,
                            collate_fn=BatchCollator(cfg["dataset"]["max_objs_per_image"], image_transform))

    writer = SummaryWriter(log_dir)
    faster_rcnn = lib.model.faster_rcnn.__dict__[cfg["model"]["name"]](
        image_size=cfg["dataset"]["resize"],
        num_classes=max(dataset.classes.keys()) + 1,
        max_objs_per_image=cfg["dataset"]["max_objs_per_image"],
        backbone_pretrained=cfg["model"]["backbone_pertrained"],
        logger=writer,
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
        else:
            gpus = cfg["train"]["gpus"].strip().split(",")
            gpus = [int(i) for i in gpus if i != ""]
        assert len(gpus) == 1, "only support one gpus now"
        cfg["train"]["gpus"] = gpus
        device = [torch.device("cuda", i) for i in gpus]
        device = device[0]
    else:
        device = torch.device("cpu")

    faster_rcnn = faster_rcnn.to(device)

    optimizer = optim.SGD(faster_rcnn.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[8, 10])
    warmup_scheduler = WarmingUpScheduler(optimizer, init_factor=0.1, steps=100)
    epochs = cfg["train"]["epochs"]
    epoch = 0
    if cfg["train"]["resume_from"] is not None and cfg["train"]["resume_from"] != "":
        state_dict = torch.load(cfg["train"]["resume_from"])
        faster_rcnn.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        epoch = state_dict["epoch"] + 1

    iters_per_epoch = len(dataloader)
    iter_width = len(str(iters_per_epoch))
    for epoch in range(epoch, epochs):
        faster_rcnn.train()
        iteration = 0
        for images, labels, bboxes, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss = faster_rcnn(images, labels, bboxes)

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

            log_string = "epoch {:03} iter {:0" + str(iter_width) + "}/{} "
            log_string += "lr {:0.6f} "
            log_string += "rpn_cls_loss {:6.4f} rpn_reg_loss {:6.4f} "
            log_string += "rcnn_cls_loss {:6.4f} rcnn_reg_loss {:6.4f} "
            log_string += "total_loss {:6.4f}"
            print(log_string.format(
                epoch, iteration, len(dataloader),
                _lr,
                rpn_cls_loss,
                rpn_reg_loss,
                rcnn_cls_loss,
                rcnn_reg_loss,
                total_loss,
            ))
            iteration += 1

        save_dict = {
            "model": faster_rcnn.state_dict(),
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

        evaluate(faster_rcnn, dataloader, device)

    writer.close()
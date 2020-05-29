import numpy as np
import time
import torch
import torch.nn as nn
from lib.dataset.coco import COCODataset
from lib.transform import Resize, BatchCollator, Compose, FlipLeftRight
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

from train_net import evaluate


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

    if cfg["train"]["gpus"] is None or len(cfg["train"]["gpus"]) == 0:
        cfg["train"]["gpus"] = None
        device = torch.device("cpu")
        cfg["train"]["multi_process"] = False

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
        assert len(cfg["train"]["gpus"]) > 0

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
    val_transform = Compose([
        Resize(cfg["dataset"]["resize"]),
    ])

    train_data = cfg["dataset"]["train_data"]
    train_dataset = COCODataset(train_data["root"], train_data["annFile"], val_transform, debug=cfg["debug"])
    num_classes = len(train_dataset.classes.keys())
    val_data = cfg["dataset"]["val_data"]
    val_dataset = COCODataset(val_data["root"], val_data["annFile"], val_transform, debug=cfg["debug"])

    if isinstance(device, list) and not cfg["train"]["multi_process"]:
        batch_size = cfg["dataset"]["batch_size"] * len(device)
    else:
        batch_size = cfg["dataset"]["batch_size"]

    if cfg["train"]["multi_process"]:
        work_size = int(os.environ["WORLD_SIZE"])
        local_rank = cfg["train"]["local_rank"]
        train_sampler = DistributedSampler(train_dataset, num_replicas=work_size, rank=local_rank, shuffle=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=work_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=cfg["dataset"]["num_works"], sampler=train_sampler,
                            collate_fn=BatchCollator(cfg["dataset"]["max_objs_per_image"], image_transform))
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["dataset"]["batch_size"], shuffle=False,
                                  num_workers=cfg["dataset"]["num_works"], sampler=val_sampler,
                                collate_fn=BatchCollator(cfg["dataset"]["max_objs_per_image"], image_transform))

    writer = None
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

    if cfg["train"]["gpus"] is None or len(cfg["train"]["gpus"]) == 0:
        faster_rcnn = faster_rcnn.to(device)
    elif cfg["train"]["multi_process"]:
        work_size = int(os.environ["WORLD_SIZE"])
        assert len(cfg["train"]["gpus"]) == work_size
        device = device[cfg["train"]["local_rank"]]
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl', world_size=work_size, rank=cfg["train"]["local_rank"])
        faster_rcnn = faster_rcnn.to(device)
        # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
        # This error indicates that your module has parameters that were not used in producing loss.
        # You can enable unused parameter detection by
        #   (1) passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`;
        #   (2) making sure all `forward` function outputs participate in calculating loss.
        # If you already have done the above two steps, then the distributed data parallel module wasn't able to
        # locate the output tensors in the return value of your module's `forward` function.
        # Please include the loss function and the structure of the return value of `forward` of your module
        # when reporting this issue (e.g. list, dict, iterable).
        faster_rcnn = DDP(faster_rcnn, device_ids=[device, ], output_device=device, find_unused_parameters=True)
    else:
        if len(cfg["train"]["gpus"]) == 1:
            device = device[0]
            faster_rcnn = faster_rcnn.to(device)
        else:
            faster_rcnn = faster_rcnn.to(device[0])
            faster_rcnn = nn.DataParallel(faster_rcnn, device_ids=device, output_device=device[0])
            device = device[0]   # 只需把输入传到device[0]，DataParallel会自动把输入分发到各个device中

    if cfg["train"]["multi_process"]:
        result_file_pattern = "val" + "-{}.json".format(cfg["train"]["local_rank"])
    else:
        result_file_pattern = "val.json"
    evaluate(faster_rcnn, val_dataloader, device, os.path.join(result_dir, result_file_pattern), cfg)

    if cfg["train"]["multi_process"]:
        dist.destroy_process_group()

import argparse
import json
import os

import yaml
from pycocotools.cocoeval import COCOeval

from lib.dataset.coco import COCODataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--cfg", default="cfg.yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, "r")as f:
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
        cfg = yaml.load(f, Loader)

    result_dir = cfg["train"]["result"]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_dir = os.path.join(os.getcwd(), result_dir)

    val_data = cfg["dataset"]["val_data"]
    dataset = COCODataset(val_data["root"], val_data["annFile"], None, debug=cfg["debug"])

    result_file = args.filename
    files = []
    for file in os.listdir(result_dir):
        if os.path.isfile(os.path.join(result_dir, file)):
            if file.startswith(result_file):
                files.append(file)

    results = []
    for file in files:
        print("load result file", file)
        with open(os.path.join(result_dir, file), 'r') as f:
            results.extend(json.load(f))

    if len(results) == 0:
        print("There is no object detected")
        exit()

    cocoGT = dataset.coco
    imgIds = dataset.ids
    cocoDT = cocoGT.loadRes(results)
    cocoEval = COCOeval(cocoGT, cocoDT, "bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()



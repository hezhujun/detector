import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


class COCODataset(object):

    def __init__(self, root, annFile, transform=None, debug=False):
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.debug = debug

        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.getImgIds())
        if debug:
            self.ids = self.ids[:100]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = self.coco.loadImgs(image_id)[0]
        image_path = self.parse_image_path(self.root, image)
        img = Image.open(image_path)
        label = []
        bbox = []
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            label.append(ann["category_id"])
            x1, y1, w, h = ann["bbox"]
            bbox.append([x1, y1, x1 + w, y1 + h])

        label = np.array(label)
        bbox = np.array(bbox)
        sample = {"image_id": image_id, "img": img, "label": label, "bbox": bbox}

        if self.transform is not None:
            return self.transform(sample)

        return sample

    def parse_image_path(self, root, item):
        return os.path.join(root, item["file_name"])

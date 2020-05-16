import os
import numpy as np
from PIL import Image, ImageFile
from collections import OrderedDict
from pycocotools.coco import COCO
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassTransform(object):

    def encode(self, class_id):
        raise NotImplementedError()

    def decode(self, class_id):
        raise NotImplementedError()


class ClassTransformImpl(ClassTransform):
    def __init__(self, classe_ids:list):
        self.map = {class_id:i for i, class_id in enumerate(classe_ids)}
        self.map_reverse = {v:k for k,v in self.map.items()}

    def encode(self, class_id):
        return self.map[class_id]

    def decode(self, class_id):
        return self.map_reverse[class_id]


class COCODataset(Dataset):

    def __init__(self, root, annFile, transform=None, debug=False):
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.debug = debug

        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.getImgIds())
        if debug:
            self.ids = self.ids[:10]

        self.class_transform = ClassTransformImpl(sorted(self.coco.getCatIds()))

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
            class_id = self.class_transform.encode(ann["category_id"])
            label.append(class_id)
            x1, y1, w, h = ann["bbox"]
            bbox.append([x1, y1, x1 + w, y1 + h])

        label = np.array(label)
        bbox = np.array(bbox)
        sample = OrderedDict(image_path=image_path, image_id=image_id, img=img, label=label, bbox=bbox)

        if self.transform is not None:
            return self.transform(sample)

        return sample

    def parse_image_path(self, root, item):
        return os.path.join(root, item["file_name"])

    @property
    def classes(self):
        cat_ids = sorted(self.coco.getCatIds())
        classes_info = self.coco.loadCats(cat_ids)
        return {int(c["id"]): c["name"] for c in classes_info}
import random
import numpy as np
from PIL import Image
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Resize(object):
    def __init__(self, size, keep_ratio=True):
        """

        :param size: (w, h)
        :param keep_ratio:
        """
        assert isinstance(size, (tuple, list))
        assert len(size) == 2
        self.size = tuple(size)
        self.keep_ratio = keep_ratio
        self.size_ratio = size[0] / size[1]

    def __call__(self, sample):
        img = sample["img"]
        label = sample["label"]
        bbox = sample["bbox"]
        size = img.size

        if not self.keep_ratio:
            img = img.resize(self.size)
        else:
            new_w, new_h = self.size
            if self.size_ratio > (size[0] / size[1]):
                new_w = size[1] * self.size_ratio
            else:
                new_h = size[0] / self.size_ratio
            new_w, new_h = int(new_w), int(new_h)
            img = img.resize((new_w, new_h))

        resize = img.size
        # (old_w / new_w, old_h / new_h)
        scale = (size[0]/resize[0], size[1]/resize[1])
        scale = scale + scale
        scale = np.array(scale)
        bbox = bbox / scale

        if self.keep_ratio:
            new_img = Image.new(img.mode, self.size)
            new_img.paste(img, (0, 0, resize[0], resize[1]))
            img = new_img

        sample["img"] = img
        sample["bbox"] = bbox
        sample["size"] = size
        sample["resize"] = resize
        return sample


class FlipLeftRight(object):

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        if random.random() < self.ratio:
            img = sample["img"]
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            size = img.size
            size = np.array([size[0], size[1], size[0], size[1]])
            bbox = sample["bbox"]
            bbox[:, 0] = size[0] - bbox[:, 0] - 1
            bbox[:, 2] = size[0] - bbox[:, 2] - 1
            sample["img"] = img
            sample["bbox"] = bbox
            sample["flip_left_right"] = True
        else:
            sample["flip_left_right"] = False
        return sample


class FlipTopBottom(object):

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        if random.random() < self.ratio:
            img = sample["img"]
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            size = img.size
            size = np.array([size[0], size[1], size[0], size[1]])
            bbox = sample["bbox"]
            bbox[:, 1] = size[1] - bbox[:, 1] - 1
            bbox[:, 3] = size[1] - bbox[:, 3] - 1
            sample["img"] = img
            sample["bbox"] = bbox
            sample["flip_top_bottom"] = True
        else:
            sample["flip_top_bottom"] = False
        return sample


class BatchCollator(object):

    def __init__(self, max_objs_per_image, image_transform):
        self.max_objs = max_objs_per_image
        self.image_transform = image_transform

    def __call__(self, samples):
        images = torch.stack([self.image_transform(sample["img"]) for sample in samples])
        num_samples = len(samples)
        labels = np.full((num_samples, self.max_objs), -1, dtype=np.int64)
        bboxes = np.full((num_samples, self.max_objs, 4), -1, dtype=np.float32)
        for i, sample in enumerate(samples):
            label = sample["label"]
            bbox = sample["bbox"]
            for j in range(min(self.max_objs, label.shape[0])):
                labels[i, j] = label[j]
                bboxes[i, j] = bbox[j]
        labels = torch.from_numpy(labels)
        bboxes = torch.from_numpy(bboxes)
        return images, labels, bboxes, samples
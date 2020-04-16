import numpy as np
from PIL import Image


class Resize(object):
    def __init__(self, size, keep_ratio=True):
        """

        :param size: (w, h)
        :param keep_ratio:
        """
        assert isinstance(size, tuple)
        self.size = size
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


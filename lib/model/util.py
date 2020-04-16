import torch


class BoxCoder(object):
    def __init__(self):
        pass

    def decode(self, anchors, bboxes):
        """

        :param anchors: (num_anchors, 4)
        :param bboxes: (BS, num_anchors, 4)
        :return:
        """
        xa1, ya1, xa2, ya2 = anchors[..., 0], anchors[..., 1], anchors[..., 2], anchors[..., 3]
        xa = (xa1 + xa2) / 2
        ya = (ya1 + ya2) / 2
        wa = xa2 - xa1
        ha = ya2 - ya1

        tx, ty, tw, th = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        x = tx * wa + xa
        y = ty * ha + ya
        w = torch.exp(tw) * wa
        h = torch.exp(th) * ha

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x1 + w
        y2 = y1 + h

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def encode(self, anchors, bboxes):
        """

        :param anchors: (H*W*num_anchors, 4)
        :param bboxes: (H*W*num_anchors, 4)
        :return:
        """

        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        xa1, ya1, xa2, ya2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        xa = (xa1 + xa2) / 2
        ya = (ya1 + ya2) / 2
        wa = xa2 - xa1
        ha = ya2 - ya1

        # x: (BS, H, W, num_anchors)
        # xa: (H, W, num_anchors)
        # wa: (H, W, num_anchors)
        tx = (x - xa) / wa
        ty = (y - ya) / ha
        tw = torch.log(w/wa + 1e-6)
        th = torch.log(h/ha + 1e-6)

        return torch.stack([tx, ty, tw, th], dim=-1)


if __name__ == '__main__':
    import numpy as np
    anchors = np.array([10.,10,50,50]).reshape((1, 1, 1, 4))
    bboxes = np.array([[8.,8,48,48],[12,12,52,52]]).reshape((2, 1, 1, 1, 4))
    anchors = torch.from_numpy(anchors)
    bboxes = torch.from_numpy(bboxes)

    box_coder = BoxCoder()
    encoded_bboxes = box_coder.encode(anchors, bboxes)
    print(encoded_bboxes.shape)
    print(encoded_bboxes.reshape(2, 4))
    decoded_bboxes = box_coder.decode(anchors, encoded_bboxes)
    print(decoded_bboxes.shape)
    print(decoded_bboxes.reshape(2, 4))

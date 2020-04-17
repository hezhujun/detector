import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from .anchor_generator import generate_anchor
from .util import BoxCoder


class RegionProposalNetwork(nn.Module):

    def __init__(self, strides, sizes, scales, ratios,
                 in_channels,
                 pre_nms_top_n_in_train, post_nms_top_n_in_train,
                 pre_nms_top_n_in_test, post_nms_top_n_in_test,
                 nms_thresh=0.7, fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                 num_samples=256, positive_fraction=0.5):
        """

        :param strides:
        :param sizes:
        :param scales:
        :param ratios:
        :param in_channels:

        strides: tuple, 步长，每个步长对应一个特征层，如
            (2**2,2**3,2**4,2**5,2**6)
            分别对应p2, p3, p4, p5, p6

        sizes: tuple, 生成anchors的网格大小，分别对应一个特征层，如
            ((80,80), (40,40), (20,20), (10,10), (5,5))

        scales: tuple, 面积，分别对应一个特征层，如
            ((32**2,), (64**2,), (128**2,), (256**2,), (512**2,))
            每个特征层只有一种尺度

        ratios: tuple, 宽高比，分别对应一个特征层，如
            ((0.5, 1, 2),(0.5, 1, 2),(0.5, 1, 2),(0.5, 1, 2),(0.5, 1, 2))
            每个特征层只有3中宽高比
        """
        super(RegionProposalNetwork, self).__init__()

        assert len(strides) == len(sizes) == len(scales) == len(ratios)
        num_anchors = 0
        for i in range(len(strides)):
            assert isinstance(strides[i], int)
            assert isinstance(sizes[i], tuple)
            assert isinstance(scales[i], tuple)
            assert isinstance(ratios[i], tuple)
            num_anchors = len(scales[i]) * len(ratios[i])
            # the shape of anchor is (size[0], size[1], num_anchors, 4)
            anchor = torch.from_numpy(generate_anchor(strides[i], scales[i], ratios[i], sizes[i]))
            self.register_buffer("anchor%i"%i, anchor)

        self.pre_nms_top_n_in_train = pre_nms_top_n_in_train
        self.post_nms_top_n_in_train = post_nms_top_n_in_train
        self.pre_nms_top_n_in_test = pre_nms_top_n_in_test
        self.post_nms_top_n_in_test = post_nms_top_n_in_test
        self.nms_thresh = nms_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction
        self.num_pos = int(num_samples * positive_fraction)
        self.num_neg = num_samples - self.num_pos

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.box_coder = BoxCoder()

    def forward(self, features, labels=None, gt_bboxes=None):
        """

        :param features: OrderDict. The shape of each item is (BS, C_i, H_i, W_i)
        :param labels: shape (BS, n_objs)
        :param gt_bboxes: shape (BS, n_objs, 4)
        :return:
        """
        total_anchors = []
        total_cls_pred = []
        total_reg_pred = []
        total_cls_scores = []
        total_reg_bboxes = []
        for i, feat in enumerate(features.values()):
            x = F.relu(self.conv(feat))
            cls_pred = self.cls(x)  # (BS, num_anchors, H, W)
            reg_pred = self.reg(x)  # (BS, num_anchors*4, H, W)

            BS, num_anchors, H, W = cls_pred.shape
            # (BS, H, W, num_anchors)
            cls_pred = cls_pred.permute(0, 2, 3, 1)
            # (BS, H, W, num_anchors, 4)
            reg_pred = reg_pred.permute(0, 2, 3, 1).reshape((BS, H, W, num_anchors, 4))
            # (H, W, num_anchors, 4)
            anchors = self._buffers["anchor%i"%i]

            # (BS, H, W, num_anchors) -> (BS, H*W*num_anchors)
            cls_pred = cls_pred.reshape((BS, -1))
            # (BS, H, W, num_anchors, 4) -> (BS, H*W*num_anchors, 4)
            reg_pred = reg_pred.reshape((BS, -1, 4))
            # (H, W, num_anchors, 4) -> (H*W*num_anchors, 4)
            anchors = anchors.reshape((-1, 4))

            total_anchors.append(anchors)
            total_cls_pred.append(cls_pred)
            total_reg_pred.append(reg_pred)

            with torch.no_grad():
                # 修正anchors
                reg_bboxes = self.box_coder.decode(anchors, reg_pred.detach())
                # 计算分数
                cls_scores = torch.sigmoid(cls_pred.detach())

                total_cls_scores.append(cls_scores)
                total_reg_bboxes.append(reg_bboxes)

        # (-1, 4)
        anchors = torch.cat(total_anchors, dim=0)
        # (BS, -1)
        cls_pred = torch.cat(total_cls_pred, dim=1)
        # (BS, -1, 4)
        reg_pred = torch.cat(total_reg_pred, dim=1)
        # (BS, -1)
        cls_scores = torch.cat(total_cls_scores, dim=1)
        # (BS, -1, 4)
        reg_bboxes = torch.cat(total_reg_bboxes, dim=1)

        if self.training:
            pre_nms_top_n = self.pre_nms_top_n_in_train
            post_nms_top_n = self.post_nms_top_n_in_train
        else:
            pre_nms_top_n = self.pre_nms_top_n_in_test
            post_nms_top_n = self.post_nms_top_n_in_test

        # NMS
        BS = cls_pred.shape[0]
        keep_bboxes = []
        for i in range(BS):
            dtype = reg_bboxes.dtype
            device = reg_bboxes.device
            _bboxes = torch.full((post_nms_top_n, 4), -1, dtype=dtype, device=device)
            keep = ops.nms(reg_bboxes[i], cls_scores[i], self.nms_thresh)
            n_keep = keep.shape[0]
            n_keep = min(n_keep, post_nms_top_n)
            keep = keep[:n_keep]
            _bboxes[:n_keep] = reg_bboxes[i][keep]
            keep_bboxes.append(_bboxes)

        bboxes = torch.stack(keep_bboxes)  # (BS, post_nms_top_n, 4)

        if self.training:
            total_cls_pred = []
            total_reg_pred = []
            total_reg_target = []
            total_fg_bg_mask = []
            for i in range(BS):
                # 为每个anchor分配label
                ious = ops.box_iou(anchors, gt_bboxes[i])  # (num_total_anchors, num_gt_bboxes) (N, M) for short
                # the anchor/anchors with the highest Intersection-over-Union (IoU)
                # overlap with a ground-truth box
                iou_max_gt, indices = torch.max(ious, dim=0)
                # 不考虑gt_bboxes中填充的部分
                iou_max_gt = torch.where(labels[i] == -1, torch.ones_like(iou_max_gt), iou_max_gt)
                highest_mask = (ious == iou_max_gt)
                fg_mask = torch.any(highest_mask, dim=1)
                # an anchor that has an IoU overlap higher than fg_iou_thresh with any ground-truth box
                iou_max, matched_idx = torch.max(ious, dim=1)
                # 1 for foreground -1 for background 0 for ignore
                fg_bg_mask = torch.zeros_like(iou_max)
                # confirm positive samples
                fg_bg_mask = torch.where(iou_max > self.fg_iou_thresh, torch.ones_like(iou_max), fg_bg_mask)
                fg_bg_mask = torch.where(fg_mask, torch.ones_like(iou_max), fg_bg_mask)
                # confirm negetive samples
                fg_bg_mask = torch.where(iou_max < self.bg_iou_thresh, torch.full_like(iou_max, -1), fg_bg_mask)

                # 采样
                indices = fg_bg_mask.argsort(descending=True)
                fg_bg_mask = fg_bg_mask[indices]
                fg_indices = indices[:self.num_pos]
                fg_mask = fg_bg_mask[:self.num_pos]
                bg_indices = indices[-self.num_neg:]
                bg_mask = fg_bg_mask[-self.num_neg:]

                indices = torch.cat([fg_indices, bg_indices], dim=0)
                fg_bg_mask = torch.cat([fg_mask, bg_mask], dim=0)

                matched_idx = matched_idx[indices]
                _anchors = anchors[indices]

                total_cls_pred.append(cls_pred[i][indices])
                total_reg_pred.append(reg_pred[i][indices])
                total_fg_bg_mask.append(fg_bg_mask)
                total_reg_target.append(self.box_coder.encode(_anchors, gt_bboxes[i][matched_idx]))

            # (BS, num_samples)
            cls_pred = torch.stack(total_cls_pred)
            # (BS, num_samples, 4)
            reg_pred = torch.stack(total_reg_pred)
            # (BS, num_samples)
            fg_bg_mask = torch.stack(total_fg_bg_mask)
            # (BS, num_samples, 4)
            reg_target = torch.stack(total_reg_target)

            cls_label = torch.where(fg_bg_mask == 1, torch.ones_like(cls_pred), torch.zeros_like(cls_pred))
            cls_loss = F.binary_cross_entropy_with_logits(cls_pred[fg_bg_mask != 0], cls_label[fg_bg_mask != 0])
            reg_loss = F.smooth_l1_loss(reg_pred[fg_bg_mask == 1], reg_target[fg_bg_mask == 1])

            return bboxes, cls_loss, reg_loss

        return bboxes, None, None
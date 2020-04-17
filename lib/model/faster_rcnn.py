import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
import torch.hub as hub
from .rpn import RegionProposalNetwork
import torch.nn.functional as F
from .util import BoxCoder


class FasterRCNN(nn.Module):
    
    def __init__(self,
                 backbone, roi_head, dim_roi_features,
                 image_size, num_classes,
                 strides, sizes, scales, ratios,
                 rpn_in_channels,
                 rpn_pre_nms_top_n_in_train=2000, rpn_post_nms_top_n_in_train=1000,
                 rpn_pre_nms_top_n_in_test=2000, rpn_post_nms_top_n_in_test=1000,
                 rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 roi_pooling="roi_align", roi_pooling_output_size=7,
                 nms_thresh=0.5, fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                 num_samples=128, positive_fraction=0.25,
                 max_objs_per_image=50, obj_thresh=0.1):
        super(FasterRCNN, self).__init__()
        self.image_size = image_size
        self.backbone = backbone
        self.rpn = RegionProposalNetwork(strides=strides,
                                         sizes=sizes,
                                         scales=scales,
                                         ratios=ratios,
                                         in_channels=rpn_in_channels,
                                         pre_nms_top_n_in_train=rpn_pre_nms_top_n_in_train,
                                         post_nms_top_n_in_train=rpn_post_nms_top_n_in_train,
                                         pre_nms_top_n_in_test=rpn_pre_nms_top_n_in_test,
                                         post_nms_top_n_in_test=rpn_post_nms_top_n_in_test,
                                         nms_thresh=rpn_nms_thresh,
                                         fg_iou_thresh=rpn_fg_iou_thresh,
                                         bg_iou_thresh=rpn_bg_iou_thresh,
                                         num_samples=rpn_num_samples,
                                         positive_fraction=rpn_positive_fraction)
        self.roi_head = roi_head
        self.cls = nn.Linear(dim_roi_features, num_classes + 1)
        self.reg = nn.Linear(dim_roi_features, num_classes * 4)

        self.cls.weight.data.normal_(std=0.01)
        self.cls.bias.data.zero_()
        self.reg.weight.data.normal_(std=0.01)
        self.reg.bias.data.zero_()

        self.strides = strides
        self.num_classes = num_classes
        self.box_coder = BoxCoder()
        self.nms_thresh = nms_thresh
        self.max_objs_per_image = max_objs_per_image
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction
        self.num_pos = int(num_samples * positive_fraction)
        self.num_neg = num_samples - self.num_pos
        self.roi_pooling = roi_pooling
        self.roi_pooling_output_size = roi_pooling_output_size
        self.obj_thresh = obj_thresh

    def forward(self, images, labels=None, gt_bboxes=None):
        """

        :param images: shape (BS, C, H, W)
        :param labels: shape (BS, n_objs)
        :param gt_bboxes: shape (BS, n_objs, 4)
        :return:
        """

        feats = self.backbone(images)
        # rois shape (BS, num_rois, 4)
        rois, rpn_cls_loss, rpn_reg_loss = self.rpn(feats, labels, gt_bboxes)

        rois[..., 0].clamp_(0, self.image_size[0])
        rois[..., 1].clamp_(0, self.image_size[1])
        rois[..., 2].clamp_(0, self.image_size[0])
        rois[..., 3].clamp_(0, self.image_size[1])

        if self.training:
            # 把gt bboxes加入到rois中
            rois = torch.cat([rois, gt_bboxes], dim=1)

        # rois 添加batch_id维
        BS, num_rois, _ = rois.shape
        batch_id = torch.stack([
            torch.full_like(rois[i, :, :1], i) for i in range(BS)
        ], dim=0)
        # (BS, num_rois, 5)
        rois = torch.cat([batch_id, rois], dim=2)
        # (BS*num_rois, 5)
        rois = rois.reshape((-1, 5))

        # roi pooling in each feature map
        if self.roi_pooling == "roi_align":
            roi_pooling = ops.roi_align
        elif self.roi_pooling == "roi_pool":
            roi_pooling = ops.roi_pool
        else:
            raise Exception("{} is not support".format(self.roi_pooling))

        if len(feats) == 1:
            _, feat = feats.popitem()
            roi_feats = roi_pooling(feat, rois, self.roi_pooling_output_size, 1/self.strides[0])
        else:
            feat_levels = np.log2(self.strides).astype(np.int64)
            feat_names = [n for n in feats.keys()]
            assert len(feat_levels) == len(feat_names)

            w = rois[:, 3] - rois[:, 1]
            h = rois[:, 4] - rois[:, 2]
            roi_levels = torch.floor(4 + torch.log2(torch.sqrt(w * h) / 224 + 1e-6))

            _f = feats[feat_names[0]]
            C = _f.shape[1]
            device = _f.device
            dtype = _f.dtype
            roi_feats = torch.zeros((BS*num_rois, C, self.roi_pooling_output_size, self.roi_pooling_output_size),
                                    dtype=dtype, device=device)

            for i, (feat_level, feat_name) in enumerate(zip(feat_levels, feat_names)):
                mask_in_level = roi_levels == feat_level
                _roi_feats = roi_pooling(feats[feat_name], rois[mask_in_level], self.roi_pooling_output_size, 1/self.strides[i])
                roi_feats[mask_in_level] = _roi_feats

        # roi_feats shape (BS*num_rois, C, self.roi_pooling_output_size, self.roi_pooling_output_size)

        # roi head
        # (BS*num_rois, num_vector)
        box_feats = self.roi_head(roi_feats)
        # (BS*num_rois, num_classes+1)
        cls_pred = self.cls(box_feats)
        # (BS*num_rois, num_classes*4)
        reg_pred = self.reg(box_feats)
        # (BS, num_rois, num_classes+1)
        cls_pred = cls_pred.reshape((BS, num_rois, -1))
        # (BS, num_rois, num_classes, 4)
        reg_pred = reg_pred.reshape((BS, num_rois, -1, 4))

        if self.training:
            # (BS, num_rois, 5)
            rois = rois.reshape((BS, num_rois, 5))

            total_cls_pred = []
            total_reg_pred = []
            total_fg_bg_mask = []
            total_labels = []
            total_reg_target = []

            for i in range(BS):
                # 为每个anchor分配label
                # (num_rois, num_gt_bboxes)
                ious = ops.box_iou(rois[i][:, 1:], gt_bboxes[i])
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
                # (num_samples)
                label = labels[i][matched_idx]
                # 把标签-1变成0，F。one_hot不支持负数
                _label = label.clone()
                _label[_label == -1] = 0
                # (num_samples, num_classes)
                label_mask = F.one_hot(_label, self.num_classes)

                # reg_pred[i][indices]: (num_samples, num_classes, 4)
                label_mask = torch.stack([label_mask]*4, dim=2).to(torch.bool)
                # (num_samples, 4)
                _reg_pred = reg_pred[i][indices][label_mask].reshape((-1, 4))
                _rois = rois[i][:, 1:][indices]
                total_cls_pred.append(cls_pred[i][indices])
                total_reg_pred.append(_reg_pred)
                total_fg_bg_mask.append(fg_bg_mask)
                total_labels.append(label)
                total_reg_target.append(self.box_coder.encode(_rois, gt_bboxes[i][matched_idx]))

            # (BS, num_samples, num_classes+1)
            cls_pred = torch.stack(total_cls_pred)
            # (BS, num_samples, 4)
            reg_pred = torch.stack(total_reg_pred)
            # (BS, num_samples)
            fg_bg_mask = torch.stack(total_fg_bg_mask)
            # (BS, num_samples)
            labels = torch.stack(total_labels)
            # (BS, num_samples, 4)
            reg_target = torch.stack(total_reg_target)

            rcnn_reg_loss = F.smooth_l1_loss(reg_pred[fg_bg_mask==1], reg_target[fg_bg_mask==1])

            cls_label = labels + 1  # 0 for background class
            cls_label = cls_label.reshape((-1,))
            cls_pred = cls_pred.reshape((-1, self.num_classes+1))
            fg_bg_mask = fg_bg_mask.reshape(-1,)
            rcnn_cls_loss = F.cross_entropy(cls_pred[fg_bg_mask != 0], cls_label[fg_bg_mask != 0])

            return rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss

        cls_scores = F.softmax(cls_pred, dim=2)
        # (BS, num_rois, num_classes)
        cls_scores = cls_scores[:, :, 1:]

        # rois: (BS*num_rois, 5)
        # reg_pred: (BS, num_rois, num_classes, 4)
        _reg_pred = reg_pred.permute((2, 0, 1, 3)).reshape(self.num_classes, BS*num_rois, 4)
        # (num_classes, BS*num_rois, 4)
        reg_bboxes = self.box_coder.decode(rois[:, 1:], _reg_pred)
        # (BS, num_rois, num_classes, 4)
        reg_bboxes = reg_bboxes.permute((1, 0, 2)).reshape((BS, num_rois, self.num_classes, 4))

        # (num_rois, num_classes)
        classes_id = torch.cat([
            # (num_rois, 1)
            torch.full_like(cls_scores[0, :, :1], i) for i in range(self.num_classes)
        ], dim=1)
        # (num_rois*num_classes)
        classes_id = classes_id.reshape((-1,))
        # (BS, num_rois*num_classes)
        cls_scores = cls_scores.reshape((BS, -1))
        # (BS, num_rois*num_classes, 4)
        reg_bboxes = reg_bboxes.reshape((BS, -1, 4))

        scores = []
        bboxes = []
        labels = []
        for i in range(BS):
            _scores = torch.full((self.max_objs_per_image,), -1, dtype=cls_scores.dtype, device=cls_scores.device)
            _labels = torch.full((self.max_objs_per_image,), -1, dtype=classes_id.dtype, device=classes_id.device)
            _bboxes = torch.full((self.max_objs_per_image, 4), -1, dtype=reg_bboxes.dtype, device=reg_bboxes.device)
            keep_mask = cls_scores[i] > self.obj_thresh
            keep = ops.boxes.batched_nms(reg_bboxes[i][keep_mask], cls_scores[i][keep_mask], classes_id[keep_mask], self.nms_thresh)
            n_keep = keep.shape[0]
            n_keep = min(n_keep, self.max_objs_per_image)
            keep = keep[:n_keep]
            _scores[:n_keep] = cls_scores[i][keep]
            _labels[:n_keep] = classes_id[keep]
            _bboxes[:n_keep] = reg_bboxes[i][keep]

            scores.append(_scores)
            labels.append(_labels)
            bboxes.append(_bboxes)

        scores = torch.stack(scores)  # (BS, max_objs)
        labels = torch.stack(labels)  # (BS, max_objs)
        bboxes = torch.stack(bboxes)  # (BS, max_objs, 4)

        bboxes[..., 0].clamp_(0, self.image_size[0])
        bboxes[..., 1].clamp_(0, self.image_size[1])
        bboxes[..., 2].clamp_(0, self.image_size[0])
        bboxes[..., 3].clamp_(0, self.image_size[1])

        return scores, labels, bboxes


def faster_rcnn_resnet(backbone_name, image_size, num_classes, max_objs_per_image, backbone_pretrained=False):
    resnet = models.resnet.__dict__[backbone_name](pretrained=backbone_pretrained)
    # return_layers = {'layer1': 'c2', 'layer2': 'c3', 'layer3': 'c4', 'layer4': 'c5'}
    backbone = models._utils.IntermediateLayerGetter(resnet, {'layer3': 'c4'})

    roi_head = nn.Sequential()
    C5 = None
    for name, module in resnet.named_children():
        if name == "layer4":
            C5 = module
            roi_head.add_module("0", module)
            break
    assert C5 is not None
    roi_head.add_module("1", nn.modules.flatten.Flatten())
    fc = nn.Linear(512 * 4 * 4, 1024)  # 自己计算roi_pooling后roi的特征数
    roi_head.add_module("2", fc)
    roi_head.add_module("3", nn.BatchNorm1d(1024))
    roi_head.add_module("4", nn.ReLU())

    strides = (2 ** 4,)  # C4的步长
    sizes = ((50, 50),)  # C4 feature map的大小
    scales = ((128 ** 2, 256 ** 2, 512 ** 2,),)
    ratios = ((0.5, 1, 2),)

    rpn_in_channels = 256  # C4的channels
    dim_roi_features = 1024  # roi特征向量长度

    return FasterRCNN(
        backbone=backbone,
        roi_head=roi_head,
        dim_roi_features=dim_roi_features,
        image_size=image_size,
        num_classes=num_classes,
        strides=strides,
        sizes=sizes,
        scales=scales,
        ratios=ratios,
        rpn_in_channels=rpn_in_channels,
        max_objs_per_image=max_objs_per_image,
        roi_pooling="roi_align",
        roi_pooling_output_size=7,
    )


def faster_rcnn_resnet18(image_size, num_classes, max_objs_per_image, backbone_pretrained=False):
    return faster_rcnn_resnet(
        "resnet18",
        image_size=image_size,
        num_classes=num_classes,
        max_objs_per_image=max_objs_per_image,
        backbone_pretrained=backbone_pretrained
    )
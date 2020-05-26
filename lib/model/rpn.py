import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from .anchor_generator import generate_anchor
from .util import BoxCoder


class RegionProposalNetwork(nn.Module):

    def __init__(self,
                 device,
                 strides, sizes, scales, ratios,
                 in_channels,
                 image_size,
                 pre_nms_top_n_in_train, post_nms_top_n_in_train,
                 pre_nms_top_n_in_test, post_nms_top_n_in_test,
                 nms_thresh=0.7, fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                 num_samples=256, positive_fraction=0.5,
                 nms_per_layer=True,
                 logger=None):
        """
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

        nms_per_layer: 是否在每个特征层上进行nms。如果不是fine-tune，训练初期分类器弱，nms得到的rois可能都属于同一层特种层，
            rois的尺度空间变化小。比如，rois都是来自p2层。
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
            anchor[..., 0].clamp_(0, image_size[0])
            anchor[..., 1].clamp_(0, image_size[1])
            anchor[..., 2].clamp_(0, image_size[0])
            anchor[..., 3].clamp_(0, image_size[1])
            self.register_buffer("anchor%i"%i, anchor)

        self.image_size = image_size
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
        self.logger = logger
        self.nms_per_layer = nms_per_layer
        self.device = device

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

        if self.training:
            pre_nms_top_n = self.pre_nms_top_n_in_train
            post_nms_top_n = self.post_nms_top_n_in_train
        else:
            pre_nms_top_n = self.pre_nms_top_n_in_test
            post_nms_top_n = self.post_nms_top_n_in_test

        if self.training:
            labels = labels.to(self.device)
            gt_bboxes = gt_bboxes.to(self.device)

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
                reg_bboxes[..., 0].clamp_(0, self.image_size[0])
                reg_bboxes[..., 1].clamp_(0, self.image_size[1])
                reg_bboxes[..., 2].clamp_(0, self.image_size[0])
                reg_bboxes[..., 3].clamp_(0, self.image_size[1])
                # 计算分数
                cls_scores = torch.sigmoid(cls_pred.detach())

                if not self.nms_per_layer:
                    total_cls_scores.append(cls_scores)
                    total_reg_bboxes.append(reg_bboxes)
                else:
                    # NMS per layer
                    BS = cls_scores.shape[0]
                    keep_bboxes = []
                    keep_scores = []
                    for i in range(BS):
                        dtype = reg_bboxes.dtype
                        device = reg_bboxes.device
                        _bboxes = torch.full((post_nms_top_n//len(features), 4), -1, dtype=dtype, device=device)
                        _scores = torch.full((post_nms_top_n//len(features), ), -1, dtype=cls_scores.dtype, device=cls_scores.device)
                        # pre_nms_top_n_indices = torch.argsort(cls_scores[i], descending=True)
                        # _num_anchors = pre_nms_top_n_indices.shape[0]
                        # _pre_nms_top_n = pre_nms_top_n // len(features) if _num_anchors > pre_nms_top_n // len(features) else _num_anchors
                        # pre_nms_top_n_indices = pre_nms_top_n_indices[:_pre_nms_top_n]
                        # keep = ops.nms(reg_bboxes[i][pre_nms_top_n_indices], cls_scores[i][pre_nms_top_n_indices], self.nms_thresh)
                        keep = ops.nms(reg_bboxes[i], cls_scores[i], self.nms_thresh)
                        n_keep = keep.shape[0]
                        n_keep = min(n_keep, post_nms_top_n//len(features))
                        keep = keep[:n_keep]
                        _bboxes[:n_keep] = reg_bboxes[i][keep]
                        _scores[:n_keep] = cls_scores[i][keep]
                        keep_bboxes.append(_bboxes)
                        keep_scores.append(_scores)

                    total_reg_bboxes.append(torch.stack(keep_bboxes))
                    total_cls_scores.append(torch.stack(keep_scores))

        # (-1, 4)
        anchors = torch.cat(total_anchors, dim=0)
        # (BS, -1)
        cls_pred = torch.cat(total_cls_pred, dim=1)
        # (BS, -1, 4)
        reg_pred = torch.cat(total_reg_pred, dim=1)

        if not self.nms_per_layer:
            # (BS, -1)
            cls_scores = torch.cat(total_cls_scores, dim=1)
            # (BS, -1, 4)
            reg_bboxes = torch.cat(total_reg_bboxes, dim=1)

            # NMS
            BS = cls_pred.shape[0]
            keep_bboxes = []
            for i in range(BS):
                dtype = reg_bboxes.dtype
                device = reg_bboxes.device
                _bboxes = torch.full((post_nms_top_n, 4), -1, dtype=dtype, device=device)
                # pre_nms_top_n_indices = torch.argsort(cls_scores[i], descending=True)
                # _num_anchors = pre_nms_top_n_indices.shape[0]
                # _pre_nms_top_n = pre_nms_top_n if _num_anchors > pre_nms_top_n else _num_anchors
                # pre_nms_top_n_indices = pre_nms_top_n_indices[:_pre_nms_top_n]
                # keep = ops.nms(reg_bboxes[i][pre_nms_top_n_indices], cls_scores[i][pre_nms_top_n_indices], self.nms_thresh)
                keep = ops.nms(reg_bboxes[i], cls_scores[i], self.nms_thresh)
                n_keep = keep.shape[0]
                n_keep = min(n_keep, post_nms_top_n)
                keep = keep[:n_keep]
                _bboxes[:n_keep] = reg_bboxes[i][keep]
                keep_bboxes.append(_bboxes)

            bboxes = torch.stack(keep_bboxes)  # (BS, post_nms_top_n, 4)
        else:
            # (BS, post_nms_top_n)
            cls_scores = torch.cat(total_cls_scores, dim=1)
            # (BS, post_nms_top_n)
            bboxes = torch.cat(total_reg_bboxes, dim=1)
            # 根据scores大小对bboxes(rois)进行降序排序
            # 可能的原因：
            # rois的顺序会影响rcnn，比如在rcnn的nms时，rcnn更倾向于选择前面的rois
            # rois的rpn_scores高，rcnn_scores的分数也高
            # 当rois的分数差不多时，位于前面的rois会在nms时抑制后面的rois
            for i in range(cls_scores.shape[0]):
                sorted_indices = torch.argsort(cls_scores[i], descending=True)
                bboxes[i] = bboxes[i][sorted_indices]

        if self.training:
            total_cls_pred = []
            total_reg_pred = []
            total_reg_target = []
            total_fg_bg_mask = []

            all_cls_pred = []
            all_fg_bg_mask = []

            BS = gt_bboxes.shape[0]
            for i in range(BS):
                # 为每个anchor分配label
                areas = ops.boxes.box_area(anchors)
                ious = ops.box_iou(anchors, gt_bboxes[i])  # (num_total_anchors, num_gt_bboxes) (N, M) for short
                # 把nan换成0
                zero_mask = (areas == 0).reshape(-1, 1).expand_as(ious)
                ious[zero_mask] = 0

                if torch.any(torch.isnan(ious)):
                    raise Exception("some elements in ious is nan")

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
                fg_bg_mask = torch.where(iou_max >= self.fg_iou_thresh, torch.ones_like(iou_max), fg_bg_mask)
                fg_bg_mask = torch.where(fg_mask, torch.ones_like(iou_max), fg_bg_mask)
                # confirm negetive samples
                fg_bg_mask = torch.where(iou_max <= self.bg_iou_thresh, torch.full_like(iou_max, -1), fg_bg_mask)

                all_cls_pred.append(cls_pred[i].detach())
                all_fg_bg_mask.append(fg_bg_mask.detach())

                # 随机采样
                indices = torch.arange(fg_bg_mask.shape[0], dtype=torch.int64, device=fg_bg_mask.device)
                rand_indices = torch.rand_like(fg_bg_mask).argsort()
                fg_bg_mask = fg_bg_mask[rand_indices]    # 打乱顺序，实现“随机”
                indices = indices[rand_indices]

                sorted_indices = fg_bg_mask.argsort(descending=True)
                fg_bg_mask = fg_bg_mask[sorted_indices]
                indices = indices[sorted_indices]
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

                # from lib import debug
                # debug.rpn_pos_bboxes.append(_anchors[fg_bg_mask == 1])
                # print(cls_pred[i][indices][fg_bg_mask == 1].detach().cpu().numpy())

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
            if torch.any(torch.isnan(reg_target[fg_bg_mask == 1])):
                raise Exception("some elements in reg_target is nan")
            if torch.any(torch.isnan(reg_pred[fg_bg_mask == 1])):
                raise Exception("some elements in reg_pred is nan")
            if torch.any(fg_bg_mask == 1):
                reg_loss = F.smooth_l1_loss(reg_pred[fg_bg_mask == 1], reg_target[fg_bg_mask == 1])
            else:  # 没有正样本
                reg_loss = torch.zeros_like(cls_loss)

            cls_pred = cls_pred >= 0.5
            cls_label = cls_label == 1
            acc = torch.mean((cls_label == cls_pred)[fg_bg_mask != 0].to(torch.float))
            num_pos = (fg_bg_mask == 1).sum()
            num_neg = (fg_bg_mask == -1).sum()

            TP = (cls_pred == True)[fg_bg_mask == 1].sum().to(torch.float32)
            FP = (cls_pred == True)[fg_bg_mask == -1].sum().to(torch.float32)
            # TN = (cls_pred == False)[fg_bg_mask == -1].sum()
            FN = (cls_pred == False)[fg_bg_mask == 1].sum().to(torch.float32)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            all_cls_pred = torch.stack(all_cls_pred)
            all_fg_bg_mask = torch.stack(all_fg_bg_mask)
            all_cls_pred = all_cls_pred >= 0
            all_TP = (all_cls_pred == True)[all_fg_bg_mask == 1].sum().to(torch.float32)
            all_FP = (all_cls_pred == True)[all_fg_bg_mask == -1].sum().to(torch.float32)
            all_FN = (all_cls_pred == False)[all_fg_bg_mask == 1].sum().to(torch.float32)
            all_precision = all_TP / (all_TP + all_FP)
            all_recall = all_TP / (all_TP + all_FN)

            if self.logger is not None:
                # print("TP {} FP {} FN {}".format(TP.detach().cpu().item(), FP.detach().cpu().item(), FN.detach().cpu().item()))
                # print("all_TP {} all_FP {} all_FN {}".format(all_TP.detach().cpu().item(), all_FP.detach().cpu().item(), all_FN.detach().cpu().item()))
                # print("precision {} recall {} all_precision {} all_recall {}".format(precision.detach().cpu().item(),
                #                                                                      recall.detach().cpu().item(),
                #                                                                      all_precision.detach().cpu().item(),
                #                                                                      all_recall.detach().cpu().item()))
                self.logger.add_scalar("rpn/TP", TP.detach().cpu().item())
                self.logger.add_scalar("rpn/FP", FP.detach().cpu().item())
                self.logger.add_scalar("rpn/FN", FN.detach().cpu().item())
                self.logger.add_scalar("rpn/all_TP", all_TP.detach().cpu().item())
                self.logger.add_scalar("rpn/all_FP", all_FP.detach().cpu().item())
                self.logger.add_scalar("rpn/all_FN", all_FN.detach().cpu().item())
                self.logger.add_scalar("rpn/acc", acc.detach().cpu().item())
                self.logger.add_scalar("rpn/num_pos", num_pos.detach().cpu().item())
                self.logger.add_scalar("rpn/num_neg", num_neg.detach().cpu().item())
                self.logger.add_scalar("rpn/precision", precision.detach().cpu().item())
                self.logger.add_scalar("rpn/recall", recall.detach().cpu().item())
                self.logger.add_scalar("rpn/all_precision", all_precision.detach().cpu().item())
                self.logger.add_scalar("rpn/all_recall", all_recall.detach().cpu().item())

            return bboxes, cls_loss, reg_loss

        return bboxes, None, None
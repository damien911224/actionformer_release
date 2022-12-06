# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
import math, os
import copy
import numpy as np
import itertools
from torch import nn

from ..utils import box_ops, segment_ops
from ..utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .matcher import build_matcher
from .segmentation import (dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .cdn_components import prepare_for_cdn, cdn_post_process


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DINO(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """

    def __init__(self, transformer, num_classes, num_queries,
                 pos_1d_embeds, pos_2d_embeds, num_feature_levels, input_dim,
                 aux_loss=True, with_box_refine=True, two_stage=False,
                 use_dab=True, num_patterns=0, random_refpoints_xy=False,
                 dn_number=100, dn_box_noise_scale=0.4, dn_label_noise_ratio=0.5, dn_labelbook_size=100):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            use_dab: using dynamic anchor boxes formulation
            num_patterns: number of pattern embeddings
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.pos_1d_embeds = pos_1d_embeds
        self.pos_2d_embeds = pos_2d_embeds
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.input_dim = input_dim
        self.max_input_len = 1024
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        if not use_dab:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        else:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)  # for indicator
            self.refpoint_embed = nn.Embedding(num_queries, 4)
            if random_refpoints_xy:
                # import ipdb; ipdb.set_trace()
                self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
                self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        if num_feature_levels > 1:
            input_proj_list = []
            # input_proj_list = [
            #     nn.Sequential(
            #         nn.Conv1d(2048, hidden_dim, kernel_size=1),
            #         nn.GroupNorm(32, hidden_dim))]

            for _ in range(num_feature_levels):
                input_proj_list.append(nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(max(min(hidden_dim // 8, 32), 1), hidden_dim)))

            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(2048, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None


    def forward(self, features, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        raw_pos_1d = self.pos_1d_embeds.repeat(features[0].size(0), 1, 1)
        raw_pos_2d = self.pos_2d_embeds.repeat(features[0].size(0), 1, 1, 1)

        srcs = []
        pos_1d = []
        pos_2d = []
        for l, feat in enumerate(features):
            src = self.input_proj[l](feat)
            n, c, t = src.shape
            src = src.unsqueeze(-1)
            pos_1d_l = F.interpolate(raw_pos_1d, size=t, mode="linear")
            pos_2d_l = F.interpolate(raw_pos_2d, size=(t, t), mode="bilinear")
            pos_1d.append(pos_1d_l)
            pos_2d.append(pos_2d_l)
            srcs.append(src)

        if self.use_dab:
            if self.num_patterns == 0:
                tgt_all_embed = self.tgt_embed.weight  # nq, 256
                refanchor = self.refpoint_embed.weight  # nq, 4
            else:
                # multi patterns is not used in this version
                assert NotImplementedError
        else:
            assert NotImplementedError

        input_query_label = self.tgt_embed.weight.unsqueeze(0).repeat(features[0].size(0), 1, 1)
        input_query_bbox = self.refpoint_embed.weight.unsqueeze(0).repeat(features[0].size(0), 1, 1)

        # prepare for dn
        if self.dn_number > 0 and self.training:
            dino_query_label, dino_query_bbox, attn_mask, dn_meta = \
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training, num_queries=self.num_queries, num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim, label_enc=self.label_enc)
            input_query_label = torch.cat((dino_query_label, input_query_label), dim=1)
            input_query_bbox = torch.cat((dino_query_bbox, input_query_bbox), dim=1)
        else:
            attn_mask = dn_meta = None

        query_embeds = torch.cat((input_query_label, input_query_bbox), dim=2)

        hs, init_reference, inter_references, _, _ = \
            self.transformer(srcs, pos_1d, pos_2d, query_embeds, attn_mask, self.label_enc)

        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # dn post process
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord = \
                cdn_post_process(outputs_class, outputs_coord, dn_meta, self.aux_loss, self._set_aux_loss)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion_DINO(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, layer=None, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        if indices is not None:
            idx = self._get_src_permutation_idx(indices)
            if layer is None:
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices) if len(t["labels"])])
            else:
                target_classes_o = torch.cat(
                    [t["labels"].repeat(2 ** (5 - layer))[J] for t, (_, J) in zip(targets, indices)])
        else:
            num_boxes = torch.ones_like(num_boxes)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if indices is not None:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        if layer is not None:
            num_boxes = num_boxes * (2 ** (5 - layer))
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log and indices is not None:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, layer=None):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, layer=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        if indices is None:
            return {}
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        if layer is None:
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices) if len(t["boxes"])], dim=0)
        else:
            target_boxes = torch.cat([t['boxes'].repeat(2 ** (5 - layer), 1)[i] for t, (_, i) in zip(targets, indices)],
                                     dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        if layer is not None:
            num_boxes = num_boxes * (2 ** (5 - layer))

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = ((1 - torch.diag(segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_boxes[..., 2:]),
            segment_ops.segment_cw_to_t1t2(target_boxes[..., 2:])))) +
                     (1 - torch.diag(segment_ops.segment_iou(
                         src_boxes[..., :2], target_boxes[..., :2])))) / 2
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses["loss_xy"] = loss_bbox[..., :2].sum() / num_boxes
            losses["loss_hw"] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            # 'dn_labels': self.loss_dn_labels,
            # 'dn_boxes': self.loss_dn_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        # base_len = 192
        # num_levels = 6
        # reg_ranges = np.array([0, 4, 8, 16, 32, 64, 10000]) / base_len
        # multiscale_outputs = list()
        # multiscale_targets = list()
        # prev_start_idx = 0
        # for l in range(num_levels):
        #     target_len = base_len // (2 ** l)
        #     this_pred_boxes = outputs_without_aux["pred_boxes"][:, prev_start_idx:prev_start_idx + target_len]
        #     this_pred_logits = outputs_without_aux["pred_logits"][:, prev_start_idx:prev_start_idx + target_len]
        #     this_outputs = dict({"pred_boxes": this_pred_boxes, "pred_logits": this_pred_logits})
        #     multiscale_outputs.append(this_outputs)
        #
        #     this_targets = list()
        #     for t in targets:
        #         target_dict = dict({"boxes": list(), "labels": list()})
        #         this_target_boxes = t["boxes"]
        #         this_target_labels = t["labels"]
        #         for b_i, box in enumerate(this_target_boxes):
        #             if reg_ranges[l] <= box[-1] < reg_ranges[l + 1]:
        #                 target_dict["boxes"].append(box)
        #                 target_dict["labels"].append(this_target_labels[b_i])
        #         if len(target_dict["boxes"]):
        #             target_dict["boxes"] = torch.stack(target_dict["boxes"], dim=0)
        #             target_dict["labels"] = torch.stack(target_dict["labels"], dim=0)
        #
        #         this_targets.append(target_dict)
        #     multiscale_targets.append(this_targets)
        #
        #     prev_start_idx += target_len

        # if return_indices:
        #     indices0_copy = indices
        #     indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # for l in range(num_levels):
        #     this_outputs = multiscale_outputs[l]
        #     this_targets = multiscale_targets[l]
        #     this_num_boxes = sum(len(t["labels"]) for t in this_targets)
        #     this_num_boxes = torch.as_tensor([this_num_boxes], dtype=torch.float, device=device)
        #     if is_dist_avail_and_initialized():
        #         torch.distributed.all_reduce(this_num_boxes)
        #     if this_num_boxes <= 0.0:
        #         this_indices = None
        #     else:
        #         this_indices = self.matcher(this_outputs, this_targets)
        #     for loss in self.losses:
        #         l_dict = self.get_loss(loss, this_outputs, this_targets, this_indices, this_num_boxes)
        #         l_dict = {'s{:02d}_'.format(l + 1) + k: v for k, v in l_dict.items()}
        #         losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                # indices = self.matcher(outputs_without_props, targets, layer=idx)
                # if return_indices:
                #     indices_list.append(indices)

                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        kwargs['log'] = False
                    # kwargs['layer'] = idx
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                # base_len = 192
                # num_levels = 6
                # reg_ranges = np.array([0, 4, 8, 16, 32, 64, 10000]) / base_len
                # multiscale_outputs = list()
                # multiscale_targets = list()
                # prev_start_idx = 0
                # for l in range(num_levels):
                #     target_len = base_len // (2 ** l)
                #     this_pred_boxes = aux_outputs["pred_boxes"][:, prev_start_idx:prev_start_idx + target_len]
                #     this_pred_logits = aux_outputs["pred_logits"][:, prev_start_idx:prev_start_idx + target_len]
                #     this_outputs = dict({"pred_boxes": this_pred_boxes, "pred_logits": this_pred_logits})
                #     multiscale_outputs.append(this_outputs)
                #
                #     this_targets = list()
                #     for t in targets:
                #         target_dict = dict({"boxes": list(), "labels": list()})
                #         this_target_boxes = t["boxes"]
                #         this_target_labels = t["labels"]
                #         for b_i, box in enumerate(this_target_boxes):
                #             if reg_ranges[l] <= box[-1] < reg_ranges[l + 1]:
                #                 target_dict["boxes"].append(box)
                #                 target_dict["labels"].append(this_target_labels[b_i])
                #         if len(target_dict["boxes"]):
                #             target_dict["boxes"] = torch.stack(target_dict["boxes"], dim=0)
                #             target_dict["labels"] = torch.stack(target_dict["labels"], dim=0)
                #
                #         this_targets.append(target_dict)
                #     multiscale_targets.append(this_targets)
                #
                #     prev_start_idx += target_len
                #
                # for l in range(num_levels):
                #     this_outputs = multiscale_outputs[l]
                #     this_targets = multiscale_targets[l]
                #     this_num_boxes = sum(len(t["labels"]) for t in this_targets)
                #     this_num_boxes = torch.as_tensor([this_num_boxes], dtype=torch.float, device=device)
                #     if is_dist_avail_and_initialized():
                #         torch.distributed.all_reduce(this_num_boxes)
                #     if this_num_boxes <= 0.0:
                #         this_indices = None
                #     else:
                #         this_indices = self.matcher(this_outputs, this_targets)
                #     for loss in self.losses:
                #         l_dict = self.get_loss(loss, this_outputs, this_targets, this_indices, this_num_boxes)
                #         l_dict = {'s{:02d}_'.format(l + 1) + k + f'_{idx}': v for k, v in l_dict.items()}
                #         losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar,
                                                    **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # if return_indices:
        #     indices_list.append(indices0_copy)
        #     return losses, indices_list

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args["num_classes"]
    device = "cuda"

    # Pick a pretrained model
    # backbone_model_name = "slowfast_r50"
    # backbone = torch.hub.load("facebookresearch/pytorchvideo:main", model=backbone_model_name, pretrained=True,
    #                           slowfast_fusion_conv_stride=[8, 1, 1], norm=FrozenBatchNorm3d)
    # backbone.blocks = backbone.blocks[:-2]
    # backbone.num_channels = 2048

    T, C = 100, args["hidden_dim"]
    t_embeds = nn.Embedding(T, C)
    s_embeds = nn.Embedding(T, C // 2)
    e_embeds = nn.Embedding(T, C // 2)
    nn.init.uniform_(t_embeds.weight)
    nn.init.uniform_(s_embeds.weight)
    nn.init.uniform_(e_embeds.weight)

    s_embeds = s_embeds.weight.unsqueeze(1).repeat(1, T, 1)
    e_embeds = e_embeds.weight.unsqueeze(0).repeat(T, 1, 1)
    pos_1d_embeds = t_embeds.weight.t().unsqueeze(0).to(device)
    pos_2d_embeds = torch.cat((s_embeds, e_embeds), dim=-1).permute(2, 0, 1).unsqueeze(0).to(device)

    transformer = build_deforamble_transformer(args)

    dn_labelbook_size = num_classes

    model = DINO(
        transformer,
        num_classes=args["num_classes"],
        num_queries=args["num_queries"],
        aux_loss=True,
        pos_1d_embeds=pos_1d_embeds,
        pos_2d_embeds=pos_2d_embeds,
        num_feature_levels=args["num_feature_levels"],
        input_dim=args["input_dim"],
        dn_number=args["dn_number"] if args["use_dn"] else 0,
        dn_box_noise_scale=args["dn_box_noise_scale"],
        dn_label_noise_ratio=args["dn_label_noise_ratio"],
        dn_labelbook_size=dn_labelbook_size
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args["weight_loss_ce"],
                   'loss_bbox': args["weight_loss_bbox"],
                   'loss_giou': args["weight_loss_giou"]}
    clean_weight_dict = copy.deepcopy(weight_dict)

    if args["use_dn"]:
        weight_dict['loss_ce_dn'] = args["weight_loss_ce"]
        weight_dict['loss_bbox_dn'] = args["weight_loss_bbox"]
        weight_dict['loss_giou_dn'] = args["weight_loss_giou"]
    # TODO this is a hack
    if args["aux_loss"]:
        aux_weight_dict = {}
        for i in range(args["dec_layers"] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    # losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion_DINO(num_classes, matcher=matcher, weight_dict=weight_dict,
                                  focal_alpha=args["focal_alpha"], losses=losses)
    criterion.to(device)

    return model, criterion

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
# from .transformer import build_deforamble_transformer
from .deformable_transformer import build_deforamble_transformer
from .cdn_components import prepare_for_cdn, cdn_post_process

from .ops.roi_align import ROIAlign
from ..utils.nms import dynamic_nms

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DINO(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """

    def __init__(self, transformer, num_classes, num_queries,
                 pos_1d_embeds, pos_2d_embeds, num_feature_levels, input_dim,
                 aux_loss=True, with_box_refine=True, two_stage=False,
                 use_dab=True, num_patterns=0, random_refpoints_xy=False,
                 dn_number=100, dn_box_noise_scale=0.4, dn_label_noise_ratio=0.5, dn_labelbook_size=100,
                 with_act_reg=True):
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
        # self.bbox_mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        # self.class_mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.class_embed = nn.Linear(hidden_dim, num_classes * 1)
        # self.start_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        # self.end_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        self.num_feature_levels = num_feature_levels
        self.input_dim = input_dim
        self.max_input_len = 1024
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.prop_label_enc = nn.Embedding(200 + 1, hidden_dim)
        self.prop_score_enc = nn.Linear(1, hidden_dim)
        self.prop_box_enc = nn.Linear(2, hidden_dim)
        self.prop_level_enc = nn.Embedding(6, hidden_dim)
        self.query_type_enc = nn.Embedding(2, hidden_dim)
        if not use_dab:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        else:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)  # for indicator
            self.refpoint_embed = nn.Embedding(num_queries, 2)
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
        self.class_embed.bias.data = torch.ones(num_classes * 1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.start_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.start_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.end_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.end_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[-1:], -2.0)
            # self.mask_embed = _get_clones(self.bbox_embed, num_pred)
            # self.start_embed = _get_clones(self.start_embed, num_pred)
            # self.end_embed = _get_clones(self.end_embed, num_pred)
            # nn.init.constant_(self.start_embed[0].layers[-1].bias.data[-1:], -2.0)
            # nn.init.constant_(self.end_embed[0].layers[-1].bias.data[-1:], -2.0)
            # hack implementation for iterative bounding box refinement
            # self.transformer.decoder.mask_embed = self.mask_embed
            self.transformer.decoder.bbox_embed = self.bbox_embed
            self.transformer.decoder.class_embed = self.class_embed
            # self.transformer.decoder.start_embed = self.start_embed
            # self.transformer.decoder.end_embed = self.end_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[-1:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            # self.bbox_mask_embed = nn.ModuleList([self.bbox_mask_embed for _ in range(num_pred)])
            # self.class_mask_embed = nn.ModuleList([self.class_mask_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        self.with_act_reg = with_act_reg
        if with_act_reg:
            # RoIAlign params
            self.roi_size = 16
            self.roi_scale = 0
            self.roi_extractor = ROIAlign(self.roi_size, self.roi_scale)
            self.actionness_pred = nn.Sequential(
                nn.Linear(self.num_feature_levels * self.roi_size * hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, features, proposals, targets=None):
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
        # features = list()
        # backbone_features = self.backbone([samples[:, :, ::8], samples])
        # for l, feat in enumerate(backbone_features):
        #     feat = feat.mean(dim=(3, 4))
        #     features.append(feat.unsqueeze(-1))

        raw_pos_1d = self.pos_1d_embeds.repeat(features[0].size(0), 1, 1)
        raw_pos_2d = self.pos_2d_embeds.repeat(features[0].size(0), 1, 1, 1)

        srcs = []
        pos_1d = []
        pos_2d = []
        points = []
        scales = []
        anchors = []
        for l, feat in enumerate(features):
            src = self.input_proj[l](feat)
            n, c, t = src.shape
            # this_max_len = self.max_input_len // (2 ** l)
            # if t > this_max_len:
            #     src = F.interpolate(src, size=this_max_len, mode="linear")
            #     t = this_max_len
            src = src.unsqueeze(-1)
            pos_1d_l = F.interpolate(raw_pos_1d, size=t, mode="linear")
            pos_2d_l = F.interpolate(raw_pos_2d, size=(t, t), mode="bilinear")
            pos_1d.append(pos_1d_l)
            pos_2d.append(pos_2d_l)
            srcs.append(src)
            this_points = torch.linspace(0.5, t - 0.5, t, dtype=torch.float32, device=src.device) / t
            points.append(this_points)
            this_scales = torch.ones_like(this_points) * (1.0 / (2 ** (len(features) - l - 1)))
            scales.append(this_scales)
            this_anchors = torch.stack((torch.clamp(this_points - this_scales / 8.0, 0.0, 1.0),
                                          torch.clamp(this_points + this_scales / 8.0, 0.0, 1.0)), dim=-1)
            anchors.append(this_anchors)

        # box_srcs = []
        # box_pos_1d = []
        # box_pos_2d = []
        # for l, feat in enumerate(proposals):
        #     prop_boxes = feat[..., 1:3]
        #     prop_labels = feat[..., 0]
        #     prop_scores = feat[..., -1].unsqueeze(-1)
        #     prop_box_embeds = self.prop_box_enc(prop_boxes)
        #     prop_label_embeds = self.prop_label_enc(torch.zeros_like(prop_labels.long()))
        #     prop_score_embeds = self.prop_score_enc(prop_scores)
        #     prop_level_embeds = self.prop_level_enc.weight[l].view(1, 1, -1)
        #     # box_src = (prop_label_embeds + prop_score_embeds + prop_level_embeds).permute(0, 2, 1)
        #     box_src = (prop_box_embeds + prop_label_embeds + prop_score_embeds + prop_level_embeds).permute(0, 2, 1)
        #     n, c, t = box_src.shape
        #     box_src = box_src.unsqueeze(-1)
        #     # this_max_len = self.max_input_len // (2 ** l)
        #     # if t > this_max_len:
        #     #     box_src = F.interpolate(box_src, size=this_max_len, mode="linear")
        #     #     t = this_max_len
        #     pos_1d_l = F.interpolate(raw_pos_1d, size=t, mode="linear")
        #     # box_src = box_src + pos_1d_l
        #     pos_2d_l = F.interpolate(raw_pos_2d, size=(t, t), mode="bilinear")
        #     box_pos_1d.append(pos_1d_l)
        #     box_pos_2d.append(pos_2d_l)
        #     box_srcs.append(box_src)

        box_srcs = srcs
        box_pos_1d = pos_1d
        box_pos_2d = pos_2d

        if self.use_dab:
            if self.num_patterns == 0:
                tgt_all_embed = self.tgt_embed.weight  # nq, 256
                refanchor = self.refpoint_embed.weight  # nq, 4
            else:
                # multi patterns is not used in this version
                assert NotImplementedError
        else:
            assert NotImplementedError

        points = torch.cat(points, dim=0)[None, :, None].repeat(features[0].size(0), 1, 1)
        scales = torch.cat(scales, dim=0)[None, :, None].repeat(features[0].size(0), 1, 1)
        anchors = torch.cat(anchors, dim=0)[None, :].repeat(features[0].size(0), 1, 1)
        refpoint_embed = torch.cat((anchors, points, scales), dim=-1)

        input_query_label = self.tgt_embed.weight.unsqueeze(0).repeat(features[0].size(0), 1, 1)
        # input_query_label = input_query_label + self.query_type_enc.weight[0].view(1, 1, -1)
        input_query_bbox = self.refpoint_embed.weight.unsqueeze(0).repeat(features[0].size(0), 1, 1)
        # input_query_bbox = refpoint_embed
        # points = inverse_sigmoid(torch.cat(points, dim=0)[None, :, None].repeat(features[0].size(0), 1, 1))
        # scales = inverse_sigmoid(torch.cat(scales, dim=0)[None, :, None].repeat(features[0].size(0), 1, 1))
        # input_query_bbox = torch.cat((input_query_bbox, points, scales), dim=-1)

        # proposals = torch.cat(proposals, dim=1)
        # prop_labels = proposals[..., 0]
        # prop_scores = proposals[..., -1].unsqueeze(-1)
        # prop_label_embeds = self.query_label_enc(torch.zeros_like(prop_labels.long()))
        # prop_score_embeds = self.query_score_enc(prop_scores)
        # prop_query_label = prop_label_embeds + prop_score_embeds
        # prop_query_label = torch.cat(box_srcs, dim=-1).permute(0, 2, 1)
        # prop_query_label = prop_query_label + self.query_type_enc.weight[1].view(1, 1, -1)
        # prop_query_bbox = torch.cat([proposals[..., 1:-1],
        #                               ((proposals[..., 1] + proposals[..., 2]) / 2.0).unsqueeze(-1),
        #                               (proposals[..., 2] - proposals[..., 1]).unsqueeze(-1)], dim=-1)
        # prop_query_bbox = inverse_sigmoid(prop_query_bbox)

        # input_query_label = torch.cat((input_query_label, prop_query_label), dim=1)
        # input_query_bbox = torch.cat((input_query_bbox, prop_query_bbox), dim=1)

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

            # input_query_label = self.tgt_embed.weight.unsqueeze(0).repeat(features.size(0), 1, 1)
            # input_query_bbox = self.refpoint_embed.weight.unsqueeze(0).repeat(features.size(0), 1, 1)

        # query_embeds = torch.cat((input_query_label, input_query_bbox), dim=2)

        proposals = torch.cat(proposals, dim=1)
        prop_query_label = self.prop_label_enc(proposals[..., 0].long())
        prop_query_label = prop_query_label + self.prop_score_enc(proposals[..., -1].unsqueeze(-1))
        # prop_query_bbox = torch.cat([proposals[..., 1:-1],
        #                              ((proposals[..., 1] + proposals[..., 2]) / 2.0).unsqueeze(-1),
        #                              (proposals[..., 2] - proposals[..., 1]).unsqueeze(-1)], dim=-1)
        # points = torch.cat(points, dim=0)[None, :, None].repeat(features[0].size(0), 1, 1)
        # scales = torch.cat(scales, dim=0)[None, :, None].repeat(features[0].size(0), 1, 1)
        # prop_query_bbox = torch.cat([proposals[..., 1:-1], points, scales], dim=-1)
        # prop_query_bbox = torch.stack((inverse_sigmoid(proposals[..., 1:-1].flatten(1)),
        #                                input_query_bbox.flatten(1)), dim=-1)
        prop_query_bbox = torch.stack((proposals[..., 1:-1].flatten(1),
                                       torch.cat(((proposals[..., 1] - proposals[..., 0] + 1.0) / 2.0,
                                                  (proposals[..., 0] - proposals[..., 1] + 1.0) / 2.0), dim=-1)),
                                       dim=-1)
        prop_query_label = prop_query_label.repeat(1, 2, 1)
        # prop_query_bbox = torch.cat([proposals[..., 1:-1]], dim=-1)
        prop_query_bbox = inverse_sigmoid(prop_query_bbox)
        prop_query_embeds = torch.cat((prop_query_label, prop_query_bbox), dim=2)
        # query_embeds = torch.cat((query_embeds, prop_query_embeds), dim=1)
        query_embeds = prop_query_embeds

        hs, init_reference, inter_references, memory, _ = \
            self.transformer(srcs, box_srcs, pos_1d, pos_2d, box_pos_1d, box_pos_2d,
                             query_embeds, attn_mask, self.label_enc)
        # hs, init_reference, inter_references, memory = self.transformer(srcs, pos_1d, query_embeds)

        # In case num object=0
        # hs[0] += self.label_enc.weight[0, 0] * 0.0

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
        # for lvl in range(hs[0].shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # # N, Q, C
            # outputs_bbox_mask = self.bbox_mask_embed[lvl](hs[lvl])
            # outputs_class_mask = self.class_mask_embed[lvl](hs[lvl])
            # # N, Q, T
            # outputs_bbox_mask = torch.bmm(outputs_bbox_mask, memory.permute(0, 2, 1)).softmax(-1)
            # outputs_class_mask = torch.bmm(outputs_class_mask, memory.permute(0, 2, 1)).softmax(-1)
            #
            # F_mask = outputs_class_mask
            # B_mask = outputs_bbox_mask
            #
            # F_hs = torch.bmm(F_mask, memory)
            # B_hs = torch.bmm(B_mask, memory)
            #
            # outputs_coord = self.bbox_embed[lvl](B_hs).sigmoid()
            # outputs_class = self.class_embed[lvl](F_hs)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                # tmp += reference
                tmp += reference[..., :2]
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            # outputs_coord = tmp.sigmoid()
            outputs_coord = torch.stack((torch.minimum(tmp[..., 0].sigmoid(), tmp[..., 0].sigmoid() + tmp[..., 1].tanh()),
                                         torch.maximum(tmp[..., 0].sigmoid(), tmp[..., 0].sigmoid() + tmp[..., 1].tanh())),
                                        dim=-1)
            outputs_coord = torch.clamp(outputs_coord, 0.0, 1.0)

            # if lvl == 0:
            #     reference = init_reference
            # else:
            #     reference = inter_references[lvl - 1]
            # reference = inverse_sigmoid(reference)
            # start_tmp = self.start_embed[lvl](hs[1][lvl])
            # end_tmp = self.end_embed[lvl](hs[2][lvl])
            # start_tmp += reference[..., 0][..., None]
            # end_tmp += reference[..., 1][..., None]
            # outputs_coord = torch.cat([start_tmp.sigmoid(), end_tmp.sigmoid()], dim=-1)

            outputs_class = self.class_embed[lvl](hs[lvl])
            # outputs_class = self.class_embed[lvl](hs[0][lvl])
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # dn post process
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord = \
                cdn_post_process(outputs_class, outputs_coord, dn_meta, self.aux_loss, self._set_aux_loss)


        if not self.with_act_reg:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        else:
            # perform RoIAlign
            roi_features = list()
            N, Q = outputs_coord[-1].shape[:2]
            # prev_query_start_index = 0
            for l_i, this_memory in enumerate(memory):
                origin_feat = this_memory.permute(0, 2, 1)
                # this_coord = outputs_coord[-1][:, prev_query_start_index:prev_query_start_index + Q]
                # prev_query_start_index += Q

                rois = self._to_roi_align_format(outputs_coord[-1], origin_feat.shape[2], scale_factor=1.5)
                this_roi_features = self.roi_extractor(origin_feat, rois)
                this_roi_features = this_roi_features.view((N, Q, -1))
                roi_features.append(this_roi_features)
            roi_features = torch.concat(roi_features, dim=-1)
            pred_actionness = self.actionness_pred(roi_features)

            last_layer_cls = outputs_class[-1]
            last_layer_reg = outputs_coord[-1]

            out = {'pred_logits': last_layer_cls,
                   'pred_boxes': last_layer_reg, 'pred_actionness': pred_actionness}

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

    def _to_roi_align_format(self, rois, T, scale_factor=1):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = ((rois[..., 0] + rois[..., 1]) / 2.0).unsqueeze(-1)
        rois_size = rois[..., -1][..., None] * scale_factor
        rois_abs = torch.cat(
            (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * T
        # expand the RoIs
        rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
        batch_ind = batch_ind.repeat(1, N, 1)
        rois_abs = torch.cat((batch_ind, rois_abs), dim=2)
        # NOTE: stop gradient here to stablize training
        return rois_abs.view((B*N, 3)).detach()


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

        # boxes = outputs['pred_boxes'].detach().cpu()
        # scores, labels = torch.max(src_logits.detach().cpu(), dim=-1)
        #
        # src_segments = outputs['pred_boxes'].view((-1, 2))
        # target_segments = torch.cat([t['boxes'] for t in targets], dim=0)
        #
        # iou_mat = segment_ops.segment_iou(src_segments, target_segments[..., :2])
        # gt_iou = iou_mat.max(dim=1)[0]
        # scores = gt_iou.view(src_logits.shape[:2]).detach().cpu()

        # valid_masks = list()
        # for n_i, (b, l, s) in enumerate(zip(boxes, labels, scores)):
        #     # 2: batched nms (only implemented on CPU)
        #     nms_indices = dynamic_nms(
        #         b.contiguous(), s.contiguous(), l.contiguous(),
        #         iou_threshold=0.70,
        #         min_score=0.0,
        #         max_seg_num=1000,
        #         use_soft_nms=False,
        #         multiclass=False,
        #         sigma=0.75,
        #         voting_thresh=0.0)
        #     valid_mask = torch.isin(torch.arange(len(b)), nms_indices).float()
        #     valid_masks.append(valid_mask)
        # # N, Q, 1
        # valid_masks = torch.stack(valid_masks, dim=0).cuda()

        target_classes = torch.full(src_logits.shape[:2], self.num_classes * 1, dtype=torch.int64, device=src_logits.device)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        for i, this_indices in enumerate(indices):
            idx = self._get_src_permutation_idx(this_indices)
            if layer is None:
                target_classes_o = \
                    torch.cat([t["labels"][J] for t, (_, J) in zip(targets, this_indices) if len(t["labels"])])
            else:
                target_classes_o = torch.cat(
                    [t["labels"].repeat(2 ** (5 - layer))[J] for t, (_, J) in zip(targets, this_indices)])

            target_classes[idx] = target_classes_o
            # target_classes[idx] = target_classes_o + i * self.num_classes
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1.0)
            target_classes_onehot[idx] = target_classes_onehot[idx] * (1.0 - 0.2 * i)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        if layer is not None:
            num_boxes = num_boxes * (2 ** (5 - layer))
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2,
        #                              mask=valid_masks)

        losses = {'loss_ce': loss_ce}

        if log and False:
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

        losses = {}
        losses["loss_xy"] = 0.0
        # losses["loss_hw"] = 0.0
        losses["loss_bbox"] = 0.0
        losses["loss_giou"] = 0.0
        for i, this_indices in enumerate(indices):
            idx = self._get_src_permutation_idx(this_indices)
            src_boxes = outputs['pred_boxes'][idx]
            if layer is None:
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, this_indices) if len(t["boxes"])],
                                         dim=0)
            else:
                target_boxes = torch.cat([t['boxes'].repeat(2 ** (5 - layer), 1)[i]
                                          for t, (_, i) in zip(targets, this_indices)], dim=0)

            loss_bbox = F.l1_loss(src_boxes, target_boxes[..., :2], reduction='none')
            if layer is not None:
                num_boxes = num_boxes * (2 ** (5 - layer))
            losses["loss_bbox"] += (loss_bbox.sum() / num_boxes) * (1.0 - 0.2 * i)

            # loss_giou = ((1 - torch.diag(segment_ops.segment_iou(
            #     segment_ops.segment_cw_to_t1t2(src_boxes[..., 2:]),
            #     segment_ops.segment_cw_to_t1t2(target_boxes[..., 2:])))) +
            #              (1 - torch.diag(segment_ops.segment_iou(
            #                  src_boxes[..., :2], target_boxes[..., :2])))) / 2
            loss_giou = 1 - torch.diag(segment_ops.segment_iou(src_boxes[..., :2], target_boxes[..., :2]))
            losses["loss_giou"] += (loss_giou.sum() / num_boxes) * (1.0 - 0.2 * i)

            # calculate the x,y and h,w loss
            with torch.no_grad():
                losses["loss_xy"] += (loss_bbox[..., :2].sum() / num_boxes) * (1.0 - 0.2 * i)
                # losses["loss_hw"] += (loss_bbox[..., 2:].sum() / num_boxes) * (1.0 - 0.2 * i)

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

    def loss_actionness(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_boxes' in outputs
        assert 'pred_actionness' in outputs
        src_segments = outputs['pred_boxes'].view((-1, 2))
        target_segments = torch.cat([t['boxes'] for t in targets], dim=0)

        losses = {}

        iou_mat = segment_ops.segment_iou(src_segments, target_segments[..., :2])

        gt_iou = iou_mat.max(dim=1)[0]
        pred_actionness = outputs['pred_actionness']
        loss_actionness = F.l1_loss(pred_actionness.view(-1), gt_iou.view(-1).detach())

        losses['loss_actionness'] = loss_actionness
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
            'actionness': self.loss_actionness,
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
                    if loss == "actionness":
                        continue
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
        dn_labelbook_size=dn_labelbook_size,
        with_act_reg=args["with_act_reg"]
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

    if args["with_act_reg"]:
        weight_dict['loss_actionness'] = args["weight_loss_act"]
        losses.append('actionness')

    criterion = SetCriterion_DINO(num_classes, matcher=matcher, weight_dict=weight_dict,
                                  focal_alpha=args["focal_alpha"], losses=losses)
    criterion.to(device)

    return model, criterion

# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
TadTR model and criterion classes.
"""
import math
import copy

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import segment_ops
from ..utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .matcher import build_matcher
from .position_encoding import build_position_encoding
from ..utils.custom_loss import sigmoid_focal_loss
from .dab_transformer import build_transformer

# if not cfg.disable_cuda:
#     from models.ops.roi_align import ROIAlign


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == 'gn':
        assert num_groups is not None, 'num_groups must be specified'
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class DABDETR(nn.Module):
    """ This is the TadTR module that performs temporal action detection """

    def __init__(self, position_embedding, transformer, num_classes, num_queries,
                 aux_loss=True, with_segment_refine=True, with_act_reg=False,
                 random_refpoints_xy=True, query_dim=2):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See deformable_transformer.py
            num_classes: number of action classes
            num_queries: number of action queries, ie detection slot. This is the maximal number of actions
                         TadTR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_segment_refine: iterative segment refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.speed_embed = nn.Linear(1, hidden_dim)

        self.query_dim = query_dim

        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :1].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :1] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :1])
            self.refpoint_embed.weight.data[:, :1].requires_grad = False

            # self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            # self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            # self.refpoint_embed.weight.data[:, :2].requires_grad = False

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2048, hidden_dim, kernel_size=1),
                # nn.Conv1d(512, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])
        # self.backbone = backbone
        self.position_embedding = position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine
        self.with_act_reg = with_act_reg

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_segment_refine:
            # hack implementation for segment refinement
            self.transformer.decoder.segment_embed = self.segment_embed

        if with_act_reg:
            # RoIAlign params
            self.roi_size = 16
            self.roi_scale = 0
            self.roi_extractor = ROIAlign(self.roi_size, self.roi_scale)
            self.actionness_pred = nn.Sequential(
                nn.Linear(self.roi_size * hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def _to_roi_align_format(self, rois, T, scale_factor=1):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1]
        rois_size = rois[:, :, 1:2] * scale_factor
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

    def forward(self, samples, speeds=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            or a tuple of tensors and mask

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (n, c, t)

        pos = self.position_embedding(samples)
        src, mask = samples.tensors, samples.mask
        src = self.input_proj[0](src)
        if speeds is not None:
            # N, C, 1
            speed_embed = self.speed_embed(speeds.unsqueeze(-1)).unsqueeze(-1)
            src = src + speed_embed

        embedweight = self.refpoint_embed.weight
        hs, reference, memory, Q_weights, K_weights, C_weights = \
            self.transformer(src, mask, embedweight, pos)

        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.segment_embed(hs)
        tmp[..., :self.query_dim] += reference_before_sigmoid
        outputs_coord = tmp.sigmoid()
        # outputs_coord = segment_ops.segment_t1t2_to_cw(tmp.sigmoid())

        outputs_class = self.class_embed(hs)
        # outputs_speed = self.speed_embed(torch.mean(memory.permute(1, 0, 2), dim=1)).sigmoid()

        # normalized_Q_weights = Q_weights[0]
        # for i in range(len(Q_weights) - 1):
        #     normalized_Q_weights = torch.sqrt(torch.bmm(normalized_Q_weights, Q_weights[i + 1].transpose(1, 2)))
        #     normalized_Q_weights = normalized_Q_weights / torch.sum(normalized_Q_weights, dim=-1, keepdim=True)
        # normalized_K_weights = K_weights[0]
        # for i in range(len(K_weights) - 1):
        #     normalized_K_weights = torch.sqrt(torch.bmm(normalized_K_weights, K_weights[i + 1].transpose(1, 2)))
        #     normalized_K_weights = normalized_K_weights / torch.sum(normalized_K_weights, dim=-1, keepdim=True)

        # print(torch.argsort(-normalized_Q_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())
        # print(torch.max(normalized_Q_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())
        # print(torch.argsort(-normalized_K_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())
        # print(torch.max(normalized_K_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())

        # out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
        #        'Q_weights': Q_weights[-1], 'K_weights': K_weights[-1], 'C_weights': C_weights[-1]}
        # out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
        #        'Q_weights': torch.mean(Q_weights, dim=0), 'K_weights': torch.mean(K_weights, dim=0),
        #        'C_weights': C_weights[-1]}
        # out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
        #        'Q_weights': Q_weights, 'K_weights': K_weights, 'C_weights': C_weights[-1]}
        # out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
        #        'Q_weights': torch.mean(Q_weights, dim=0), 'K_weights': torch.mean(K_weights, dim=0),
        #        'C_weights': torch.mean(C_weights, dim=0)}
        # out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
        #        'Q_weights': normalized_Q_weights, 'K_weights': normalized_K_weights, 'C_weights': C_weights[-1]}
        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
               'Q_weights': Q_weights, 'K_weights': K_weights, 'C_weights': C_weights}

        if self.with_act_reg:
            # perform RoIAlign
            B, N = outputs_coord[-1].shape[:2]
            origin_feat = memory.permute(1, 2, 0)

            rois = self._to_roi_align_format(
                outputs_coord[-1], origin_feat.shape[2], scale_factor=1.5)
            roi_features = self.roi_extractor(origin_feat, rois)
            roi_features = roi_features.view((B, N, -1))
            pred_actionness = self.actionness_pred(roi_features)

            last_layer_cls = outputs_class[-1]
            last_layer_reg = outputs_coord[-1]

            out['pred_actionness'] = pred_actionness

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        # return [{'pred_logits': a, 'pred_segments': b, 'Q_weights': c, 'K_weights': d, 'C_weights': e}
        #         for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
        #                                  Q_weights[:-1], K_weights[:-1], C_weights[:-1])]
        # return [{'pred_logits': a, 'pred_segments': b, 'Q_weights': c, 'K_weights': d, 'C_weights': e}
        #         for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1], Q_weights[:-1], K_weights[:-1],
        #                                  [C_weights[-1]] * len(outputs_class[:-1]))]


class SetCriterion(nn.Module):
    """ This class computes the loss for TadTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
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

    def loss_labels(self, outputs, targets, indices, num_segments, log=True, layer=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        for this_indices in indices:
            idx = self._get_src_permutation_idx(this_indices)
            src_segments = outputs['pred_segments'][idx]
            if layer is None:
                target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, this_indices)], dim=0)
            else:
                if layer <= 1:
                    target_segments = torch.cat([t['segments'].repeat(10, 1)[i] for t, (_, i) in zip(targets, this_indices)], dim=0)
                else:
                    target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, this_indices)], dim=0)
            IoUs = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments),
                                           segment_ops.segment_cw_to_t1t2(target_segments))
            IoUs = torch.diag(IoUs).detach()

            if layer is None:
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, this_indices)])
            else:
                if layer <= 1:
                    target_classes_o = torch.cat([t["labels"].repeat(10)[J] for t, (_, J) in zip(targets, this_indices)])
                else:
                    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, this_indices)])
            target_classes[idx] = target_classes_o
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            # target_classes_onehot[idx] = target_classes_onehot[idx]
            target_classes_onehot[idx] = target_classes_onehot[idx] * IoUs.unsqueeze(-1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]  # nq
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

            # N, Q
            probs = torch.max(src_logits.sigmoid(), dim=-1)[0]
            top_k_indices = torch.argsort(-probs, dim=-1)
            top_1_indices = top_k_indices[..., 0]
            top_2_indices = top_k_indices[..., 1]
            # score_gap = torch.mean(probs[top_1_indices] - probs[top_2_indices], dim=0)
            score_gap = torch.mean(probs[torch.arange(len(top_1_indices)),
                                         top_1_indices[torch.arange(len(top_1_indices))]] -
                                   probs[torch.arange(len(top_2_indices)),
                                         top_2_indices[torch.arange(len(top_2_indices))]], dim=0)

            losses['score_gap'] = score_gap

        return losses

    def loss_segments(self, outputs, targets, indices, num_segments, layer=None):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        indices = indices[0]
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        if layer is None:
            target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        else:
            if layer <= 1:
                target_segments = torch.cat([t['segments'].repeat(10, 1)[i] for t, (_, i) in zip(targets, indices)], dim=0)
            else:
                target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_segment = F.l1_loss(src_segments, target_segments, reduction='none')

        losses = {}
        losses['loss_segments'] = loss_segment.sum() / num_segments

        loss_iou = 1 - torch.diag(segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_segments),
            segment_ops.segment_cw_to_t1t2(target_segments)))
        losses['loss_iou'] = loss_iou.sum() / num_segments
        return losses

    def loss_actionness(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        assert 'pred_actionness' in outputs
        src_segments = outputs['pred_segments'].view((-1, 2))
        target_segments = torch.cat([t['segments'] for t in targets], dim=0)

        losses = {}

        iou_mat = segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_segments),
            segment_ops.segment_cw_to_t1t2(target_segments))

        gt_iou = iou_mat.max(dim=1)[0]
        pred_actionness = outputs['pred_actionness']
        loss_actionness = F.l1_loss(pred_actionness.view(-1), gt_iou.view(-1).detach())   

        losses['loss_iou'] = loss_actionness
        return losses

    def loss_QQ(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'Q_weights' in outputs
        assert 'C_weights' in outputs
        assert 'pred_segments' in outputs

        # IoUs = list()
        # for n_i in range(len(targets)):
        #     src_segments = outputs['pred_segments'][n_i]
        #     tgt_segments = targets[n_i]['segments']
        #     # this_IoUs = list()
        #     # for t_i in range(len(tgt_segments)):
        #     #     this_IoU = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(tgt_segments[t_i]),
        #     #                                        segment_ops.segment_cw_to_t1t2(src_segments))
        #     #     this_IoU = torch.max(this_IoU, dim=1)[0]
        #     #     this_IoUs.append(this_IoU)
        #     this_IoU = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(tgt_segments),
        #                                        segment_ops.segment_cw_to_t1t2(src_segments))
        #     this_IoU = torch.max(this_IoU, dim=1)[0]
        #     this_IoU = torch.mean(this_IoU, dim=0)
        #     IoUs.append(this_IoU)
        #     print(this_IoU)
        # IoUs = torch.stack(IoUs).detach()
        # IoU_weight = IoUs / 0.5
        # print(IoU_weight.shape)
        # exit()

        # Q_weights = torch.mean(outputs["Q_weights"], dim=0)
        # Q_weights = outputs["Q_weights"]
        # normalized_Q_weights = Q_weights[0]
        # for i in range(len(Q_weights) - 1):
        #     normalized_Q_weights = torch.sqrt(
        #         torch.bmm(normalized_Q_weights, Q_weights[i + 1].transpose(1, 2)) + 1.0e-7)
        #     normalized_Q_weights = normalized_Q_weights / torch.sum(normalized_Q_weights, dim=-1, keepdim=True)
        # Q_weights = normalized_Q_weights

        Q_weights = outputs["Q_weights"].flatten(0, 1)

        # C_weights = outputs["C_weights"][-1].detach()
        # QQ_weights = torch.bmm(C_weights, C_weights.transpose(1, 2))
        # QQ_weights = torch.sqrt(QQ_weights)
        # target_Q_weights = QQ_weights / torch.sum(QQ_weights, dim=-1, keepdim=True)

        # C_weights = outputs["C_weights"].detach()
        # normalized_QQ_weights = torch.sqrt(torch.bmm(C_weights[0], C_weights[0].transpose(1, 2)) + 1.0e-7)
        # normalized_QQ_weights = normalized_QQ_weights / torch.sum(normalized_QQ_weights, dim=-1, keepdim=True)
        # for i in range(len(C_weights) - 1):
        #     QQ_weights = torch.sqrt(torch.bmm(C_weights[i + 1], C_weights[i + 1].transpose(1, 2)) + 1.0e-7)
        #     QQ_weights = QQ_weights / torch.sum(QQ_weights, dim=-1, keepdim=True)
        #     normalized_QQ_weights = torch.sqrt(torch.bmm(normalized_QQ_weights, QQ_weights) + 1.0e-7)
        #     normalized_QQ_weights = normalized_QQ_weights / torch.sum(normalized_QQ_weights, dim=-1, keepdim=True)
        # target_Q_weights = normalized_QQ_weights

        C_weights = outputs["C_weights"].flatten(0, 1).detach()
        QQ_weights = torch.sqrt(torch.bmm(C_weights, C_weights.transpose(1, 2)) + 1.0e-7)
        target_Q_weights = QQ_weights / torch.sum(QQ_weights, dim=-1, keepdim=True)

        src_QQ = (Q_weights.flatten(0, 1) + 1.0e-7).log()
        tgt_QQ = (target_Q_weights.flatten(0, 1) + 1.0e-7).log()

        losses = {}

        loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1)
        # loss_QQ = loss_QQ * IoU_weight.unsqueeze(-1)
        # loss_QQ = loss_QQ.sum() / loss_QQ
        loss_QQ = loss_QQ.mean()

        losses['loss_QQ'] = loss_QQ
        return losses

    def loss_KK(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'K_weights' in outputs
        assert 'C_weights' in outputs

        K_weights = torch.mean(outputs["K_weights"], dim=0)

        # K_weights = outputs["K_weights"]
        # normalized_K_weights = K_weights[0]
        # for i in range(len(K_weights) - 1):
        #     normalized_K_weights = torch.sqrt(
        #         torch.bmm(normalized_K_weights, K_weights[i + 1].transpose(1, 2)) + 1.0e-7)
        #     normalized_K_weights = normalized_K_weights / torch.sum(normalized_K_weights, dim=-1, keepdim=True)
        # K_weights = normalized_K_weights

        # C_weights = outputs["C_weights"][-1].detach()
        # KK_weights = torch.bmm(C_weights.transpose(1, 2), C_weights)
        # KK_weights = torch.sqrt(KK_weights)
        # target_K_weights = KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)

        # C_weights = outputs["C_weights"].detach()
        # normalized_KK_weights = torch.sqrt(torch.bmm(C_weights[0].transpose(1, 2), C_weights[0]) + 1.0e-7)
        # normalized_KK_weights = normalized_KK_weights / torch.sum(normalized_KK_weights, dim=-1, keepdim=True)
        # for i in range(len(C_weights) - 1):
        #     KK_weights = torch.sqrt(torch.bmm(C_weights[i + 1].transpose(1, 2), C_weights[i + 1]) + 1.0e-7)
        #     KK_weights = KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)
        #     normalized_KK_weights = torch.sqrt(torch.bmm(normalized_KK_weights, KK_weights) + 1.0e-7)
        #     normalized_KK_weights = normalized_KK_weights / torch.sum(normalized_KK_weights, dim=-1, keepdim=True)
        # target_K_weights = normalized_KK_weights

        # print(torch.argsort(-target_K_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())
        # print(torch.max(target_K_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())
        # print(torch.max(C_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())
        # print(target_K_weights[0, 0].detach().cpu().numpy())
        # print((torch.max(C_weights) - torch.max(target_K_weights)).detach().cpu().numpy())

        # NK, K

        C_weights = torch.mean(outputs["C_weights"], dim=0).detach()
        # C_weights = outputs["C_weights"][-1].detach()
        KK_weights = torch.bmm(C_weights.transpose(1, 2), C_weights)
        KK_weights = torch.sqrt(KK_weights + 1.0e-7)
        target_K_weights = KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)

        src_KK = (K_weights.flatten(0, 1) + 1.0e-7).log()
        tgt_KK = (target_K_weights.flatten(0, 1) + 1.0e-7).log()

        losses = {}

        loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1)
        loss_KK = loss_KK.mean()

        losses['loss_KK'] = loss_KK
        return losses

    def loss_speed(self, outputs, targets, indices, num_segments):
        assert 'pred_speeds' in outputs
        src_speeds = outputs['pred_speeds'].squeeze(-1)
        tgt_speeds = torch.stack([t['speeds'] for t in targets], dim=0)

        loss_speed = F.l1_loss(src_speeds, tgt_speeds, reduction='none')

        losses = {}
        losses['loss_speed'] = loss_speed.mean()

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

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'segments': self.loss_segments,
            'actionness': self.loss_actionness,
            "QQ": self.loss_QQ,
            "KK": self.loss_KK,
            "speed": self.loss_speed,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                layer = None
                indices = self.matcher(aux_outputs, targets, layer=layer)
                for loss in self.losses:
                    # we do not compute actionness loss for aux outputs
                    if 'QQ' in loss or 'KK' in loss:
                        continue

                    if 'speed' in loss:
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False

                    if layer is not None and loss in ('labels', 'segments'):
                        kwargs['layer'] = layer
                        if layer <= 1:
                            num_segments = sum(len(t["labels"].repeat(10)) for t in targets)
                        else:
                            num_segments = sum(len(t["labels"]) for t in targets)
                    else:
                        num_segments = sum(len(t["labels"]) for t in targets)
                    num_segments = torch.as_tensor([num_segments], dtype=torch.float,
                                                   device=next(iter(outputs.values())).device)
                    if is_dist_avail_and_initialized():
                        torch.distributed.all_reduce(num_segments)
                    num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        self.indices = indices
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the TADEvaluator"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, fuse_score=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the duration of each video of the batch
        """
        out_logits, out_segments = outputs['pred_logits'], outputs['pred_segments']

        assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 1

        prob = out_logits.sigmoid()   # [bs, nq, C]
        if fuse_score:
            prob *= outputs['pred_actionness']

        segments = segment_ops.segment_cw_to_t1t2(out_segments)   # bs, nq, 2

        if cfg.postproc_rank == 1:     # default
            # sort across different instances, pick top 100 at most
            topk_values, topk_indexes = torch.topk(prob.view(
                out_logits.shape[0], -1), min(cfg.postproc_ins_topk, prob.shape[1]*prob.shape[2]), dim=1)
            scores = topk_values
            topk_segments = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]

            # bs, nq, 2; bs, num, 2
            segments = torch.gather(
                segments, 1, topk_segments.unsqueeze(-1).repeat(1, 1, 2))
            query_ids = topk_segments
        else:
            # pick topk classes for each query
            # pdb.set_trace()
            scores, labels = torch.topk(prob, cfg.postproc_cls_topk, dim=-1)
            scores, labels = scores.flatten(1), labels.flatten(1)
            # (bs, nq, 1, 2)
            segments = segments[:, [
                i//cfg.postproc_cls_topk for i in range(cfg.postproc_cls_topk*segments.shape[1])], :]
            query_ids = (torch.arange(0, cfg.postproc_cls_topk*segments.shape[1], 1, dtype=labels.dtype,
                         device=labels.device) // cfg.postproc_cls_topk)[None, :].repeat(labels.shape[0], 1)

        # from normalized [0, 1] to absolute [0, length] coordinates
        vid_length = target_sizes
        scale_fct = torch.stack([vid_length, vid_length], dim=1)
        segments = segments * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'segments': b, 'query_ids': q}
                   for s, l, b, q in zip(scores, labels, segments, query_ids)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    pos_embed = build_position_encoding(args)
    transformer = build_transformer(args)
    num_classes = 1

    model = DABDETR(
        pos_embed,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        with_segment_refine=True,
        with_act_reg=False)

    matcher = build_matcher(args)
    losses = ['labels', 'segments']

    weight_dict = {
        'loss_ce': args.weight_loss_ce,
        'loss_segments': args.weight_loss_bbox,
        'loss_iou': args.weight_loss_giou}

    # if args.act_reg:
    #     weight_dict['loss_actionness'] = args.act_loss_coef
    #     losses.append('actionness')

    if args.use_KK:
        weight_dict["loss_KK"] = args.weight_loss_KK
        losses.append("KK")

    if args.use_QQ:
        weight_dict["loss_QQ"] = args.weight_loss_QQ
        losses.append("QQ")

    # if True:
    #     weight_dict["loss_speed"] = 1.0
    #     losses.append("speed")

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)

    # postprocessor = PostProcess()
    # return model, criterion, postprocessor

    return model, criterion

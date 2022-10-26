# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
import math, os
import copy
from torch import nn

from util import box_ops, segment_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DABDeformableDETR(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """

    def __init__(self, transformer, num_classes, num_queries, in_channels,
                 pos_1d_embeds, pos_2d_embeds, num_feature_levels,
                 aux_loss=True, with_box_refine=True, two_stage=False,
                 use_dab=True, num_patterns=0, random_refpoints_xy=False):
        """ Initializes the model.
        Parameters:
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
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pos_1d_embeds = pos_1d_embeds
        self.pos_2d_embeds = pos_2d_embeds
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.iou_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.num_feature_levels = num_feature_levels
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        # dn label enc
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # if not two_stage:
        if not use_dab:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        else:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim - 1)  # for indicator
            self.refpoint_embed = nn.Embedding(num_queries, 4)
            if random_refpoints_xy:
                # import ipdb; ipdb.set_trace()
                self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
                self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = num_feature_levels
            input_proj_list = []
            for _ in range(num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim)))

            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_specific_embed.bias.data = torch.ones(num_classes) * bias_value
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
            # self.iou_embed = _get_clones(self.iou_embed, num_pred)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

    def forward(self, features, proposals, dn_args=None):
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
        raw_pos_1d = self.pos_1d_embeds.repeat(features[0].shape[0], 1, 1)
        raw_pos_2d = self.pos_2d_embeds.repeat(features[0].shape[0], 1, 1, 1)

        srcs = []
        pos_1d = []
        pos_2d = []
        for l, feat in enumerate(features):
            src = self.input_proj[l](feat)
            n, c, t, _ = src.shape
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

        # prepare for dn
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dn(dn_args, tgt_all_embed, refanchor, features[0].size(0),
                           self.training, self.num_queries, self.num_classes, self.hidden_dim, self.label_enc)

        if proposals is not None:
            proposal_labels = proposals[..., 0]
            proposal_label_embed = self.label_enc(proposal_labels.long())
            indicator1 = torch.ones_like(proposal_labels).unsqueeze(-1)
            proposal_query_label = torch.cat([proposal_label_embed, indicator1], dim=-1)

            input_query_label = proposal_query_label
            input_query_bbox = torch.cat([proposals[..., 1:-1],
                                          ((proposals[..., 1] + proposals[..., 2]) / 2.0).unsqueeze(-1),
                                          (proposals[..., 2] - proposals[..., 1]).unsqueeze(-1)], dim=-1)

        query_embeds = torch.cat((input_query_label, input_query_bbox), dim=2)

        hs, init_reference, inter_references, (output_proposal, output_video), _ = \
            self.transformer(srcs, pos_1d, pos_2d, query_embeds, attn_mask, self.label_enc)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # dn post process
        outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion_DN(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
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

        empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight = torch.ones(2)
        empty_weight[-1] = 0.1
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes_o_bin = torch.zeros_like(target_classes_o)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # target_classes[idx] = target_classes_o_bin

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # # target_classes_o_bin = torch.zeros_like(target_classes_o)
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        # # target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o
        # # target_classes[idx] = target_classes_o_bin
        #
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        # target_weights = torch.full(src_logits.shape[:2], 0.1, dtype=torch.float32, device=src_logits.device)
        # target_weights[idx] = 1.0
        # src_logits_flat = src_logits.flatten(0, 1)
        # targeted_src_logits = src_logits_flat[torch.arange(len(src_logits_flat)), target_classes.flatten(0, 1)]
        # loss_ce = torch.mean((target_weights.flatten(0, 1) * (1.0 - targeted_src_logits)) ** 2)

        losses = {'loss_ce': loss_ce}

        # assert 'pred_embeds' in outputs
        #
        # src_embeds = outputs['pred_embeds'][idx]
        # target_embeds = torch.cat([t["embeddings"][J] for t, (_, J) in zip(targets, indices)])

        # loss_embed = F.mse_loss(src_embeds, target_embeds, reduction="none")
        # losses['loss_embed'] = loss_embed.sum() / num_boxes

        # loss_embed = 1.0 - F.cosine_similarity(src_embeds, target_embeds)
        # losses['loss_embed'] = loss_embed.sum() / num_boxes

        # assert 'pred_sims' in outputs

        # src_sims = outputs['pred_sims'][idx]
        # target_sims = torch.cat([t['similarity'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # loss_sim = 1.0 - F.cosine_similarity(src_sims, target_sims)
        # losses['loss_similarty'] = loss_sim.sum() / num_boxes

        # loss_sim = F.cross_entropy(src_sims, target_sims.softmax(dim=-1), reduction="none")
        # losses['loss_similarity'] = loss_sim.sum() / num_boxes

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            # losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o_bin)[0]
            # losses['class_error'] = 100 - accuracy(F.softmax(src_sims, dim=-1), target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
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

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes)))
        # losses['loss_giou'] = loss_giou.sum() / num_boxes

        loss_giou = ((1 - torch.diag(segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_boxes[..., 2:]),
            segment_ops.segment_cw_to_t1t2(target_boxes[..., 2:])))) + \
                     (1 - torch.diag(segment_ops.segment_iou(
                         src_boxes[..., :2], target_boxes[..., :2])))) / 2
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_ious(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_boxes' in outputs and 'pred_ious' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        src_ious = outputs['pred_ious']
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        target_ious = torch.zeros_like(src_ious)
        matched_ious = torch.stack((torch.diag(segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_boxes[..., 2:]),
            segment_ops.segment_cw_to_t1t2(target_boxes[..., 2:]))),
                                    torch.diag(segment_ops.segment_iou(
                                        src_boxes[..., :2], target_boxes[..., :2]))), dim=-1)
        target_ious[idx] = matched_ious
        loss_iou = F.l1_loss(src_ious, target_ious, reduction='none')
        losses['loss_iou'] = loss_iou.sum() / num_boxes

        return losses

    def loss_embeddings(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_embeddings' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_embeds = outputs['pred_embeddings'][idx]
        target_embeds = torch.cat([t['embeddings'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_embed = F.cosine_embedding_loss(src_embeds, target_embeds, reduction="none")

        losses = {}
        losses['loss_embed'] = loss_embed.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_encoder_outputs(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # src_segmentation = outputs["pred_segmentation"]
        # target_segmentation = torch.stack([t["masks"] for t in targets], dim=0)
        # loss_segmentation = F.cross_entropy(src_segmentation.permute(0, 2, 1), target_segmentation)
        #
        # src_video = outputs["pred_video"]
        # target_max_class = torch.stack([torch.mode(t["labels"], dim=0)[0] for t in targets], dim=0)
        # loss_video = F.cross_entropy(src_video, target_max_class)
        #
        # losses = {"loss_segmentation": loss_segmentation,
        #           "loss_video": loss_video}

        src_logits = outputs["pred_proposals"][..., 0]
        target_classes_onehot = torch.stack([t["proposals"][..., 0] for t in targets], dim=0)

        valid_flags = target_classes_onehot.flatten(0, 1) > 0.0
        num_positives = torch.sum(valid_flags.float())

        loss_actionness = \
            sigmoid_focal_loss(src_logits, target_classes_onehot, num_positives, alpha=self.focal_alpha, gamma=2) * \
            src_logits.shape[1]

        src_boxes = outputs["pred_proposals"][..., 1:].flatten(0, 1)[valid_flags].sigmoid()
        target_boxes = torch.stack([t["proposals"][..., 1:] for t in targets], dim=0).flatten(0, 1)[valid_flags]

        loss_proposal = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_proposal = loss_proposal.sum() / num_positives

        losses = {"loss_actionness": loss_actionness, "loss_proposal": loss_proposal}

        if outputs["pred_video"] is not None:
            src_video = outputs["pred_video"]
            target_max_class = torch.stack([torch.mode(t["labels"], dim=0)[0] for t in targets], dim=0)
            loss_video = F.cross_entropy(src_video, target_max_class)
            losses["loss_video"] = loss_video

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
            'ious': self.loss_ious
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            if os.environ.get('IPDB_SHILONG_DEBUG') == 'INFO':
                import ipdb;
                ipdb.set_trace()
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # dn loss computation
        aux_num = 0
        if 'aux_outputs' in outputs:
            aux_num = len(outputs['aux_outputs'])
        dn_losses = compute_dn_loss(mask_dict, self.training, aux_num, self.focal_alpha)
        losses.update(dn_losses)

        return losses


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


def build_dn():
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    args = {
        "num_classes": 200,
        "num_queries": 30,
        "hidden_dim": 256,
        "in_channels": 256,
        "num_feature_levels": 7,
        "aux_loss": True,
        "enc_layers": 2,
        "dec_layers": 4,
        "enc_n_points": 4,
        "dec_n_points": 4,
        "eos_coef": 0.1,
        "set_cost_class": 6,
        "set_cost_bbox": 5,
        "set_cost_giou": 2,
        "weight_loss_ce": 2,
        "weight_loss_bbox": 5,
        "weight_loss_giou": 2,
        "nhead": 8,
        "dropout": 0.1,
        "dim_feedforward": 256 * 4,
        "pre_norm": False,
        "scalar": 5,
        "label_noise_scale": 0.2,
        "box_noise_scale": 0.4,
        "num_patterns": 0
    }

    num_classes = args["num_classes"]
    device = "cuda"

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

    model = DABDeformableDETR(
        transformer,
        num_classes=args["num_classes"],
        num_queries=args["num_queries"],
        in_channels=args["in_channels"],
        aux_loss=True,
        pos_1d_embeds=pos_1d_embeds,
        pos_2d_embeds=pos_2d_embeds,
        num_feature_levels=args["num_feature_levels"],
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args["weight_loss_ce"], 'loss_bbox': args["weight_loss_bbox"],
                   'loss_giou': args["weight_loss_giou"], 'loss_similarity': args["weight_loss_similarity"]}
    weight_dict['tgt_loss_ce'] = args["weight_loss_ce"]
    weight_dict['tgt_loss_bbox'] = args["weight_loss_bbox"]
    weight_dict['tgt_loss_giou'] = args["weight_loss_giou"]
    # TODO this is a hack
    if args["aux_loss"]:
        aux_weight_dict = {}
        for i in range(args["dec_layers"] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion_DN(num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)

    return model, criterion

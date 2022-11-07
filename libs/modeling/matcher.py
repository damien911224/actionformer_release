# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..utils.segment_ops import segment_cw_to_t1t2, segment_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, layer=None):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            # out_embeds = outputs["pred_embeds"].flatten(0, 1)
            # out_sims = outputs["pred_sims"].flatten(0, 1).softmax(-1)
            # out_prob = out_prob[..., 0].unsqueeze(-1) * out_embeds
            # out_prob = out_sims
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            # tgt_ids = torch.cat([v["labels"] for v in targets])
            if layer is None:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])
            else:
                tgt_ids = torch.cat([v["labels"].repeat(2 ** (3 - layer)) for v in targets])
                tgt_bbox = torch.cat([v["boxes"].repeat(2 ** (3 - layer), 1) for v in targets])

            # tgt_sims = torch.cat([v["similarity"] for v in targets]) # [num_boxes, num_classes]
            # NB, K = tgt_sims.shape

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - prob[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]
            # cost_class = -torch.matmul(out_embeds, tgt_embeds.transpose(0, 1))
            # cost_class = -(out_prob[:, torch.zeros_like(tgt_ids)] + torch.matmul(out_embeds, tgt_embeds.t()))
            # cost_class = -(torch.matmul(out_embeds, tgt_embeds.t()))
            # cost_class = \
            #     -(out_prob[:, tgt_ids] +
            #       F.cross_entropy(out_prob.unsqueeze(-1).repeat(1, 1, NB),
            #                       tgt_sims.softmax(dim=1).t().unsqueeze(0).repeat(bs * num_queries, 1, 1),
            #                       reduction="none"))
            # cost_class = \
            #     -(out_prob[:, tgt_ids] +
            #       torch.sum(tgt_sims.softmax(dim=1).t().unsqueeze(0).repeat(bs * num_queries, 1, 1) *
            #                 torch.log(out_prob[..., :-1].unsqueeze(-1).repeat(1, 1, NB) + 1.0e-7), dim=1))
            # cost_class = \
            #     -(out_prob[:, 0].unsqueeze(-1) +
            #       torch.sum(tgt_sims.softmax(dim=1).t().unsqueeze(0).repeat(bs * num_queries, 1, 1) *
            #                 torch.log(out_sims.unsqueeze(-1).repeat(1, 1, NB) + 1.0e-7), dim=1))

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            cost_giou = -(segment_iou(segment_cw_to_t1t2(out_bbox[..., 2:]), segment_cw_to_t1t2(tgt_bbox[..., 2:])) +
                          segment_iou(out_bbox[..., :2], tgt_bbox[..., :2])) / 2.0

            # Final cost matrix
            # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            if layer is None:
                sizes = [len(v["boxes"]) for v in targets]
            else:
                sizes = [len(v["boxes"].repeat(2 ** (3 - layer), 1)) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args["set_cost_class"], cost_bbox=args["set_cost_bbox"])
import os
import shutil
import time
import pickle

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from ..utils import batched_nms

################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    # assert len(param_dict.keys() - union_params) == 0, \
    #     "parameters %s were not separated into either decay/no_decay set!" \
    #     % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None, copy_model=True):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        if copy_model:
            self.module = deepcopy(model)
        else:
            self.module = model
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch_zoom_in(
        train_loader,
        detr,
        detr_optimizer,
        detr_scheduler,
        detr_criterion,
        curr_epoch,
        data_type="rgb",
        detr_model_ema=None,
        tb_writer=None,
        print_freq=20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    detr.train()

    # main training loop
    print("\n[Train|Phase 1]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        detr_optimizer.zero_grad(set_to_none=True)

        detr_target_dict = list()
        for b_i in range(len(video_list)):
            batch_dict = dict()
            batch_dict["labels"] = torch.zeros_like(video_list[b_i]["labels"]).cuda()
            boxes = (video_list[b_i]["segments"] * video_list[b_i]["feat_stride"] +
                     0.5 * video_list[b_i]["feat_num_frames"]) / video_list[b_i]["fps"] / video_list[b_i]["duration"]
            boxes = torch.clamp(boxes, 0.0, 1.0)
            batch_dict["boxes"] = torch.cat((boxes,
                                             ((boxes[..., 0] + boxes[..., 1]) / 2.0).unsqueeze(-1),
                                             (boxes[..., 1] - boxes[..., 0]).unsqueeze(-1)), dim=-1).cuda()
            detr_target_dict.append(batch_dict)

        # features = [feat.detach() for feat in backbone_features]
        features = [torch.stack([x["feats"] for x in video_list], dim=0).cuda()]

        detr_predictions = detr(features, detr_target_dict)
        detr_loss_dict = detr_criterion(detr_predictions, detr_target_dict)
        weight_dict = detr_criterion.weight_dict
        detr_loss = sum(detr_loss_dict[k] * weight_dict[k] for k in detr_loss_dict.keys() if k in weight_dict)

        final_loss = detr_loss

        final_loss.backward()
        # gradient cliping (to stabilize training if necessary)
        torch.nn.utils.clip_grad_norm_(detr.parameters(), 0.1)
        detr_optimizer.step()
        detr_scheduler.step()

        if detr_model_ema is not None:
            detr_model_ema.update(detr)

        losses = dict()
        for k, v in detr_loss_dict.items():
            if "dn" not in k and k[-2] != "_":
                losses["detr_" + k] = v

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = detr_scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                # tag_dict = {}
                for key, value in losses.items():
                    # if key != "final_loss":
                    #     tag_dict[key] = value.val
                    tb_writer.add_scalar(
                        "train/" + key,
                        value.item(),
                        global_step
                    )

                # tb_writer.add_scalars(
                #     'train/all_losses',
                #     tag_dict,
                #     global_step
                # )
                # # final loss
                # tb_writer.add_scalar(
                #     'train/final_loss',
                #     losses_tracker['final_loss'].val,
                #     global_step
                # )

            # print to terminal
            block1 = 'Phase 1|{}|Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                data_type.upper(), curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            # block3 = 'Loss {:.2f} ({:.2f})\n'.format(
            #     losses_tracker['rgb_final_loss'].val,
            #     losses_tracker['rgb_final_loss'].avg,
            #     losses_tracker['flow_final_loss'].val,
            #     losses_tracker['flow_final_loss'].avg,
            # )
            block4 = ''
            for key, value in losses_tracker.items():
                block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                    key, value.val, value.avg
                )

            print('\t'.join([block1, block2, block4]))

    # finish up and print
    lr = detr_scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return

def train_one_epoch(
        train_loader,
        backbone,
        backbone_optimizer,
        backbone_scheduler,
        detr,
        detr_optimizer,
        detr_scheduler,
        detr_criterion,
        curr_epoch,
        data_type="rgb",
        backbone_model_ema=None,
        detr_model_ema=None,
        clip_grad_l2norm=-1,
        tb_writer=None,
        print_freq=20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    backbone.train()
    detr.train()

    # main training loop
    print("\n[Train|Phase 1]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        backbone_optimizer.zero_grad(set_to_none=True)
        detr_optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        backbone_losses, backbone_results, backbone_features = backbone(video_list, data_type=data_type, nms=False)
        backbone_loss = backbone_losses["final_loss"]

        detr_target_dict = list()
        for b_i in range(len(video_list)):
            batch_dict = dict()
            batch_dict["labels"] = torch.zeros_like(video_list[b_i]["labels"]).cuda()
            boxes = (video_list[b_i]["segments"] * video_list[b_i]["feat_stride"] +
                     0.5 * video_list[b_i]["feat_num_frames"]) / video_list[b_i]["fps"] / video_list[b_i]["duration"]
            boxes = torch.clamp(boxes, 0.0, 1.0)
            batch_dict["boxes"] = torch.cat((boxes,
                                             ((boxes[..., 0] + boxes[..., 1]) / 2.0).unsqueeze(-1),
                                             (boxes[..., 1] - boxes[..., 0]).unsqueeze(-1)), dim=-1).cuda()
            detr_target_dict.append(batch_dict)

        features = [feat for feat in backbone_features]
        # features = [torch.stack([x["feats"] for x in video_list], dim=0).cuda()]

        labels = list()
        scores = list()
        segments = list()
        for p, x in zip(backbone_results, video_list):
            this_labels = p["labels"].float()
            this_scores = p["scores"].float()
            this_segments = p["segments"] / x["duration"]

            # sorted_indices = torch.argsort(this_scores, descending=True)[:100]
            # this_labels = this_labels[sorted_indices]
            # this_scores = this_scores[sorted_indices]
            # this_segments = this_segments[sorted_indices]

            # if len(this_labels) < 100:
            #     this_labels = F.pad(this_labels, (0, 100 - len(this_labels)))
            #     this_scores = F.pad(this_scores, (0, 100 - len(this_scores)))
            #     this_segments = F.pad(this_segments, (0, 0, 0, 100 - len(this_segments)))

            labels.append(this_labels)
            scores.append(this_scores)
            segments.append(this_segments)
        labels = torch.stack(labels, dim=0)
        scores = torch.stack(scores, dim=0)
        segments = torch.stack(segments, dim=0)
        proposals = torch.cat((labels.unsqueeze(-1), segments, scores.unsqueeze(-1)), dim=-1).cuda()

        start_index = 0
        pyramidal_proposals = list()
        for feat in backbone_features:
            this_len = feat.size(2)
            this_proposals = proposals[:, start_index:start_index + this_len]
            pyramidal_proposals.append(this_proposals)
            start_index += this_len

        # detr_predictions = detr(features, proposals, detr_target_dict)
        detr_predictions = detr(features, pyramidal_proposals, detr_target_dict)
        detr_loss_dict = detr_criterion(detr_predictions, detr_target_dict)
        weight_dict = detr_criterion.weight_dict
        detr_loss = sum(detr_loss_dict[k] * weight_dict[k] for k in detr_loss_dict.keys() if k in weight_dict)

        # final_loss = backbone_loss * 0.1 + detr_loss
        final_loss = detr_loss
        # final_loss = backbone_loss

        final_loss.backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), clip_grad_l2norm)
        torch.nn.utils.clip_grad_norm_(detr.parameters(), 0.1)
        # backbone_optimizer.step()
        detr_optimizer.step()
        # backbone_scheduler.step()
        detr_scheduler.step()

        if backbone_model_ema is not None:
            backbone_model_ema.update(backbone)
        if detr_model_ema is not None:
            detr_model_ema.update(detr)

        losses = dict()
        for k, v in backbone_losses.items():
            losses["backbone_" + k] = v

        for k, v in detr_loss_dict.items():
            if "dn" not in k and k[-2] != "_":
                losses["detr_" + k] = v

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = backbone_scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                # tag_dict = {}
                for key, value in losses.items():
                    # if key != "final_loss":
                    #     tag_dict[key] = value.val
                    tb_writer.add_scalar(
                        "train/" + key,
                        value.item(),
                        global_step
                    )

                # tb_writer.add_scalars(
                #     'train/all_losses',
                #     tag_dict,
                #     global_step
                # )
                # # final loss
                # tb_writer.add_scalar(
                #     'train/final_loss',
                #     losses_tracker['final_loss'].val,
                #     global_step
                # )

            # print to terminal
            block1 = 'Phase 1|{}|Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                data_type.upper(), curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            # block3 = 'Loss {:.2f} ({:.2f})\n'.format(
            #     losses_tracker['rgb_final_loss'].val,
            #     losses_tracker['rgb_final_loss'].avg,
            #     losses_tracker['flow_final_loss'].val,
            #     losses_tracker['flow_final_loss'].avg,
            # )
            block4 = ''
            for key, value in losses_tracker.items():
                block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                    key, value.val, value.avg
                )

            print('\t'.join([block1, block2, block4]))

    # finish up and print
    lr = backbone_scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return

def train_one_epoch_phase_1(
        train_loader,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        data_type="rgb",
        model_ema=None,
        clip_grad_l2norm=-1,
        tb_writer=None,
        print_freq=20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train|Phase 1]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses, _, _ = model(video_list, data_type=data_type)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                # tag_dict = {}
                for key, value in losses.items():
                    # if key != "final_loss":
                    #     tag_dict[key] = value.val
                    tb_writer.add_scalar(
                        "train/" + key,
                        value.item(),
                        global_step
                    )

                # tb_writer.add_scalars(
                #     'train/all_losses',
                #     tag_dict,
                #     global_step
                # )
                # # final loss
                # tb_writer.add_scalar(
                #     'train/final_loss',
                #     losses_tracker['final_loss'].val,
                #     global_step
                # )

            # print to terminal
            block1 = 'Phase 1|{}|Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                data_type.upper(), curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            # block3 = 'Loss {:.2f} ({:.2f})\n'.format(
            #     losses_tracker['rgb_final_loss'].val,
            #     losses_tracker['rgb_final_loss'].avg,
            #     losses_tracker['flow_final_loss'].val,
            #     losses_tracker['flow_final_loss'].avg,
            # )
            block4 = ''
            for key, value in losses_tracker.items():
                if "final_loss" in key:
                    block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return

def train_one_epoch_phase_2(
        train_loader,
        detr,
        detr_model_ema,
        criterion,
        optimizer,
        scheduler,
        data_types,
        proposal_models,
        curr_epoch,
        tb_writer=None,
        print_freq=20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    detr.train()
    for proposal_model in proposal_models:
        proposal_model.train()

    # main training loop
    print("\n[Train|Phase 2]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        proposals = list()
        backbone_features = list()
        with torch.no_grad():
            for m_i, proposal_model in enumerate(proposal_models):
                backbone_losses, results, this_backbone_features = proposal_model(video_list, data_type=data_types[m_i])
                backbone_features.extend(this_backbone_features)
                labels = list()
                scores = list()
                segments = list()
                for p, x in zip(results, video_list):
                    this_labels = p["labels"].float()
                    this_scores = p["scores"].float()
                    this_segments = p["segments"] / x["duration"]
                    # if len(this_labels) < 378:
                    #     this_labels = F.pad(this_labels, (0, 378 - len(this_labels)))
                    #     this_scores = F.pad(this_scores, (0, 378 - len(this_scores)))
                    #     this_segments = F.pad(this_segments, (0, 0, 0, 378 - len(this_segments)))
                    # elif len(this_labels) > 378:
                    #     sorted_indices = torch.argsort(this_scores, dim=0, descending=True)[:378]
                    #     this_labels = this_labels[sorted_indices]
                    #     this_scores = this_scores[sorted_indices]
                    #     this_segments = this_segments[sorted_indices]
                    labels.append(this_labels)
                    scores.append(this_scores)
                    segments.append(this_segments)
                labels = torch.stack(labels, dim=0)
                scores = torch.stack(scores, dim=0)
                segments = torch.stack(segments, dim=0)
                this_proposals = torch.cat((labels.unsqueeze(-1), segments, scores.unsqueeze(-1)), dim=-1).cuda()
                proposals.append(this_proposals)
        proposals = torch.cat(proposals, dim=1)

        detr_target_dict = list()
        for b_i in range(len(video_list)):
            batch_dict = dict()
            batch_dict["labels"] = torch.zeros_like(video_list[b_i]["labels"]).cuda()
            boxes = (video_list[b_i]["segments"] * video_list[b_i]["feat_stride"] +
                     0.5 * video_list[b_i]["feat_num_frames"]) / video_list[b_i]["fps"] / video_list[b_i]["duration"]
            boxes = torch.clamp(boxes, 0.0, 1.0)
            batch_dict["boxes"] = torch.cat((boxes,
                                             ((boxes[..., 0] + boxes[..., 1]) / 2.0).unsqueeze(-1),
                                             (boxes[..., 1] - boxes[..., 0]).unsqueeze(-1)), dim=-1).cuda()
            detr_target_dict.append(batch_dict)

        # features = [torch.stack([x["feats"] for x in video_list], dim=0).cuda()]
        # features = [feat for feat in features]
        # features = torch.stack([x["feats"] for x in video_list], dim=0).cuda()
        # features = torch.stack([F.interpolate(x["feats"].unsqueeze(0),
        #                                       size=192, mode='linear', align_corners=False).squeeze(0)
        #                         for x in video_list], dim=0).cuda()
        # features = [features]
        features = [feat.detach() for feat in backbone_features]

        start_index = 0
        pyramidal_proposals = list()
        for feat in backbone_features:
            this_len = feat.size(2)
            this_proposals = proposals[:, start_index:start_index + this_len]
            pyramidal_proposals.append(this_proposals)
            start_index += this_len

        detr_predictions = detr(features, pyramidal_proposals, detr_target_dict)
        loss_dict = criterion(detr_predictions, detr_target_dict)
        weight_dict = criterion.weight_dict
        detr_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        detr_losses.backward()
        torch.nn.utils.clip_grad_norm_(detr.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        detr_model_ema.update(detr)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in loss_dict.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                # tag_dict = {}
                for key, value in loss_dict.items():
                    # if key != "final_loss":
                    #     tag_dict[key] = value.val
                    tb_writer.add_scalar(
                        "train/" + key,
                        value.item(),
                        global_step
                    )

                # tb_writer.add_scalars(
                #     'train/all_losses',
                #     tag_dict,
                #     global_step
                # )
                # # final loss
                # tb_writer.add_scalar(
                #     'train/final_loss',
                #     losses_tracker['final_loss'].val,
                #     global_step
                # )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            # block3 = 'Loss {:.2f} ({:.2f})\n'.format(
            #     losses_tracker['rgb_final_loss'].val,
            #     losses_tracker['rgb_final_loss'].avg,
            #     losses_tracker['flow_final_loss'].val,
            #     losses_tracker['flow_final_loss'].avg,
            # )
            block4 = ''
            for key, value in losses_tracker.items():
                if key[-2] != "_":
                    block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return

def valid_one_epoch_zoom_in(
        val_loader,
        detr,
        curr_epoch,
        test_cfg,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    detr.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            features = torch.stack([x["feats"] for x in video_list], dim=0).cuda()

            feat_len = features.size(-1)
            boxes = list()
            labels = list()
            scores = list()
            num_levels = np.minimum(np.log2(max(feat_len // 192, 2)).astype(np.int32) + 1, 4)
            for l_i in range(num_levels):
                try:
                    start_indices = np.arange(0, feat_len, feat_len // max(2 ** l_i, 1))
                except:
                    print(feat_len)
                    print(l_i)
                for s_i, start_index in enumerate(start_indices):
                    if s_i >= len(start_indices) - 1:
                        this_features = features[..., start_index:]
                    else:
                        end_index = start_indices[s_i + 1]
                        this_features = features[..., start_index:end_index]

                    this_features = F.interpolate(this_features, size=192, mode='linear', align_corners=False)

                    detr_predictions = detr([this_features])

                    this_boxes = detr_predictions["pred_boxes"].detach().cpu()
                    this_boxes = (this_boxes[..., :2] +
                                  torch.stack((torch.clamp(this_boxes[..., 2] - this_boxes[..., 3] / 2.0, 0.0, 1.0),
                                               torch.clamp(this_boxes[..., 2] + this_boxes[..., 3] / 2.0, 0.0, 1.0)),
                                              dim=-1)) / 2.0
                    this_boxes = this_boxes / (2 ** l_i) + (1 / 2 ** l_i) * s_i
                    this_logits = detr_predictions["pred_logits"].detach().cpu().sigmoid()
                    this_scores, this_labels = torch.max(this_logits, dim=-1)

                    boxes.append(this_boxes)
                    labels.append(this_labels)
                    scores.append(this_scores)

            boxes = torch.cat(boxes, dim=1)
            labels = torch.cat(labels, dim=1)
            scores = torch.cat(scores, dim=1)

            durations = [x["duration"] for x in video_list]
            boxes = boxes * torch.Tensor(durations)

            nmsed_boxes = list()
            nmsed_labels = list()
            nmsed_scores = list()
            for b, l, s in zip(boxes, labels, scores):
                if test_cfg['nms_method'] != 'none':
                    # 2: batched nms (only implemented on CPU)
                    b, s, l = batched_nms(
                        b.contiguous(), s.contiguous(), l.contiguous(),
                        test_cfg['iou_threshold'],
                        test_cfg['min_score'],
                        test_cfg['max_seg_num'],
                        use_soft_nms=(test_cfg['nms_method'] == 'soft'),
                        multiclass=test_cfg['multiclass_nms'],
                        sigma=test_cfg['nms_sigma'],
                        voting_thresh=test_cfg['voting_thresh']
                    )
                nmsed_boxes.append(b)
                nmsed_labels.append(l)
                nmsed_scores.append(s)
            boxes = torch.stack(nmsed_boxes, dim=0)
            boxes = torch.where(boxes.isnan(), torch.zeros_like(boxes), boxes)
            labels = torch.stack(nmsed_labels, dim=0)
            labels = torch.where(labels.isnan(), torch.zeros_like(labels), labels)
            scores = torch.stack(nmsed_scores, dim=0)
            scores = torch.where(scores.isnan(), torch.zeros_like(scores), scores)

            # upack the results into ANet format
            num_vids = len(boxes)
            for vid_idx in range(num_vids):
                if boxes[vid_idx].shape[0] > 0:
                    results['video-id'].extend(
                        [video_list[vid_idx]['video_id']] *
                        boxes[vid_idx].shape[0]
                    )
                    results['t-start'].append(boxes[vid_idx][:, 0])
                    results['t-end'].append(boxes[vid_idx][:, 1])
                    results['label'].append(labels[vid_idx])
                    results['score'].append(scores[vid_idx])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP

def valid_one_epoch(
        val_loader,
        backbone,
        detr,
        data_type,
        curr_epoch,
        test_cfg,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    backbone.eval()
    detr.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }
    backbone_results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output, backbone_features = backbone(video_list, data_type=data_type, nms=False)

            labels = list()
            scores = list()
            segments = list()
            for p, x in zip(output, video_list):
                this_labels = p["labels"].float()
                this_scores = p["scores"]
                this_segments = p["segments"] / x["duration"]

                # sorted_indices = torch.argsort(this_scores, descending=True)[:100]
                # this_labels = this_labels[sorted_indices]
                # this_scores = this_scores[sorted_indices]
                # this_segments = this_segments[sorted_indices]
                #
                # if len(this_labels) < 100:
                #     this_labels = F.pad(this_labels, (0, 100 - len(this_labels)))
                #     this_scores = F.pad(this_scores, (0, 100 - len(this_scores)))
                #     this_segments = F.pad(this_segments, (0, 0, 0, 100 - len(this_segments)))

                labels.append(this_labels)
                scores.append(this_scores)
                segments.append(this_segments)
            labels = torch.stack(labels, dim=0)
            scores = torch.stack(scores, dim=0)
            segments = torch.stack(segments, dim=0)
            proposals = torch.cat((labels.unsqueeze(-1), segments, scores.unsqueeze(-1)), dim=-1).cuda()

            features = [feat for feat in backbone_features]
            # features = [torch.stack([x["feats"] for x in video_list], dim=0).cuda()]

            start_index = 0
            pyramidal_proposals = list()
            for feat in backbone_features:
                this_len = feat.size(2)
                this_proposals = proposals[:, start_index:start_index + this_len]
                pyramidal_proposals.append(this_proposals)
                start_index += this_len

            # detr_predictions = detr(features, proposals)
            detr_predictions = detr(features, pyramidal_proposals)

            boxes = detr_predictions["pred_boxes"].detach().cpu()
            # boxes = (boxes[..., :2] +
            #          torch.stack((torch.clamp(boxes[..., 2] - boxes[..., 3] / 2.0, 0.0, 1.0),
            #                       torch.clamp(boxes[..., 2] + boxes[..., 3] / 2.0, 0.0, 1.0)), dim=-1)) / 2.0
            # boxes = boxes[..., :2]
            logits = detr_predictions["pred_logits"].detach().cpu().sigmoid()
            # logits = (logits[..., 0] * 1.0 + logits[..., 1] * 0.5 + logits[..., 1] * 0.2).unsqueeze(-1)
            scores, labels = torch.max(logits, dim=-1)
            if "pred_actionness" in detr_predictions.keys():
                actionness = detr_predictions["pred_actionness"].detach().cpu()
                scores = scores * actionness.squeeze(-1)

            proposals = proposals.cpu()
            backbone_boxes = proposals[..., 1:3]
            backbone_scores = proposals[..., -1]
            backbone_labels = proposals[..., 0].long()

            # boxes = torch.clamp((boxes + backbone_boxes) / 2.0, 0.0, 1.0)
            # scores = scores * backbone_scores

            durations = [x["duration"] for x in video_list]
            boxes = boxes * torch.Tensor(durations)

            nmsed_boxes = list()
            nmsed_labels = list()
            nmsed_scores = list()
            for b, l, s in zip(boxes, labels, scores):
                if test_cfg['nms_method'] != 'none':
                    # 2: batched nms (only implemented on CPU)
                    b, s, l = batched_nms(
                        b.contiguous(), s.contiguous(), l.contiguous(),
                        test_cfg['iou_threshold'],
                        test_cfg['min_score'],
                        test_cfg['max_seg_num'],
                        use_soft_nms=(test_cfg['nms_method'] == 'soft'),
                        multiclass=test_cfg['multiclass_nms'],
                        sigma=test_cfg['nms_sigma'],
                        voting_thresh=test_cfg['voting_thresh']
                    )
                nmsed_boxes.append(b)
                nmsed_labels.append(l)
                nmsed_scores.append(s)
            boxes = torch.stack(nmsed_boxes, dim=0)
            boxes = torch.where(boxes.isnan(), torch.zeros_like(boxes), boxes)
            labels = torch.stack(nmsed_labels, dim=0)
            labels = torch.where(labels.isnan(), torch.zeros_like(labels), labels)
            scores = torch.stack(nmsed_scores, dim=0)
            scores = torch.where(scores.isnan(), torch.zeros_like(scores), scores)

            # upack the results into ANet format
            num_vids = len(boxes)
            for vid_idx in range(num_vids):
                if boxes[vid_idx].shape[0] > 0:
                    results['video-id'].extend(
                        [video_list[vid_idx]['video_id']] *
                        boxes[vid_idx].shape[0]
                    )
                    results['t-start'].append(boxes[vid_idx][:, 0])
                    results['t-end'].append(boxes[vid_idx][:, 1])
                    results['label'].append(labels[vid_idx])
                    results['score'].append(scores[vid_idx])

            durations = [x["duration"] for x in video_list]
            backbone_boxes = backbone_boxes * torch.Tensor(durations)

            nmsed_boxes = list()
            nmsed_labels = list()
            nmsed_scores = list()
            for b, l, s in zip(backbone_boxes, backbone_labels, backbone_scores):
                if test_cfg['nms_method'] != 'none':
                    # 2: batched nms (only implemented on CPU)
                    b, s, l = batched_nms(
                        b.contiguous(), s.contiguous(), l.contiguous(),
                        test_cfg['iou_threshold'],
                        test_cfg['min_score'],
                        test_cfg['max_seg_num'],
                        use_soft_nms=(test_cfg['nms_method'] == 'soft'),
                        multiclass=test_cfg['multiclass_nms'],
                        sigma=test_cfg['nms_sigma'],
                        voting_thresh=test_cfg['voting_thresh']
                    )
                nmsed_boxes.append(b)
                nmsed_labels.append(l)
                nmsed_scores.append(s)
            backbone_boxes = torch.stack(nmsed_boxes, dim=0)
            backbone_boxes = torch.where(backbone_boxes.isnan(), torch.zeros_like(backbone_boxes), backbone_boxes)
            backbone_labels = torch.stack(nmsed_labels, dim=0)
            backbone_labels = torch.where(backbone_labels.isnan(), torch.zeros_like(backbone_labels), backbone_labels)
            backbone_scores = torch.stack(nmsed_scores, dim=0)
            backbone_scores = torch.where(backbone_scores.isnan(), torch.zeros_like(backbone_scores), backbone_scores)

            # upack the results into ANet format
            num_vids = len(backbone_boxes)
            for vid_idx in range(num_vids):
                if backbone_boxes[vid_idx].shape[0] > 0:
                    backbone_results['video-id'].extend(
                        [video_list[vid_idx]['video_id']] *
                        backbone_boxes[vid_idx].shape[0]
                    )
                    backbone_results['t-start'].append(backbone_boxes[vid_idx][:, 0])
                    backbone_results['t-end'].append(backbone_boxes[vid_idx][:, 1])
                    backbone_results['label'].append(backbone_labels[vid_idx])
                    backbone_results['score'].append(backbone_scores[vid_idx])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    backbone_results['t-start'] = torch.cat(backbone_results['t-start']).numpy()
    backbone_results['t-end'] = torch.cat(backbone_results['t-end']).numpy()
    backbone_results['label'] = torch.cat(backbone_results['label']).numpy()
    backbone_results['score'] = torch.cat(backbone_results['score']).numpy()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
            backbone_results = postprocess_results(backbone_results, ext_score_file)
        # call the evaluator
        _, mAP = evaluator.evaluate(results, verbose=True)
        _, backbone_mAP = evaluator.evaluate(backbone_results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0
        backbone_mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)
        tb_writer.add_scalar('validation/backbone_mAP', backbone_mAP, curr_epoch)

    return mAP

def valid_one_epoch_phase_1(
        val_loader,
        models,
        data_types,
        curr_epoch,
        test_cfg,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    for model in models:
        model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            proposals = list()
            for m_i, model in enumerate(models):
                data_type = data_types[m_i]
                output, backbone_features = model(video_list, data_type=data_type, nms=len(models) == 1)
                # output, backbone_features = model(video_list, data_type=data_type, nms=False)

                labels = list()
                scores = list()
                segments = list()
                for p, x in zip(output, video_list):
                    this_labels = p["labels"].float()
                    this_scores = p["scores"]
                    this_segments = p["segments"] / x["duration"]
                    # if len(this_labels) < 378:
                    #     this_labels = F.pad(this_labels, (0, 378 - len(this_labels)))
                    #     this_scores = F.pad(this_scores, (0, 378 - len(this_scores)))
                    #     this_segments = F.pad(this_segments, (0, 0, 0, 378 - len(this_segments)))
                    # elif len(this_labels) > 378:
                    #     sorted_indices = torch.argsort(this_scores, dim=0, descending=True)[:378]
                    #     this_labels = this_labels[sorted_indices]
                    #     this_scores = this_scores[sorted_indices]
                    #     this_segments = this_segments[sorted_indices]
                    labels.append(this_labels)
                    scores.append(this_scores)
                    segments.append(this_segments)
                labels = torch.stack(labels, dim=0)
                scores = torch.stack(scores, dim=0)
                segments = torch.stack(segments, dim=0)
                this_proposals = torch.cat((labels.unsqueeze(-1), segments, scores.unsqueeze(-1)), dim=-1)
                proposals.append(this_proposals)
            proposals = torch.mean(torch.stack(proposals, dim=0), dim=0)

            boxes = proposals[..., 1:3]
            durations = [x["duration"] for x in video_list]
            boxes = boxes * torch.Tensor(durations)
            scores = proposals[..., -1]
            labels = proposals[..., 0].long()

            nmsed_boxes = list()
            nmsed_labels = list()
            nmsed_scores = list()
            for b, l, s in zip(boxes, labels, scores):
                if test_cfg['nms_method'] != 'none':
                    # 2: batched nms (only implemented on CPU)
                    b, s, l = batched_nms(
                        b.contiguous(), s.contiguous(), l.contiguous(),
                        test_cfg['iou_threshold'],
                        test_cfg['min_score'],
                        test_cfg['max_seg_num'],
                        use_soft_nms=(test_cfg['nms_method'] == 'soft'),
                        multiclass=test_cfg['multiclass_nms'],
                        sigma=test_cfg['nms_sigma'],
                        voting_thresh=test_cfg['voting_thresh']
                    )
                nmsed_boxes.append(b)
                nmsed_labels.append(l)
                nmsed_scores.append(s)
            boxes = torch.stack(nmsed_boxes, dim=0)
            labels = torch.stack(nmsed_labels, dim=0)
            scores = torch.stack(nmsed_scores, dim=0)

            # upack the results into ANet format
            num_vids = len(boxes)
            for vid_idx in range(num_vids):
                if boxes[vid_idx].shape[0] > 0:
                    results['video-id'].extend(
                        [video_list[vid_idx]['video_id']] *
                        boxes[vid_idx].shape[0]
                    )
                    results['t-start'].append(boxes[vid_idx][:, 0])
                    results['t-end'].append(boxes[vid_idx][:, 1])
                    results['label'].append(labels[vid_idx])
                    results['score'].append(scores[vid_idx])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/backbone_mAP', mAP, curr_epoch)

    return mAP

def valid_one_epoch_phase_2(
        val_loader,
        detr,
        data_types,
        proposal_models,
        curr_epoch,
        test_cfg,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    detr.eval()
    for model in proposal_models:
        model.eval()
    # dict for results (for our evaluation code)
    detr_results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            proposals = list()
            backbone_features = list()
            for m_i, model in enumerate(proposal_models):
                data_type = data_types[m_i]
                output, this_backbone_features = model(video_list, data_type=data_type)
                backbone_features.extend(this_backbone_features)

                labels = list()
                scores = list()
                segments = list()
                for p, x in zip(output, video_list):
                    this_labels = p["labels"].float()
                    this_scores = p["scores"]
                    this_segments = p["segments"] / x["duration"]
                    # if len(this_labels) < 378:
                    #     this_labels = F.pad(this_labels, (0, 378 - len(this_labels)))
                    #     this_scores = F.pad(this_scores, (0, 378 - len(this_scores)))
                    #     this_segments = F.pad(this_segments, (0, 0, 0, 378 - len(this_segments)))
                    # elif len(this_labels) > 378:
                    #     sorted_indices = torch.argsort(this_scores, dim=0, descending=True)[:378]
                    #     this_labels = this_labels[sorted_indices]
                    #     this_scores = this_scores[sorted_indices]
                    #     this_segments = this_segments[sorted_indices]
                    labels.append(this_labels)
                    scores.append(this_scores)
                    segments.append(this_segments)
                labels = torch.stack(labels, dim=0)
                scores = torch.stack(scores, dim=0)
                segments = torch.stack(segments, dim=0)
                this_proposals = torch.cat((labels.unsqueeze(-1), segments, scores.unsqueeze(-1)), dim=-1)
                proposals.append(this_proposals)
            cat_proposals = torch.cat(proposals, dim=1).cuda()

            # features = [torch.stack([x["resize_feats"] for x in video_list], dim=0).cuda()]
            # features = [feat for feat in features]
            # features = torch.stack([x["feats"] for x in video_list], dim=0).cuda()
            # features = torch.stack([F.interpolate(x["feats"].unsqueeze(0),
            #                                       size=192, mode='linear', align_corners=False).squeeze(0)
            #                         for x in video_list], dim=0).cuda()
            # features = [features]
            features = [feat.detach() for feat in backbone_features]

            start_index = 0
            pyramidal_proposals = list()
            for feat in backbone_features:
                this_len = feat.size(2)
                this_proposals = cat_proposals[:, start_index:start_index + this_len]
                pyramidal_proposals.append(this_proposals)
                start_index += this_len

            detr_predictions = detr(features, pyramidal_proposals)

            boxes = detr_predictions["pred_boxes"].detach().cpu()
            boxes = (boxes[..., :2] +
                     torch.stack((torch.clamp(boxes[..., 2] - boxes[..., 3] / 2.0, 0.0, 1.0),
                                  torch.clamp(boxes[..., 2] + boxes[..., 3] / 2.0, 0.0, 1.0)), dim=-1)) / 2.0
            # boxes = boxes[..., :2]
            logits = detr_predictions["pred_entire_logits"].detach().cpu().sigmoid()
            detr_scores, labels = torch.max(logits, dim=-1)
            scores = detr_scores

            boxes = boxes[:, :100]
            labels = labels[:, :100]
            scores = scores[:, :100]

            durations = [x["duration"] for x in video_list]
            boxes = boxes * torch.Tensor(durations)

            # print(boxes[0, 50:100])
            # print(torch.argsort(scores, dim=1, descending=True)[:10])
            # print(scores[0, 50:100])

            # mean_proposals = torch.mean(torch.stack(proposals, dim=0), dim=0)
            # dense_boxes = mean_proposals[..., 1:3].contiguous()
            # durations = [x["duration"] for x in video_list]
            # dense_boxes = dense_boxes * torch.Tensor(durations)
            # dense_scores = mean_proposals[..., -1].contiguous()
            # dense_labels = mean_proposals[..., 0].long()
            # dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
            # dense_scores = dense_scores * scores[:, :100].max()

            # sorted_indices = torch.argsort(scores, dim=1, descending=True)[:, :5]
            # boxes = boxes[np.arange(len(sorted_indices)), sorted_indices[np.arange(len(sorted_indices))]]
            # labels = labels[np.arange(len(sorted_indices)), sorted_indices[np.arange(len(sorted_indices))]]
            # scores = scores[np.arange(len(sorted_indices)), sorted_indices[np.arange(len(sorted_indices))]]

            # boxes = torch.cat((boxes, dense_boxes), dim=1)
            # scores = torch.cat((scores, dense_scores), dim=1)
            # labels = torch.cat((labels, dense_labels), dim=1)

            # boxes = dense_boxes
            # scores = dense_scores
            # labels = dense_labels

            # dense_onehot = F.one_hot(dense_labels, num_classes=20).sum(dim=1)
            # labels = torch.argsort(dense_onehot, dim=-1, descending=True)[..., 0].unsqueeze(1).repeat(1, labels.size(1))
            # top_2_labels = torch.argsort(dense_onehot, dim=-1, descending=True)[..., 1].unsqueeze(1).repeat(1, labels.size(1))

            # scores = (scores - scores.min()) / (scores.max() - scores.min())
            # dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())

            # boxes = torch.cat((boxes, boxes, dense_boxes), dim=1)
            # scores = torch.cat((scores, scores, dense_scores), dim=1)
            # labels = torch.cat((labels, top_2_labels, dense_labels), dim=1)

            # labels = dense_labels
            # scores = scores * dense_scores

            nmsed_boxes = list()
            nmsed_labels = list()
            nmsed_scores = list()
            for b, l, s in zip(boxes, labels, scores):
                if test_cfg['nms_method'] != 'none':
                    # 2: batched nms (only implemented on CPU)
                    b, s, l = batched_nms(
                        b, s, l,
                        test_cfg['iou_threshold'],
                        test_cfg['min_score'],
                        test_cfg['max_seg_num'],
                        use_soft_nms=(test_cfg['nms_method'] == 'soft'),
                        multiclass=test_cfg['multiclass_nms'],
                        sigma=test_cfg['nms_sigma'],
                        voting_thresh=test_cfg['voting_thresh'])
                nmsed_boxes.append(b)
                nmsed_labels.append(l)
                nmsed_scores.append(s)
            boxes = torch.stack(nmsed_boxes, dim=0)
            boxes = torch.where(boxes.isnan(), torch.zeros_like(boxes), boxes)
            labels = torch.stack(nmsed_labels, dim=0)
            labels = torch.where(labels.isnan(), torch.zeros_like(labels), labels)
            scores = torch.stack(nmsed_scores, dim=0)
            scores = torch.where(scores.isnan(), torch.zeros_like(scores), scores)

            # upack the results into ANet format
            num_vids = len(boxes)
            for vid_idx in range(num_vids):
                if boxes[vid_idx].shape[0] > 0:
                    detr_results['video-id'].extend(
                        [video_list[vid_idx]['video_id']] *
                        boxes[vid_idx].shape[0]
                    )
                    detr_results['t-start'].append(boxes[vid_idx][:, 0])
                    detr_results['t-end'].append(boxes[vid_idx][:, 1])
                    detr_results['label'].append(labels[vid_idx])
                    detr_results['score'].append(scores[vid_idx])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    detr_results['t-start'] = torch.cat(detr_results['t-start']).numpy()
    detr_results['t-end'] = torch.cat(detr_results['t-end']).numpy()
    detr_results['label'] = torch.cat(detr_results['label']).numpy()
    detr_results['score'] = torch.cat(detr_results['score']).numpy()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            detr_results = postprocess_results(detr_results, ext_score_file)
        # call the evaluator
        _, detr_mAP = evaluator.evaluate(detr_results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(detr_results, f)
        detr_mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/detr_mAP', detr_mAP, curr_epoch)

    return detr_mAP

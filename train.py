# python imports
import argparse
import os
import time
import datetime
import glob
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)
from libs.modeling.detr import build_dino

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    # data_types = ["rgb", "flow"]
    data_types = ["fusion"]
    models = list()
    optimizers = list()
    schedulers = list()
    model_emas = list()
    num_iters_per_epoch = len(train_loader)
    for data_type in data_types:
        this_cfg = dict(cfg)
        if data_type in ["rgb", "flow"]:
            this_cfg['model']['input_dim'] = this_cfg['dataset']['input_dim'] // 2
        model = make_meta_arch(this_cfg['model_name'], **this_cfg['model'])
        model_ = make_meta_arch(this_cfg['model_name'], **this_cfg['model'])
        # not ideal for multi GPU training, ok for now
        model = nn.DataParallel(model, device_ids=this_cfg['devices'])
        model_ = nn.DataParallel(model_, device_ids=this_cfg['devices'])
        model_.load_state_dict(model.state_dict())
        # optimizer
        optimizer = make_optimizer(model, this_cfg['opt'])
        # schedule
        scheduler = make_scheduler(optimizer, this_cfg['opt'], num_iters_per_epoch)
        # enable model EMA
        # model_ema = ModelEma(model)
        model_ema = ModelEma(model_, copy_model=False)

        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        model_emas.append(model_ema)

    """ DETR """
    cfg['detr']['num_feature_levels'] *= len(data_types)
    detr, detr_criterion = build_dino(cfg['detr'])
    detr_, _ = build_dino(cfg['detr'])
    detr_.load_state_dict(detr.state_dict())
    detr = detr.cuda()
    detr_ = detr_.cuda()
    detr_model_ema = ModelEma(detr_, copy_model=False)
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
    detr_param_dicts = [
        # non-backbone, non-offset
        {
            "params":
                [p for n, p in detr.named_parameters()
                 if not match_name_keywords(n, cfg['detr']["lr_linear_proj_names"]) and p.requires_grad],
            "lr": cfg['detr']["lr"],
            "initial_lr": cfg['detr']["lr"]
        },
        # offset
        {
            "params": [p for n, p in detr.named_parameters() if
                       match_name_keywords(n, cfg['detr']["lr_linear_proj_names"]) and p.requires_grad],
            "lr": cfg['detr']["lr"] * cfg['detr']["lr_linear_proj_mult"],
            "initial_lr": cfg['detr']["lr"] * cfg['detr']["lr_linear_proj_mult"]
        }
    ]
    detr_optimizer = torch.optim.AdamW(detr_param_dicts, lr=cfg['detr']["lr"], weight_decay=cfg['detr']["weight_decay"])
    detr_scheduler = make_scheduler(detr_optimizer, cfg['opt'], num_iters_per_epoch)

    # """4. Resume from model / Misc"""
    # resume from a checkpoint?
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         # load ckpt, reset epoch / best rmse
    #         checkpoint = torch.load(args.resume,
    #             map_location = lambda storage, loc: storage.cuda(
    #                 cfg['devices'][0]))
    #         args.start_epoch = checkpoint['epoch'] + 1
    #         model.load_state_dict(checkpoint['state_dict'])
    #         model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
    #         # also load the optimizer / scheduler if necessary
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #         print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
    #             args.resume, checkpoint['epoch']
    #         ))
    #         del checkpoint
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    #         return

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_root_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_root_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_root_folder):
        os.mkdir(ckpt_root_folder)

    # save the current config
    # with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
    #     pprint(cfg, stream=fid)
    #     fid.flush()

    # set up evaluator
    output_file = None
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds=val_db_vars['tiou_thresholds']
    )

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])

    backbone_ckpt_folder = os.path.join(ckpt_root_folder, "backbone")
    if not os.path.exists(backbone_ckpt_folder):
        os.mkdir(backbone_ckpt_folder)
    detr_ckpt_folder = os.path.join(ckpt_root_folder, "detr")
    if not os.path.exists(detr_ckpt_folder):
        os.mkdir(detr_ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_root_folder, 'logs'))

    is_best = False
    best_mAP = -1
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            models[0],
            optimizers[0],
            schedulers[0],
            detr,
            detr_optimizer,
            detr_scheduler,
            detr_criterion,
            epoch,
            data_type=data_types[0],
            backbone_model_ema=model_emas[0],
            detr_model_ema=detr_model_ema,
            tb_writer=tb_writer,
            print_freq=args.print_freq)

        if (epoch >= 0 and epoch % 1 == 0) or epoch == max_epochs - 1:
            mAP = valid_one_epoch(
                val_loader,
                model_emas[0].module,
                detr_model_ema.module,
                data_types[0],
                epoch,
                cfg['test_cfg'],
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=tb_writer,
                print_freq=args.print_freq
            )

            is_best = mAP >= best_mAP
            if is_best:
                best_mAP = mAP

        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
                is_best or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0) and
                        (epoch > 0)
                )
        ):
            backbone_save_states = {
                'epoch': epoch,
                'state_dict': models[0].state_dict(),
                'scheduler': schedulers[0].state_dict(),
                'optimizer': optimizers[0].state_dict()
            }
            detr_save_states = {
                'epoch': epoch,
                'state_dict': detr.state_dict(),
                'scheduler': detr_scheduler.state_dict(),
                'optimizer': detr_optimizer.state_dict()
            }

            backbone_save_states['state_dict_ema'] = model_emas[0].module.state_dict()
            save_checkpoint(
                backbone_save_states,
                is_best,
                file_folder=backbone_ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )
            detr_save_states['state_dict_ema'] = detr_model_ema.module.state_dict()
            save_checkpoint(
                detr_save_states,
                is_best,
                file_folder=detr_ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    # wrap up
    tb_writer.close()

    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)

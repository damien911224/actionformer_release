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
from libs.utils import (train_one_epoch_phase_1, train_one_epoch_phase_2,
                        valid_one_epoch_phase_1, valid_one_epoch_phase_2, ANETdetection,
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
    rgb_model = make_meta_arch(cfg['model_name'], **cfg['model'])
    flow_model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    rgb_model = nn.DataParallel(rgb_model, device_ids=cfg['devices'])
    flow_model = nn.DataParallel(flow_model, device_ids=cfg['devices'])
    # optimizer
    rgb_optimizer = make_optimizer(rgb_model, cfg['opt'])
    flow_optimizer = make_optimizer(flow_model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    rgb_scheduler = make_scheduler(rgb_optimizer, cfg['opt'], num_iters_per_epoch)
    flow_scheduler = make_scheduler(flow_optimizer, cfg['opt'], num_iters_per_epoch)

    """ DETR """
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

    # enable model EMA
    print("Using model EMA ...")
    rgb_model_ema = ModelEma(rgb_model)
    flow_model_ema = ModelEma(flow_model)

    models = (rgb_model, flow_model)
    optimizers = (rgb_optimizer, flow_optimizer)
    schedulers = (rgb_scheduler, flow_scheduler)
    model_emas = (rgb_model_ema, flow_model_ema)

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

    folder_names = ["rgb", "flow", "detr"]
    tb_writers = list()
    for folder_name in folder_names:
        ckpt_folder = os.path.join(ckpt_root_folder, folder_name)
        if not os.path.exists(ckpt_folder):
            os.mkdir(ckpt_folder)

        # tensorboard writer
        tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))
        tb_writers.append(tb_writer)

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
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in range(args.start_epoch, max_epochs):
        for m_i, (model, optimizer, scheduler, model_ema) in enumerate(zip(models, optimizers, schedulers, model_emas)):
            data_type = ["rgb", "flow"][m_i]
            # train for one epoch
            train_one_epoch_phase_1(
                train_loader,
                model,
                optimizer,
                scheduler,
                epoch,
                data_type=data_type,
                model_ema=model_ema,
                clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
                tb_writer=tb_writers[m_i],
                print_freq=args.print_freq)

            # save ckpt once in a while
            if (
                (epoch == max_epochs - 1) or
                (
                    (args.ckpt_freq > 0) and
                    (epoch % args.ckpt_freq == 0) and
                    (epoch > 0)
                )
            ):
                save_states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'detr': detr.state_dict()
                }

                save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='epoch_{:03d}.pth.tar'.format(epoch)
                )

        if (epoch >= 0 and epoch % 1 == 0) or epoch == max_epochs - 1:
            valid_one_epoch_phase_1(
                val_loader,
                [m.module for m in model_emas],
                epoch,
                cfg['test_cfg'],
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=tb_writers[-1],
                print_freq=args.print_freq
            )

        # train for one epoch
        train_one_epoch_phase_2(
            train_loader,
            detr,
            detr_model_ema,
            detr_criterion,
            detr_optimizer,
            detr_scheduler,
            [m.module for m in model_emas],
            epoch,
            tb_writer=tb_writers[-1],
            print_freq=args.print_freq)

        if (epoch >= 0 and epoch % 1 == 0) or epoch == max_epochs - 1:
            valid_one_epoch_phase_2(
                val_loader,
                detr_model_ema.module,
                models,
                epoch,
                cfg['test_cfg'],
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=tb_writers[-1],
                print_freq=args.print_freq
            )

        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0) and
                        (epoch > 0)
                )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': detr.state_dict(),
                'scheduler': detr_scheduler.state_dict(),
                'optimizer': detr_optimizer.state_dict(),
                'detr': detr.state_dict()
            }

            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

        for tb_writer in tb_writers:
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

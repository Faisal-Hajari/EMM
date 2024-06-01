# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import autocast
from torch.cuda.amp import GradScaler
from timm.utils import AverageMeter

from config_wandb import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

import wandb 
import json 

def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    # parser.add_argument(
    #     "--opts",
    #     help="Modify config options by adding 'KEY VALUE' pairs. ",
    #     default=None,
    #     nargs='+',
    # )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local-rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main():
    config = dict(wandb.config)
    data_loader_train = build_loader(config, logger, is_pretrain=True)

    logger.info(f"Creating model:{config["MODEL_TYPE"]}/{config["MODEL_NAME"]}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    grad_scaler = GradScaler() if config["AMP"] else None  
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config["LOCAL_RANK"]], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config["TRAIN_AUTO_RESUME"]:
        resume_file = auto_resume_helper(config["OUTPUT"], logger)
        if resume_file:
            if config["MODEL_RESUME"]:
                logger.warning(f"auto-resume changing resume file from {config["MODEL_RESUME"]} to {resume_file}")
            config["MODEL_RESUME"] = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config["OUTPUT"]}, ignoring auto resume')

    if config["MODEL_RESUME"]:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config['TRAIN_START_EPOCH'], config['TRAIN_EPOCHS']):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, grad_scaler)
        if dist.get_rank() == 0 and (epoch % config["SAVE_FREQ"] == 0 or epoch == (config["TRAIN_EPOCHS"] - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, grad_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img, mask, _) in enumerate(data_loader):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        if config["TRAIN_ACCUMULATION_STEPS"] > 1:
            if config["AMP"]:
                with autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(img, mask)
                loss = loss / config["TRAIN_ACCUMULATION_STEPS"]
                grad_scaler.scale(loss).backward()
            else:
                loss = model(img, mask)
                loss = loss / config["TRAIN_ACCUMULATION_STEPS"]
                loss.backward()

            if (idx + 1) % config["TRAIN_ACCUMULATION_STEPS"] == 0:
                if config["AMP"]:
                    grad_scaler.unscale_(optimizer)
                if config["TRAIN_CLIP_GRAD"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN_CLIP_GRAD"])
                else: 
                    grad_norm = get_grad_norm(model.parameters())

                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)
                if config["AMP"]:
                    grad_scaler.update()
                optimizer.zero_grad()
            else: 
                grad_norm = get_grad_norm(model.parameters()) #we add it here to maintain logs

        else:
            optimizer.zero_grad()
            if config["AMP"]:
                with autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(img, mask)
                grad_scaler.unscale_(optimizer)
                if config["TRAIN_CLIP_GRAD"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN_CLIP_GRAD"])
                else: 
                    grad_norm = get_grad_norm(model.parameters())
            else:
                loss = model(img, mask)
                loss.backward()
                if config["TRAIN_CLIP_GRAD"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN_CLIP_GRAD"])
                else: 
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)
            if config["AMP"]:
                grad_scaler.update()

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config["PRINT_FREQ"] == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config["TRAIN_EPOCHS"]}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            
            wandb.log(
                {
                    "epoch":epoch,
                    "step":(epoch*num_steps)+idx, 
                    "time": batch_time.val,
                    "time (avg)": batch_time.avg, 
                    "loss": loss_meter.val, 
                    "loss (avg)": loss_meter.avg, 
                    "grad_norm": norm_meter.val, 
                    "grad_norm (avg)": norm_meter.avg, 
                    "mem": memory_used, 
                }
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    wandb.log(
                {
                    "epoch":epoch,
                    "epoch_time":int(datetime.timedelta(seconds=int(epoch_time)).seconds), 
                }
            )


if __name__ == '__main__':
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config["LOCAL_RANK"])
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config["SEED"] + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config["TRAIN_BASE_LR"] * config["DATA_BATCH_SIZE"] * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config["TRAIN_WARMUP_LR"] * config["DATA_BATCH_SIZE"] * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config["TRAIN_MIN_LR"] * config["DATA_BATCH_SIZE"] * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config["TRAIN_ACCUMULATION_STEPS"] > 1:
        linear_scaled_lr = linear_scaled_lr * config["TRAIN_ACCUMULATION_STEPS"]
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config["TRAIN_ACCUMULATION_STEPS"]
        linear_scaled_min_lr = linear_scaled_min_lr * config["TRAIN_ACCUMULATION_STEPS"]
    config["TRAIN_BASE_LR"] = linear_scaled_lr
    config["TRAIN_WARMUP_LR"] = linear_scaled_warmup_lr
    config["TRAIN_MIN_LR"] = linear_scaled_min_lr

    os.makedirs(config["OUTPUT"], exist_ok=True)
    logger = create_logger(output_dir=config["OUTPUT"], dist_rank=dist.get_rank(), name=f"{config["MODEL_NAME"]}")

    if dist.get_rank() == 0:
        path = os.path.join(config["OUTPUT"], "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Full config saved to {path}")
        run = wandb.init(project=F"{config["MODEL_NAME"]}_{config["TAG"]}", config=config)

    # print config
    logger.info(config)

    main()
sweep_config = {
    "method": "random",  # Choose a search strategy (e.g., random, grid)
    "metric": {"goal": "maximize", "name": "max_accuracy"},  # Define the metric to optimize
      "parameters": {
        'AUG_COLOR_JITTER': {'values': [0.19142765, 0.1939659,0.2687049,0.4,0.5312951,0.6060341,0.60857236]},
        'AUG_CUTMIX': {'values': [0.073544264,0.35426086,0.64957076,0.751639,0.85458434,0.9037231,1.0]},
        'AUG_MIXUP': {'values': [0.0,0.2096451,0.23680645,0.3550117,0.7190215,0.9890876,1.0]},
        'AUG_MIXUP_PROB': {'values': [0.12280595,0.30467975,0.31916678,0.40000468,0.4958151,0.7025622,1.0]},
        'AUG_MIXUP_SWITCH_PROB': {'values': [0.022307009,0.13213614,0.38405824,0.5,0.61594176,0.8678639,0.97769296,1.0]},
        'AUG_REPROB': {'values': [0.08310923,0.21976,0.25,0.58310926,0.71976,0.7710203,0.9768015,1.0]},
        'DATA_BATCH_SIZE': {'values': [64, 128, 256, 512, 1024]},
        'MODEL_DROP_PATH_RATE': {'values': [0.030991517,0.1,0.1690085,0.36810523,0.5681052,0.7711458,0.97114587,0.95]},
        'MODEL_DROP_RATE': {'values': [0.0,0.049390446,0.25506154,0.32669935,0.3506588,0.38954917,0.51296324,0.95]},
        'MODEL_LABEL_SMOOTHING': {'values': [0.026220776,0.1,0.22622079,0.52637655,0.7263766,0.8404708,0.95]},
        'TRAIN_ACCUMULATION_STEPS': {'values': [0, 1, 2, 5, 7, 8, 9]},
        'TRAIN_BASE_LR': {'values': [0.00255859375,0.026838038,0.031955227,0.7334642,0.73858136,0.99940324,1.0]},
        'TRAIN_CLIP_GRAD': {'values': [1, 3, 4, 5, 6, 9, 13]},
        'TRAIN_EPOCHS': {'values': [36, 41, 44, 45, 46, 49, 54]},
        'TRAIN_LAYER_DECAY': {'values': [0.2460284,0.3720761,0.4017095,0.65,0.89829046,0.92792386,0.95]},
        'TRAIN_WARMUP_EPOCHS': {'values': [1, 2, 3, 4, 5, 7, 8, 11, 14]},
        'TRAIN_LR_SCHEDULER_DECAY_EPOCHS': {'values': [17, 22, 24, 25, 26, 28, 33]},
        'TRAIN_LR_SCHEDULER_DECAY_RATE': {'values': [0.1,0.27690786,0.47690785,0.6295608,0.8295609,0.83221257,0.95]},
        'TRAIN_LR_SCHEDULER_GAMMA': {'values': [0.1,0.20474148,0.34909925,0.40474147,0.5197809,0.54909927,0.7197809,0.95]},
        'TRAIN_MIN_LR': {'values': [5.117187499999999e-07,0.050085,0.05008602,0.38906166,0.38906267,0.9322572,0.93225825]},
        'TRAIN_OPTIMIZER_MOMENTUM': {'values': [0.032524765,0.38258517,0.60187066,0.62120306,0.64276683,0.7398006,0.9,0.95]},
    }
}
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
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import autocast
from torch.cuda.amp import GradScaler

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from config_wandb import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

import wandb

def parse_option():
    parser = argparse.ArgumentParser("Swin Transformer training and evaluation script", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file", )
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--pretrained", type=str, help="path to pre-trained model")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--use-checkpoint", action="store_true",
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"],
                        help="mixed precision opt level, if O0, no amp is used")
    parser.add_argument("--output", default="output", type=str, metavar="PATH",
                        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)")
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")

    # distributed training
    parser.add_argument("--local-rank", type=int, required=True, help="local rank for DistributedDataParallel")

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main():
    wandb.init(config=config)
    config_wandb = dict(wandb.config)
    for k, v in config_wandb.items(): 
        config[k]= v

    logger.info(config)
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, logger, is_pretrain=False)

    logger.info(f'Creating model:{config["MODEL_TYPE"]}/{config["MODEL_NAME"]}')
    model = build_model(config, is_pretrain=False)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    
    grad_scaler = GradScaler() if config["AMP"] else None     
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config["LOCAL_RANK"]], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config["AUG_MIXUP"] > 0 or config["AUG_CUTMIX"] > 0. or config["AUG_CUTMIX_MINMAX"] is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config["MODEL_LABEL_SMOOTHING"] > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config["MODEL_LABEL_SMOOTHING"])
    else:
        criterion = torch.nn.CrossEntropyLoss()


    max_accuracy = 0.0

    if config["TRAIN_AUTO_RESUME"]:
        resume_file = auto_resume_helper(config["OUTPUT"], logger)
        if resume_file:
            if config["MODEL_RESUME"]:
                logger.warning(f'auto-resume changing resume file from {config["MODEL_RESUME"]} to {resume_file}')
            config["MODEL_RESUME"] = resume_file
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f'no checkpoint found in {config["OUTPUT"]}, ignoring auto resume')

    if config["MODEL_RESUME"]:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config["EVAL_MODE"]:
            return
    elif config["PRETRAINED"]:
        load_pretrained(config, model_without_ddp, logger)

    if config["THROUGHPUT_MODE"]:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    patiance = config['PATIANCE']
    for epoch in range(config["TRAIN_START_EPOCH"], config["TRAIN_EPOCHS"]):
        data_loader_train.sampler.set_epoch(epoch)

        flag = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, grad_scaler)
        if dist.get_rank() == 0 and (epoch % config["SAVE_FREQ"] == 0 or epoch == (config["TRAIN_EPOCHS"] - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if max(max_accuracy, acc1) == max_accuracy: 
            patiance -= 1 
        else: 
            patiance = config['PATIANCE']
            
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

        wandb.log(
            {
                "epoch":epoch, 
                "acc@1": acc1,
                "acc@5": acc5, 
                "validation loss": loss
            }
        )
        if flag or patiance<=0 or max_accuracy<=10.0: 
            break 

    wandb.log({"max_accuracy":max_accuracy})
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, grad_scaler):
    model.train()
    optimizer.zero_grad()
    
    logger.info(f"Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}")

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
    
        samples = samples.cuda(non_blocking=True)
        targets = targets.type(torch.LongTensor).cuda(non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        if config["TRAIN_ACCUMULATION_STEPS"] > 1:
            if config["AMP"]:
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

                loss = loss / config["TRAIN_ACCUMULATION_STEPS"]
                if torch.isnan(loss):
                    return True
                grad_scaler.scale(loss).backward()
                if config["TRAIN_CLIP_GRAD"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN_CLIP_GRAD"])
                else: 
                    grad_norm = get_grad_norm(model.parameters())   

            else:
                outputs = model(samples)
                loss = criterion(outputs, targets)
                loss = loss / config["TRAIN_ACCUMULATION_STEPS"]
                if torch.isnan(loss):
                    return True 
                loss.backward()
                if config["TRAIN_CLIP_GRAD"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN_CLIP_GRAD"])
                else:
                    grad_norm = get_grad_norm(model.parameters())
            
            if (idx + 1) % config["TRAIN_ACCUMULATION_STEPS"] == 0:
                if config["AMP"]:
                    grad_scaler.step(optimizer)
                else: 
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
                if config["AMP"]:
                    grad_scaler.update()

        else:
            optimizer.zero_grad()
            if config["AMP"]:
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    return True
                grad_scaler.scale(loss).backward()
                if config["TRAIN_CLIP_GRAD"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN_CLIP_GRAD"])
                else: 
                    grad_norm = get_grad_norm(model.parameters())   

            else:
                outputs = model(samples)
                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    return True
                loss.backward()
                if config["TRAIN_CLIP_GRAD"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN_CLIP_GRAD"])
                else:
                    grad_norm = get_grad_norm(model.parameters())
            
            if config["AMP"]:
                grad_scaler.step(optimizer)
            else: 
                optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config["PRINT_FREQ"] == 0:
            lr = optimizer.param_groups[-1]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config["TRAIN_EPOCHS"]}][{idx}/{num_steps}]\t'
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB")
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
    return False

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config["PRINT_FREQ"] == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB")
    logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

if __name__ == "__main__": 
    _, config = parse_option()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config["SEED"] + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config["OUTPUT"], exist_ok=True)
    logger = create_logger(output_dir=config["OUTPUT"], dist_rank=dist.get_rank(), name=f'{config["MODEL_NAME"]}')

    # run = wandb.init()
    sweep_id = wandb.sweep(sweep_config, project=f'{config["MODEL_NAME"]}_{config["TAG"]}')
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config["TRAIN_BASE_LR"] * config["DATA_BATCH_SIZE"] * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config["TRAIN_WARMUP_LR"] * config["DATA_BATCH_SIZE"] * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config["TRAIN_MIN_LR"] * config["DATA_BATCH_SIZE"] * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config["TRAIN_ACCUMULATION_STEPS"] > 1:
        linear_scaled_lr = linear_scaled_lr * config["TRAIN_ACCUMULATION_STEPS"]
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config["TRAIN_ACCUMULATION_STEPS"]
        linear_scaled_min_lr = linear_scaled_min_lr * config["TRAIN_ACCUMULATION_STEPS"]
    config.update({
        "TRAIN_BASE_LR":linear_scaled_lr, 
        "TRAIN_WARMUP_LR":linear_scaled_warmup_lr, 
        "TRAIN_MIN_LR":linear_scaled_min_lr}, allow_val_change=True)
    if dist.get_rank() == 0:
        path = os.path.join(config["OUTPUT"], "config.json")
        with open(path, "w+") as f:
            json.dump(dict(config), f, indent=4)
        logger.info(f"Full config saved to {path}")

    wandb.agent(sweep_id, function=main, count=500)
    # main()

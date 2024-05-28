# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import os
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _str_to_pil_interpolation
import torch 

def build_loader_finetune(config, logger):
    dataset_train, config["MODEL_NUM_CLASSES"] = build_dataset(is_train=True, config=config, logger=logger)
    dataset_val, _ = build_dataset(is_train=False, config=config, logger=logger)
    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config["DATA_BATCH_SIZE"],
        num_workers=config["DATA_NUM_WORKERS"],
        pin_memory=config["DATA_PIN_MEMORY"],
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config["DATA_BATCH_SIZE"],
        num_workers=config["DATA_NUM_WORKERS"],
        pin_memory=config["DATA_PIN_MEMORY"],
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config["AUG_MIXUP"] > 0 or config["AUG_CUTMIX"] > 0 or config["AUG_CUTMIX_MINMAX"] is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config["AUG_MIXUP"], cutmix_alpha=config["AUG_CUTMIX"], cutmix_minmax=config["AUG_CUTMIX_MINMAX"],
            prob=config["AUG_MIXUP_PROB"], switch_prob=config["AUG_MIXUP_SWITCH_PROB"], mode=config["AUG_MIXUP_MODE"],
            label_smoothing=config["MODEL_LABEL_SMOOTHING"], num_classes=config["MODEL_NUM_CLASSES"])

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config, logger):
    transform = build_transform(is_train, config)
    logger.info(f"Fine-tune data transform, is_train={is_train}:\n{transform}")
    
    if config["DATA_DATASET"] == "imagenet":
        prefix = "train" if is_train else "val"
        root = os.path.join(config["DATA_DATA_PATH"], prefix)
        dataset = datasets.ImageFolder(
            root,
            "/mnt/c/Users/hiiam/Documents/MIM_Mask/SimMIM/images",
             transform=transform)
        nb_classes = 1000
    elif config["DATA_DATASET"] == "cifar10":
        dataset = datasets.CIFAR10(config["DATA_DATA_PATH"], train=is_train, 
                                    transform=transform, 
                                    download=True,
                                    target_transform=transforms.Compose([transforms.Lambda(lambda x: torch.tensor(x).to(torch.long))])
                                    )
        
        nb_classes=10
    else:
        raise NotImplementedError("We only support ImageNet and CIFAR10 Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config["DATA_IMG_SIZE"] > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config["DATA_IMG_SIZE"],
            is_training=True,
            color_jitter=config["AUG_COLOR_JITTER"] if config["AUG_COLOR_JITTER"] > 0 else None,
            auto_augment=config["AUG_AUTO_AUGMENT"] if config["AUG_AUTO_AUGMENT"] != "none" else None,
            re_prob=config["AUG_REPROB"],
            re_mode=config["AUG_REMODE"],
            re_count=config["AUG_RECOUNT"],
            interpolation=config["DATA_INTERPOLATION"],
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config["DATA_IMG_SIZE"], padding=4)
        return transform

    t = []
    if resize_im:
        if config["TEST_CROP"]:
            size = int((256 / 224) * config["DATA_IMG_SIZE"])
            t.append(
                transforms.Resize(size, interpolation=_str_to_pil_interpolation[config["DATA_INTERPOLATION"]]),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config["DATA_IMG_SIZE"]))
        else:
            t.append(
                transforms.Resize((config["DATA_IMG_SIZE"], config["DATA_IMG_SIZE"]),interpolation=_str_to_pil_interpolation[config["DATA_INTERPOLATION"]])
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
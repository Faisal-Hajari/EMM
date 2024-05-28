import wandb
import os 
import yaml
config = {
    # Data settings
    "DATA_BATCH_SIZE": 128,
    "DATA_DATA_PATH": "",
    "DATA_DATASET": "imagenet",
    "DATA_IMG_SIZE": 224,
    "DATA_INTERPOLATION": "bicubic",
    "DATA_PIN_MEMORY": True,
    "DATA_NUM_WORKERS": os.cpu_count(),
    "DATA_MASK_PATCH_SIZE": 32,  # SimMIM specific
    "DATA_MASK_RATIO": 0.6,       # SimMIM specific
    "DATA_MASK_STRAT": "random",  # SimMIM specific
    # Model settings
    "MODEL_TYPE": "swin",
    "MODEL_NAME": "swin_tiny_patch4_window7_224",
    "MODEL_RESUME": "",
    "MODEL_NUM_CLASSES": 1000,
    "MODEL_DROP_RATE": 0.0,
    "MODEL_DROP_PATH_RATE": 0.1,
    "MODEL_LABEL_SMOOTHING": 0.1,
        # #SWIN setting
        # "MODEL_SWIN_PATCH_SIZE": 4,
        # "MODEL_SWIN_IN_CHANS": 3,
        # "MODEL_SWIN_EMBED_DIM": 96,
        # "MODEL_SWIN_DEPTHS": [2, 2, 6, 2],
        # "MODEL_SWIN_NUM_HEADS": [3, 6, 12, 24],
        # "MODEL_SWIN_WINDOW_SIZE": 7,
        # "MODEL_SWIN_MLP_RATIO": 4.0,
        # "MODEL_SWIN_QKV_BIAS": True,
        # "MODEL_SWIN_QK_SCALE": None,
        # "MODEL_SWIN_APE": False,
        # "MODEL_SWIN_PATCH_NORM": True,
        
        #VIT settings
        "MODEL_VIT_PATCH_SIZE": 16,
        "MODEL_VIT_IN_CHANS": 3,
        "MODEL_VIT_EMBED_DIM": 768,
        "MODEL_VIT_DEPTH": 12,
        "MODEL_VIT_NUM_HEADS": 12,
        "MODEL_VIT_MLP_RATIO": 4,
        "MODEL_VIT_QKV_BIAS": True,
        "MODEL_VIT_INIT_VALUES": 0.1,
        "MODEL_VIT_USE_APE": False,
        "MODEL_VIT_USE_RPB": False,
        "MODEL_VIT_USE_SHARED_RPB": True,
        "MODEL_VIT_USE_MEAN_POOLING": False,
    # Training settings
    "TRAIN_START_EPOCH": 0,
    "TRAIN_EPOCHS": 300,
    "TRAIN_WARMUP_EPOCHS": 20,
    "TRAIN_WEIGHT_DECAY": 0.05,
    "TRAIN_BASE_LR": 5e-4,
    "TRAIN_WARMUP_LR": 5e-7,
    "TRAIN_MIN_LR": 5e-6,
    "TRAIN_CLIP_GRAD": 5.0,
    "TRAIN_AUTO_RESUME": True,
    "TRAIN_ACCUMULATION_STEPS": 0,  
    "TRAIN_USE_CHECKPOINT": False,
        # LR_SCHEDULER settings:
        "TRAIN_LR_SCHEDULER_NAME": "cosine",
        "TRAIN_LR_SCHEDULER_DECAY_EPOCHS": 30,
        "TRAIN_LR_SCHEDULER_DECAY_RATE": 0.1,
        "TRAIN_LR_SCHEDULER_GAMMA": 0.1,
        "TRAIN_LR_SCHEDULER_MULTISTEPS": [],
        # OPTIMIZER settings:
        "TRAIN_OPTIMIZER_NAME": "adamw",
        "TRAIN_OPTIMIZER_EPS": 1e-8,
        "TRAIN_OPTIMIZER_BETAS": (0.9, 0.999),
        "TRAIN_OPTIMIZER_MOMENTUM": 0.9,
    "TRAIN_LAYER_DECAY": 1.0, 
    # Augmentation settings
    "AUG_COLOR_JITTER": 0.4,
    "AUG_AUTO_AUGMENT": "rand-m9-mstd0.5-inc1",
    "AUG_REPROB": 0.25,
    "AUG_REMODE": "pixel",
    "AUG_RECOUNT": 1,
    "AUG_MIXUP": 0.8,
    "AUG_CUTMIX": 1.0,
    "AUG_CUTMIX_MINMAX": None,
    "AUG_MIXUP_PROB": 1.0,
    "AUG_MIXUP_SWITCH_PROB": 0.5,
    "AUG_MIXUP_MODE": "batch",
    # Testing
    "TEST_CROP":True,
    # Misc settings
    "AMP": False,  # Can be overwritten by command line argument
    "OUTPUT": "",  # Can be overwritten by command line argument
    "TAG": "default",  # Can be overwritten by command line argument
    "SAVE_FREQ": 1,
    "PRINT_FREQ": 10,
    "SEED": 0,
    "EVAL_MODE": False,  # Can be overwritten by command line argument
    "THROUGHPUT_MODE": False,  # Can be overwritten by command line argument
    "LOCAL_RANK": 0,
    "PRETRAINED": "",  # SimMIM specific
}


def flatten_dict(data, prefix=""):
  """
  Flattens a cascaded dictionary into a single level dictionary.

  Args:
      data: The cascaded dictionary to flatten.
      prefix: An optional prefix to prepend to the keys (default: "").

  Returns:
      A new dictionary with flattened keys and values.
  """
  flattened = {}
  for key, value in data.items():
    new_key = prefix + key if prefix else key
    if isinstance(value, dict):
      flattened.update(flatten_dict(value, new_key + "_"))  # Recursive call for nested dictionaries
    else:
      flattened[new_key] = value
  return flattened


def _update_config_from_file(config, cfg_file):
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    flattened_yaml_cfg = flatten_dict(yaml_cfg)
    for k, v in flattened_yaml_cfg.items(): 
        config[k] = v
    print("=> merge config from {}".format(cfg_file))
    

def update_config(config, args):
    _update_config_from_file(config, args.cfg)    
    def _check_args(name):
        if hasattr(args, name) and eval(f"args.{name}"):
            return True
        return False
    
    # merge from specific arguments
    if _check_args("batch_size"):
        config["DATA_BATCH_SIZE"] = args.batch_size
    if _check_args("data_path"):
        config["DATA_DATA_PATH"] = args.data_path
    if _check_args("resume"):
        config["MODEL_RESUME"] = args.resume
    if _check_args("pretrained"):
        config["PRETRAINED"] = args.pretrained
    if _check_args("accumulation_steps"):
        config["TRAIN_ACCUMULATION_STEPS"] = args.accumulation_steps
    if _check_args("use_checkpoint"):
        config["TRAIN_USE_CHECKPOINT"] = True
    if _check_args("amp"):
        config["AMP"] = args.amp
    if _check_args("output"):
        config["OUTPUT"] = args.output
    if _check_args("tag"):
        config["TAG"] = args.tag
    if _check_args("eval"):
        config["EVAL_MODE"] = True
    if _check_args("throughput"):
        config["THROUGHPUT_MODE"] = True

    # set local rank for distributed training
    config["LOCAL_RANK"] = int(os.environ["LOCAL_RANK"])

    # output folder
    config["OUTPUT"] = os.path.join(config["OUTPUT"], config["MODEL_NAME"], config["TAG"])



def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    update_config(config, args)

    return config
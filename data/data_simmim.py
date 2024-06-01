import numpy as np 
import torch 

from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR10
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
    
    def collate_fn(self, batch):
        if not isinstance(batch[0][0], tuple):
            return default_collate(batch)
        else:
            batch_num = len(batch)
            ret = []
            for item_idx in range(len(batch[0][0])):
                if batch[0][0][item_idx] is None:
                    ret.append(None)
                else:
                    ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
            ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
            return ret


class ExpandingMaskGenerator(MaskGenerator): 
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        super().__init__(input_size, mask_patch_size, model_patch_size, mask_ratio)
        """Note, here we relay on the image to be squares. for different aspact ratios 
        indexing logic might need to be changed.
        assums 0 is masked 1 is unmasked """ 
        self.n_rows = self.n_col = input_size//model_patch_size
        self.input_size = (input_size//model_patch_size)**2
        self.masks = self.__generate_mask()
    
    def __generate_mask(self): 
        masks = [] 
        top_left = bottom_right = (self.n_rows//2)
        masked_idxs_all = self.__cover_outline(top_left, bottom_right)
        for _ in range(self.n_rows//2):
            mask = np.ones((self.n_rows, self.n_rows), dtype=int)
            mask[masked_idxs_all[:, 0], masked_idxs_all[:, 1]] = 0
            masks.append(mask)
            top_left = torch.min(masked_idxs_all)-1
            bottom_right = torch.max(masked_idxs_all)+1 
            masked_idxs_all = torch.unique(
                torch.concat([masked_idxs_all, self.__cover_outline(top_left, bottom_right)], dim=0)
                , dim=0
                )
        masks = np.array(masks)
        return torch.tensor(masks)
    
    def __cover_outline(self, top_left:int, bottom_right:int): 
        top = [torch.tensor([p, top_left]) for p in range(top_left, bottom_right+1)]
        left = [torch.tensor([top_left, p]) for p in range(top_left, bottom_right+1)]
        right = [torch.tensor([bottom_right, p]) for p in range(top_left, bottom_right+1)]
        bottom = [torch.tensor([p, bottom_right]) for p in range(top_left, bottom_right+1)]
        mask_idx = torch.stack(top+left+right+bottom+ [torch.tensor([top_left,top_left]), torch.tensor([bottom_right, bottom_right])])
        mask_idx = torch.unique(mask_idx, dim=0)
        return mask_idx
    
    def __call__(self): 
        return self.masks
    
    def collate_fn(self, batch):
        batch = super().collate_fn(batch)
        GB, MB, C, H, W = batch[0].shape
        batch[0] = batch[0].view(GB*MB, C, H, W)
        
        GB, MB, SQ1, SQ2 = batch[1].shape
        batch[1] = batch[1].view(GB*MB, SQ1, SQ2)
        return batch

class ExpandingMaskTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.RandomResizedCrop(config["DATA_IMG_SIZE"], scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        self.mask_generator = ExpandingMaskGenerator(
            input_size=config["DATA_IMG_SIZE"],
            mask_patch_size=config["DATA_MASK_PATCH_SIZE"],
            model_patch_size=config["MODEL_VIT_PATCH_SIZE"],
            mask_ratio=config["DATA_MASK_RATIO"],
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        img = torch.stack([img for _ in range(len(mask))])
        return img,mask


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.RandomResizedCrop(config["DATA_IMG_SIZE"], scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])        
        self.mask_generator = MaskGenerator(
            input_size=config["DATA_IMG_SIZE"],
            mask_patch_size=config["DATA_MASK_PATCH_SIZE"],
            model_patch_size=config["MODEL_VIT_PATCH_SIZE"],
            mask_ratio=config["DATA_MASK_RATIO"],
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask
    
def build_loader_pretrain(config, logger):
    if config["DATA_MASK_STRAT"] == "random": 
        transform = SimMIMTransform(config)
    elif  config["DATA_MASK_STRAT"] == "expand":
        transform = ExpandingMaskTransform(config)
    else: 
        raise NotImplementedError
    
    logger.info(f"Pre-train data transform:\n{transform}")

    if config["DATA_DATASET"]=="cifar10":
        dataset = CIFAR10(config["DATA_DATA_PATH"], train=True, transform=transform, download=True)
    elif config["DATA_DATASET"]=="imagefolder":
        dataset = ImageFolder(config["DATA_DATA_PATH"], transform)

     
    
    logger.info(f"Build dataset: train images = {len(dataset)}")
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config["DATA_BATCH_SIZE"], sampler=sampler, 
                            num_workers=config["DATA_NUM_WORKERS"], pin_memory=True, drop_last=True, 
                            collate_fn=transform.mask_generator.collate_fn)

    return dataloader  
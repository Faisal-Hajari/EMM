# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .vision_transformer import VisionTransformer


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction="none")

        #wiegh the loss by how much of the image is masked (the more it is masked the lower the wieght)
        loss_wieght = mask.view(mask.shape[0], -1).sum(dim=1).to(loss_recon.dtype).softmax(0)
        loss_recon= loss_recon*loss_wieght

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    encoder = VisionTransformerForSimMIM(
        img_size=config["DATA_IMG_SIZE"],
        patch_size=config["MODEL_VIT_PATCH_SIZE"],
        in_chans=config["MODEL_VIT_IN_CHANS"],
        num_classes=0,
        embed_dim=config["MODEL_VIT_EMBED_DIM"],
        depth=config["MODEL_VIT_DEPTH"],
        num_heads=config["MODEL_VIT_NUM_HEADS"],
        mlp_ratio=config["MODEL_VIT_MLP_RATIO"],
        qkv_bias=config["MODEL_VIT_QKV_BIAS"],
        drop_rate=config["MODEL_DROP_RATE"],
        drop_path_rate=config["MODEL_DROP_PATH_RATE"],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=config["MODEL_VIT_INIT_VALUES"],
        use_abs_pos_emb=config["MODEL_VIT_USE_APE"],
        use_rel_pos_bias=config["MODEL_VIT_USE_RPB"],
        use_shared_rel_pos_bias=config["MODEL_VIT_USE_SHARED_RPB"],
        use_mean_pooling=config["MODEL_VIT_USE_MEAN_POOLING"])
    encoder_stride = config['MODEL_VIT_PATCH_SIZE']
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)
    return model

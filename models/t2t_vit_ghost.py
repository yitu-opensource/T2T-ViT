# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT-Ghost
"""
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from .transformer_block import Block, get_sinusoid_encoding
from .t2t_vit import T2T_module, _cfg


default_cfgs = {
    'T2t_vit_16_ghost': _cfg(),
}

class Mlp_ghost(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.ratio = hidden_features//in_features
        self.cheap_operation2 = nn.Conv1d(in_features, in_features, kernel_size=1, groups=in_features, bias=False)
        self.cheap_operation3 = nn.Conv1d(in_features, in_features, kernel_size=1, groups=in_features, bias=False)

    def forward(self, x):  # x: [B, N, C]
        x1 = self.fc1(x)   # x1: [B, N, C]
        x1 = self.act(x1)

        x2 = self.cheap_operation2(x1.transpose(1,2))  # x2: [B, N, C]
        x2 = x2.transpose(1,2)
        x2 = self.act(x2)

        x3 = self.cheap_operation3(x1.transpose(1, 2))  # x3: [B, N, C]
        x3 = x3.transpose(1, 2)
        x3 = self.act(x3)

        x = torch.cat((x1, x2, x3), dim=2)  # x: [B, N, 3C]
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention_ghost(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        half_dim = int(0.5*dim)
        self.q = nn.Linear(dim, half_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, half_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, half_dim, bias=qkv_bias)

        self.cheap_operation_q = nn.Conv1d(half_dim, half_dim, kernel_size=1, groups=half_dim, bias=False)
        self.cheap_operation_k = nn.Conv1d(half_dim, half_dim, kernel_size=1, groups=half_dim, bias=False)
        self.cheap_operation_v = nn.Conv1d(half_dim, half_dim, kernel_size=1, groups=half_dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q1 = self.cheap_operation_q(q.transpose(1,2)).transpose(1,2)
        k1 = self.cheap_operation_k(k.transpose(1,2)).transpose(1,2)
        v1 = self.cheap_operation_v(v.transpose(1,2)).transpose(1,2)

        q = torch.cat((q, q1), dim=2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = torch.cat((k, k1), dim=2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = torch.cat((v, v1), dim=2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_ghost(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_ghost(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class T2T_ViT_Ghost(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def t2t_vit_16_ghost(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT_Ghost(tokens_type='performer', embed_dim=384, depth=16, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_16_ghost']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

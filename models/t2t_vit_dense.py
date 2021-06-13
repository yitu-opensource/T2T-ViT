# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT-Dense
"""
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from .transformer_block import Mlp, Block, get_sinusoid_encoding
from .t2t_vit import T2T_module, _cfg

default_cfgs = {
    't2t_vit_dense': _cfg(),
}

class Transition(nn.Module):
    def __init__(self, in_features, out_features,  act_layer=nn.GELU):
        super(Transition, self).__init__()
        self.act =  act_layer()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, growth_rate, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  #, out_features=growth_rate
        self.dense_linear = nn.Linear(dim, growth_rate)

    def forward(self, x):
        new_x = x + self.drop_path(self.attn(self.norm1(x)))
        new_x = new_x + self.drop_path(self.mlp(self.norm2(new_x)))
        new_x = self.dense_linear(new_x)
        x = torch.cat([x, new_x], 2)  # dense connnection: concate all the old features with new features in channel dimension
        return x

class T2T_ViT_Dense(nn.Module):
    def __init__(self, growth_rate=32, tokens_type='performer', block_config=(3, 4, 6, 3), img_size=224, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
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
        self.blocks = nn.ModuleList()

        start_dim = embed_dim
        for i, num_layers in enumerate(block_config):
            for j in range(num_layers):
                new_dim = start_dim + j * growth_rate
                block = Block(growth_rate=growth_rate, dim=new_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                self.blocks.append(block)
            if i != len(block_config)-1:
                transition = Transition(new_dim+growth_rate, (new_dim+growth_rate)//2)
                self.blocks.append(transition)
                start_dim = int((new_dim+growth_rate)//2)
        out_dim = new_dim + growth_rate
        print(f'end dim:{out_dim}')
        self.norm = norm_layer(out_dim)

        # Classifier head
        self.head = nn.Linear(out_dim, num_classes) if num_classes > 0 else nn.Identity()

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

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed # self.pos_embed
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
def t2t_vit_dense(pretrained=False, **kwargs):
    model = T2T_ViT_Dense(growth_rate=64, block_config=(3, 6, 6, 4), embed_dim=128, num_heads=8, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['t2t_vit_dense']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

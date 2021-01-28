# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Take Performer as T2T Transformer
"""
import math
import torch
import torch.nn as nn

class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        emb = dim * head_cnt
        self.kqv = nn.Linear(dim, 3 * dim)
        self.dp = nn.Dropout(dp1)
        self.proj1 = nn.Linear(in_dim, dim)
        self.proj2 = nn.Linear(emb, emb)
        self.head_cnt = head_cnt
        self.dim = dim
        self.ln1 = nn.LayerNorm(emb)
        self.ln2 = nn.LayerNorm(emb)

        self.mlp = nn.Sequential(
            nn.Linear(emb, 1 * emb),
            nn.GELU(),
            nn.Linear(1 * emb, emb),
            nn.Dropout(dp2),
        )

        self.m = int(dim * kernel_ratio)
        self.w = torch.randn(self.m, emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    # updating

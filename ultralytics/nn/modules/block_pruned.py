"""Pruned building blocks for YOLO26 structured channel pruning."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Attention
from ultralytics.nn.modules.conv import Conv


class BottleneckPruned(nn.Module):
    """Pruned Bottleneck (shared by C3k2 / C3k inner blocks)."""

    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, k[0], 1)
        self.cv2 = Conv(cv1out, cv2out, k[1], 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3kPruned(nn.Module):
    """Pruned C3k (CSP bottleneck used inside C3k2 when c3k=True)."""

    def __init__(
        self,
        cv1in: int,
        cv1out: int,
        cv2out: int,
        cv3in: int,
        cv3out: int,
        inner_cv1outs: list[int],
        inner_cv2outs: list[int],
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ):
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in, cv3out, 1, 1)
        self.m = nn.Sequential(
            *(
                BottleneckPruned(cv1out, inner_cv1outs[i], inner_cv2outs[i], shortcut, g, k=(k, k), e=1.0)
                for i in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2Pruned(nn.Module):
    """Pruned C3k2 with Bottleneck inner blocks (c3k=False)."""

    def __init__(
        self,
        cv1in: int,
        cv1out: int,
        cv1_split_sections: list[int],
        inner_cv1outs: list[int],
        inner_cv2outs: list[int],
        cv2out: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = cv1_split_sections[1]
        self.cv1_split_sections = cv1_split_sections
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        if shortcut and all(o == self.c for o in inner_cv2outs):
            self.m = nn.ModuleList(
                BottleneckPruned(self.c, inner_cv1outs[i], self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                for i in range(n)
            )
            cv2_in = sum(cv1_split_sections) + n * self.c
        else:
            self.m = nn.ModuleList()
            c_hidden = self.c
            for i in range(n):
                self.m.append(
                    BottleneckPruned(c_hidden, inner_cv1outs[i], inner_cv2outs[i], False, g, k=((3, 3), (3, 3)), e=1.0)
                )
                c_hidden = inner_cv2outs[i]
            cv2_in = sum(cv1_split_sections) + sum(inner_cv2outs)
        self.cv2 = Conv(cv2_in, cv2out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2C3kPruned(nn.Module):
    """Pruned C3k2 with C3k inner blocks (c3k=True)."""

    def __init__(
        self,
        cv1in: int,
        cv1out: int,
        cv1_split_sections: list[int],
        c3k_args: list[tuple],
        cv2out: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = cv1_split_sections[1]
        self.cv1_split_sections = cv1_split_sections
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.m = nn.ModuleList(C3kPruned(*c3k_args[i]) for i in range(n))
        c3k_outs = [a[4] for a in c3k_args]  # cv3out per C3kPruned
        cv2_in = sum(cv1_split_sections) + sum(c3k_outs)
        self.cv2 = Conv(cv2_in, cv2out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class PSABlockPruned(nn.Module):
    """Pruned PSABlock: attention at width c; FFN convs pruned."""

    def __init__(
        self,
        c: int,
        ffn_cv1out: int,
        ffn_cv2out: int,
        attn_ratio: float = 0.5,
        num_heads: int = 4,
        shortcut: bool = True,
    ):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, ffn_cv1out, 1), Conv(ffn_cv1out, ffn_cv2out, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C3k2AttnPruned(nn.Module):
    """Pruned C3k2 with attn=True (Sequential(Bottleneck, PSABlock) per repeat)."""

    def __init__(
        self,
        cv1in: int,
        cv1out: int,
        cv1_split_sections: list[int],
        inner_cv1out: int,
        inner_cv2out: int,
        psa_ffn_cv1out: int,
        psa_ffn_cv2out: int,
        cv2out: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        attn_ratio: float = 0.5,
        num_heads: int = 1,
    ):
        super().__init__()
        self.c = cv1_split_sections[1]
        self.cv1_split_sections = cv1_split_sections
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        # Bottleneck output: inner_cv2out (may differ from self.c after pruning)
        # PSABlock must match Bottleneck output, not self.c
        bot_out = self.c if (shortcut and self.c == inner_cv2out) else inner_cv2out
        self.m = nn.ModuleList(
            nn.Sequential(
                BottleneckPruned(self.c, inner_cv1out, inner_cv2out, shortcut, g, k=((3, 3), (3, 3)), e=1.0),
                PSABlockPruned(bot_out, psa_ffn_cv1out, psa_ffn_cv2out, attn_ratio, num_heads, shortcut=True),
            )
            for _ in range(n)
        )
        cv2_in = sum(cv1_split_sections) + n * bot_out
        self.cv2 = Conv(cv2_in, cv2out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2PSAPruned(nn.Module):
    """Pruned C2PSA (backbone attention block)."""

    def __init__(
        self,
        cv1in: int,
        cv1out: int,
        cv1_split_sections: list[int],
        psa_ffn_cv1outs: list[int],
        psa_ffn_cv2outs: list[int],
        cv2out: int,
        n: int = 1,
        e: float = 0.5,
        attn_ratio: float = 0.5,
    ):
        super().__init__()
        self.c = cv1_split_sections[1]
        self.cv1_split_sections = cv1_split_sections
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 1, 1)
        num_heads = max(self.c // 64, 1)
        self.m = nn.Sequential(
            *(
                PSABlockPruned(self.c, psa_ffn_cv1outs[i], psa_ffn_cv2outs[i], attn_ratio, num_heads, shortcut=True)
                for i in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split(self.cv1_split_sections, dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class SPPFPruned(nn.Module):
    """Pruned SPPF for YOLO26 (n=3 pooling branches, optional residual)."""

    def __init__(self, cv1in: int, cv1out: int, cv2out: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1, act=False)  # matches SPPF.cv1 (block.py:226): no activation
        self.cv2 = Conv(cv1out * (n + 1), cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and cv1in == cv2out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y

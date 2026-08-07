"""Pruned detection head for YOLO26 (end-to-end, reg_max=1, DWConv cls branch)."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.utils.tal import dist2bbox, make_anchors


class DetectPruned(nn.Module):
    """Pruned YOLO26 Detect head (reg_max=1, optional end-to-end one-to-one branch)."""

    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(
        self,
        cv2x0_outs: list[int],
        cv2x1_outs: list[int],
        cv3x0_outs: list[int],
        cv3x1_outs: list[int],
        nc: int = 80,
        ch: tuple = (),
        reg_max: int = 1,
        end2end: bool = True,
    ):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self._end2end = end2end

        def _box_branch(x, c20, c21):
            return nn.Sequential(
                Conv(x, c20, 3),
                Conv(c20, c21, 3),
                nn.Conv2d(c21, 4 * self.reg_max, 1),
            )

        def _cls_branch(x, c30, c31):
            return nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c30, 1)),
                nn.Sequential(DWConv(c30, c30, 3), Conv(c30, c31, 1)),
                nn.Conv2d(c31, self.nc, 1),
            )

        self.cv2 = nn.ModuleList(_box_branch(x, cv2x0_outs[i], cv2x1_outs[i]) for i, x in enumerate(ch))
        self.cv3 = nn.ModuleList(_cls_branch(x, cv3x0_outs[i], cv3x1_outs[i]) for i, x in enumerate(ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Whether one-to-one (end-to-end) branch is active."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one_cv2")

    @end2end.setter
    def end2end(self, value):
        self._end2end = value

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self._end2end and hasattr(self, "one2one_cv2"):
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        o2o = preds["one2one"] if self._end2end and isinstance(preds, dict) else preds
        y = self._inference(o2o)
        if self._end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x):
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        # Matches Detect.decode_bboxes (head.py:215): for end2end heads, dist2bbox must
        # decode straight to xyxy — postprocess() (below) assumes [x1,y1,x2,y2,...] input,
        # not xywh. Hardcoding xywh=True here (regardless of end2end) was producing
        # xywh-format boxes that postprocess() then silently mis-split as xyxy.
        xywh = not self._end2end
        dbox = dist2bbox(self.dfl(x["boxes"]), self.anchors.unsqueeze(0), xywh=xywh, dim=1) * self.strides
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def postprocess(self, preds):
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores, max_det):
        batch_size, anchors, nc = scores.shape
        k = max_det if self.export else min(max_det, anchors)
        if getattr(self, "agnostic_nms", False):
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]
        return scores[..., None], (index % nc)[..., None].float(), idx

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        if self._end2end and hasattr(self, "one2one_cv2"):
            for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

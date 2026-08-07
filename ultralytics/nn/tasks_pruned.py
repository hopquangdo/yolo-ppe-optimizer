"""Detection model builder for pruned YOLO26 architectures."""

from __future__ import annotations

import contextlib
from copy import deepcopy

import torch
import torch.nn as nn

from ultralytics.nn.modules.block_pruned import (
    C2PSAPruned,
    C3k2AttnPruned,
    C3k2C3kPruned,
    C3k2Pruned,
    SPPFPruned,
)
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights, scale_img


class DetectionModelPruned(BaseModel):
    """YOLO26 detection model built from pruned yaml + ``maskbndict``."""

    def __init__(self, maskbndict, cfg, ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg
        if nc and nc != self.yaml.get("nc"):
            self.yaml["nc"] = nc
        self.model, self.save, self.current_to_prev = parse_model_pruned(maskbndict, deepcopy(cfg), ch)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, DetectPruned):
            s = 256
            m.inplace = self.inplace

            def _forward(x):
                output = self.forward(x)
                if self.end2end and isinstance(output, dict):
                    output = output["one2many"]
                return output["feats"]

            self.model.eval()
            m.training = True
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    @property
    def end2end(self):
        """Return whether the model uses end-to-end NMS-free detection."""
        if not hasattr(self, "model") or not len(self.model):
            return self.yaml.get("end2end", True)
        return getattr(self.model[-1], "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self.set_head_attr(end2end=value)

    def set_head_attr(self, **kwargs):
        """Set attributes on the detection head (DetectPruned)."""
        head = self.model[-1]
        for k, v in kwargs.items():
            if not hasattr(head, k):
                LOGGER.warning(f"Head has no attribute '{k}'.")
                continue
            setattr(head, k, v)

    def init_criterion(self):
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)

    def _predict_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        p[:, :4] /= scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))
        e = 1
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))
        y[0] = y[0][..., :-i]
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]
        return y


def _mask_sum(maskbndict, name):
    return torch.sum(maskbndict[name]).int().item()


def parse_model_pruned(maskbndict, d, ch, verbose=True):
    import ast

    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    reg_max = d.get("reg_max", 1)
    end2end = d.get("end2end", True)
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    else:
        max_channels = float("inf")
    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<50}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    current_to_prev = {}
    idx_to_bn_layer_name = {}
    prev_module = None
    prev_bn_layer_name = None

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else globals()[m]
        ) if isinstance(m, str) else m
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        base_name = f"model.{i}"

        if m in [Conv]:
            c1 = ch[f]
            bn_layer_name = base_name + ".bn"
            mask = maskbndict[bn_layer_name]
            c2 = _mask_sum(maskbndict, bn_layer_name)
            args = [c1, c2, *args[1:]]
            if i == 0:
                prev_bn_layer_name = bn_layer_name
            else:
                current_to_prev[bn_layer_name] = prev_bn_layer_name
                prev_bn_layer_name = bn_layer_name
            idx_to_bn_layer_name[i] = bn_layer_name

        elif m in [C3k2Pruned]:
            c1, args, c2, links = _parse_c3k2_bottleneck(maskbndict, base_name, n, ch, f, args, i, idx_to_bn_layer_name, prev_bn_layer_name, prev_module)
            current_to_prev.update(links["current_to_prev"])
            prev_bn_layer_name = links["prev_bn"]
            idx_to_bn_layer_name[i] = links["idx_bn"]
            n = 1

        elif m in [C3k2C3kPruned]:
            c1, args, c2, links = _parse_c3k2_c3k(maskbndict, base_name, n, ch, f, args, i, idx_to_bn_layer_name, prev_bn_layer_name, prev_module)
            current_to_prev.update(links["current_to_prev"])
            prev_bn_layer_name = links["prev_bn"]
            idx_to_bn_layer_name[i] = links["idx_bn"]
            n = 1

        elif m in [C3k2AttnPruned]:
            c1, args, c2, links = _parse_c3k2_attn(maskbndict, base_name, n, ch, f, args, i, idx_to_bn_layer_name, prev_bn_layer_name, prev_module)
            current_to_prev.update(links["current_to_prev"])
            prev_bn_layer_name = links["prev_bn"]
            idx_to_bn_layer_name[i] = links["idx_bn"]
            n = 1

        elif m in [C2PSAPruned]:
            c1 = ch[f]
            cv1_bn = base_name + ".cv1.bn"
            cv2_bn = base_name + ".cv2.bn"
            cv1_mask = maskbndict[cv1_bn]
            cv1out = _mask_sum(maskbndict, cv1_bn)
            cv1_split = [torch.sum(cv1_mask.chunk(2, 0)[0]).int().item(), torch.sum(cv1_mask.chunk(2, 0)[1]).int().item()]
            psa_ffn_cv1outs, psa_ffn_cv2outs = [], []
            for pi in range(n):
                ffn0_bn = base_name + f".m.{pi}.ffn.0.bn"
                ffn1_bn = base_name + f".m.{pi}.ffn.1.bn"
                psa_ffn_cv1outs.append(_mask_sum(maskbndict, ffn0_bn))
                psa_ffn_cv2outs.append(_mask_sum(maskbndict, ffn1_bn))
            cv2out = _mask_sum(maskbndict, cv2_bn)
            args = [c1, cv1out, cv1_split, psa_ffn_cv1outs, psa_ffn_cv2outs, cv2out, n, *args[1:]]
            c2 = cv2out
            current_to_prev[cv1_bn] = prev_bn_layer_name
            prev_bn = cv1_bn
            for pi in range(n):
                ffn0_bn = base_name + f".m.{pi}.ffn.0.bn"
                ffn1_bn = base_name + f".m.{pi}.ffn.1.bn"
                current_to_prev[ffn0_bn] = prev_bn
                current_to_prev[ffn1_bn] = ffn0_bn
                prev_bn = ffn1_bn
            current_to_prev[cv2_bn] = prev_bn
            prev_bn_layer_name = cv2_bn
            idx_to_bn_layer_name[i] = cv2_bn

        elif m in [SPPFPruned]:
            cv1in = ch[f]
            cv1_bn = base_name + ".cv1.bn"
            cv2_bn = base_name + ".cv2.bn"
            cv1out = _mask_sum(maskbndict, cv1_bn)
            cv2out = _mask_sum(maskbndict, cv2_bn)
            k = args[1] if len(args) > 1 else 5
            pool_n = args[2] if len(args) > 2 else 3
            shortcut = args[3] if len(args) > 3 else False
            args = [cv1in, cv1out, cv2out, k, pool_n, shortcut]
            c2 = cv2out
            current_to_prev[cv1_bn] = prev_bn_layer_name
            current_to_prev[cv2_bn] = cv1_bn
            prev_bn_layer_name = cv2_bn
            idx_to_bn_layer_name[i] = cv2_bn

        elif m in [nn.Upsample]:
            c2 = ch[f]
            idx_to_bn_layer_name[i] = idx_to_bn_layer_name[i - 1]
            prev_bn_layer_name = idx_to_bn_layer_name[i]

        elif m in [Concat]:
            c2 = sum(ch[x] for x in f)
            fx = [x if x != -1 else i + x for x in f]
            # Resolve layer indices to actual BN names
            concat_bns = []
            for ix in fx:
                bn = idx_to_bn_layer_name[ix]
                if isinstance(bn, list):
                    concat_bns.extend(bn)
                else:
                    concat_bns.append(bn)
            idx_to_bn_layer_name[i] = concat_bns
            prev_bn_layer_name = concat_bns

        elif m in [DetectPruned]:
            args.append([ch[x] for x in f])
            cv2x0 = [base_name + f".cv2.{k}.0.bn" for k in range(3)]
            cv2x1 = [base_name + f".cv2.{k}.1.bn" for k in range(3)]
            cv3x0 = [base_name + f".cv3.{k}.0.1.bn" for k in range(3)]
            cv3x1 = [base_name + f".cv3.{k}.1.1.bn" for k in range(3)]
            cv2x0_outs = [_mask_sum(maskbndict, x) for x in cv2x0]
            cv2x1_outs = [_mask_sum(maskbndict, x) for x in cv2x1]
            cv3x0_outs = [_mask_sum(maskbndict, x) for x in cv3x0]
            cv3x1_outs = [_mask_sum(maskbndict, x) for x in cv3x1]
            args = [cv2x0_outs, cv2x1_outs, cv3x0_outs, cv3x1_outs, *args, reg_max, end2end]
            for ix, (a0, b0) in enumerate(zip(cv2x0, cv3x0)):
                current_to_prev[a0] = idx_to_bn_layer_name[f[ix]]
                current_to_prev[b0] = idx_to_bn_layer_name[f[ix]]
            for ix in range(3):
                current_to_prev[cv2x1[ix]] = cv2x0[ix]
                current_to_prev[cv3x1[ix]] = cv3x0[ix]
            for ix in range(3):
                feat_bn = idx_to_bn_layer_name[f[ix]]
                current_to_prev[base_name + f".cv2.{ix}.2"] = cv2x1[ix]
                current_to_prev[base_name + f".cv3.{ix}.2"] = cv3x1[ix]
                # DWConv BNs in cls branch (cv3): DWConv input = feature/prev conv output
                current_to_prev[base_name + f".cv3.{ix}.0.0.bn"] = feat_bn
                current_to_prev[base_name + f".cv3.{ix}.1.0.bn"] = cv3x0[ix]
                if end2end:
                    o2o_cv2x0 = base_name + f".one2one_cv2.{ix}.0.bn"
                    o2o_cv2x1 = base_name + f".one2one_cv2.{ix}.1.bn"
                    o2o_cv3x0 = base_name + f".one2one_cv3.{ix}.0.1.bn"
                    o2o_cv3x1 = base_name + f".one2one_cv3.{ix}.1.1.bn"
                    current_to_prev[o2o_cv2x0] = feat_bn
                    current_to_prev[o2o_cv3x0] = feat_bn
                    current_to_prev[o2o_cv2x1] = o2o_cv2x0
                    current_to_prev[o2o_cv3x1] = o2o_cv3x0
                    current_to_prev[base_name + f".one2one_cv2.{ix}.2"] = o2o_cv2x1
                    current_to_prev[base_name + f".one2one_cv3.{ix}.2"] = o2o_cv3x1
                    current_to_prev[base_name + f".one2one_cv3.{ix}.0.0.bn"] = feat_bn
                    current_to_prev[base_name + f".one2one_cv3.{ix}.1.0.bn"] = o2o_cv3x0

        else:
            raise ValueError(f"ERROR module {m} not supported in parse_model_pruned.")

        prev_module = m
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<50}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), current_to_prev


def _parse_c3k2_bottleneck(maskbndict, base_name, n, ch, f, args, i, idx_to_bn, prev_bn, prev_module):
    cv1in = ch[f]
    cv1_bn = base_name + ".cv1.bn"
    inner_cv1_bns = [base_name + f".m.{k}.cv1.bn" for k in range(n)]
    inner_cv2_bns = [base_name + f".m.{k}.cv2.bn" for k in range(n)]
    cv2_bn = base_name + ".cv2.bn"
    cv1_mask = maskbndict[cv1_bn]
    cv1out = _mask_sum(maskbndict, cv1_bn)
    cv1_split = [torch.sum(cv1_mask.chunk(2, 0)[0]).int().item(), torch.sum(cv1_mask.chunk(2, 0)[1]).int().item()]
    inner_cv1outs = [_mask_sum(maskbndict, x) for x in inner_cv1_bns]
    inner_cv2outs = [_mask_sum(maskbndict, x) for x in inner_cv2_bns]
    cv2out = _mask_sum(maskbndict, cv2_bn)
    # NOTE: yaml args[1] here is NOT shortcut — it's the `c3k` flag (see tasks.py's
    # parse_model: args = [c1, c2, *args[1:]]; args.insert(2, n) means arg positions
    # after c1,c2,n map to c3k,e,attn,g,shortcut). yolo26.yaml never overrides
    # `shortcut` for any C3k2 variant, so it is always the class default True.
    shortcut = True
    out_args = [cv1in, cv1out, cv1_split, inner_cv1outs, inner_cv2outs, cv2out, n, shortcut]
    links = _link_c3k2_inner(cv1_bn, cv2_bn, inner_cv1_bns, inner_cv2_bns, n, prev_bn, prev_module, i, f, idx_to_bn)
    return cv1in, out_args, cv2out, links


def _parse_c3k2_c3k(maskbndict, base_name, n, ch, f, args, i, idx_to_bn, prev_bn, prev_module):
    cv1in = ch[f]
    cv1_bn = base_name + ".cv1.bn"
    cv2_bn = base_name + ".cv2.bn"
    cv1_mask = maskbndict[cv1_bn]
    cv1out = _mask_sum(maskbndict, cv1_bn)
    cv1_split = [torch.sum(cv1_mask.chunk(2, 0)[0]).int().item(), torch.sum(cv1_mask.chunk(2, 0)[1]).int().item()]
    c_hidden = cv1_split[1]
    c3k_n = 2
    c3k_args = []
    for j in range(n):
        p = base_name + f".m.{j}"
        cv1o = _mask_sum(maskbndict, p + ".cv1.bn")
        cv2o = _mask_sum(maskbndict, p + ".cv2.bn")
        cv3o = _mask_sum(maskbndict, p + ".cv3.bn")
        in1 = [_mask_sum(maskbndict, p + f".m.{k}.cv1.bn") for k in range(c3k_n)]
        in2 = [_mask_sum(maskbndict, p + f".m.{k}.cv2.bn") for k in range(c3k_n)]
        c3k_args.append((c_hidden, cv1o, cv2o, cv1o + cv2o, cv3o, in1, in2, c3k_n, True, 1, 0.5, 3))
        c_hidden = cv3o
    cv2out = _mask_sum(maskbndict, cv2_bn)
    # NOTE: yaml args[1] here is NOT shortcut — it's the `c3k` flag (see tasks.py's
    # parse_model: args = [c1, c2, *args[1:]]; args.insert(2, n) means arg positions
    # after c1,c2,n map to c3k,e,attn,g,shortcut). yolo26.yaml never overrides
    # `shortcut` for any C3k2 variant, so it is always the class default True.
    shortcut = True
    out_args = [cv1in, cv1out, cv1_split, c3k_args, cv2out, n, shortcut]
    current_to_prev = {cv1_bn: prev_bn}
    prev = cv1_bn
    for j in range(n):
        p = base_name + f".m.{j}"
        c3k_input = prev  # input to this C3k module
        current_to_prev[p + ".cv1.bn"] = c3k_input  # cv1 takes x
        current_to_prev[p + ".cv2.bn"] = c3k_input  # cv2 takes x (same input)
        for k in range(c3k_n):
            bot_input = p + ".cv1.bn" if k == 0 else p + f".m.{k - 1}.cv2.bn"
            current_to_prev[p + f".m.{k}.cv1.bn"] = bot_input
            current_to_prev[p + f".m.{k}.cv2.bn"] = p + f".m.{k}.cv1.bn"
        last_bot_out = p + f".m.{c3k_n - 1}.cv2.bn"
        current_to_prev[p + ".cv3.bn"] = [last_bot_out, p + ".cv2.bn"]
        prev = p + ".cv3.bn"
    current_to_prev[cv2_bn] = [cv1_bn] + [base_name + f".m.{j}.cv3.bn" for j in range(n)]
    return cv1in, out_args, cv2out, {"current_to_prev": current_to_prev, "prev_bn": cv2_bn, "idx_bn": cv2_bn}


def _parse_c3k2_attn(maskbndict, base_name, n, ch, f, args, i, idx_to_bn, prev_bn, prev_module):
    cv1in = ch[f]
    cv1_bn = base_name + ".cv1.bn"
    cv2_bn = base_name + ".cv2.bn"
    cv1_mask = maskbndict[cv1_bn]
    cv1out = _mask_sum(maskbndict, cv1_bn)
    cv1_split = [torch.sum(cv1_mask.chunk(2, 0)[0]).int().item(), torch.sum(cv1_mask.chunk(2, 0)[1]).int().item()]
    b0_cv1 = base_name + ".m.0.0.cv1.bn"
    b0_cv2 = base_name + ".m.0.0.cv2.bn"
    ffn0 = base_name + ".m.0.1.ffn.0.bn"
    ffn1 = base_name + ".m.0.1.ffn.1.bn"
    inner_cv1out = _mask_sum(maskbndict, b0_cv1)
    inner_cv2out = _mask_sum(maskbndict, b0_cv2)
    psa_ffn_cv1out = _mask_sum(maskbndict, ffn0)
    psa_ffn_cv2out = _mask_sum(maskbndict, ffn1)
    cv2out = _mask_sum(maskbndict, cv2_bn)
    # NOTE: yaml args[1] here is NOT shortcut — it's the `c3k` flag (see tasks.py's
    # parse_model: args = [c1, c2, *args[1:]]; args.insert(2, n) means arg positions
    # after c1,c2,n map to c3k,e,attn,g,shortcut). yolo26.yaml never overrides
    # `shortcut` for any C3k2 variant, so it is always the class default True.
    shortcut = True
    e = args[2] if len(args) > 2 else 0.5
    attn = args[3] if len(args) > 3 else True
    # num_heads must match original C3k2's attn branch: max(self.c // 64, 1)
    # (block.py: PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))).
    # self.c == cv1_split[1] (see C3k2AttnPruned.__init__). Hardcoding this to 1 silently
    # changes the multi-head attention split and corrupts the block's output.
    num_heads = max(cv1_split[1] // 64, 1)
    out_args = [cv1in, cv1out, cv1_split, inner_cv1out, inner_cv2out, psa_ffn_cv1out, psa_ffn_cv2out, cv2out, n, shortcut, 1, e, 0.5, num_heads]
    # cv2 input = cat(cv1_split_0, cv1_split_1, sequential_output)
    # Sequential(Bottleneck, PSABlock) output = b0_cv2 channels (PSABlock shortcut preserves)
    current_to_prev = {cv1_bn: prev_bn, b0_cv1: cv1_bn, b0_cv2: b0_cv1, ffn0: b0_cv2, ffn1: ffn0, cv2_bn: [cv1_bn, b0_cv2]}
    return cv1in, out_args, cv2out, {"current_to_prev": current_to_prev, "prev_bn": cv2_bn, "idx_bn": cv2_bn}


def _link_c3k2_inner(cv1_bn, cv2_bn, inner_cv1_bns, inner_cv2_bns, n, prev_bn, prev_module, i, f, idx_to_bn):
    current_to_prev = {cv1_bn: prev_bn}
    prev = cv1_bn
    prev_list = [cv1_bn]
    for k in range(n):
        current_to_prev[inner_cv1_bns[k]] = prev
        current_to_prev[inner_cv2_bns[k]] = inner_cv1_bns[k]
        prev = inner_cv2_bns[k]
        prev_list.append(inner_cv2_bns[k])
    current_to_prev[cv2_bn] = prev_list
    return {"current_to_prev": current_to_prev, "prev_bn": cv2_bn, "idx_bn": cv2_bn}

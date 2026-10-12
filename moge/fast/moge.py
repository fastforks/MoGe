"""MoGe-2 fast path with custom kernels (adapted from exp 004 mono_fast/moge.py @ 2171802): same math as `MoGeModel.infer`
(resolution_level -> num_tokens, mask, metric scale, focal/shift recovery), any input size, nothing compiled per size.

Changes vs upstream (fp32 by default; `infer` default is fp16 autocast):
- DINOv2 encoder in pure fp16 (`vit.FastViT`) with fp16-accumulate GEMMs and the head_dim-64 flash-attention kernel with fp16
  accumulation (vendored fastbb)
- the 4 output projections (1x1 convs, summed) as 4 accumulating GEMMs on the token layout (no NCHW permutes)
- neck + heads in fp16 channels-last; the constant UV inputs are folded into the 1x1 input convs (one GEMM + a per-size bias)
- the normal head is not run (depth, points and mask do not use it); no normal output
- the points / mask final 1x1 convs in fp32 (upstream: points fp32_output_projection); output resize + remap in fp32
- with `kernels`: ReLU + replicate padding and the 2x bilinear upsample as prebuilt NHWC kernels (`ops`), and the 3x3 / 1x1
  convolutions as fp16-accumulate cuDNN graphs (pip install nvidia-cudnn-frontend)

`kernels=False` (or the environment variable MOGE_FAST_KERNELS=0) keeps plain PyTorch ops only (SDPA, F.interpolate, F.pad,
cuDNN via PyTorch); a kernel that cannot be loaded on this machine falls back to its plain op with a warning.
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift
from . import ops
from .vit import FastViT

try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d


def _probe_kernels():
    """Which custom kernels can run here: (NHWC bilinear / padding kernels, flash attention kernel, cuDNN fp16-acc convs)."""
    bil = ops.available()
    try:
        from .fastbb_vendored import kernels as fk

        fk.Kernels.get()
        flash = True
    except Exception:
        flash = False
    try:
        from .fastbb_vendored import cudnn_conv

        conv = cudnn_conv.cudnn is not None
    except Exception:
        conv = False
    return bil, flash, conv


class FastConvStack:
    """ConvStack forward in fp16 NHWC. kern: ReLU + replicate padding and the 2x bilinear upsample as prebuilt kernels;
    hacc: 3x3 / 1x1 convs as fp16-accumulate cuDNN graphs."""

    def __init__(self, cs, kern=False, hacc=False):
        self.cs = cs
        self.kern = kern
        self.hacc = hacc
        for m in cs.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.to(torch.float16)
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)

    def _conv2d(self, m, x, pad):
        if self.hacc:
            from .fastbb_vendored.cudnn_conv import conv_half_acc

            return conv_half_acc(x, m.weight, m.bias, [pad, pad], list(m.stride))
        return F.conv2d(x, m.weight, m.bias, m.stride, pad)

    def conv(self, m, x, relu=False):
        if isinstance(m, nn.Conv2d) and m.padding_mode == "replicate" and m.padding != (0, 0):
            p = m.padding[0]
            assert m.padding == (p, p)
            if self.kern:
                x = ops.pad_replicate(x, p, relu)
            else:
                x = F.pad(F.relu(x) if relu else x, (p, p, p, p), mode="replicate")
            return self._conv2d(m, x, 0)
        if relu:
            x = F.relu(x)
        if isinstance(m, nn.Conv2d) and m.groups == 1:
            return self._conv2d(m, x, m.padding[0])
        return m(x)

    def resblock(self, rb, x):
        L = rb.layers  # [norm(Identity), ReLU, conv, norm(Identity), ReLU, conv]
        assert isinstance(L[0], nn.Identity) and isinstance(L[3], nn.Identity)
        y = self.conv(L[2], x, relu=True)
        y = self.conv(L[5], y, relu=True)
        return y.add_(x) if isinstance(rb.skip_connection, nn.Identity) else y.add_(rb.skip_connection(x))

    def resample(self, r, x):
        if isinstance(r[0], nn.Upsample):
            if self.kern and r[0].mode == "bilinear":
                x = ops.interp_bilinear(x, (x.shape[-2] * 2, x.shape[-1] * 2), False)
            else:
                align = False if r[0].mode == "bilinear" else None
                x = F.interpolate(x, scale_factor=2, mode=r[0].mode, align_corners=align)
            return self.conv(r[1], x)
        return self.conv(r[1], r[0](x))

    def forward(self, feats, adds=None, skip_input0=False, out_fp32=()):
        """feats[i]: channels-last fp16 input of level i or None; adds[i]: extra tensor added to the level-i input block output.
        skip_input0: feats[0] is the output of the level-0 input block already (the neck: UV folded in)."""
        cs = self.cs
        outs = []
        x = None
        n = len(cs.res_blocks)
        for i in range(n):
            f = feats[i]
            if f is not None and not (i == 0 and skip_input0) and not isinstance(cs.input_blocks[i], nn.Identity):
                f = cs.input_blocks[i](f)
            if adds is not None and adds[i] is not None:
                f = adds[i] if f is None else f + adds[i]
            x = f if i == 0 else (x if f is None else x + f)
            for rb in cs.res_blocks[i]:
                x = self.resblock(rb, x)
            ob = cs.output_blocks[i]
            if isinstance(ob, nn.Identity):
                outs.append(x)
            elif i in out_fp32:
                outs.append(F.conv2d(x.float(), ob.weight.float(), ob.bias.float()))
            else:
                outs.append(ob(x))
            if i < n - 1:
                x = self.resample(cs.resamplers[i], x)
        return outs


class FastMoGe:
    def __init__(self, model, gacc=True, kernels=None):
        """model: moge.model.v2.MoGeModel on cuda (eval); its encoder, neck and heads are converted to fp16 in place.
        gacc: torch's allow_fp16_accumulation in the encoder GEMMs (a PyTorch flag, no custom kernel).
        kernels: custom kernels (flash attention with fp16 accumulation, fp16-accumulate cuDNN convolutions, NHWC padding /
        bilinear kernels); None = on unless MOGE_FAST_KERNELS=0. Each kernel that cannot be loaded falls back to its plain op.
        """
        self.m = model.eval()
        if kernels is None:
            kernels = os.environ.get("MOGE_FAST_KERNELS", "1") != "0"
        self.kernels = bool(kernels)
        bil, attn, hacc = False, None, False
        if self.kernels:
            bil, flash, hacc = _probe_kernels()
            attn = "hh" if flash else None
            missing = [
                n
                for n, ok in (("NHWC kernels", bil), ("flash attention", flash), ("cudnn-frontend convs", hacc))
                if not ok
            ]
            if missing:
                warnings.warn("moge.fast: falling back to plain PyTorch for " + ", ".join(missing))
        dt = torch.float16
        enc = model.encoder
        self.vit = FastViT(enc.backbone, dtype=dt, gacc=gacc, attn=attn)
        self.layers = list(enc.intermediate_layers)
        # output projections: [C_out, C_in] fp16 matrices + summed bias
        self.proj_w = [p.weight.detach()[:, :, 0, 0].to(dt) for p in enc.output_projections]
        self.proj_b = sum(p.bias.detach().float() for p in enc.output_projections).to(dt)
        self.neck = FastConvStack(model.neck, bil, hacc)
        self.heads = {
            h: FastConvStack(getattr(model, h), bil, hacc) for h in ("points_head", "mask_head") if hasattr(model, h)
        }
        # neck input block 0 = 1x1 conv over [features | uv(2)] -> split
        ib0 = model.neck.input_blocks[0]
        w = ib0.weight.detach()[:, :, 0, 0]
        self.neck0_wf = w[:, :-2].to(dt)
        self.neck0_wuv = w[:, -2:].float()
        self.neck0_b = ib0.bias.detach().float()
        self.mean = enc.image_mean.float()
        self.std = enc.image_std.float()

    @torch.inference_mode()
    def forward(self, image, num_tokens):
        """image: [B, 3, H, W] float in [0, 1]. Returns the raw network-resolution maps (points 3ch fp32, mask logit, ...)."""
        B, _, H, W = image.shape
        dev = image.device
        aspect = W / H
        bh, bw = round((num_tokens / aspect) ** 0.5), round((num_tokens * aspect) ** 0.5)
        x = F.interpolate(image.float(), (bh * 14, bw * 14), mode="bilinear", align_corners=False, antialias=True)
        x = ((x - self.mean) / self.std).to(torch.float16)
        feats = self.vit.intermediate(x, self.layers)
        cls = feats[-1][1]
        # summed 1x1 projections on tokens [B*N, C]
        f0 = torch.addmm(self.proj_b, feats[0][0].reshape(B * bh * bw, -1), self.proj_w[0].t())
        for (t, _), wp in zip(feats[1:], self.proj_w[1:]):
            f0.addmm_(t.reshape(B * bh * bw, -1), wp.t())

        def uv(w, h):
            return normalized_view_plane_uv(width=w, height=h, aspect_ratio=aspect, dtype=torch.float32, device=dev)

        # neck input 0: f0 @ Wf^T + (uv @ Wuv^T + b); inputs of levels 1..4 are the UV maps through 1x1 convs (per-size constants)
        bias0 = torch.addmm(self.neck0_b, uv(bw, bh).reshape(-1, 2), self.neck0_wuv.t()).to(torch.float16)
        n0 = torch.mm(f0, self.neck0_wf.t()).view(B, bh * bw, -1).add_(bias0)
        n0 = n0.view(B, bh, bw, -1).permute(0, 3, 1, 2)  # NCHW view of NHWC memory (channels-last)
        neck = self.m.neck
        adds = [None]
        for lv in range(1, len(neck.res_blocks)):
            ib = neck.input_blocks[lv]
            a = torch.addmm(
                ib.bias.float(), uv(bw * 2**lv, bh * 2**lv).reshape(-1, 2), ib.weight[:, :, 0, 0].float().t()
            )
            adds.append(a.to(torch.float16).view(1, bh * 2**lv, bw * 2**lv, -1).permute(0, 3, 1, 2))
        nf = self.neck.forward([n0] + [None] * (len(neck.res_blocks) - 1), adds, skip_input0=True)
        out = {h: fs.forward(nf, out_fp32=(len(nf) - 1,))[-1] for h, fs in self.heads.items()}
        out["metric_scale"] = self.m.scale_head(cls.float()).squeeze(1).exp() if hasattr(self.m, "scale_head") else None
        return out

    @torch.inference_mode()
    def infer(self, image, num_tokens=None, resolution_level=9, fov_x=None, apply_mask=True, force_projection=True):
        """image [3, H, W] or [B, 3, H, W] in [0, 1] (or uint8). Same outputs as MoGeModel.infer except for the normal."""
        single = image.dim() == 3
        if single:
            image = image[None]
        if image.dtype == torch.uint8:
            image = image.float() / 255
        B, _, H, W = image.shape
        aspect = W / H
        if num_tokens is None:
            lo, hi = self.m.num_tokens_range
            num_tokens = int(lo + (resolution_level / 9) * (hi - lo))
        o = self.forward(image, num_tokens)
        pm = torch.cat([o["points_head"], o["mask_head"]], 1) if "mask_head" in o else o["points_head"]
        pm = F.interpolate(pm.float(), (H, W), mode="bilinear", align_corners=False, antialias=False)
        points = self.m._remap_points(pm[:, :3].permute(0, 2, 3, 1))
        mask_binary = (pm[:, 3] > 0) if pm.shape[1] > 3 else None  # sigmoid(x) > 0.5
        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
        else:
            fov = torch.as_tensor(fov_x, device=points.device, dtype=points.dtype)
            focal = aspect / (1 + aspect**2) ** 0.5 / torch.tan(torch.deg2rad(fov / 2))
            if focal.ndim == 0:
                focal = focal[None].expand(B)
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        fx, fy = focal / 2 * (1 + aspect**2) ** 0.5 / aspect, focal / 2 * (1 + aspect**2) ** 0.5
        one_half = torch.tensor(0.5, device=points.device)
        intrinsics = utils3d.pt.intrinsics_from_focal_center(fx, fy, one_half, one_half)
        depth = points[..., 2] + shift[:, None, None]
        if mask_binary is not None:
            mask_binary &= depth > 0
        if o["metric_scale"] is not None:
            depth = depth * o["metric_scale"][:, None, None]
        res = dict(depth=depth, intrinsics=intrinsics, mask=mask_binary)
        if force_projection:
            res["points"] = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)
        if apply_mask and mask_binary is not None:
            res["depth"] = torch.where(mask_binary, depth, torch.inf)
            if "points" in res:
                res["points"] = torch.where(mask_binary[..., None], res["points"], torch.inf)
        res = {k: v for k, v in res.items() if v is not None}
        return {k: v[0] for k, v in res.items()} if single else res


def load_fast(path, **opts):
    """Light loader: = MoGeModel.from_pretrained(path).cuda().eval() (same config and weights) wrapped in a FastMoGe, but the
    network is constructed on the GPU (upstream: random init of ViT-L on the CPU, ~2.6 s) and the checkpoint is memory-mapped.
    path: a local checkpoint file (model.pt of a MoGe-2 model). opts go to FastMoGe."""
    from ..model.v2 import MoGeModel

    ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    with torch.device("cuda"):
        model = MoGeModel(**ck["model_config"])
    _, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, unexpected[:5]
    return FastMoGe(model.eval(), **opts)

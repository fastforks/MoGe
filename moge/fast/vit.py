"""Fast inference for the DINOv2 ViT inside MoGe: pure fp16 weights and activations, attention through the vendored fastbb
flash kernel (fp16 accumulation, head_dim 64) or torch SDPA, optional fp16-accumulate cuBLAS GEMMs
(`torch.backends.cuda.matmul.allow_fp16_accumulation`), residual + LayerScale as one in-place addcmul per branch, and the
positional-embedding interpolation (bicubic, fp32, upstream formula incl. interpolate_offset) cached per token grid.
The block math is the upstream DINOv2 Block (eval, no drop path); only the precision / kernel choice changes.
"""

import contextlib
import math

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    """torch.backends.cuda.matmul.allow_fp16_accumulation (2x tensor-core rate on GeForce) for the duration."""
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


class FastViT:
    """Wraps a DinoVisionTransformer (patch_embed, cls_token, pos_embed, blocks, norm); the module is converted in place.
    attn: fastbb flash-kernel variant ("hh" fp16 accumulation, "fh", "ff"); None = torch SDPA."""

    def __init__(self, vit, dtype=torch.float16, gacc=True, attn=None):
        self.vit = vit.to(dtype).eval()
        self.dtype = dtype
        self.gacc = gacc
        self.attn = attn
        self.ps = vit.patch_size if isinstance(vit.patch_size, int) else vit.patch_size[0]
        self.nreg = getattr(vit, "num_register_tokens", 0) or 0
        self._pos = {}
        blocks = []
        for b in vit.blocks:
            blocks.extend(
                list(b) if isinstance(b, torch.nn.ModuleList) or b.__class__.__name__ == "BlockChunk" else [b]
            )
        self.blocks = [b for b in blocks if not isinstance(b, torch.nn.Identity)]
        self.pos_fp32 = vit.pos_embed.detach().float()

    def pos_embed(self, h0, w0):
        key = (h0, w0)
        if key not in self._pos:
            vit = self.vit
            pe = self.pos_fp32
            M = int(math.sqrt(pe.shape[1] - 1))
            if M * M == h0 * w0 and h0 == w0:
                out = pe
            else:
                off = getattr(vit, "interpolate_offset", 0.1)
                kw = dict(scale_factor=((h0 + off) / M, (w0 + off) / M)) if off else dict(size=(h0, w0))
                grid = pe[:, 1:].reshape(1, M, M, -1).permute(0, 3, 1, 2)
                p = F.interpolate(grid, mode="bicubic", antialias=getattr(vit, "interpolate_antialias", False), **kw)
                out = torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)
            self._pos[key] = out.to(self.dtype)
        return self._pos[key]

    def tokens(self, x):
        vit = self.vit
        B, _, h, w = x.shape
        t = vit.patch_embed(x.to(self.dtype))
        t = torch.cat([vit.cls_token.to(self.dtype).expand(B, -1, -1), t], 1)
        t = t + self.pos_embed(h // self.ps, w // self.ps)
        if self.nreg:
            t = torch.cat([t[:, :1], vit.register_tokens.to(self.dtype).expand(B, -1, -1), t[:, 1:]], 1)
        return t

    def sdpa(self, q, k, v):
        """softmax(q k^T / sqrt(d)) v for fp16 [B, H, N, d]; the fastbb kernel for d == 64 when enabled."""
        if self.attn and q.shape[-1] == 64:
            from .fastbb_vendored.attention import attention_bhnd

            return attention_bhnd(q, k, v, variant=self.attn)
        return F.scaled_dot_product_attention(q, k, v)

    def block(self, b, x):
        a = b.attn
        B, N, C = x.shape
        H = a.num_heads
        qkv = F.linear(b.norm1(x), a.qkv.weight, a.qkv.bias).view(B, N, 3, H, C // H)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        o = F.linear(self.sdpa(q, k, v).transpose(1, 2).reshape(B, N, C), a.proj.weight, a.proj.bias)
        g1 = b.ls1.gamma if hasattr(b.ls1, "gamma") else None
        x = torch.addcmul(x, o, g1) if g1 is not None else x + o
        m = b.mlp
        o = F.linear(F.gelu(F.linear(b.norm2(x), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        g2 = b.ls2.gamma if hasattr(b.ls2, "gamma") else None
        return x.addcmul_(o, g2) if g2 is not None else x.add_(o)

    def intermediate(self, x, layers):
        """-> list of (patch tokens [B, N, C] fp16, cls token [B, C]) after the given block indices (normed)."""
        out = []
        with fp16_accumulate(self.gacc):
            t = self.tokens(x)
            for i, b in enumerate(self.blocks):
                t = self.block(b, t)
                if i in layers:
                    o = self.vit.norm(t)
                    out.append((o[:, 1 + self.nreg :], o[:, 0]))
        return out

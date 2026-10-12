"""CPU tests for the export-friendly paths. Needs a MoGe-2 checkpoint: set MOGE_TEST_CKPT (e.g. the ViT-S model.pt)
or the tests are skipped."""

import os

import pytest
import torch
import torch.nn.functional as F

CKPT = os.environ.get("MOGE_TEST_CKPT")
needs_ckpt = pytest.mark.skipif(CKPT is None, reason="MOGE_TEST_CKPT not set")


@pytest.fixture(scope="module")
def model():
    from moge.model.v2 import MoGeModel

    return MoGeModel.from_pretrained(CKPT).eval()


@pytest.mark.parametrize("n,m,offset", [(16, 37, 0.1), (24, 37, 0.1), (37, 37, 0.1), (50, 37, 0.1), (20, 37, 0.0)])
def test_bicubic_matrix_matches_interpolate(n, m, offset):
    from moge.model.modules.dinov2.models.vision_transformer import _bicubic_matrix

    x = torch.randn(1, 5, m, m)
    r = _bicubic_matrix(n, offset, m, x.device)
    got = r @ x @ r.T
    scale = (n + offset) / m if offset > 0 else None
    kw = dict(scale_factor=(scale, scale)) if scale else dict(size=(n, n))
    want = F.interpolate(x, mode="bicubic", antialias=False, **kw)
    assert got.shape == want.shape
    assert (got - want).abs().max() < 1e-4


def test_export_native_grid_is_identity():
    """the dynamic export keeps the eager native-grid shortcut (37 x 37 tokens -> pos_embed unchanged)"""
    from moge.model.modules.dinov2.models.vision_transformer import DinoVisionTransformer

    m = DinoVisionTransformer(img_size=518, patch_size=14, embed_dim=32, depth=1, num_heads=2, interpolate_offset=0.1).eval()
    m.pos_embed.data = torch.randn_like(m.pos_embed)

    class Wrap(torch.nn.Module):
        def forward(self, grid):  # [1, 1, gh, gw] token-grid carrier
            gh, gw = grid.shape[-2:]
            return m.interpolate_pos_encoding(torch.zeros(1, gh * gw + 1, 32), gh * 14, gw * 14)

    gh, gw = torch.export.Dim("gh", min=2, max=60), torch.export.Dim("gw", min=2, max=60)
    ep = torch.export.export(Wrap(), (torch.zeros(1, 1, 19, 28),), dynamic_shapes={"grid": {2: gh, 3: gw}}).module()
    for g in ((19, 28), (37, 20), (37, 37)):
        x = torch.zeros(1, 1, *g)
        assert torch.allclose(ep(x), Wrap()(x), atol=1e-6)
    assert torch.equal(ep(torch.zeros(1, 1, 37, 37)), m.pos_embed)


@needs_ckpt
def test_forward_net_matches_forward(model):
    torch.manual_seed(0)
    img = torch.rand(1, 3, 252, 252)  # square: grid aspect == image aspect
    with torch.inference_mode():
        out = model.forward(img, 324)  # 18 x 18 tokens
        image_14 = F.interpolate(img, (252, 252), mode="bilinear", align_corners=False, antialias=True)
        points, normal, mask, scale = model.forward_net(image_14)
    assert torch.allclose(scale.squeeze(1).exp(), out["metric_scale"], atol=1e-6)
    want = F.interpolate(mask, (252, 252), mode="bilinear", align_corners=False).squeeze(1).sigmoid()
    assert torch.allclose(want, out["mask"], atol=1e-6)
    assert normal.shape[1] == 3 and points.shape[1] == 3


@needs_ckpt
def test_return_normal(model):
    torch.manual_seed(0)
    img = torch.rand(3, 224, 336)
    with torch.inference_mode():
        a = model.infer(img, num_tokens=150, use_fp16=False)
        b = model.infer(img, num_tokens=150, use_fp16=False, return_normal=False)
    assert "normal" in a and "normal" not in b
    for k in ("depth", "points", "mask", "intrinsics"):
        assert torch.equal(torch.nan_to_num(a[k].float(), posinf=0), torch.nan_to_num(b[k].float(), posinf=0))


@needs_ckpt
def test_export_dynamic_hw(model):
    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 336)
    hh = torch.export.Dim("hh", min=8, max=64)
    ww = torch.export.Dim("ww", min=8, max=64)

    class Net(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.forward_net(x, return_normal=False)[0::2]

    net = Net(model).eval()
    ep = torch.export.export(net, (x,), dynamic_shapes={"x": {2: 14 * hh, 3: 14 * ww}}, strict=False)
    y = torch.rand(1, 3, 280, 224)
    with torch.inference_mode():
        want = net(y)
        got = ep.module()(y)
    for g, w_ in zip(got, want):
        assert g.shape == w_.shape
        assert (g - w_).abs().max() < 1e-4

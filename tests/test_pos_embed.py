import pytest
import torch
import torch.nn.functional as F

from moge.model.modules.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
    _bicubic_matrix,
)


@pytest.mark.parametrize(
    "n,m,offset",
    [(16, 37, 0.1), (24, 37, 0.1), (37, 37, 0.1), (50, 37, 0.1), (20, 37, 0.0)],
)
def test_bicubic_matrix_matches_interpolate(n, m, offset):
    x = torch.randn(1, 5, m, m)
    r = _bicubic_matrix(n, offset, m, x.device)
    kw = dict(scale_factor=((n + offset) / m,) * 2) if offset > 0 else dict(size=(n, n))
    want = F.interpolate(x, mode="bicubic", antialias=False, **kw)
    assert (r @ x @ r.T - want).abs().max() < 1e-4


def test_pos_embed_exports_with_dynamic_hw():
    vit = DinoVisionTransformer(
        img_size=518,
        patch_size=14,
        embed_dim=32,
        depth=1,
        num_heads=2,
        interpolate_offset=0.1,
    ).eval()

    class Wrap(torch.nn.Module):
        def forward(self, x):
            tokens = x.flatten(2).transpose(1, 2)
            return vit.interpolate_pos_encoding(
                torch.cat([tokens[:, :1], tokens], 1),
                x.shape[-2] * 14,
                x.shape[-1] * 14,
            )

    x = torch.randn(1, 32, 20, 24)
    h, w = torch.export.Dim("h", min=4, max=60), torch.export.Dim("w", min=4, max=60)
    ep = torch.export.export(Wrap(), (x,), dynamic_shapes={"x": {2: h, 3: w}})
    for hh, ww in [(20, 24), (30, 17)]:
        x2 = torch.randn(1, 32, hh, ww)
        assert torch.allclose(ep.module()(x2), Wrap()(x2), atol=1e-5)


def test_export_native_grid_is_identity():
    """the dynamic export keeps the eager native-grid shortcut (37 x 37 tokens -> pos_embed unchanged)"""
    m = DinoVisionTransformer(
        img_size=518,
        patch_size=14,
        embed_dim=32,
        depth=1,
        num_heads=2,
        interpolate_offset=0.1,
    ).eval()
    m.pos_embed.data = torch.randn_like(m.pos_embed)

    class Wrap(torch.nn.Module):
        def forward(self, grid):  # [1, 1, gh, gw] token-grid carrier
            gh, gw = grid.shape[-2:]
            return m.interpolate_pos_encoding(
                torch.zeros(1, gh * gw + 1, 32), gh * 14, gw * 14
            )

    gh, gw = torch.export.Dim("gh", min=2, max=60), torch.export.Dim(
        "gw", min=2, max=60
    )
    ep = torch.export.export(
        Wrap(), (torch.zeros(1, 1, 19, 28),), dynamic_shapes={"grid": {2: gh, 3: gw}}
    ).module()
    for g in ((19, 28), (37, 20), (37, 37)):
        x = torch.zeros(1, 1, *g)
        assert torch.allclose(ep(x), Wrap()(x), atol=1e-6)
    assert torch.equal(ep(torch.zeros(1, 1, 37, 37)), m.pos_embed)

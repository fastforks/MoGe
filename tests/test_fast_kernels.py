"""GPU tests of moge.fast: the NHWC kernels against PyTorch ops, FastMoGe against MoGeModel.infer. The model test needs a
MoGe-2 checkpoint in MOGE_TEST_CKPT (the ViT-S one is enough)."""

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CKPT = os.environ.get("MOGE_TEST_CKPT")


def _nhwc(*shape):
    return torch.randn(*shape, device="cuda").half().contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize(
    "hw,size,align", [((17, 23), (34, 46), False), ((16, 24), (64, 96), False), ((9, 13), (40, 50), True)]
)
def test_bilinear_kernel_matches_interpolate(hw, size, align):
    from moge.fast import ops

    if not ops.available():
        pytest.skip("kernel not loadable")
    x = _nhwc(2, 64, *hw)
    got = ops.interp_bilinear(x, size, align)
    want = F.interpolate(x, size=size, mode="bilinear", align_corners=align)
    assert got.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(got, want)


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("p", [1, 2])
def test_pad_replicate_kernel(relu, p):
    from moge.fast import ops

    if not ops.available():
        pytest.skip("kernel not loadable")
    x = _nhwc(2, 32, 11, 15)
    got = ops.pad_replicate(x, p, relu)
    want = F.pad(F.relu(x) if relu else x, (p, p, p, p), mode="replicate")
    assert torch.equal(got, want)


def test_kernels_fall_back_for_other_layouts():
    from moge.fast import ops

    x = torch.randn(1, 12, 8, 8, device="cuda")  # fp32, C % 8 != 0
    assert torch.equal(ops.interp_bilinear(x, (16, 16), False), F.interpolate(x, size=(16, 16), mode="bilinear"))
    assert torch.equal(ops.pad_replicate(x, 1), F.pad(x, (1, 1, 1, 1), mode="replicate"))


@pytest.mark.skipif(CKPT is None, reason="MOGE_TEST_CKPT not set")
@pytest.mark.parametrize("kernels,tol", [(True, 2e-2), (False, 5e-3)])  # fp16 accumulation on a noise image: ~1 %
def test_fast_infer_matches_upstream(kernels, tol):
    from moge.fast import load_fast
    from moge.model.v2 import MoGeModel

    torch.manual_seed(0)
    img = torch.rand(3, 252, 364, device="cuda")
    ref = MoGeModel.from_pretrained(CKPT).cuda().eval()
    with torch.inference_mode():
        want = ref.infer(img, num_tokens=300, use_fp16=False)
    out = load_fast(CKPT, kernels=kernels).infer(img, num_tokens=300)
    assert "normal" not in out
    both = want["mask"] & out["mask"]
    assert (want["mask"] == out["mask"]).float().mean() > 0.99
    rel = (out["depth"][both] - want["depth"][both]).abs() / want["depth"][both]
    assert rel.mean() < tol

import os

import pytest
import torch

CKPT = os.environ.get("MOGE_TEST_CKPT")


@pytest.mark.skipif(CKPT is None, reason="MOGE_TEST_CKPT (a MoGe-2 model.pt) not set")
def test_return_normal_false_keeps_geometry():
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(CKPT).eval()
    img = torch.rand(3, 224, 280)
    with torch.inference_mode():
        full = model.infer(img, num_tokens=320, use_fp16=False)
        lean = model.infer(img, num_tokens=320, use_fp16=False, return_normal=False)
    assert "normal" in full and "normal" not in lean
    for k in ("points", "depth", "mask", "intrinsics"):
        assert torch.equal(full[k], lean[k]), k


@pytest.mark.skipif(os.environ.get("MOGE3_TEST_CKPT") is None, reason="MOGE3_TEST_CKPT (a MoGe-3 model.pt) not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoGe-3 inference needs CUDA")
def test_return_normal_false_keeps_geometry_v3():
    from moge.model.v3 import MoGeModel

    model = MoGeModel.from_pretrained(os.environ["MOGE3_TEST_CKPT"]).cuda().eval()
    img = torch.rand(3, 224, 280, device="cuda")
    with torch.inference_mode():
        full = model.infer(img, num_tokens=320)
        lean = model.infer(img, num_tokens=320, return_normal=False)
    assert "normal" in full and "normal" not in lean
    assert torch.equal(full["mask"], lean["mask"])
    for k in ("points", "depth", "intrinsics"):  # MoGe-3 on CUDA is not bitwise deterministic between identical calls
        torch.testing.assert_close(full[k], lean[k], rtol=1e-2, atol=1e-3, msg=k)

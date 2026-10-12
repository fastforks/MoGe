"""Custom-kernel inference path for MoGe-2 (cuda-kernels branch): `FastMoGe`, `load_fast`.

    from moge.fast import load_fast
    fast = load_fast("/path/to/moge-2-vitl-normal/model.pt")
    out = fast.infer(img_float_chw_cuda)     # dict: depth, intrinsics, mask, points (no normal)

`FastMoGe(model, kernels=False)` (or MOGE_FAST_KERNELS=0) switches the custom kernels off and runs plain PyTorch ops.
"""

from .moge import FastMoGe, load_fast  # noqa: F401

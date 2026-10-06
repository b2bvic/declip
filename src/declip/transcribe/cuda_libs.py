"""Preload Linux pip-wheel CUDA libraries before importing CTranslate2."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_HANDLES = []
_LOADED = set()


def preload() -> tuple[bool, str]:
    if sys.platform != "linux":
        return True, "CUDA preload is Linux-only"
    try:
        directories = [Path(p) / "nvidia" for p in sys.path if p]
        cublas = sorted(
            {p for d in directories for p in d.glob("*/lib/libcublas.so.12")}
        )
        cudnn = sorted({p for d in directories for p in d.glob("*/lib/libcudnn*.so.9")})
        # cuBLAS Lt must precede cuBLAS; cuDNN's base must precede its components.
        lt = sorted({p for d in directories for p in d.glob("*/lib/libcublasLt.so.12")})
        base = [p for p in cudnn if p.name == "libcudnn.so.9"]
        files = lt + cublas + base + [p for p in cudnn if p not in base]
        if not cublas or not base:
            # Also support a system CUDA installation through the loader search path.
            for name in ("libcublas.so.12", "libcudnn.so.9"):
                if name not in _LOADED:
                    _HANDLES.append(ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL))
                    _LOADED.add(name)
        else:
            for path in files:
                if str(path) not in _LOADED:
                    _HANDLES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
                    _LOADED.add(str(path))
        return True, "CUDA 12 and cuDNN 9 loaded"
    except Exception as exc:
        return False, f"CUDA library preload failed: {exc}"

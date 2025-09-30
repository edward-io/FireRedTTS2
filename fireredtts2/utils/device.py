import warnings
from typing import Optional, Union

import torch


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(getattr(backend, "is_available", lambda: False)())


def _mps_built() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(getattr(backend, "is_built", lambda: False)())


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Return a torch.device, preferring MPS over CUDA when no device is requested."""
    requested: Optional[torch.device]
    if isinstance(device, torch.device):
        requested = device
    elif isinstance(device, str) and device:
        try:
            requested = torch.device(device)
        except (TypeError, RuntimeError) as error:
            warnings.warn(f"Unable to interpret device '{device}': {error}")
            requested = None
    else:
        requested = None

    if requested is not None:
        if requested.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to default device.")
            requested = None
        elif requested.type == "mps" and not (_mps_built() and _mps_available()):
            warnings.warn("MPS requested but not available; falling back to default device.")
            requested = None

    if requested is None:
        if _mps_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return requested


def empty_cache(device: torch.device | None = None) -> None:
    """Best-effort cache clearing for the provided device."""
    target = resolve_device(device) if device is not None else None
    if target is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and _mps_available():
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        return

    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif target.type == "mps" and hasattr(torch, "mps") and _mps_available():
        torch.mps.empty_cache()  # type: ignore[attr-defined]


def configure_inference_kernels(device: torch.device) -> None:
    """Enable the fastest available attention kernels for the target device."""
    try:
        # "medium" opts into TF32 on Ampere+ while keeping fp32 fallback elsewhere.
        precision = "medium" if device.type == "cuda" else "high"
        torch.set_float32_matmul_precision(precision)
    except Exception as exc:  # pragma: no cover - best effort tuning
        warnings.warn(f"Failed to set matmul precision: {exc}")

    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(False)
        except Exception as exc:  # pragma: no cover - best effort tuning
            warnings.warn(f"Failed to configure CUDA SDPA kernels: {exc}")


def maybe_compile_callable(
    fn,
    *,
    description: str,
    device: torch.device,
    dynamic: bool = True,
    mode: str = "reduce-overhead",
):
    """Wrap a callable with torch.compile when supported, else return it unchanged."""

    compiler = getattr(torch, "compile", None)
    if compiler is None:
        return fn

    if device.type not in {"cuda", "cpu"}:
        return fn

    try:
        return compiler(fn, dynamic=dynamic, mode=mode)
    except Exception as exc:  # pragma: no cover - dynamo can bail per backend
        warnings.warn(f"torch.compile skipped for {description}: {exc}")
        return fn

import contextlib
import gc

import torch


def cuda_build_version():
    return tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])


def setup_device(require_gpu=False):
    if not torch.cuda.is_available():
        if require_gpu:
            raise RuntimeError("STOP: no CUDA GPU found")
        return torch.device("cpu")
    name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    if cap >= (12, 0) and cuda_build_version() < (12, 8):
        raise RuntimeError(f"STOP: {name} (sm_{cap[0]}{cap[1]}) needs PyTorch built with CUDA 12.8+, this one has "
                           f"{torch.version.cuda}")
    try:
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        native_bf16 = torch.cuda.is_bf16_supported()
    if cap < (8, 0) or not native_bf16:
        raise RuntimeError(f"STOP: {name} (compute capability {cap[0]}.{cap[1]}) has no native bf16; this project "
                           "trains in bf16 and needs an Ampere-or-newer GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def device_name(device):
    return torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"


def device_memory_gb(device):
    return torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == "cuda" else 0.0


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def exact_fp32():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

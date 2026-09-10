# -*- coding: utf-8 -*-
"""YuE2 on Windows / ComfyUI 兼容补丁。

问题:Windows 版 torch 2.10+cu130 构建未编译内置 flash-attention 内核,
官方 yue2_infer 的 CUDA Graph 解码(GraphAR)在 attention_backend="auto" 时
会直接调用 torch.ops.aten._flash_attention_forward 并报
"USE_FLASH_ATTENTION was not enabled for build"。

方案:运行时探测该算子是否真的可用;不可用则把 GraphAR 切到 cuDNN SDPA 后端
(实测支持 GQA、bool mask、且可被 CUDA Graph 捕获)。Linux 正常构建不受影响。

另外 yue2 管线构造时会设置进程级 torch.cuda.set_per_process_memory_fraction,
会限制同进程 ComfyUI 其他模型的显存分配,构造完成后必须复位。
"""
from __future__ import annotations

import contextlib

import torch

_probe_cache: dict = {}


def builtin_flash_available(device="cuda") -> bool:
    """探测 torch 内置 varlen flash-attention 算子是否真实可用(带缓存)。"""
    key = str(device)
    if key in _probe_cache:
        return _probe_cache[key]
    ok = False
    if torch.cuda.is_available() and hasattr(torch.ops.aten, "_flash_attention_forward"):
        try:
            dev = torch.device(device)
            q = torch.zeros(1, 1, 4, 128, device=dev, dtype=torch.bfloat16)
            k = torch.zeros(1, 1, 2, 128, device=dev, dtype=torch.bfloat16)
            v = torch.zeros(1, 1, 2, 128, device=dev, dtype=torch.bfloat16)
            cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
            cu_k = torch.tensor([0, 128], dtype=torch.int32, device=dev)
            used = torch.tensor([128], dtype=torch.int32, device=dev)
            out = torch.ops.aten._flash_attention_forward(
                q, k, v, cu_q, cu_k, 1, 128, 0.0, False, False, seqused_k=used)
            ok = bool(torch.isfinite(out[0].float()).all().item())
        except Exception:
            ok = False
    _probe_cache[key] = ok
    return ok


def apply_yue2_windows_patch() -> bool:
    """给 yue2.cuda_graph.GraphAR 打自动后端降级补丁(幂等)。"""
    try:
        import yue2.cuda_graph as cg
    except ImportError:
        return False
    if getattr(cg, "_comfy_yue2_flash_patch", False):
        return True
    if builtin_flash_available("cuda"):
        cg._comfy_yue2_flash_patch = True
        return True
    orig_init = cg.GraphAR.__init__

    def patched_init(self, model, prefixes, max_tokens, *, capture=True,
                     attention_backend="auto", fuse_projections=False):
        if attention_backend == "auto":
            dev = None
            try:
                dev = next(model.parameters()).device
            except StopIteration:
                pass
            if dev is not None and dev.type == "cuda" and not builtin_flash_available(dev):
                attention_backend = "cudnn"
        return orig_init(self, model, prefixes, max_tokens, capture=capture,
                         attention_backend=attention_backend,
                         fuse_projections=fuse_projections)

    cg.GraphAR.__init__ = patched_init
    cg._comfy_yue2_flash_patch = True
    print("[ComfyUI-YuE2] YuE2 补丁: 内置 flash-attention 不可用, "
          "CUDA Graph 解码改用 cuDNN SDPA 后端")
    return True


def reset_cuda_memory_fraction():
    """yue2 管线构造会把进程显存上限压到 memory_budget 附近,必须复位为不限制,
    否则同进程的 ComfyUI 视频等大模型会莫名 OOM。"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(1.0, i)


@contextlib.contextmanager
def yue2_runtime_flags():
    """生成期间应用 yue2 官方要求的全局精度/确定性开关,结束后恢复 ComfyUI 原值。"""
    saved = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.get_float32_matmul_precision(),
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        (torch.backends.cudnn.benchmark,
         torch.backends.cudnn.deterministic,
         torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction) = saved[:5]
        torch.set_float32_matmul_precision(saved[5])

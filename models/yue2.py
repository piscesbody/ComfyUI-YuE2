# -*- coding: utf-8 -*-
"""YuE2 生成管线（加载、缓存、生成）。

管线构造有两个会**影响同进程其他模型**的副作用，必须处理：
- ``torch.cuda.set_per_process_memory_fraction`` 把进程显存上限压到
  ``memory_budget_gib`` 附近 → 构造完立即复位，否则 ComfyUI 加载视频等大模型会莫名 OOM
- 全局 TF32 / matmul 精度开关 → 用 :func:`runtime_flags` 上下文包裹生成过程并恢复
"""
from __future__ import annotations

import contextlib
import gc

import torch

from .paths import resolve, vae_dir

# 进程级管线缓存（键为构造参数）
_cache: dict = {}


@contextlib.contextmanager
def runtime_flags():
    """生成期间应用 YuE2 要求的全局精度开关，结束后恢复原值。"""
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


def _reset_memory_fraction() -> None:
    """复位被 YuE2 管线压低的进程显存配额。"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(1.0, i)
            except Exception:
                pass


def _apply_windows_patch() -> None:
    """Windows/torch 构建缺少内置 flash 内核时，把 CUDA Graph 解码切到 cuDNN。"""
    from .yue2_patch import apply_yue2_windows_patch
    apply_yue2_windows_patch()


def load(model_name: str, vae_name: str = "YuE2-Vae", device: str = "cuda",
         memory_budget_gib: int = 24, offload_ar: bool = False,
         offline: bool = False):
    """创建或复用 YuE2 管线（按参数缓存）。"""
    import yue2

    _apply_windows_patch()

    model_path = resolve("yue2", model_name)
    vae_path = vae_dir(model_path, vae_name)

    key = (model_path, vae_path, device, int(memory_budget_gib), bool(offload_ar), bool(offline))
    cached = _cache.get("pipeline")
    if cached is not None and _cache.get("key") == key:
        return cached

    if cached is not None:
        close()
        torch.cuda.empty_cache()
        gc.collect()

    pipe = yue2.YuE2Pipeline.from_pretrained(
        model_path, vae=vae_path, device=device,
        memory_budget_gib=int(memory_budget_gib), offload_ar=bool(offload_ar),
        local_files_only=bool(offline), progress=False,
    )
    _reset_memory_fraction()
    _cache["pipeline"] = pipe
    _cache["key"] = key
    return pipe


def close() -> None:
    """释放管线并清空缓存。"""
    pipe = _cache.pop("pipeline", None)
    _cache.pop("key", None)
    if pipe is not None:
        try:
            pipe.close()
        except Exception:
            pass
    torch.cuda.empty_cache()
    gc.collect()


def _with_ode_steps(pipe, ode_steps: int | None):
    """``GenerationConfig`` 是 frozen dataclass，需构造新实例而非就地赋值。"""
    if not ode_steps:
        return
    import dataclasses
    config = pipe.generation_config
    if int(config.ode_steps) == int(ode_steps):
        return
    pipe.generation_config = dataclasses.replace(config, ode_steps=int(ode_steps))


def generate(pipe, *, style: str, lyrics: str, cot: str = "full", seed: int = 831001,
             abc: str | None = None, cfg_scale: float | None = None,
             abc_sampling: dict | None = None, semantic_sampling: dict | None = None,
             ode_steps: int | None = None, on_progress=None):
    """生成歌曲，返回 ``SongResult``。``on_progress`` 在每生成一个 token 时回调。"""
    with runtime_flags():
        _with_ode_steps(pipe, ode_steps)
        return pipe(
            style=style, lyrics=lyrics, cot=cot, seed=int(seed),
            abc=abc, cfg_scale=cfg_scale,
            abc_sampling=abc_sampling, semantic_sampling=semantic_sampling,
            on_token=on_progress,
        )


def plan(pipe, *, style: str, lyrics: str, cot: str = "full", seed: int = 831001,
         abc: str | None = None, cfg_scale: float | None = None,
         abc_sampling: dict | None = None, on_progress=None):
    """只生成符号规划（ABC 乐谱），不合成音频。"""
    with runtime_flags():
        request = pipe._request(style=style, lyrics=lyrics, cot=cot, seed=int(seed),
                                abc=abc, cfg_scale=cfg_scale)
        return pipe.plan(request=request, abc_sampling=abc_sampling, on_token=on_progress)

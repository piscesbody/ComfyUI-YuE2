# -*- coding: utf-8 -*-
"""SheetSage2 在 transformers 5.x 下的兼容处理。

通用 API 差异（``_tied_weights_keys``、``tie_weights`` 签名等）由
``compat.transformers5`` 统一处理；这里只保留 SheetSage2 特有的两项：

1. ``BartDecoder.__init__`` 的 ``embed_tokens`` 关键字参数（4.x 有，5.x 已移除）
2. MERT2 编码器 ``RotaryEmbedding.inv_freq`` buffer 被 5.x 的 meta 初始化
   写入垃圾数据——**这是最隐蔽的一项**：模型不会报错，只是转录结果里
   完全没有旋律（melody 字段恒为 0），听感上像"只识别出节奏和和弦"。

对照验证：transformers 4.45.2（官方锁定版本）输出 vocal=23 / ins=38 音符；
修复后 5.3.0 输出与之完全一致。
"""
from __future__ import annotations

import torch

from .transformers5 import apply_transformers5_compat


def _fix_bart_decoder() -> None:
    from transformers.models.bart import modeling_bart as mb

    if getattr(mb.BartDecoder, "_ss2_compat_patched", False):
        return
    orig_init = mb.BartDecoder.__init__

    def patched_init(self, config, embed_tokens=None):
        orig_init(self, config)
        if embed_tokens is not None:
            # 4.x 语义: decoder 与外部 embedding 共享同一权重
            self.embed_tokens.weight = embed_tokens.weight

    mb.BartDecoder.__init__ = patched_init
    mb.BartDecoder._ss2_compat_patched = True


def fix_rotary_inv_freq(model) -> int:
    """重算被 transformers 5.x meta 初始化污染的 inv_freq buffer（幂等）。"""
    fixed = 0
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        head_dim = getattr(module, "head_dim", None)
        base = getattr(module, "base", None)
        if inv_freq is None or head_dim is None or base is None:
            continue
        if getattr(module, "_ss2_inv_freq_ok", False):
            continue
        expected = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        if not torch.isfinite(inv_freq).all() or not torch.allclose(
                inv_freq.float().cpu(), expected, atol=1e-6):
            with torch.no_grad():
                module.inv_freq = expected.to(inv_freq.device)
            fixed += 1
        module._ss2_inv_freq_ok = True
    return fixed


def apply_sheetsage2_compat(model_path: str):
    """应用补丁并返回 SheetSage2Model 类（幂等）。"""
    apply_transformers5_compat(verbose=False)
    _fix_bart_decoder()

    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module(
        "modeling_sheetsage2.SheetSage2Model", model_path)
    if not getattr(cls, "_ss2_tie_patched", False):
        orig_tie = cls.tie_weights

        def patched_tie(self, *args, **kwargs):
            kwargs.pop("recompute_mapping", None)
            kwargs.pop("missing_keys", None)
            return orig_tie(self)

        cls.tie_weights = patched_tie
        cls._ss2_tie_patched = True
    return cls


def load_sheetsage2(model_path: str, device: str = "cuda", dtype=torch.bfloat16):
    """加载 SheetSage2（处理 LoRA 合并、权重绑定、inv_freq 修复）。"""
    cls = apply_sheetsage2_compat(model_path)
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    # 5.x 下该绑定可能解绑，与模型自身 tie_weights 的语义保持一致
    model.output_projection.weight = model.token_embedding.weight
    fix_rotary_inv_freq(model)
    model = model.to(device=device, dtype=dtype)
    fix_rotary_inv_freq(model)
    return model

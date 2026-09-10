"""ComfyUI-YuE2 — 音乐生成 / 翻唱 / 歌词整理一体化节点包。

包含两套模型能力，互相衔接构成完整工作流：

- **YuE2**      : 风格 + 歌词 → 48 kHz 完整歌曲（可编辑 ABC 乐谱）
- **SheetSage2**: 歌曲 → ABC 乐谱 + 曲式结构（翻唱、改谱）

歌词建议手填，用 ``歌词结构化`` 节点自动加 [verse]/[chorus] 段落标签。
纯时间戳歌词（如 SRT）可用 ``歌词格式化`` 节点按曲式分段。

模型放在 ``ComfyUI/models/`` 下的约定目录（见 ``models/paths.py``），
也可用 ``extra_model_paths.yaml`` 重定向到别处。

本包为自包含实现：YuE2/SheetSage2 的推理代码与兼容层都在包内，
只需按 README 安装依赖并放置模型权重。
"""
from __future__ import annotations

import os
import sys

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _try_register(module_name: str, mapping: dict) -> None:
    """注册节点模块；依赖缺失时只禁用该组节点，不影响其余功能。"""
    import importlib

    try:
        module = importlib.import_module(f".{module_name}", __package__)
    except Exception as exc:  # noqa: BLE001
        mapping[module_name] = f"{type(exc).__name__}: {exc}"
        print(f"[ComfyUI-YuE2] 跳过 {module_name}: {exc}")
        return
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))


_failed: dict = {}

_try_register("nodes.yue2", _failed)
_try_register("nodes.sheetsage2", _failed)

if not NODE_CLASS_MAPPINGS:
    print("[ComfyUI-YuE2] 未注册任何节点；请检查依赖安装与上面的错误信息")
else:
    print(f"[ComfyUI-YuE2] 已注册 {len(NODE_CLASS_MAPPINGS)} 个节点")
    for name, err in _failed.items():
        print(f"[ComfyUI-YuE2]   未加载 {name}: {err}")

WEB_DIRECTORY = None  # 暂无前端扩展

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

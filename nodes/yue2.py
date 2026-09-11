# -*- coding: utf-8 -*-
"""YuE2 生成节点：加载、生成歌曲、只出乐谱、卸载。"""
from __future__ import annotations

import os

import torch

from ..models import yue2 as yue2_model
from ..models.paths import list_snapshots, vae_options
from .utils import advance, audio_dict, progress_bar, safe_stem, timestamp_dir

COT_MODES = ["full", "melody", "off"]
VAE_DECODE_MODES = ["tiled", "full"]
ATTN_BACKENDS = ["auto", "external-flash", "cudnn", "sdpa"]
QUANT_MODES = ["none", "fp8"]
FRAMES_PER_SECOND = 25      # YuE2 语义 token 帧率（与 comfy 官方实现一致）


class YuE2Loader:
    """加载 YuE2 生成模型与解码器（VAE）。"""

    @classmethod
    def INPUT_TYPES(cls):
        models = [n for n in list_snapshots("yue2") if "vae" not in n.lower()]
        if not models:
            models = ["YuE2-3B"]
        return {
            "required": {
                "model": (models, {"tooltip": "models/YuE2/ 下的模型目录名"}),
                "vae": (vae_options(), {"tooltip": "解码器; 听感用 YuE2-Vae, 复现基准用 legacy"}),
                "device": (["cuda", "cpu"],),
                "memory_budget_gib": ("INT", {"default": 24, "min": 6, "max": 96, "step": 1,
                    "tooltip": "显存预算; 24GB 卡保持默认即可"}),
                "offload_ar": ("BOOLEAN", {"default": False,
                    "tooltip": "NAR 阶段临时卸载 AR 模块以省显存（更慢）"}),
                "offline": ("BOOLEAN", {"default": True,
                    "tooltip": "只用本地文件; 关闭则在缺模型时联网下载"}),
                "attention_backend": (ATTN_BACKENDS, {"default": "auto",
                    "tooltip": "AR 解码 attention 内核。auto=自动选择(Windows 无内置 "
                               "flash 时用 cuDNN); external-flash=用环境里的 "
                               "pip flash-attn(需已安装); 实测与 cuDNN 速度相当"}),
                "quantization": (QUANT_MODES, {"default": "none",
                    "tooltip": "fp8=AR 线性层 FP8 量化(官方实验性; 实测省 ~1.3GB "
                               "显存但慢 ~6 倍, 因禁用 CUDA Graph; 仅限显存极端受限)"}),
            },
        }

    RETURN_TYPES = ("YUE2_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "YuE2"

    def load(self, model, vae, device, memory_budget_gib, offload_ar, offline,
             attention_backend="auto", quantization="none"):
        pipe = yue2_model.load(model, vae, device=device,
                               memory_budget_gib=memory_budget_gib,
                               offload_ar=offload_ar, offline=offline,
                               attention_backend=attention_backend,
                               quantization=quantization)
        return (pipe,)


class YuE2Unload:
    """卸载 YuE2 管线释放显存。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": ("YUE2_PIPE",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "YuE2"

    def unload(self, pipeline):
        yue2_model.close()
        return ()


class YuE2Sampler:
    """YuE2 歌曲生成：风格 + 歌词（+可选 ABC 乐谱）-> 48 kHz 完整歌曲。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("YUE2_PIPE",),
                "style": ("STRING", {"default":
                    "Mandarin pop, warm female vocal, clean electric guitar, soft drums",
                    "multiline": True,
                    "tooltip": "曲风/乐器/人声/语言/速度等标签"}),
                "lyrics": ("STRING", {"default":
                    "[verse]\n夜色温柔 星光落在窗台\n梦想还在 不必匆忙离开\n\n"
                    "[chorus]\n风带着我 往更远的地方\n心跳的方向 就是回答",
                    "multiline": True,
                    "tooltip": "带 [verse]/[chorus] 等段落标签的歌词; 可用歌词格式化节点生成"}),
                "cot": (COT_MODES, {"tooltip":
                    "full=旋律+和声规划; melody=仅旋律(翻唱推荐); off=无乐谱直接生成"}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": 2**63 - 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01,
                    "tooltip": "文本引导强度; 1.0=官方默认(off 模式默认 1.01)"}),
                "ode_steps": ("INT", {"default": 32, "min": 4, "max": 64, "step": 1,
                    "tooltip": "声学流匹配步数; 官方默认 32"}),
                "max_duration": ("FLOAT", {"default": 360.0, "min": 4.0, "max": 640.0, "step": 1.0,
                    "tooltip": "歌曲时长预算(秒)。语义 token 按 25帧/秒 换算; "
                               "生成可能在预算内提前结束。缩短可显著提速"}),
                "abc_max_tokens": ("INT", {"default": 4096, "min": 64, "max": 8192, "step": 32}),
                "semantic_max_tokens": ("INT", {"default": 9000, "min": 200, "max": 16384, "step": 32,
                    "tooltip": "语义 token 上限(高级)。与 max_duration 联动: "
                               "取 min(本值, max_duration×25)"}),
                "abc_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.01}),
                "abc_top_p": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
                "abc_top_k": ("INT", {"default": 30, "min": 1, "max": 32768, "step": 1,
                    "tooltip": "ABC 规划采样参数; 默认取 comfy 官方 PR 调优值 "
                               "(top_p 0.9 / top_k 30 / rep 1.005)"}),
                "abc_repetition_penalty": ("FLOAT", {"default": 1.005, "min": 0.01, "max": 10.0, "step": 0.005}),
                "semantic_temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "save_flac": ("BOOLEAN", {"default": True,
                    "tooltip": "另存 48 kHz FLAC 到 output/YuE2"}),
                "save_abc": ("BOOLEAN", {"default": True,
                    "tooltip": "保存 ABC 乐谱到 output/YuE2"}),
                "vae_decode": (VAE_DECODE_MODES, {"default": "tiled",
                    "tooltip": "full=整曲一次解码(大显存更快, 不足自动回退); "
                               "tiled=分块解码(显存友好)"}),
                "vae_tile_frames": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 128,
                    "tooltip": "tiled 块大小(latent帧); 0=默认(小预算512/大预算1024); "
                               "8GB 显存建议 256"}),
            },
            "optional": {
                "abc": ("STRING", {"forceInput": True,
                    "tooltip": "外部 ABC 乐谱(来自 SheetSage2 或手工编辑); 需 cot=melody/full"}),
                "abort_after_plan": ("BOOLEAN", {"default": False,
                    "tooltip": "只生成 ABC 乐谱，不合成音频（可先看谱再决定）"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING",)
    RETURN_NAMES = ("audio", "abc_score", "info",)
    FUNCTION = "generate"
    CATEGORY = "YuE2"

    def generate(self, pipeline, style, lyrics, cot, seed, cfg_scale, ode_steps,
                 max_duration, abc_max_tokens, semantic_max_tokens, abc_temperature,
                 abc_top_p, abc_top_k, abc_repetition_penalty,
                 semantic_temperature, save_flac, save_abc,
                 vae_decode="tiled", vae_tile_frames=0,
                 abc=None, abort_after_plan=False):
        style, lyrics = (style or "").strip(), (lyrics or "").strip()
        if not style:
            raise ValueError("style 不能为空")
        if not lyrics:
            raise ValueError("lyrics 不能为空")
        abc_text = (abc or "").strip() or None
        if abc_text and cot == "off":
            raise ValueError("外部 ABC 需要 cot=melody 或 full")

        # YuE2 语义帧率 25 token/秒；时长预算与 token 上限取较小者
        sem_budget = min(int(semantic_max_tokens),
                         max(200, int(round(max_duration * FRAMES_PER_SECOND))))
        abc_sampling = {"temperature": abc_temperature, "max_tokens": int(abc_max_tokens),
                        "top_p": abc_top_p, "top_k": int(abc_top_k),
                        "repetition_penalty": abc_repetition_penalty}
        semantic_sampling = {"temperature": semantic_temperature,
                             "max_tokens": sem_budget}

        bar = progress_bar(int(abc_max_tokens) + sem_budget)

        def on_token(_phase, _token):
            if bar is not None:
                try:
                    bar.update(1)
                except Exception:
                    pass

        vae_bar_holder: list = []

        def on_vae_progress(completed, total):
            """VAE 分块解码进度：首次回调时按真实块数建第二条进度条。"""
            if not vae_bar_holder:
                vae_bar_holder.append(progress_bar(int(total) if total else 1))
            advance(vae_bar_holder[0],
                    max(0, int(completed) - getattr(vae_bar_holder[0], "_yue2_done", 0)))
            if vae_bar_holder[0] is not None:
                vae_bar_holder[0]._yue2_done = int(completed)

        # 只看乐谱：走 plan 分支，不合成音频
        if abort_after_plan:
            plan = yue2_model.plan(pipeline, style=style, lyrics=lyrics, cot=cot,
                                   seed=seed, abc=abc_text, cfg_scale=cfg_scale,
                                   abc_sampling=abc_sampling, on_progress=on_token)
            score = plan.abc or ""
            out_dir = timestamp_dir("YuE2") if save_abc else None
            if out_dir and score:
                with open(os.path.join(out_dir, f"{safe_stem(style)}_s{seed}.abc"),
                          "w", encoding="utf-8") as fh:
                    fh.write(score)
            info = f"仅生成乐谱 | cot={cot} seed={seed} | abc_tokens={len(plan.abc_ids)}"
            print(f"[YuE2] {info}")
            silence = audio_dict(torch.zeros(1, 160), 48000)
            return (silence, score, info)

        song = yue2_model.generate(
            pipeline, style=style, lyrics=lyrics, cot=cot, seed=seed,
            abc=abc_text, cfg_scale=cfg_scale,
            abc_sampling=abc_sampling, semantic_sampling=semantic_sampling,
            ode_steps=ode_steps, on_progress=on_token,
            vae_decode=vae_decode,
            vae_tile_frames=int(vae_tile_frames) if vae_tile_frames else None,
            on_vae_progress=on_vae_progress)

        out_dir = timestamp_dir("YuE2") if (save_flac or save_abc) else None
        stem = f"{safe_stem(style)}_s{seed}"
        saved = []
        if out_dir:
            if save_flac:
                saved.append(song.save(os.path.join(out_dir, stem + ".flac")))
            if save_abc and song.abc:
                abc_path = os.path.join(out_dir, stem + ".abc")
                with open(abc_path, "w", encoding="utf-8") as fh:
                    fh.write(song.abc)
                saved.append(abc_path)

        seconds = len(song.audio) / song.sample_rate
        info = (f"{seconds:.1f}s @{song.sample_rate}Hz | cot={cot} seed={seed} | "
                f"abc_tokens={song.timing['abc'].get('output_tokens', 0)} "
                f"sem_tokens={song.timing['semantic'].get('output_tokens', 0)} | "
                f"truncated={song.truncated} | e2e={song.timing['e2e_seconds']:.1f}s")
        if saved:
            info += f" | saved: {out_dir}"
        print(f"[YuE2] {info}")

        return (audio_dict(torch.from_numpy(song.audio).T, song.sample_rate),
                song.abc or "", info)


NODE_CLASS_MAPPINGS = {
    "YuE2Loader": YuE2Loader,
    "YuE2Sampler": YuE2Sampler,
    "YuE2Unload": YuE2Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2Loader": "YuE2 加载模型",
    "YuE2Sampler": "YuE2 歌曲生成",
    "YuE2Unload": "YuE2 卸载",
}

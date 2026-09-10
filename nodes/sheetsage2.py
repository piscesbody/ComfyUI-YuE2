# -*- coding: utf-8 -*-
"""SheetSage2 转录节点 + 歌词格式化节点。

SheetSage2: 歌曲 -> ABC 乐谱 + 曲式结构（供翻唱与歌词分段）
歌词格式化: 带时间戳的字幕 + 曲式结构 -> 带段落标签的 YuE2 歌词
"""
from __future__ import annotations

import os
import re

from ..models import sheetsage2 as ss2_model
from ..models.paths import list_snapshots
from .utils import mono_waveform, timestamp_dir

# SheetSage2 的结构名 -> YuE2 段落标签（未识别的原样保留）
_SECTION_ALIASES = {
    "silence": "intro",
    "instrumental": "interlude",
    "solo": "interlude",
}

# 纯音乐段落（歌词结构化时不分配歌词行）
_NON_SINGING = {"intro", "silence", "interlude", "instrumental", "solo"}


def parse_structure_spans(text: str, duration_hint: float):
    """解析结构文本为 ``[(起点, 终点, 名)]``，合并相邻同名段。

    支持 ``起点<tab>终点<tab>名``、``起-止: 名``、``起点: 名`` 三种行格式。
    """
    spans: list[tuple[float, float, str]] = []
    points: list[tuple[float, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r"\t+", line)
        if len(parts) == 3:
            try:
                spans.append((float(parts[0]), float(parts[1]), parts[2].strip()))
                continue
            except ValueError:
                pass
        m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
        if m:
            spans.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
        if m:
            points.append((float(m.group(1)), m.group(2).strip()))

    if not spans and points:
        points.sort()
        for i, (start, name) in enumerate(points):
            end = points[i + 1][0] if i + 1 < len(points) else duration_hint
            spans.append((start, end, name))

    # 合并相邻同名段
    merged: list[tuple[float, float, str]] = []
    for start, end, name in sorted(spans):
        if merged and merged[-1][2] == name and abs(merged[-1][1] - start) < 1e-3:
            merged[-1] = (merged[-1][0], end, name)
        else:
            merged.append((start, end, name))
    return merged


class SheetSage2Loader:
    """加载 SheetSage2 歌曲转录模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        models = list_snapshots("sheetsage2") or ["SheetSage2"]
        return {
            "required": {
                "model": (models,),
                "device": (["cuda", "cpu"],),
                "dtype": (["bfloat16", "float32"],),
            },
        }

    RETURN_TYPES = ("SS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "YuE2/SheetSage2"

    def load(self, model, device, dtype):
        return (ss2_model.load(model, device=device, dtype=dtype),)


class SheetSage2Transcribe:
    """歌曲转录: 音频 -> ABC 乐谱 + 曲式结构。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SS2_MODEL",),
                "audio": ("AUDIO",),
                "melody_only": ("BOOLEAN", {"default": False,
                    "tooltip": "输出不含和弦的旋律谱（翻唱配 cot=melody 用）"}),
                "preset": (["default", "paper"],),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0,
                                          "step": 1.0, "tooltip": "只转录前 N 秒; 0=完整"}),
                "save_outputs": ("BOOLEAN", {"default": True,
                    "tooltip": "保存 score.abc / midi / lab 到 output/SheetSage2"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("abc", "structure", "info",)
    OUTPUT_NODE = True
    FUNCTION = "transcribe"
    CATEGORY = "YuE2/SheetSage2"

    def transcribe(self, model, audio, melody_only, preset, max_seconds, save_outputs):
        wav, rate = mono_waveform(audio)
        out_dir = timestamp_dir("SheetSage2") if save_outputs else None

        abc_text, structure_text, result = ss2_model.transcribe(
            model, wav, rate, melody_only=melody_only, preset=preset,
            max_seconds=max_seconds, output_dir=out_dir)

        chords = sum(1 for e in (result.get("events") or [])
                     if (e.get("values") or {}).get("chord"))
        info = (f"duration={result.get('duration_seconds', 0):.1f}s "
                f"vocal_notes={result.get('vocal_notes', 0)} "
                f"ins_notes={result.get('instrumental_notes', 0)} "
                f"chords={chords} sections={len(structure_text.splitlines())} "
                f"elapsed={result.get('elapsed_seconds', 0):.1f}s")
        if result.get("abc_error"):
            info += f" | abc_error: {result['abc_error']}"
        if out_dir:
            info += f" | saved: {out_dir}"
        print(f"[SheetSage2] {info}")
        return (abc_text, structure_text, info)


class LyricsFormatter:
    """把带时间戳的歌词字幕按曲式结构分段，整理成可交给 YuE2 的歌词。

    输入（二者都可用纯文本手填）:
    - ``subtitles``: 每行 ``起-止: 文本``，或标准 SRT
    - ``structure``: 每行 ``起点<tab>终点<tab>段落名``（SheetSage2 节点输出）

    处理: 解析时间 -> 按时间中点归入结构区间 -> 段首插 ``[verse]``/``[chorus]``
    等标签 -> 无词区间输出空标签（间奏） -> 清理标点与相邻重复行。
    """

    _PUNCT = str.maketrans("", "", "。！？，、；：,.!?;:\"'`()[]{}<>~…·—“”‘’")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subtitles": ("STRING", {"forceInput": True,
                    "tooltip": "带时间戳的字幕（每行 '起-止: 文本'）或 SRT 文本"}),
                "structure": ("STRING", {"forceInput": True,
                    "tooltip": "SheetSage2 节点的 structure 输出；留空则全部归入 [verse]"}),
                "clean_punct": ("BOOLEAN", {"default": True,
                    "tooltip": "去掉歌词文本中的标点"}),
                "dedupe": ("BOOLEAN", {"default": True,
                    "tooltip": "去掉相邻的完全重复行（副歌重复不在此列）"}),
                "min_line_chars": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1,
                    "tooltip": "短于此长度的行丢弃（哼声/气声误识别）; 0=不过滤"}),
                "map_sections": ("BOOLEAN", {"default": True,
                    "tooltip": "把 SheetSage2 的结构名映射为 YuE2 标签（如 silence->intro）"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("lyrics", "report",)
    FUNCTION = "format"
    CATEGORY = "YuE2/SheetSage2"

    # ── 解析 ──────────────────────────────────────────────────────────

    def parse_subtitles(self, text: str):
        """支持 ``起-止: 文本``、标准 SRT、LRC 三种格式，返回 ``[(start, end, text)]``。

        对时间戳不可靠的行做兜底：强制对齐器在纯音乐/气声片段上可能给出
        ``0.00-0.00`` 这类零长度区间，若直接使用会让这些行全部落到第一段。
        这里把无效时间标记为 ``None``，交由 :meth:`format` 按行序补位。
        """
        rows: list[tuple[float | None, float | None, str]] = []
        pending_start = None
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
            if m:
                rows.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
                continue
            # LRC: [mm:ss.xx]歌词（同一行可有多个连续时间标签）。
            # 只有行起点没有终点 → end 记 None，末尾统一延伸到下一行起点。
            lrc = re.match(r"^((?:\[\d+:\d+(?:\.\d+)?\])+)(.*)$", line)
            if lrc:
                text_part = lrc.group(2).strip()
                if text_part:
                    for tag in re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", lrc.group(1)):
                        t = int(tag[0]) * 60 + float(tag[1])
                        rows.append((t, None, text_part))
                continue
            m = re.match(
                r"^(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)$", line)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
                rows.append((start, end, ""))
                pending_start = start
                continue
            # SRT 的正文行：补上一条时间轴
            if pending_start is not None and rows and rows[-1][2] == "":
                s, e, _ = rows[-1]
                rows[-1] = (s, e, line)
                pending_start = None

        out = []
        for start, end, text in rows:
            if not text:
                continue
            # LRC 头部元数据行（歌名/词曲作者等），不是歌词
            if re.match(r"^(词|曲|编曲|作词|作曲|编|混音|母带|和声|吉他|钢琴|"
                        r"制作|监制|歌词|专辑|歌手|title|artist|album|by)[:：]",
                        text, re.IGNORECASE):
                continue
            # "歌名 - 歌手" 形式的标题行（LRC 首行惯例: 时间 0 且含 " - "）
            if start == 0 and " - " in text:
                continue
            # 零长度、倒挂或 (0,0) 哨兵：对齐器在歌声上失败的产物，时间视为未知。
            # end 为 None 的 LRC 行是"只有起点"的正常形态，下一循环补终点。
            if start is None or (end is not None and (end <= start or
                                                      (start == 0.0 and end == 0.0))):
                out.append((None, None, text))
            else:
                out.append((start, end, text))
        # LRC 行只有起点：延伸到下一行起点（末行 +8s），保证区间有效
        fixed = []
        for i, (s, e, t) in enumerate(out):
            if s is not None and e is None:
                nxt = next((out[j][0] for j in range(i + 1, len(out))
                            if out[j][0] is not None), None)
                e = nxt if (nxt is not None and nxt > s) else s + 8.0
            fixed.append((s, e, t))
        return fixed

    def _fill_missing_times(self, rows):
        """为时间无效的行按行序插值补位，使其仍能归入合理的段落。

        对齐器在歌声上常出现成片失败（尤其副歌与纯音乐段）。做法是找出连续
        的无效区段，在左右两个可靠锚点之间均匀铺开——而不是按固定步长从锚点
        回推，后者会把整段未对齐歌词挤到锚点附近。
        """
        n = len(rows)
        valid_idx = [i for i, (s, _, _) in enumerate(rows) if s is not None]

        if not valid_idx:
            # 完全没有任何可靠时间（对齐器在纯音乐/长段歌声上可能全失败）。
            # 此时歌词本身仍然正确，绝不能丢弃：按行序均匀铺在 [0, 1) 上，
            # 保证全部归入第一段并可被人工编辑。
            total = max(n, 1)
            return [(i / total, (i + 1) / total, t) for i, (_, _, t) in enumerate(rows)]

        # 兜底步长：可靠行的平均时长，用于无法在锚点间铺开的极端情形
        fallback = sum(rows[i][1] - rows[i][0] for i in valid_idx) / len(valid_idx)
        fallback = max(float(fallback), 0.05)

        filled = list(rows)
        # 在每段连续无效区段的两侧找锚点，然后均匀铺开
        bounds = [-1] + valid_idx + [n]
        for left, right in zip(bounds, bounds[1:]):
            lo, hi = left + 1, right          # 无效区段 [lo, hi)
            if lo >= hi:
                continue
            count = hi - lo
            anchor_l = filled[left][1] if left >= 0 else 0.0
            anchor_r = filled[right][0] if right < n else None
            if anchor_r is None:              # 尾部：从左侧锚点按兜底步长铺开
                step = fallback
                base = anchor_l
            else:
                span = max(anchor_r - anchor_l, 0.0)
                step = span / count if span > 0 else fallback
                base = anchor_l

            for k in range(count):
                s = base + step * k
                filled[lo + k] = (s, s + step, filled[lo + k][2])
        return filled

    def parse_structure(self, text: str, duration_hint: float):
        """兼容保留：见模块级 :func:`parse_structure_spans`。"""
        return parse_structure_spans(text, duration_hint)

    # ── 执行 ──────────────────────────────────────────────────────────

    def format(self, subtitles, structure, clean_punct, dedupe, min_line_chars,
               map_sections):
        sub_rows = self.parse_subtitles(subtitles)
        if not sub_rows:
            raise ValueError(
                "subtitles 为空或无法解析；每行应为 '起-止: 文本' 或标准 SRT")

        # 先按文本过滤/清理，再补时间（避免为将被丢弃的行插值）
        lines = []
        for start, end, text in sub_rows:
            cleaned = text.translate(self._PUNCT).strip() if clean_punct else text.strip()
            if not cleaned or (min_line_chars > 0 and len(cleaned) < min_line_chars):
                continue
            lines.append((start, end, cleaned))
        if dedupe:
            deduped = []
            for row in lines:
                if deduped and deduped[-1][2] == row[2]:
                    continue
                deduped.append(row)
            lines = deduped
        if not lines:
            raise ValueError("清理后没有剩下任何歌词行；请检查 min_line_chars 设置")

        # 统计对齐失败、时间靠插值推断的行数，用于在 report 中如实告知
        inferred = sum(1 for s, _, _ in lines if s is None)
        lines = self._fill_missing_times(lines)
        timed = [(s, e, t) for s, e, t in lines if s is not None]
        duration = max((e for _, e, _ in timed), default=float(len(lines)))

        spans = self.parse_structure(structure or "", duration + 1.0)
        if not spans:
            # 没有结构信息：全部归入一个段落
            sections = [("verse", [t for _, _, t in lines])]
        else:
            sections = []
            for idx, (start, end, name) in enumerate(spans):
                label = _SECTION_ALIASES.get(name.lower(), name) if map_sections else name
                last = idx == len(spans) - 1
                bucket = []
                for s, e, t in lines:
                    if s is None:
                        continue
                    mid = (s + e) / 2
                    if start <= mid < end or (last and mid >= start):
                        bucket.append(t)
                if sections and sections[-1][0] == label:
                    sections[-1][1].extend(bucket)
                else:
                    sections.append((label, bucket))
            # 歌词掉进纯音乐段（间奏/前奏）说明边界有偏移：并入下一个歌唱段，
            # 间奏保持为空标签（YuE2 把空段当器乐）。
            singing = [i for i, (name, _) in enumerate(sections)
                       if (name or "").lower() not in _NON_SINGING]
            if singing:
                def nearest_singing(i):
                    return min(singing, key=lambda j: (abs(j - i), j))
                for i, (name, bucket) in enumerate(sections):
                    if (name or "").lower() in _NON_SINGING and bucket:
                        sections[nearest_singing(i)][1].extend(bucket)
                        bucket.clear()

        out_lines: list[str] = []
        for name, bucket in sections:
            out_lines.append(f"[{(name or 'verse').strip().lower()}]")
            out_lines.extend(bucket)
            out_lines.append("")
        lyrics = "\n".join(out_lines).rstrip() + "\n"

        report = (f"lines={len(sub_rows)} kept={len(lines)} "
                  f"sections={len(sections)} chars={sum(len(t) for _, _, t in lines)}")
        if inferred:
            # 对齐器在歌声上大面积失败时，段落归属是推断出来的，必须告知用户
            report += (f" | 警告: {inferred}/{len(lines)} 行无可靠时间戳"
                       "(对齐器对歌声失效), 段落归属为推断值, 建议人工核对")
        print(f"[歌词格式化] {report}")
        return (lyrics, report)


class LyricsStructurer:
    """把纯文本歌词自动加上 [verse]/[chorus] 段落标签，直接交给 YuE2。

    两种用法（structure 留空时按行数自动规划）：
    - 只填 ``lyrics_text``：每 ``verse_lines`` 行算一段，全部标 [verse]
    - 填 ``section_plan``：逗号分隔的段落序列，如 ``verse,chorus,verse,chorus``
      （可带行数 ``verse:8,chorus:8``），按顺序每段取对应行数

    也接受手工标注：文本里已带 ``[xxx]`` 行的段落原样保留。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics_text": ("STRING", {"multiline": True,
                    "tooltip": "纯文本歌词，每行一句；空行会被忽略"}),
                "structure": ("STRING", {"default": "",
                    "tooltip": "可选: SheetSage2 的 structure 输出。提供时按段落"
                               "时长比例把歌词行分配到各段（歌唱段才分词）"}),
                "verse_lines": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "structure 为空时每段的行数"}),
                "section_plan": ("STRING", {"default": "",
                    "tooltip": "段落规划, 如 verse,chorus,verse,chorus 或 "
                               "verse:8,chorus:8,verse:8,chorus:8; 留空=全部 verse"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("lyrics", "report",)
    FUNCTION = "format"
    CATEGORY = "YuE2/SheetSage2"

    def format(self, lyrics_text, structure, verse_lines, section_plan):
        # 已带 [tag] 的文本原样保留（用户手工标好就尊重）
        if re.search(r"^\s*\[[a-z]+\]\s*$", lyrics_text or "", re.MULTILINE):
            cleaned = "\n".join(
                ln for ln in (l.strip() for l in lyrics_text.splitlines())
                if ln and not re.match(r"^\[[a-z]+\]$", ln))
            report = f"检测到已有段落标签, 原样保留 (正文 {len(cleaned.splitlines())} 行)"
            print(f"[歌词结构化] {report}")
            return (lyrics_text.rstrip() + "\n", report)

        lines = [ln.strip() for ln in (lyrics_text or "").splitlines() if ln.strip()]
        if not lines:
            raise ValueError("lyrics_text 为空")

        plan = self._parse_plan(section_plan, verse_lines, len(lines))

        # 有结构信息时按歌唱段时长比例分配行数
        spans = parse_structure_spans(structure or "", 0.0) if structure else []
        if spans:
            singing = [(name, end - start) for start, end, name in spans
                       if _SECTION_ALIASES.get(name.lower(), name.lower()) not in _NON_SINGING
                       and (end - start) > 1.0]
            if singing:
                total = sum(d for _, d in singing)
                plan = [(name, max(1, round(len(lines) * dur / total)))
                        for name, dur in singing]
                plan[-1] = (plan[-1][0], len(lines) - sum(n for _, n in plan[:-1]))

        out: list[str] = []
        idx, sections = 0, 0
        for name, count in plan:
            if idx >= len(lines):
                break
            take = lines[idx:idx + count]
            idx += len(take)
            out.append(f"[{name}]")
            out.extend(take)
            out.append("")
            sections += 1
        if idx < len(lines):          # 行数超出规划：剩余全部进尾段
            out.append(f"[{plan[-1][0]}]" if not out or out[-1] else f"[{plan[-1][0]}]")
            out.extend(lines[idx:])

        lyrics = "\n".join(out).rstrip() + "\n"
        report = (f"lines={len(lines)} sections={sections} "
                  f"plan={','.join(f'{n}:{c}' for n, c in plan)}")
        print(f"[歌词结构化] {report}")
        return (lyrics, report)

    @staticmethod
    def _parse_plan(section_plan, verse_lines, n_lines):
        """解析段落规划为 ``[(段名, 行数)]``。"""
        if not section_plan.strip():
            n = max(1, -(-n_lines // max(verse_lines, 1)))
            return [("verse", verse_lines)] * n
        plan = []
        for part in section_plan.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if ":" in part:
                name, cnt = part.split(":", 1)
                plan.append((name.strip() or "verse", max(1, int(cnt))))
            else:
                plan.append((part, verse_lines))
        return plan or [("verse", verse_lines)]


NODE_CLASS_MAPPINGS = {
    "SheetSage2Loader": SheetSage2Loader,
    "SheetSage2Transcribe": SheetSage2Transcribe,
    "YuE2LyricsFormatter": LyricsFormatter,
    "YuE2LyricsStructurer": LyricsStructurer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SheetSage2Loader": "SheetSage2 加载模型",
    "SheetSage2Transcribe": "SheetSage2 歌曲转录",
    "YuE2LyricsFormatter": "歌词格式化 (时间戳→段落)",
    "YuE2LyricsStructurer": "歌词结构化 (纯文本→段落)",
}

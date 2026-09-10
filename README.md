# ComfyUI-YuE2

YuE2 音乐生成 + SheetSage2 歌曲转录的一体化 ComfyUI 节点包。

两套模型互相衔接，可以只做音乐生成，也可以串成"听歌 → 识谱 → 手填歌词 → 重编曲"的完整翻唱流水线。
包内自包含全部推理代码与 transformers 兼容层，按下面的说明装依赖、放模型即可使用。

## 能做什么

| 工作流 | 节点链路 |
|---|---|
| 文生歌（可编辑乐谱） | `YuE2 加载模型` → `YuE2 歌曲生成`（`cot=full`）|
| 无乐谱快速生成 | 同上，`cot=off` |
| 翻唱（改风格/编曲） | `SheetSage2 歌曲转录`（`melody_only`）+ `歌词结构化` → `YuE2 歌曲生成`（`cot=melody`）|
| 改编（换和声/改谱） | `SheetSage2 歌曲转录`（含和弦）→ 编辑 ABC → `YuE2 歌曲生成`（`cot=full`）|
| 只要乐谱 | `YuE2 歌曲生成` 打开 `abort_after_plan` |

歌词建议手填：`歌词结构化` 节点把纯文本歌词自动加上 `[verse]`/`[chorus]`
段落标签，比自动识别更快更准（语音 ASR 模型对歌唱歌词的识别目前不可靠）。

## 安装

### 1. 依赖

```bash
# 在 ComfyUI 的 Python 环境里执行
pip install -r requirements.txt
```

### 2. YuE2 推理库

YuE2 官方推理库随模型仓库分发，版本需与模型匹配。从模型目录安装：

```bash
pip install <模型目录>/yue2_infer-0.1.5-py3-none-any.whl
```

### 3. 模型

模型全部放在 ComfyUI 默认的 `models/` 目录下，两个子目录各管一件事。
下面写清了**每个目录放什么**，照着放即可；每一项都可以用软链接（junction）
指向已有副本，不必复制文件。

```
ComfyUI/models/
├── YuE2/                      ← 歌曲生成
│   ├── YuE2-3B/                 必装，生成模型
│   ├── YuE2-Vae/                必装，音频解码器（听感优先）
│   └── YuE2-Vae-legacy/         可选，复现论文基准
└── SheetSage2/                ← 歌曲转录（音频 → 乐谱）
    ├── config.json              必装，模型文件直接放这一层
    ├── model.safetensors
    └── ...
```

各目录对应的下载地址：

| 目录 | 放什么 | Hugging Face 仓库 |
|---|---|---|
| `YuE2/YuE2-3B/` | 生成模型 | [`m-a-p/YuE2-3B`](https://huggingface.co/m-a-p/YuE2-3B) 或 [`mrfakename/YuE2-3B`](https://huggingface.co/mrfakename/YuE2-3B) |
| `YuE2/YuE2-Vae/` | 音频解码器 | [`m-a-p/YuE2-Vae`](https://huggingface.co/m-a-p/YuE2-Vae) |
| `YuE2/YuE2-Vae-legacy/` | 可选解码器 | [`m-a-p/YuE2-Vae-legacy`](https://huggingface.co/m-a-p/YuE2-Vae-legacy) |
| `SheetSage2/`（文件直接放这层） | 转录模型 | [`m-a-p/SheetSage2`](https://huggingface.co/m-a-p/SheetSage2) |

另外 **SheetSage2 首次加载会自动下载其父模型** [`m-a-p/MERT-v2-FullSong`](https://huggingface.co/m-a-p/MERT-v2-FullSong)
到 Hugging Face 缓存（`~/.cache/huggingface/`，约 3.7 GB）——不需要手动放置，
但需要联网，或事先把该快照放进缓存。

下载示例：

```bash
huggingface-cli download m-a-p/YuE2-3B --local-dir ComfyUI/models/YuE2/YuE2-3B
```

> **`SheetSage2` 注意**：它是「仓库本身就是一个模型目录」，文件要**直接放
> 在 `models/SheetSage2/` 这一层**，不要再套一层同名目录。用
> `huggingface-cli` 下载时它会默认多套一层，需要手动把里面的文件上移：
>
> ```bash
> huggingface-cli download m-a-p/SheetSage2 --local-dir /tmp/ss2
> # 然后把 /tmp/ss2/SheetSage2/* 全部移到 ComfyUI/models/SheetSage2/
> ```
>
> 放对之后 `models/SheetSage2/config.json` 应该存在。若变成了
> `models/SheetSage2/SheetSage2/config.json` 就是多套了一层。
>
> 首次加载还会自动下载其 MERT-v2-FullSong 父模型到 HF 缓存，
> 需要联网（或事先把该快照放进缓存）。

### 重定向到其它位置

不想放在默认位置的话，在 `ComfyUI/extra_model_paths.yaml` 里加：

```yaml
yue2_music:
    base_path: F:/models/music
    yue2: YuE2/
    sheetsage2: SheetSage2/
```

`base_path` 下的子目录结构与上面 `ComfyUI/models/` 完全一致。

### 示例工作流

[`example_workflows/example_workflows.json`](example_workflows/example_workflows.json)
包含完整翻唱流水线（转录 → 歌词结构化 → 生成）。在 ComfyUI 里
`Workflows → Open` 直接导入即可；音频文件和歌词都是占位内容，
换成你自己的就能跑。

## 节点

### YuE2 音乐生成

| 节点 | 说明 |
|---|---|
| **YuE2 加载模型** | 加载生成模型与解码器；进程内按参数缓存 |
| **YuE2 歌曲生成** | 风格 + 歌词 → 48 kHz 立体声完整歌曲；输出音频、ABC 乐谱、运行信息 |
| **YuE2 卸载** | 释放显存 |

`cot` 三档：

- `full` — 生成带和弦的 ABC 乐谱后再合成（改谱/改编用）
- `melody` — 只规划旋律，伴奏自由发挥（**翻唱推荐**）
- `off` — 不做符号规划，直接生成

`style` 写曲风、乐器、人声特质、语言；`lyrics` 用 `[verse]`/`[chorus]` 等段落标签。
把 `abort_after_plan` 打开可只出乐谱、不合成音频（先看谱再决定）。

显存与性能选项（RTX 4090 实测）：

| 参数 | 选项 | 说明 |
|---|---|---|
| `vae_decode` | `tiled` / `full` | 分块解码（默认，显存友好）/ 整曲一次解码（快 ~2s，需大显存，不足自动回退）。VAE 解码只占总时长 ~2%，日常保持默认即可 |
| `vae_tile_frames` | 0=自动 / 256~4096 | 分块粒度。8GB 显存建议 256；实测 3 分钟歌各档峰值差 ~1GB |
| `attention_backend` | `auto` / `external-flash` / `cudnn` / `sdpa` | AR 解码内核。`external-flash` 用环境里 pip 安装的 flash-attn；实测与 cuDNN 速度持平（512 步 6.2ms/步） |
| `quantization` | `none` / `fp8` | FP8 量化（官方实验性）。实测省 ~1.3GB 但慢 ~6 倍（禁用 CUDA Graph），仅限显存极端受限 |

> 生成耗时的主体是 AR 自回归 token 生成（3 分钟歌约 70-90s），
> VAE 解码仅 ~2s。提速请优先调 `ode_steps`/`semantic_max_tokens`，而非解码选项。

### SheetSage2 歌曲转录

| 节点 | 说明 |
|---|---|
| **SheetSage2 加载模型** | 加载转录模型 |
| **SheetSage2 歌曲转录** | 音频 → ABC 乐谱 + 曲式结构 |

- `melody_only`：输出不含和弦的旋律谱，配 `cot=melody` 做翻唱
- `structure` 输出每行 `起点<tab>终点<tab>段落名`，供歌词节点分段
- 结果默认保存 `score.abc` / MIDI / `.lab` 到 `output/SheetSage2/`

### 歌词工具

| 节点 | 说明 |
|---|---|
| **歌词结构化 (纯文本→段落)** | 贴纯文本歌词，自动加 `[verse]`/`[chorus]` 段落标签 |
| **歌词格式化 (时间戳→段落)** | 带时间戳的字幕（SRT 等）按曲式结构分段 |

**歌词结构化** 的三种用法：

1. **最简**：只填 `lyrics_text`，每 `verse_lines` 行（默认 4）自动一段
2. **指定结构**：`section_plan` 填 `verse:8,chorus:4,verse:8,chorus:8`，
   按顺序取对应行数（只写段名不写行数时用 `verse_lines`）
3. **全自动**：`structure` 接 SheetSage2 的输出，按各歌唱段时长比例分配行数，
   段名对应真实曲式（intro/interlude 等纯音乐段自动跳过不分配歌词）

文本里已带 `[xxx]` 标签的原样保留，不会二次处理。

**歌词格式化** 面向带时间戳的来源（每行 `起-止: 文本` 或标准 SRT）：
按时间中点归入 SheetSage2 的结构区间，自动清理标点与相邻重复行；
时间戳不可靠的行按行序插值补位，不会全挤进第一段。

## 完整翻唱流水线

```
LoadAudio ─┬─→ SheetSage2 歌曲转录 (melody_only) ──→ abc ──────┐
           │                                      └structure─┐ │
           │                                                 ↓ ↓
           └─→ 歌词结构化 (lyrics_text 手填) ────→ lyrics ─→ YuE2 歌曲生成 (cot=melody) → SaveAudio
```

风格由 `style` 决定（换成 Jazz / 摇滚 / 民谣…即可换编曲），歌词用原词也可改写。

## 环境与兼容性

本包在 **Python 3.13 + torch 2.10+cu130 + transformers 5.3** 的 ComfyUI 便携包上开发验证，
并内置了让 4.x 时代模型代码在 transformers 5.x 下正确运行的兼容层（`compat/`）：

- **Windows 版 torch 构建无内置 flash-attention 内核** → YuE2 的 CUDA Graph 解码
  自动降级到 cuDNN SDPA 后端（Linux 正常构建不受影响）
- **YuE2 管线会压低进程显存配额** → 加载后立即复位，避免影响同进程其它模型
- **MERT2 的位置编码 buffer 被 5.x 写坏** → 表现为"转录只有节奏和和弦、没有旋律音高"，
  加载后自动重算

详细根因见 `compat/` 各模块的文档字符串。

## 显存参考（RTX 4090）

| 任务 | 峰值显存 | 速度 |
|---|---|---|
| YuE2 生成 3.6 分钟歌曲 | ~11 GB | ~71 秒（官方数据）|
| SheetSage2 转录 60 秒音频 | ~2 GB | ~4 秒 |
| SheetSage2 转录整首 4.6 分钟歌曲 | ~2 GB | ~17 秒 |

两个模型可同时驻留显存；生成时建议卸载不用的那个。

## 许可证

- 本包代码：见 `LICENSE`
- 模型权重各自遵循其仓库许可证；**YuE2 权重为 CC BY-NC 4.0（非商用）**

# Novel to Sound (Novel_to_sound)

将中文小说转换为角色配音有声书的自动化流水线。

## 流水线概览

```
小说文本 → 章节拆分 → LLM结构化(角色/性别/情绪) → 角色音色分配 → TTS语音合成 → WAV+字幕
```

| 阶段 | 脚本 | 功能 |
|------|------|------|
| A | `src/pipeline/stage_a_novel_cut.py` | 将小说按章节拆分为 txt |
| B | `src/pipeline/stage_b_json_converter.py` | LLM 提取角色/对白/情绪 → 结构化 JSON |
| C | `src/pipeline/stage_c_fix_narrator.py` | 确保旁白条目存在于 role_registry |
| D | `src/pipeline/stage_d_build_batches.py` | 聚合同一批次角色信息 → CSV 供用户分配音色 |
| E | `src/pipeline/stage_e_sync_voices.py` | 将用户填写的 CSV 音色选择同步到 voice_profiles.json |
| F | `src/pipeline/stage_f_tts_gen.py` | TTS 合成：生成章节 WAV + SRT 字幕 + 分段元数据 |

## 环境要求

- Python 3.10+
- CUDA GPU (推荐，用于 TTS 合成)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 模型

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API Key

复制 `.env.example` 为 `.env`，填入 DashScope API Key：

```bash
cp .env.example .env
# 编辑 .env:
# DASHSCOPE_API_KEY=sk-your-key-here
# QWEN3_TTS_ROOT=D:\Qwen3-TTS
```

也可以直接设置环境变量：
```bash
set DASHSCOPE_API_KEY=sk-your-key
set QWEN3_TTS_ROOT=D:\Qwen3-TTS
```

## 使用方法

### 准备小说文件

将小说 txt 文件放入 `data/novel/` 目录。

### 完整流水线

```bash
# Stage A: 拆分章节
python src/pipeline/stage_a_novel_cut.py data/novel/your_novel.txt -o data/chapter

# Stage B: 章节 → 结构化 JSON (批量运行)
python scripts/run_json_batches.py --batch-size 100

# Stage C: 修复旁白条目
python src/pipeline/stage_c_fix_narrator.py

# Stage D: 聚合角色 → 生成音色选择 CSV
python src/pipeline/stage_d_build_batches.py

# ⚠ 手动步骤: 编辑 data/role_batches/voice_pick_*.csv
#   填写 design_prompt (文本描述音色) 或 clone_ref_audio_path (克隆参考音频)

# Stage E: 同步 CSV 选择到 voice_profiles.json
python src/pipeline/stage_e_sync_voices.py

# Stage F: TTS 合成音频
python src/pipeline/stage_f_tts_gen.py
```

### 单独运行某个阶段

每个脚本支持 `--help` 查看完整参数：

```bash
python src/pipeline/stage_b_json_converter.py --help
```

常用参数：
- `--in_dir` / `--out_dir`: 自定义输入/输出目录
- `--start` / `--limit`: 控制处理范围
- `--force`: 强制重新生成已有文件
- `--dry_run`: 预览模式，不实际写入

## 项目结构

```
├── src/
│   ├── lib/
│   │   ├── config.py          # 路径配置 & 环境变量
│   │   └── utils.py           # 共享工具函数
│   └── pipeline/              # 6 个流水线阶段脚本
├── scripts/
│   ├── run_json_batches.py    # Stage B 批量运行器
│   └── prepare_voice_selection.py
├── data/                      # 数据目录 (gitignore 排除小说内容)
│   ├── role_batches/          # 角色批次 & 音色选择 CSV
│   └── voice_profiles.json   # 角色音色注册表
├── output/                    # TTS 输出 (gitignore)
├── deprecated/                # 废弃/参考脚本
├── .env.example               # 环境变量模板
├── requirements.txt
└── README.md
```

> **注意**: `data/novel/`、`data/chapter/`、`data/chapters_json/` 和 `output/` 不纳入版本控制。

## 音色配置说明

在 `data/role_batches/voice_pick_*.csv` 中为每个角色选择音色来源：

### 方式一：文本描述音色 (Voice Design)
在 `design_prompt` 列填写音色描述，例如：
- `温柔知性的年轻女性声音`
- `低沉沙哑的老年男性声音`
- `清脆明亮的少年音`

### 方式二：音频克隆 (Voice Clone)
填写 `clone_ref_audio_path` (参考音频路径) 和 `clone_ref_text` (参考文本) 两列。

两种方式互斥，同时填写会被跳过并标记为冲突。

## License

MIT

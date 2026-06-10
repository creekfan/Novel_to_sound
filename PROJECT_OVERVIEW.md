# NS Project Overview

## 1. Project Goal

This project turns a novel into chapter-level structured data and then into character-based audio with subtitles.

Pipeline stages:
1. **Stage A**: Split full novel text into chapter `.txt` files.
2. **Stage B**: Convert chapter text into structured chapter `.json` via LLM.
3. **Stage C**: Normalize narrator entries in chapter JSONs.
4. **Stage D**: Aggregate character information in 100-chapter batches.
5. **Stage E**: Sync user voice-pick CSV into `voice_profiles.json`.
6. **Stage F**: Generate chapter audio, subtitles, and segment metadata via TTS.


## 2. Project Structure

```
D:\NS\
├── src/
│   ├── lib/
│   │   ├── config.py                   # Central paths, API config, env vars
│   │   └── utils.py                    # Shared utilities (load_json, safe_name, etc.)
│   └── pipeline/
│       ├── stage_a_novel_cut.py        # Stage A: Novel -> chapter txt
│       ├── stage_b_json_converter.py   # Stage B: Chapter txt -> JSON (LLM)
│       ├── stage_c_fix_narrator.py     # Stage C: Ensure narrator in role_registry
│       ├── stage_d_build_batches.py    # Stage D: Aggregate roles into batches
│       ├── stage_e_sync_voices.py      # Stage E: Sync CSV choices -> voice_profiles
│       └── stage_f_tts_gen.py          # Stage F: TTS generation (WAV + SRT)
├── scripts/
│   ├── run_json_batches.py             # Batch runner for Stage B (replaces broken .bat)
│   └── prepare_voice_selection.py      # Generate design/clone request templates
├── data/
│   ├── novel/                          # Source novel text
│   ├── chapter/                        # Per-chapter txt (1455 files)
│   ├── chapters_json/                  # Structured chapter JSON (109+ files)
│   ├── role_batches/                   # Role extraction batches + voice_pick CSVs
│   └── voice_profiles.json             # Global voice registry
├── output/                             # Generated audio outputs
│   ├── tts_outputs_smoke_0001/
│   └── tts_outputs_0001_0100/
├── deprecated/                         # Old/legacy scripts (not current pipeline)
│   ├── LLM_prompt.py
│   ├── TTS.py
│   ├── chapter_tts_from_csv.py
│   ├── simple_voice_design_gender_prompt.py
│   ├── run_json_batches_100.bat        # Broken - use scripts/run_json_batches.py
│   └── README_qwen3_tts_migration.txt
├── .env.example                        # Template for env vars
├── .gitignore
├── requirements.txt
└── PROJECT_OVERVIEW.md
```


## 3. Pipeline Scripts Reference

### Stage A: Novel to Chapter Text
- **Script**: `src/pipeline/stage_a_novel_cut.py`
- Splits the source novel into per-chapter `.txt` files.
- Usage: `python src/pipeline/stage_a_novel_cut.py data/novel/lvyang.txt -o data/chapter`

### Stage B: Chapter Text to Structured JSON
- **Script**: `src/pipeline/stage_b_json_converter.py`
- Calls DashScope LLM to convert chapter `.txt` -> structured `.json`.
- Resumable via `progress.json`, auto JSON-repair retry.
- **Requires**: `DASHSCOPE_API_KEY` env variable (or `--api_key` flag).
- Batch runner: `python scripts/run_json_batches.py --batch-size 100`

### Stage C: Normalize Narrator
- **Script**: `src/pipeline/stage_c_fix_narrator.py`
- Ensures `role_registry["旁白"]` exists in every chapter JSON.

### Stage D: Extract Characters by Batch
- **Script**: `src/pipeline/stage_d_build_batches.py`
- Aggregates all characters from chapter JSONs into 100-chapter batches.
- Generates `roles_*.json` and `voice_pick_*.csv` under `data/role_batches/`.
- User fills the generated CSVs with voice assignments.

### Stage E: Sync Voice Selection
- **Script**: `src/pipeline/stage_e_sync_voices.py`
- Reads user-filled `voice_pick_*.csv` files, syncs design/clone choices into `data/voice_profiles.json`.

### Stage F: Generate Audio and Subtitles
- **Script**: `src/pipeline/stage_f_tts_gen.py`
- Loads Qwen3-TTS models, processes chapter JSONs, generates:
  - Merged chapter `.wav`
  - `.srt` subtitles
  - `.segments.json` metadata
- **Requires**: Qwen3-TTS at `QWEN3_TTS_ROOT` (env var or default `D:\Qwen3-TTS`).


## 4. Quick Start

```bash
# 1. Set up environment
copy .env.example .env
# Edit .env and add your DASHSCOPE_API_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Stage A: Split novel
python src/pipeline/stage_a_novel_cut.py data/novel/lvyang.txt -o data/chapter

# 4. Stage B: Convert to JSON (batch)
python scripts/run_json_batches.py --batch-size 100

# 5. Stage C: Fix narrators
python src/pipeline/stage_c_fix_narrator.py

# 6. Stage D: Build voice batches
python src/pipeline/stage_d_build_batches.py

# 7. (Manual) Fill voice_pick_*.csv in data/role_batches/

# 8. Stage E: Sync voices
python src/pipeline/stage_e_sync_voices.py

# 9. Stage F: Generate TTS audio
python src/pipeline/stage_f_tts_gen.py
```


## 5. Configuration

All default paths are in `src/lib/config.py`. Override via:
- **Environment variables**: `DASHSCOPE_API_KEY`, `QWEN3_TTS_ROOT`, `LLM_MODEL`
- **Command-line args**: each script accepts `--in_dir`, `--out_dir`, etc.

Shared utilities (`load_json`, `save_json`, `safe_name`, `role_key_from_name`, `dominant`, `suggest_voice`, `normalize_gender`, `map_language`) are in `src/lib/utils.py`.


## 6. Current Operational State

### Confirmed Available
- Full novel source exists in `data/novel/`.
- Full chapter text split exists in `data/chapter/` (~1455 files).
- Structured JSON exists for early chapter range in `data/chapters_json/` (~109 files).
- Role batch CSVs exist for chapters 1-100 and 101-200 in `data/role_batches/`.
- TTS output directories contain partial runs and caches in `output/`.

### Incomplete
- JSON conversion not complete for all 1455 chapters.
- Character extraction not complete for the full book.
- Voice assignment centered on batch 1-100.
- `tts_outputs_0001_0100` is partial, not a confirmed finished run.


## 7. Known Issues

- `data/chapter/` contains some bad chapter splits (regex overmatches noisy text).
- `data/chapters_json/` has at least one `.error.txt` from failed conversion.
- `data/voice_profiles.json` may display mojibake in cmd (terminal encoding issue, file is valid UTF-8).


## 8. Environment Notes

- Main runtime: Anaconda env `SN` (`D:\annaconda\envs\SN`).
- Qwen3-TTS expected at `D:\Qwen3-TTS` (or set `QWEN3_TTS_ROOT` env var).
- API key: set `DASHSCOPE_API_KEY` environment variable.

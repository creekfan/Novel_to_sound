These files were migrated from D:\Qwen3-TTS\examples.

Files:
- chapter_tts_from_csv.py
- simple_voice_design_gender_prompt.py
- role_voice_0_100_template.csv
- chapters_template.json

Runtime note:
- The scripts will import qwen_tts from D:\Qwen3-TTS by default.
- To override, set environment variable QWEN3_TTS_ROOT.

Examples:
- D:\annaconda\envs\SN\python.exe D:\NS\simple_voice_design_gender_prompt.py --gender 女 --prompt "温柔，偏年轻" --text "你好，这是测试。"
- D:\annaconda\envs\SN\python.exe D:\NS\chapter_tts_from_csv.py --roles-csv D:\NS\role_voice_0_100_template.csv --chapters D:\NS\chapters_template.json --out-dir D:\NS\outputs_chapter_tts

import os
import json
from openai import OpenAI

# 1) 初始化客户端（环境变量：OPENAI_API_KEY）
client = OpenAI(
            api_key="******",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

# 2) 读取小说章节文本
with open("第一章.txt", "r", encoding="utf-8") as f:
    chapter_text = f.read()

# 3) 你的“结构化拆分器”prompt（不使用 Structured Outputs，只靠指令约束）
SYSTEM_PROMPT = """你是“小说章节文本结构化拆分器”。你的任务：把我提供的【小说章节.txt 原文】拆分成结构化 JSON。你必须严格遵守规则，保证可解析、可追溯、可复用。

========================
总原则（最高优先级）
========================
1) 禁止改写原文：不得润色、纠错、替换同义词、调整语序、合并/删减句子。
   - 所有 lines[i].dialogue 必须是原文中的逐字连续片段（允许去掉首尾空白；不得拼接非连续片段）。
2) 禁止臆造：任何字段若无明确原文证据，必须填 null 或 []。
3) 同一角色 feature 仅在角色出现足够稳定证据后写入，若后文出现更高优先级证据（例如直接嗓音描写），允许更新 feature，并在 notes 说明“高优先级证据覆盖”。
   - 后文若出现新的特征线索：写入 lines[i].notes，不更新 role_registry 里的 feature。
4) 只输出 JSON：输出必须是严格可解析 JSON（UTF-8），不得输出任何解释、前后缀、Markdown、注释。

========================
输出 JSON 结构（必须严格遵守）
========================
{
  "chapter_title": string|null,
  "lines": [
    {
      "role": string,                 // "旁白" 或 角色名 或 职业、身份
      "gender": "男"|"女"|null,        // 严格证据
      "dialogue": string,              // 原文逐字片段
      "emotion": string|null,          // 短标签，必须可由证据支撑
      "feature": string|null,          // 仅在角色出现足够稳定证据后写入，若后文出现更高优先级证据（例如直接嗓音描写），允许更新 feature，并在 notes 说明“高优先级证据覆盖”
      "evidence": {
        "role_quotes": [string],       // 支持 role 判定的原文短引（<=30字/条）
        "gender_quotes": [string],     // 支持 gender 的原文短引（<=30字/条）
        "emotion_quotes": [string],    // 支持 emotion 的原文短引（<=30字/条）
        "feature_quotes": [string]     // 支持 feature 的原文短引（<=30字/条）
      },
      "notes": string|null             // 仅记录不确定性/新增线索/歧义原因
    }
  ],
  "role_registry": {
    "<角色名或占位名>": {
      "gender": "男"|"女"|null,
      "feature": string|null,
      "first_seen_line_index": integer
    }
  }
}

约束：
- 禁止新增顶层字段；禁止在每个 line 对象里新增字段。
- evidence 四个数组必须存在；若无证据则填 []。
- lines 顺序必须与原文出现顺序一致。

========================
拆分规则（如何生成 lines）
========================
A) 何时拆分为新 line
- 对话：出现引号（“”/「」/『』）或“某某说/道/问/喊/低声说/心想”等引导的直接引语 → 单独一行。
- 内心语言/心理独白：有明确归属（他想/她心里/我暗自/某某在心里说）→ 归属该角色，单独一行。
- 旁白叙述：环境、动作、叙述、无法确定说话者的描写 → role="旁白"。
- 若同一句包含“旁白 + 引语”，必须拆成两行：旁白部分一行，引语部分一行。

B) dialogue 字段的截取
- 每一行的 dialogue 必须是原文的连续片段。
- 允许包含原文换行与标点；不得修改标点；不得拼接两段不相邻的文本。
- 若原文中引号不完整/不规范，也不得修正；照原文截取。

========================
角色判定规则（role）
========================
1) 明确角色名：原文出现姓名/称呼并与引语绑定（如“张三说：‘…’”）→ role=“张三”。
2) 代词指代：仅当在同一小段落内能唯一确定“他/她/我/你”对应哪个角色时，才可用该角色名。
3) 说话者在原文仅以身份短语出现（如“家族长辈/掌柜/老者/小二/小斯...”），允许直接使用该短语作为 role。
4) role_quotes：必须提供至少一条支持 role 的原文短引；若 role="旁白" 则 role_quotes=[]。

========================
性别规则（gender，极严格）
========================
仅在原文出现明确性别证据时填“男/女”，否则 null。
可用证据（举例）：
- 明确词：男/女/男人/女人/少年/少女/老汉/老妇/男孩/女孩/丈夫/妻子 等
- 代词“他/她”只能在能唯一绑定到该 role 时才算证据
必须把证据摘入 gender_quotes（<=30字/条）。无证据则 gender=null 且 gender_quotes=[]。

========================
情绪规则（emotion，证据驱动）
========================
- emotion 为短标签（如：愤怒/喜悦/悲伤/惊讶/紧张/羞怯/害怕/冷漠/不耐烦/坚定/委屈/释然/焦虑…）。
- 禁止使用“介绍/解释/安抚/劝告/提醒/命令/请求/陈述”等“话语功能/行为目的”作为 emotion。
- 必须有原文依据：情绪词、语气词、动作描写、说话方式（如“怒道/冷冷地/哽咽/笑/颤抖/喊”）等。
- emotion_quotes 必须提供证据短引；否则 []。

========================
feature 规则（基础音色设定规则）
========================
feature 的目的：为角色设定“长期稳定的基础嗓音特征”，用于 TTS 的基础音色选择。
feature 不代表情绪，也不代表当前语气。
允许作为 feature 依据的证据类型（按优先级）：
1）明确嗓音描写（最高优先级）
   如：声音低沉 / 嗓音沙哑 / 声音清亮 / 尖细 / 洪亮 等
   → 直接映射为基础音色
2）年龄线索
   如：少年 / 孩子 / 老者 / 中年男子 等
   → 可推导为音色区间（如偏清亮 / 偏低沉）
3）长期性格或气质描写（必须为稳定特征）
   如：一向沉稳 / 性格暴躁 / 温婉柔和
   → 可推导为语调风格倾向
写法要求：
- 根据优先级直接描述嗓音特征（20字以内）。
- 若无上述证据 → feature = null
同一 role：
- 第一次出现该 role 时，若可填写 feature 则写入，并同步写入 role_registry。
- 后续该 role 的 lines[i].feature 必须与 role_registry 中一致（原样复用），不得新增或修改。
- 若后文出现新线索：写入 notes，例如“发现‘沙哑’线索但不更新 feature”。

========================
chapter_title 规则
========================
- 若原文开头存在明显标题/章名（如“第十二章 …”）→ 填该原文标题（逐字）。
- 否则 null。

========================
自检（在输出前必须完成）
========================
- JSON 可解析；无多余字段；必含 chapter_title、lines、role_registry。
- 每条 line 必含 role、gender、dialogue、emotion、feature、evidence、notes。
- dialogue 均为原文连续片段且不改写。
- 任何非 null 的 gender/emotion/feature 均有对应 quotes 证据；否则改为 null。
- role_registry 中每个角色的 feature 在后续 lines 中未发生变化。

========================
输入格式
========================
我将用以下标记提供原文：
<<<TEXT>>>
（原文）
<<<END>>>

现在开始处理，并只输出 JSON。

"""

user_input = f"""请按规则把以下章节拆分为 JSON（只输出 JSON）：
<<<TEXT>>>
{chapter_text}
<<<END>>>"""

# 4) 调用 Responses API
resp = client.chat.completions.create(
    model="qwen3.5-397b-a17b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ],
    # 可选：控制输出长度/随机性
    # max_output_tokens=4000,
    # temperature=0.2,
)

raw = resp.choices[0].message.content  # SDK 直接给你聚合后的文本输出
print("=== Model raw output ===")
print(raw)

# 5) 尝试解析 JSON
def parse_json_or_raise(s: str):
    return json.loads(s)

try:
    data = parse_json_or_raise(raw)
except json.JSONDecodeError:
    # 6) 解析失败：二次调用让模型“只修复 JSON 格式”，不改内容
    fix_prompt = """你将收到一段“疑似 JSON”的文本。
只做一件事：把它修复成“严格可解析的 JSON”。
禁止：新增字段、删改字段语义、改写任何 dialogue 文本内容。
只输出修复后的 JSON 本体（以 { 开始，以 } 结束）。"""

    resp2 = client.chat.completions.create(
        model="qwen3.5-397b-a17b",
        messages=[
            {"role": "system", "content": fix_prompt},
            {"role": "user", "content": raw},
        ],
    )
    fixed = resp2.choices[0].message.content
    print("=== Fixed JSON ===")
    print(fixed)
    data = parse_json_or_raise(fixed)

# 7) 本地使用 data（示例：保存为 json 文件）
with open("第一章.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Saved to 第一章.json")

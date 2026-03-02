import json
import re

# 模拟 utils_data.py 的逻辑
def check_ratio(text):
    if not text: return 0.0
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r'[a-zA-Z]', text))
    total = cn + en
    if total == 0: return 0.0
    return cn / total

# 读取你的验证集
with open("data_proc/pref_val.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        # 拼接完整文本
        full_text = "\n".join([item.get("prompt", ""), item.get("chosen", ""), item.get("rejected", "")])
        ratio = check_ratio(full_text)
        
        # 打印前 5 条所谓的“英文”数据（假设 prompt 是英文开头）
        if re.match(r'^[a-zA-Z]', item.get("prompt", "")):
            print(f"Line {i}: Prompt starts with English.")
            print(f"  - Prompt: {item.get('prompt')[:30]}...")
            print(f"  - Chosen: {item.get('chosen')[:30]}...")
            print(f"  - Calculated Ratio: {ratio:.2f}")
            print(f"  - Category: {'CN' if ratio >= 0.3 else 'EN'}")
            print("-" * 30)
            if i > 10: break
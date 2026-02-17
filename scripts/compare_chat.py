#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Arena: Base vs SFT vs DPO
功能：实时对比三个阶段模型的回答效果。
特点：自动查找最新权重、内存中动态合并 DPO 依赖。
"""

import torch
import sys
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === 配置区域 ===
# 1. 基座模型路径 (请确保正确)
BASE_MODEL_PATH = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# 2. SFT 和 DPO 的输出根目录
SFT_ROOT = "outputs/sft"
DPO_ROOT = "outputs/align"
# =================

def get_latest_adapter(base_dir: str) -> str:
    """自动寻找 base_dir/times 下最新的时间戳目录"""
    path = Path(base_dir)
    times_path = path / "times"
    
    if not times_path.exists():
        return str(path)
    
    subdirs = [d for d in times_path.iterdir() if d.is_dir()]
    if not subdirs:
        return str(path)
        
    latest = sorted(subdirs, key=lambda x: x.name, reverse=True)[0]
    return str(latest)

def load_models():
    print(f"\n[1/4] 正在加载基座模型...")
    print(f"      路径: {BASE_MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ 基座模型加载失败: {e}")
        return None, None

    # 自动查找最新路径
    sft_path = get_latest_adapter(SFT_ROOT)
    dpo_path = get_latest_adapter(DPO_ROOT)

    print(f"[2/4] 加载 SFT Adapter (用于微调效果)...")
    print(f"      路径: {sft_path}")
    try:
        # 加载 SFT 适配器，命名为 'sft'
        model = PeftModel.from_pretrained(base_model, sft_path, adapter_name="sft")
    except Exception as e:
        print(f"❌ SFT 加载失败: {e}")
        return None, None

    print(f"[3/4] 加载 DPO Adapter (用于对齐效果)...")
    print(f"      路径: {dpo_path}")
    try:
        # 加载 DPO 适配器，命名为 'dpo'
        model.load_adapter(dpo_path, adapter_name="dpo")
    except Exception as e:
        print(f"❌ DPO 加载失败: {e}")
        print("      (可能你还没跑完 DPO，或者路径不对)")
        return None, None
        
    return model, tokenizer

def generate_response(model, tokenizer, prompt, mode):
    """
    mode: 'base', 'sft', 'dpo'
    """
    messages = [
        {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    try:
        # === 核心切换逻辑 ===
        if mode == "base":
            # 1. 基座: 禁用所有适配器
            with model.disable_adapter():
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)

        elif mode == "sft":
            # 2. SFT: 启用 sft 适配器
            model.set_adapter("sft")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)

        elif mode == "dpo":
            # 3. DPO: 需要 SFT 的底子
            # 步骤 A: 激活 SFT 并合并进基座 (Base 变成 Base+SFT)
            model.set_adapter("sft")
            model.merge_adapter()
            
            # 步骤 B: 激活 DPO (现在是 Base+SFT+DPO)
            model.set_adapter("dpo")
            
            # 步骤 C: 生成
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
            
            # 步骤 D: 恢复现场 (非常重要!)
            # 必须先切回 sft 才能解合并
            model.set_adapter("sft")
            model.unmerge_adapter() 
            
    except Exception as e:
        # 异常恢复，防止影响下一轮
        try:
            model.set_adapter("sft")
            model.unmerge_adapter()
        except: pass
        return f"[Error] {str(e)}"

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def main():
    print("="*60)
    print("⚔️  LLM 竞技场：Base vs SFT vs DPO  ⚔️")
    print("="*60)
    
    model, tokenizer = load_models()
    if not model: return

    print("\n✅ 环境就绪！输入 'exit' 退出。")
    
    while True:
        try:
            query = input("\n🎤 请输入问题: ").strip()
        except EOFError: break
        
        if not query: continue
        if query.lower() in ["exit", "quit"]: break

        print("-" * 60)
        
        # 依次生成三个模型的回答
        for name, label in [("base", "🔵 Base (基座)"), ("sft", "🟢 SFT (微调)"), ("dpo", "🟣 DPO (对齐)")]:
            print(f"{label} 思考中...", end="", flush=True)
            res = generate_response(model, tokenizer, query, name)
            print(f"\r{label}:\n{res}\n")
            print("-" * 30)

if __name__ == "__main__":
    main()
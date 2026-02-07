#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Arena: Compare Base vs SFT vs DPO in real-time.
(Fixed: Use context manager for Base model to prevent state lock-up)
"""

import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === 配置区域 ===
PATHS = {
    # 1. 基座模型
    "base": "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    
    # 2. SFT Adapter
    "sft": "outputs/sft",
    
    # 3. DPO Adapter
    "dpo": "outputs/align/times/20260206_214733"
}
# ==========================================

def load_models():
    print(f"\n[1/4] 正在加载基座模型: {PATHS['base']} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(PATHS['base'], trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            PATHS['base'],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ 加载基座模型失败: {e}")
        return None, None

    print(f"[2/4] 正在加载 SFT Adapter: {PATHS['sft']} ...")
    try:
        # 加载第一个 Adapter，命名为 sft
        model = PeftModel.from_pretrained(base_model, PATHS['sft'], adapter_name="sft")
    except Exception as e:
        print(f"❌ Error loading SFT: {e}")
        return None, None

    print(f"[3/4] 正在加载 DPO Adapter: {PATHS['dpo']} ...")
    try:
        # 加载第二个 Adapter，命名为 dpo
        model.load_adapter(PATHS['dpo'], adapter_name="dpo")
    except Exception as e:
        print(f"❌ Error loading DPO: {e}")
        return None, None
        
    return model, tokenizer

def generate_response(model, tokenizer, prompt, adapter_name):
    """
    使用上下文管理器来处理 Base 模型，确保状态自动恢复
    """
    messages = [
        {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 核心修复逻辑
    try:
        if adapter_name == "base":
            # 【重要】使用上下文管理器暂时禁用 Adapter
            # 跑完这行代码后，Adapter 会自动重新开启，不会导致后续报错
            with model.disable_adapter():
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
        else:
            # 切换到指定的 Adapter
            model.set_adapter(adapter_name)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
    except Exception as e:
        return f"[系统错误] 生成失败: {str(e)}"

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    print("="*60)
    print("⚔️  LLM 竞技场：Base vs SFT vs DPO  ⚔️")
    print("="*60)
    
    model, tokenizer = load_models()
    if not model:
        print("❌ 模型加载失败，请检查路径。")
        return

    print("\n✅ 模型加载完毕！")
    print("输入 'exit' 退出。\n")

    while True:
        try:
            query = input("\n🎤 请输入测试问题: ").strip()
        except EOFError:
            break
        
        if not query: continue
        if query.lower() in ["exit", "quit"]: break

        print("-" * 60)
        
        # 1. Base
        print("🔵 [Base 基座] 思考中...", end="", flush=True)
        res_base = generate_response(model, tokenizer, query, "base")
        print(f"\r🔵 [Base 基座]:\n{res_base}\n")
        print("-" * 30)

        # 2. SFT
        print("🟢 [SFT 微调] 思考中...", end="", flush=True)
        res_sft = generate_response(model, tokenizer, query, "sft")
        print(f"\r🟢 [SFT 微调]:\n{res_sft}\n")
        print("-" * 30)

        # 3. DPO
        print("🟣 [DPO 对齐] 思考中...", end="", flush=True)
        res_dpo = generate_response(model, tokenizer, query, "dpo")
        print(f"\r🟣 [DPO 对齐]:\n{res_dpo}")
        print("-" * 60)

if __name__ == "__main__":
    main()
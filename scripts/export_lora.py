#!/usr/bin/env python
"""
Export and merge LoRA weights script.
Usage:
    python -m scripts.export_lora \
        --model_name_or_path <base_model> \
        --adapter_path <lora_path> \
        --output_path <merged_path>
"""
import argparse
import logging
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Export LoRA adapters or merged weights")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="基座模型路径或名称 (Base Model)")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA 权重目录 (SFT Checkpoint)")
    parser.add_argument("--output_path", type=str, required=True, help="合并后模型的保存路径")
    parser.add_argument("--export_format", type=str, default="saved_model", help="导出格式 (默认 saved_model)")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto, cpu, cuda)")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    LOGGER.info(f"Loading base model from: {args.model_name_or_path}")
    
    # 加载基座模型
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16, # 建议使用 fp16 以节省显存
            device_map=args.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        LOGGER.error(f"Failed to load base model: {e}")
        raise

    LOGGER.info(f"Loading tokenizer from: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True
    )

    LOGGER.info(f"Loading LoRA adapter from: {args.adapter_path}")
    # 加载 LoRA
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    LOGGER.info("Merging weights (merge_and_unload)...")
    # 核心步骤：合并权重
    model = model.merge_and_unload()

    LOGGER.info(f"Saving merged model to: {args.output_path}")
    # 保存合并后的模型
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    LOGGER.info("Export completed successfully!")

if __name__ == "__main__":
    main()
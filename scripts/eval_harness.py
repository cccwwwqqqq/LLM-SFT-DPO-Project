# # #!/usr/bin/env python
# # """Wrapper for lm-eval-harness with dry-run support."""

# # from __future__ import annotations

# # import argparse
# # import csv
# # import json
# # import logging
# # from pathlib import Path
# # import os
# # from typing import Any, Dict, Mapping, Optional, Sequence
# # from scripts.utils_data import get_latest_adapter
# # import yaml

# # LOGGER = logging.getLogger(__name__)


# # DEFAULT_CONFIG: Dict[str, Any] = {
# #     "general": {
# #         "dry_run": True,
# #         "seed": 42,
# #         "output_json": "outputs/eval/results.json",
# #         "output_csv": "outputs/eval/results.csv",
# #         "log_dir": "outputs/eval/logs",
# #     },
# #     "model": {
# #         "base_model": "Qwen/Qwen2.5-7B-Instruct",
# #         "peft_adapter": "outputs/align",
# #         "trust_remote_code": True,
# #     },
# #     "tasks": ["cmmlu", "ceval-valid-lite", "hellaswag", "winogrande"],
# #     "metrics": {
# #         "save_raw": True,
# #         "summarize": True,
# #     },
# #     "lm_eval": {
# #         "limit": None,
# #         "batch_size": 4,
# #         "use_cache": True,
# #     },
# # }


# # def parse_model_args(args_str):
# #     """辅助函数：解析 model_args 字符串并自动更新 peft 路径"""
# #     if not args_str:
# #         return ""
    
# #     # 将 "key=value,key2=val2" 拆分成字典
# #     pairs = [part.split("=", 1) for part in args_str.split(",") if "=" in part]
# #     args_dict = {k.strip(): v.strip() for k, v in pairs}
    
# #     # === [新增 2] 自动查找最新 adapter ===
# #     if "peft" in args_dict:
# #         original_path = args_dict["peft"]
# #         # 尝试自动查找 (如果路径下有 times 子目录)
# #         new_path = get_latest_adapter(original_path)
        
# #         if new_path != original_path:
# #             print(f"[Auto-Eval] Detected latest adapter: {new_path}")
# #             args_dict["peft"] = new_path
# #     # ===================================
    
# #     # 重新组装回字符串
# #     return ",".join([f"{k}={v}" for k, v in args_dict.items()])

# # def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
# #     parser = argparse.ArgumentParser(description="lm-eval-harness wrapper")
# #     parser.add_argument("--config", type=str, default="configs/eval.yaml")
# #     parser.add_argument("--dry-run", type=str, default=None)
# #     parser.add_argument("--log-level", type=str, default="INFO")
# #     return parser.parse_args(argv)


# # def load_config(path: str) -> Dict[str, Any]:
# #     config = json.loads(json.dumps(DEFAULT_CONFIG))
# #     cfg_path = Path(path)
# #     if cfg_path.exists():
# #         with cfg_path.open("r", encoding="utf-8") as f:
# #             update = yaml.safe_load(f) or {}
# #         config = deep_update(config, update)
# #     return config


# # def deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
# #     result = dict(base)
# #     for key, value in update.items():
# #         if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
# #             result[key] = deep_update(result[key], value)  # type: ignore[arg-type]
# #         else:
# #             result[key] = value
# #     return result


# # def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
# #     if args.dry_run is not None:
# #         config.setdefault("general", {})["dry_run"] = str(args.dry_run).lower() not in {"false", "0", "no"}
# #     return config


# # def dry_run_summary(config: Mapping[str, Any]) -> None:
# #     LOGGER.info("Dry-run 模式：不会实际调用 lm-eval-harness")
# #     LOGGER.info("计划评测任务：%s", ", ".join(config.get("tasks", [])))
# #     LOGGER.info("输出 JSON：%s", config["general"].get("output_json"))
# #     LOGGER.info("输出 CSV：%s", config["general"].get("output_csv"))
# #     # 预期结果 JSON schema 示例：
# #     # {
# #     #   "results": {
# #     #     "cmmlu": {
# #     #       "acc": 0.62,
# #     #       "acc_stderr": 0.02
# #     #     },
# #     #     "hellaswag": {"acc": 0.75, "acc_stderr": 0.01}
# #     #   },
# #     #   "config": {...},
# #     #   "versions": {...}
# #     # }


# # def run_evaluation(config: Mapping[str, Any]) -> None:
# #     try:  # pragma: no cover
# #         from lm_eval import evaluator
# #     except ImportError as exc:  # pragma: no cover
# #         raise RuntimeError("需要在远程环境安装 lm-eval-harness") from exc

# #     # 环境准备：HF 缓存与数据集自定义代码信任
# #     hf_home = os.environ.get("HF_HOME") or str(Path("/root/autodl-tmp/hf_cache").absolute())
# #     os.environ["HF_HOME"] = hf_home
# #     Path(hf_home).mkdir(parents=True, exist_ok=True)
# #     # 允许 datasets 运行远程自定义代码（如 cmmlu 等）
# #     os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

# #     model_cfg = config["model"]
# #     # 使用 lm-eval 的注册名 "hf"，避免直接依赖具体模块路径（不同版本有变动）
# #     hf_model_name = "hf"
# #     # 以字符串形式传入 model_args，兼容 lm-eval 的解析器
# #     arg_items = [
# #         f"pretrained={model_cfg['base_model']}",
# #         f"trust_remote_code={str(model_cfg.get('trust_remote_code', False))}",
# #     ]
# #     peft_adapter = model_cfg.get("peft_adapter")
# #     if peft_adapter:
# #         arg_items.append(f"peft={peft_adapter}")
# #     model_args_str = ",".join(arg_items)
# #     eval_tasks = config.get("tasks", [])
# #     # 处理缓存文件路径（lm-eval 0.4.9.1 期望字符串路径或 None）
# #     use_cache_cfg = config["lm_eval"].get("use_cache", True)
# #     if use_cache_cfg:
# #         cache_dir = Path(config["general"].get("log_dir", "outputs/eval/logs"))
# #         cache_dir.mkdir(parents=True, exist_ok=True)
# #         use_cache_value = str((cache_dir / "lm_cache.db").absolute())
# #     else:
# #         use_cache_value = None

# #     eval_kwargs = {
# #         "tasks": eval_tasks,
# #         "model": hf_model_name,
# #         "model_args": model_args_str,
# #         "bootstrap_iters": 100,
# #         "limit": config["lm_eval"].get("limit"),
# #         # 0.4.9.1 使用 use_cache 参数（字符串路径或 None）
# #         "use_cache": use_cache_value,
# #         "batch_size": config["lm_eval"].get("batch_size", 4),
# #         "apply_chat_template": True,
# #         "torch_random_seed": config["general"].get("seed", 42),
# #         "numpy_random_seed": config["general"].get("seed", 42),
# #         "random_seed": config["general"].get("seed", 42),
# #         "confirm_run_unsafe_code": True,
# #     }
# #     results = evaluator.simple_evaluate(**eval_kwargs)

# #     output_json = Path(config["general"]["output_json"])
# #     output_json.parent.mkdir(parents=True, exist_ok=True)
# #     output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# #     if config["metrics"].get("summarize", True):
# #         flat_rows = []
# #         for task, metrics in results.get("results", {}).items():
# #             for metric_name, value in metrics.items():
# #                 if isinstance(value, Mapping):
# #                     for sub_key, sub_val in value.items():
# #                         flat_rows.append((task, f"{metric_name}/{sub_key}", sub_val))
# #                 else:
# #                     flat_rows.append((task, metric_name, value))
# #         output_csv = Path(config["general"]["output_csv"])
# #         with output_csv.open("w", newline="", encoding="utf-8") as f:
# #             writer = csv.writer(f)
# #             writer.writerow(["task", "metric", "value"])
# #             writer.writerows(flat_rows)


# # def main(argv: Optional[Sequence[str]] = None) -> int:
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--model", type=str, default="hf")
# #     parser.add_argument("--model_args", type=str, required=True)
# #     parser.add_argument("--tasks", type=str, default="ceval-valid")
    
# #     args = parse_args(argv)
    
# #     args.model_args = parse_model_args(args.model_args)
# #     logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
# #     config = load_config(args.config)
# #     config = apply_cli_overrides(config, args)


# #     results = evaluator.simple_evaluate(
# #         model=args.model,
# #         model_args=args.model_args,
# #         tasks=args.tasks.split(","),
# #         # ...
# #     )
    
# #     if config["general"].get("dry_run", True):
# #         dry_run_summary(config)
# #         return 0

# #     run_evaluation(config)
# #     LOGGER.info("评测完成")
# #     return 0


# # if __name__ == "__main__":  # pragma: no cover
# #     raise SystemExit(main())
# #!/usr/bin/env python
# """Wrapper for lm-eval-harness with dry-run support and auto-latest-adapter."""

# from __future__ import annotations

# import argparse
# import csv
# import json
# import logging
# import os
# import sys
# from pathlib import Path
# from typing import Any, Dict, Mapping, Optional, Sequence
# import yaml

# # 确保导入这个函数
# from scripts.utils_data import get_latest_adapter

# LOGGER = logging.getLogger(__name__)

# DEFAULT_CONFIG: Dict[str, Any] = {
#     "general": {
#         "dry_run": True,
#         "seed": 42,
#         "output_json": "outputs/eval/results.json",
#         "output_csv": "outputs/eval/results.csv",
#         "log_dir": "outputs/eval/logs",
#     },
#     "model": {
#         "base_model": "Qwen/Qwen2.5-7B-Instruct",
#         # 默认指向 align 根目录，会自动查找最新的 times/ 子目录
#         "peft_adapter": "outputs/align",
#         "trust_remote_code": True,
#     },
#     "tasks": ["cmmlu", "ceval-valid-lite"],
#     "metrics": {
#         "save_raw": True,
#         "summarize": True,
#     },
#     "lm_eval": {
#         "limit": None,
#         "batch_size": 4,
#         "use_cache": True,
#     },
# }

# def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="lm-eval-harness wrapper")
#     parser.add_argument("--config", type=str, default="configs/eval.yaml")
#     parser.add_argument("--dry-run", type=str, default=None)
#     parser.add_argument("--log-level", type=str, default="INFO")
    
#     # 允许命令行覆盖 model 和 tasks
#     parser.add_argument("--model-args", type=str, help="Override model args (e.g. 'pretrained=X,peft=Y')")
#     parser.add_argument("--tasks", type=str, help="Comma separated tasks")
    
#     return parser.parse_args(argv)

# def load_config(path: str) -> Dict[str, Any]:
#     config = json.loads(json.dumps(DEFAULT_CONFIG))
#     cfg_path = Path(path)
#     if cfg_path.exists():
#         with cfg_path.open("r", encoding="utf-8") as f:
#             update = yaml.safe_load(f) or {}
#         config = deep_update(config, update)
#     return config

# def deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
#     result = dict(base)
#     for key, value in update.items():
#         if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
#             result[key] = deep_update(result[key], value)
#         else:
#             result[key] = value
#     return result

# def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
#     if args.dry_run is not None:
#         config.setdefault("general", {})["dry_run"] = str(args.dry_run).lower() not in {"false", "0", "no"}
    
#     if args.tasks:
#         config["tasks"] = args.tasks.split(",")
        
#     return config

# def dry_run_summary(config: Mapping[str, Any]) -> None:
#     LOGGER.info("Dry-run 模式：不会实际调用 lm-eval-harness")
#     LOGGER.info("计划评测任务：%s", ", ".join(config.get("tasks", [])))
#     LOGGER.info("基座模型：%s", config["model"].get("base_model"))
#     LOGGER.info("Adapter (原始配置)：%s", config["model"].get("peft_adapter"))
#     # 在 Dry-run 里也展示一下自动查找到的结果
#     if config["model"].get("peft_adapter"):
#         found = get_latest_adapter(config["model"]["peft_adapter"])
#         LOGGER.info("Adapter (自动定位)：%s", found)

# def run_evaluation(config: Mapping[str, Any], cli_model_args: Optional[str] = None) -> None:
#     try:
#         from lm_eval import evaluator
#     except ImportError as exc:
#         raise RuntimeError("需要在远程环境安装 lm-eval-harness") from exc

#     # 1. 环境准备
#     hf_home = os.environ.get("HF_HOME") or str(Path("/root/autodl-tmp/hf_cache").absolute())
#     os.environ["HF_HOME"] = hf_home
#     Path(hf_home).mkdir(parents=True, exist_ok=True)
#     os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

#     # 2. 构建 model_args
#     # 优先使用命令行传入的 model_args，如果没有，则从 config["model"] 构建
#     if cli_model_args:
#         # 如果命令行传了 model_args="pretrained=xxx,peft=outputs/align"，需要解析并自动查找
#         final_model_args = parse_and_update_model_args(cli_model_args)
#     else:
#         # 从配置文件构建
#         model_cfg = config["model"]
#         base_model = model_cfg["base_model"]
#         peft_path = model_cfg.get("peft_adapter")
#         trust_remote = str(model_cfg.get("trust_remote_code", False))
        
#         arg_list = [f"pretrained={base_model}", f"trust_remote_code={trust_remote}"]
        
#         if peft_adapter:
#             # === 关键点：自动查找最新 adapter ===
#             real_peft_path = get_latest_adapter(peft_path)
#             if real_peft_path != peft_path:
#                 LOGGER.info(f"[Auto-Eval] 将 Adapter 路径自动更新为最新: {real_peft_path}")
#             arg_list.append(f"peft={real_peft_path}")
            
#         final_model_args = ",".join(arg_list)

#     LOGGER.info(f"最终使用的 model_args: {final_model_args}")

#     # 3. 准备参数并执行
#     eval_tasks = config.get("tasks", [])
    
#     # 缓存处理
#     use_cache_cfg = config["lm_eval"].get("use_cache", True)
#     if use_cache_cfg:
#         cache_dir = Path(config["general"].get("log_dir", "outputs/eval/logs"))
#         cache_dir.mkdir(parents=True, exist_ok=True)
#         # lm-eval 0.4.x 可能需要 db 路径
#         use_cache_value = str((cache_dir / "lm_cache.db").absolute())
#     else:
#         use_cache_value = None

#     eval_kwargs = {
#         "model": "hf",
#         "model_args": final_model_args,
#         "tasks": eval_tasks,
#         "batch_size": config["lm_eval"].get("batch_size", 4),
#         "limit": config["lm_eval"].get("limit"),
#         "use_cache": use_cache_value,
#         "apply_chat_template": True,
#         "torch_random_seed": config["general"].get("seed", 42),
#         "numpy_random_seed": config["general"].get("seed", 42),
#         "random_seed": config["general"].get("seed", 42),
#         "confirm_run_unsafe_code": True,
#     }

#     # 执行评测
#     LOGGER.info("开始执行 lm-eval...")
#     results = evaluator.simple_evaluate(**eval_kwargs)

#     # 4. 保存结果
#     output_json = Path(config["general"]["output_json"])
#     output_json.parent.mkdir(parents=True, exist_ok=True)
#     # 处理不可序列化的对象
#     output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
#     LOGGER.info(f"结果已保存至: {output_json}")

#     # 保存 CSV 摘要
#     if config["metrics"].get("summarize", True) and "results" in results:
#         flat_rows = []
#         for task, metrics in results["results"].items():
#             # 过滤掉非 dict 的元数据
#             if not isinstance(metrics, dict): continue
#             for k, v in metrics.items():
#                 # 只保存数值型指标
#                 if isinstance(v, (int, float, str)):
#                     flat_rows.append((task, k, v))
        
#         output_csv = Path(config["general"]["output_csv"])
#         with output_csv.open("w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow(["Task", "Metric", "Value"])
#             writer.writerows(flat_rows)
#         LOGGER.info(f"CSV 摘要已保存至: {output_csv}")

# def parse_and_update_model_args(args_str: str) -> str:
#     """解析命令行传入的 model_args 字符串并自动更新 peft 路径"""
#     if not args_str:
#         return ""
    
#     # 简单的解析逻辑
#     parts = args_str.split(",")
#     new_parts = []
#     for part in parts:
#         if part.strip().startswith("peft="):
#             key, val = part.split("=", 1)
#             # 自动查找
#             new_val = get_latest_adapter(val.strip())
#             if new_val != val:
#                  LOGGER.info(f"[Auto-Eval] 命令行 peft 参数自动更新: {new_val}")
#             new_parts.append(f"{key}={new_val}")
#         else:
#             new_parts.append(part)
    
#     return ",".join(new_parts)

# def main(argv: Optional[Sequence[str]] = None) -> int:
#     args = parse_args(argv)
    
#     # 设置日志
#     logging.basicConfig(
#         level=getattr(logging, args.log_level.upper(), logging.INFO), 
#         format="%(asctime)s %(levelname)s %(message)s"
#     )
    
#     # 加载配置
#     config = load_config(args.config)
#     config = apply_cli_overrides(config, args)
    
#     # Dry Run 检查
#     if config["general"].get("dry_run", True):
#         dry_run_summary(config)
#         return 0

#     # 运行评测
#     # 注意：我们把命令行的 model_args 单独传进去处理
#     run_evaluation(config, cli_model_args=args.model_args)
    
#     LOGGER.info("评测流程结束")
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())

#!/usr/bin/env python
"""Wrapper for lm-eval-harness with support for DPO (Base+SFT+DPO) evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
import yaml
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 尝试导入 HFLM 用于手动包装模型
try:
    from lm_eval.models.huggingface import HFLM
except ImportError:
    HFLM = None

from lm_eval import evaluator
from scripts.utils_data import get_latest_adapter

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "dry_run": True,
        "seed": 42,
        "output_json": "outputs/eval/results.json",
        "output_csv": "outputs/eval/results.csv",
        "log_dir": "outputs/eval/logs",
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "peft_adapter": "outputs/align", # DPO adapter
        "sft_adapter": None,             # Optional: SFT adapter for merging
        "trust_remote_code": True,
    },
    "tasks": ["ceval-valid-lite"],
    "metrics": {
        "save_raw": True,
        "summarize": True,
    },
    "lm_eval": {
        "limit": None,
        "batch_size": 4,
        "use_cache": True,
    },
}

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lm-eval-harness wrapper")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--dry-run", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)

def load_config(path: str) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            update = yaml.safe_load(f) or {}
        config = deep_update(config, update)
    return config

def deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result

def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.dry_run is not None:
        config.setdefault("general", {})["dry_run"] = str(args.dry_run).lower() not in {"false", "0", "no"}
    return config

def dry_run_summary(config: Mapping[str, Any]) -> None:
    LOGGER.info("Dry-run 模式：不会实际调用 lm-eval-harness")
    LOGGER.info("任务: %s", config.get("tasks"))
    LOGGER.info("SFT Adapter: %s", config["model"].get("sft_adapter"))
    LOGGER.info("DPO Adapter: %s", config["model"].get("peft_adapter"))

def run_evaluation(config: Mapping[str, Any]) -> None:
    if HFLM is None:
        raise RuntimeError("无法导入 lm_eval.models.huggingface.HFLM，请更新 lm-eval")

    # 1. 环境准备
    hf_home = os.environ.get("HF_HOME") or str(Path("/root/autodl-tmp/hf_cache").absolute())
    os.environ["HF_HOME"] = hf_home
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

    # 2. 决定加载模式
    model_cfg = config["model"]
    base_path = model_cfg["base_model"]
    sft_path = model_cfg.get("sft_adapter")
    dpo_path = model_cfg.get("peft_adapter") # 复用 peft_adapter 字段作为最终的 adapter (DPO)

    lm_obj = None

    # === 模式 A: SFT + DPO 动态合并评估 ===
    if sft_path and dpo_path:
        LOGGER.info("检测到 SFT + DPO 配置，正在执行动态合并加载...")
        
        # 自动查找最新路径
        sft_path = get_latest_adapter(sft_path)
        dpo_path = get_latest_adapter(dpo_path)
        LOGGER.info(f"Base: {base_path}")
        LOGGER.info(f"SFT (Merge): {sft_path}")
        LOGGER.info(f"DPO (Load):  {dpo_path}")

        # 手动加载模型
        try:
            # A. 加载 Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
            
            # B. 加载 Base Model
            model = AutoModelForCausalLM.from_pretrained(
                base_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            
            # C. 加载 SFT 并合并
            model = PeftModel.from_pretrained(model, sft_path)
            model = model.merge_and_unload()
            LOGGER.info("SFT Adapter 已合并到基座")

            # D. 加载 DPO
            model = PeftModel.from_pretrained(model, dpo_path)
            LOGGER.info("DPO Adapter 已加载")

            # E. 包装为 HFLM 给 lm-eval 使用
            # 注意: pretrained 参数可以接受已加载的模型实例
            lm_obj = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=config["lm_eval"].get("batch_size", 4),
                trust_remote_code=True
            )

        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise e

    # === 模式 B: 普通评估 (Base 或 Base+SFT) ===
    else:
        LOGGER.info("执行标准评估模式 (lm-eval原生加载)...")
        # 构建 model_args 字符串
        arg_items = [
            f"pretrained={base_path}",
            f"trust_remote_code={str(model_cfg.get('trust_remote_code', False))}",
        ]
        
        if dpo_path: # 这里 dpo_path 实际上就是 peft_adapter
            real_path = get_latest_adapter(dpo_path)
            if real_path != dpo_path:
                LOGGER.info(f"自动更新 Adapter 路径: {real_path}")
            arg_items.append(f"peft={real_path}")
            
        model_args_str = ",".join(arg_items)
        
        # 使用字符串参数初始化
        lm_obj = "hf" # 告诉 simple_evaluate 使用 hf 后端
        # 注意：如果是字符串模式，args 传给 simple_evaluate 的 model_args 参数
        # 但为了代码统一，我们这里不做特殊处理，直接在下面传参时区分
        
    # 3. 执行评估
    eval_kwargs = {
        "model": lm_obj, # 可是 HFLM 对象，也可以是 "hf" 字符串
        "tasks": config.get("tasks", []),
        "batch_size": config["lm_eval"].get("batch_size", 4),
        "limit": config["lm_eval"].get("limit"),
        "apply_chat_template": True,
        "confirm_run_unsafe_code": True,
    }

    # 如果是模式 B (字符串模式)，需要传入 model_args
    if isinstance(lm_obj, str):
        eval_kwargs["model_args"] = model_args_str

    LOGGER.info("开始运行 lm-eval...")
    results = evaluator.simple_evaluate(**eval_kwargs)

    # 4. 保存结果
    output_json = Path(config["general"]["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    
    # 保存 CSV
    if config["metrics"].get("summarize", True) and "results" in results:
        flat_rows = []
        for task, metrics in results["results"].items():
            if not isinstance(metrics, dict): continue
            for k, v in metrics.items():
                if isinstance(v, (int, float, str)):
                    flat_rows.append((task, k, v))
        
        output_csv = Path(config["general"]["output_csv"])
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Task", "Metric", "Value"])
            writer.writerows(flat_rows)
        LOGGER.info(f"结果已保存: {output_csv}")

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    if config["general"].get("dry_run", True):
        dry_run_summary(config)
        return 0

    run_evaluation(config)
    return 0

if __name__ == "__main__":
    sys.exit(main())
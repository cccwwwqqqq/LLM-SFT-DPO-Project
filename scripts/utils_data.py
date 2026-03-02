# """Shared data utilities for the LLM fine-tuning pipeline.

# This module contains reusable helpers for text normalization, language ratio
# estimation, n-gram repetition detection, SHA256 deduplication and curriculum
# aware sampling. All functions are pure and safe to import in dry-run mode.

# Example
# -------
# >>> bucket = LengthBucket(name="short", min_tokens=0, max_tokens=256)
# >>> sampler = MixedBucketSampler(length_buckets=[bucket])
# >>> item = SamplingItem(
# ...     identifier="demo-1",
# ...     source="OASST1",
# ...     text_length=128,
# ...     chinese_ratio=0.9,
# ...     payload={"messages": []},
# ... )
# >>> plan = sampler.plan(total_samples=1, available_items=[item])
# >>> len(plan.selected)
# 1

# The dry-run unit tests import this module to ensure type integrity, but do not
# execute any heavy operations.
# """

# from __future__ import annotations

# import hashlib
# import logging
# import math
# import random
# import re
# from dataclasses import dataclass, field
# from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# CONTROL_CHAR_PATTERN = re.compile(r"[\u0000-\u001f\u007f]")
# WHITESPACE_PATTERN = re.compile(r"\s+")
# HAN_PATTERN = re.compile(r"[\u4e00-\u9fff]")


# def normalize_text(text: str) -> str:
#     """Normalize whitespace and strip ASCII control characters."""

#     if not text:
#         return ""
#     normalized = CONTROL_CHAR_PATTERN.sub("", text.replace("\u3000", " "))
#     normalized = WHITESPACE_PATTERN.sub(" ", normalized)
#     return normalized.strip()


# def estimate_chinese_ratio(text: str) -> float:
#     """Estimate Chinese character coverage in *text* (Optimized).
    
#     Uses regex findall for bulk matching instead of character-level iteration.
#     """
#     if not text:
#         return 0.0
    
#     # 优化点 1: 直接在 C 语言层面批量统计中文字符数，而不是写 Python for 循环
#     # HAN_PATTERN = re.compile(r"[\u4e00-\u9fff]")
#     chinese_count = len(HAN_PATTERN.findall(text))
    
#     # 优化点 2: 批量统计英文字母
#     # 这里用正则 [a-zA-Z] 近似原逻辑的 isalpha()，速度极快
#     alpha_count = len(re.findall(r'[a-zA-Z]', text))
    
#     total = chinese_count + alpha_count
#     if total == 0:
#         return 0.0
#     return chinese_count / total


# def estimate_token_length(text: str) -> int:
#     """Rough token length heuristic (4 characters per token)."""

#     if not text:
#         return 0
#     return max(1, math.ceil(len(text) / 4))


# def hash_for_text(text: str) -> str:
#     """Return a SHA256 hash for *text* encoded as UTF-8."""

#     return hashlib.sha256(text.encode("utf-8")).hexdigest()


# def merge_messages(messages: Sequence[Mapping[str, str]]) -> str:
#     """Concatenate message contents into a single newline-delimited string."""

#     return "\n".join(msg.get("content", "") for msg in messages if isinstance(msg, Mapping))


# def dedupe_by_hash(records: Iterable[Mapping[str, object]], key: str) -> List[Mapping[str, object]]:
#     """Remove duplicate entries using the SHA256 hash stored in *key*."""

#     seen: set[str] = set()
#     result: List[Mapping[str, object]] = []
#     for record in records:
#         digest = record.get(key)
#         if not isinstance(digest, str):
#             continue
#         if digest in seen:
#             continue
#         seen.add(digest)
#         result.append(record)
#     return result


# def compute_ngram_repetition(text: str, n: int = 4) -> float:
#     """Compute a simple n-gram repetition ratio (auto-adapts to Chinese)."""
#     if not text:
#         return 0.0

#     # 简单判断是否包含中文，不再全量正则搜索
#     # (如果前面已经做了 filter，这里通常可以直接假设逻辑)
#     # 优化：直接按字符切分，不做复杂的 isspace 判断循环，加速处理
#     if len(text) > 10000: # 极长文本截断检查，防止卡死
#         text = text[:10000]
        
#     tokens = list(text) # 直接转列表比列表推导式快
    
#     if len(tokens) < n:
#         return 0.0
        
#     # 使用生成器而非列表推导式，减少内存
#     ngrams_count = len(tokens) - n + 1
#     unique_ngrams = set(tuple(tokens[i : i + n]) for i in range(ngrams_count))
    
#     if ngrams_count == 0:
#         return 0.0
#     return 1.0 - len(unique_ngrams) / ngrams_count


# @dataclass(frozen=True)
# class LengthBucket:
#     """Inclusive token boundaries for curriculum-aware sampling."""

#     name: str
#     min_tokens: int
#     max_tokens: int

#     def contains(self, token_count: int) -> bool:
#         return self.min_tokens <= token_count <= self.max_tokens


# @dataclass
# class CurriculumPhase:
#     """Optional curriculum phase weighting certain buckets or sources."""

#     start: float
#     end: float
#     weights: Mapping[str, float] = field(default_factory=dict)

#     def applies(self, progress: float) -> bool:
#         return self.start <= progress <= self.end


# @dataclass
# class SamplingItem:
#     """Container for sampler metadata."""

#     identifier: str
#     source: str
#     text_length: int
#     chinese_ratio: float
#     payload: Mapping[str, object]

#     def is_chinese(self, threshold: float) -> bool:
#         return self.chinese_ratio >= threshold


# @dataclass
# class SamplingPlan:
#     """Result bundle returned from :class:`MixedBucketSampler`."""

#     selected: List[SamplingItem]
#     rejected: List[SamplingItem]
#     stats: Mapping[str, float]


# class MixedBucketSampler:
#     """Language-aware bucket sampler with curriculum schedule support."""

#     def __init__(
#         self,
#         length_buckets: Sequence[LengthBucket],
#         target_cn_ratio: float = 0.7,
#         cn_threshold: float = 0.3,
#         allow_english_fallback: bool = False,
#         curriculum: Optional[Sequence[CurriculumPhase]] = None,
#         seed: int = 42,
#     ) -> None:
#         self.length_buckets = list(length_buckets)
#         self.target_cn_ratio = target_cn_ratio
#         self.cn_threshold = cn_threshold
#         self.allow_english_fallback = allow_english_fallback
#         self.curriculum = list(curriculum or [])
#         self._rng = random.Random(seed)

#     def plan(
#         self,
#         total_samples: int,
#         available_items: Sequence[SamplingItem],
#         *,
#         progress: float = 1.0,
#         source_weights: Optional[Mapping[str, float]] = None,
#     ) -> SamplingPlan:
#         if total_samples <= 0 or not available_items:
#             return SamplingPlan(selected=[], rejected=list(available_items), stats={"requested": float(total_samples), "selected": 0.0})

#         chinese_pool: List[SamplingItem] = []
#         english_pool: List[SamplingItem] = []
#         for item in available_items:
#             (chinese_pool if item.is_chinese(self.cn_threshold) else english_pool).append(item)

#         desired_cn = int(round(total_samples * self.target_cn_ratio))
#         if not self.allow_english_fallback and len(chinese_pool) < desired_cn:
#             LOGGER.warning(
#                 "可用中文样本不足：当前=%d，目标=%d。采样量将按中文池截断；可考虑启用 allow_english_fallback 或降低 target_cn_ratio",
#                 len(chinese_pool),
#                 desired_cn,
#             )

#         if not self.allow_english_fallback:
#             total_samples = min(total_samples, len(chinese_pool))

#         target_cn = int(round(total_samples * self.target_cn_ratio))
#         if not self.allow_english_fallback:
#             target_cn = min(target_cn, len(chinese_pool))

#         weights = self._merge_weights(source_weights, progress)
#         chinese_selected = self._select_weighted(chinese_pool, target_cn, weights)
#         remaining_slots = total_samples - len(chinese_selected)

#         if self.allow_english_fallback:
#             combined_pool = [item for item in available_items if item not in chinese_selected]
#         else:
#             combined_pool = [item for item in chinese_pool if item not in chinese_selected]

#         rest_selected = self._select_weighted(combined_pool, remaining_slots, weights)

#         selected = chinese_selected + rest_selected
#         rejected = [item for item in available_items if item not in selected]
#         stats = {
#             "requested": float(total_samples),
#             "selected": float(len(selected)),
#             "chinese_selected": float(sum(item.is_chinese(self.cn_threshold) for item in selected)),
#             "english_selected": float(sum(not item.is_chinese(self.cn_threshold) for item in selected)),
#             "avg_length": float(sum(item.text_length for item in selected) / max(1, len(selected))),
#         }
#         return SamplingPlan(selected=selected, rejected=rejected, stats=stats)

#     def _bucket_label(self, item: SamplingItem) -> str:
#         for bucket in self.length_buckets:
#             if bucket.contains(item.text_length):
#                 return bucket.name
#         return "overflow"

#     def _merge_weights(self, provided: Optional[Mapping[str, float]], progress: float) -> Dict[str, float]:
#         weights = dict(provided or {})
#         for phase in self.curriculum:
#             if phase.applies(progress):
#                 for key, value in phase.weights.items():
#                     weights[key] = weights.get(key, 1.0) * float(value)
#                 break
#         return weights

#     def _select_weighted(self, pool: Sequence[SamplingItem], k: int, weights: Mapping[str, float]) -> List[SamplingItem]:
#         if k <= 0 or not pool:
#             return []
#         bucket_groups: MutableMapping[str, MutableMapping[str, List[SamplingItem]]] = {}
#         for item in pool:
#             bucket_groups.setdefault(self._bucket_label(item), {}).setdefault(item.source, []).append(item)
#         for source_map in bucket_groups.values():
#             for items in source_map.values():
#                 self._rng.shuffle(items)

#         bucket_order = list(bucket_groups.keys())
#         selected: List[SamplingItem] = []
#         while bucket_order and len(selected) < k:
#             for bucket in list(bucket_order):
#                 source_map = bucket_groups.get(bucket, {})
#                 if not source_map:
#                     bucket_order.remove(bucket)
#                     continue
#                 for source in list(source_map.keys()):
#                     items = source_map.get(source, [])
#                     if not items:
#                         source_map.pop(source, None)
#                         continue
#                     weight = max(1e-3, weights.get(source, 1.0))
#                     steps = max(1, int(round(weight)))
#                     for _ in range(steps):
#                         if not items or len(selected) >= k:
#                             break
#                         selected.append(items.pop())
#                     if not items:
#                         source_map.pop(source, None)
#                     if len(selected) >= k:
#                         break
#                 if not source_map:
#                     bucket_order.remove(bucket)
#                 if len(selected) >= k:
#                     break
#         return selected


# # Alias exported for consumers
# __all__ = [
#     "LengthBucket",
#     "CurriculumPhase",
#     "SamplingItem",
#     "SamplingPlan",
#     "MixedBucketSampler",
#     "normalize_text",
#     "estimate_chinese_ratio",
#     "estimate_token_length",
#     "hash_for_text",
#     "dedupe_by_hash",
#     "compute_ngram_repetition",
#     "merge_messages",
# ]
# LOGGER = logging.getLogger(__name__)



# """Shared data utilities for the LLM fine-tuning pipeline (Optimized + Debug)."""

# from __future__ import annotations

# import hashlib
# import logging
# import math
# import random
# import re
# import time
# from dataclasses import dataclass, field
# from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# CONTROL_CHAR_PATTERN = re.compile(r"[\u0000-\u001f\u007f]")
# WHITESPACE_PATTERN = re.compile(r"\s+")
# HAN_PATTERN = re.compile(r"[\u4e00-\u9fff]")
# ALPHA_PATTERN = re.compile(r'[a-zA-Z]')


# def normalize_text(text: str) -> str:
#     """Normalize whitespace and strip ASCII control characters."""
#     if not text:
#         return ""
#     normalized = CONTROL_CHAR_PATTERN.sub("", text.replace("\u3000", " "))
#     normalized = WHITESPACE_PATTERN.sub(" ", normalized)
#     return normalized.strip()


# def estimate_chinese_ratio(text: str) -> float:
#     """Estimate Chinese character coverage in *text* (High Performance Regex)."""
#     if not text:
#         return 0.0
#     # 优化：使用 findall 批量在 C 层面计数，避免 Python 循环
#     chinese_count = len(HAN_PATTERN.findall(text))
#     alpha_count = len(ALPHA_PATTERN.findall(text))
#     total = chinese_count + alpha_count
#     if total == 0:
#         return 0.0
#     return chinese_count / total


# def estimate_token_length(text: str) -> int:
#     """Rough token length heuristic (4 characters per token)."""
#     if not text:
#         return 0
#     return max(1, math.ceil(len(text) / 4))


# def hash_for_text(text: str) -> str:
#     """Return a SHA256 hash for *text* encoded as UTF-8."""
#     return hashlib.sha256(text.encode("utf-8")).hexdigest()


# def merge_messages(messages: Sequence[Mapping[str, str]]) -> str:
#     """Concatenate message contents into a single newline-delimited string."""
#     return "\n".join(msg.get("content", "") for msg in messages if isinstance(msg, Mapping))


# def dedupe_by_hash(records: Iterable[Mapping[str, object]], key: str) -> List[Mapping[str, object]]:
#     """Remove duplicate entries using the SHA256 hash stored in *key*."""
#     seen: set[str] = set()
#     result: List[Mapping[str, object]] = []
#     for record in records:
#         digest = record.get(key)
#         if not isinstance(digest, str):
#             continue
#         if digest in seen:
#             continue
#         seen.add(digest)
#         result.append(record)
#     return result


# def compute_ngram_repetition(text: str, n: int = 4) -> float:
#     """Compute a simple n-gram repetition ratio (Optimized)."""
#     if not text:
#         return 0.0
#     # 简单的长度截断保护，防止超长文本卡死
#     if len(text) > 20000:
#         text = text[:20000]
    
#     # 直接按字符切分，速度更快
#     tokens = list(text)
#     if len(tokens) < n:
#         return 0.0
    
#     ngrams_count = len(tokens) - n + 1
#     if ngrams_count <= 0:
#         return 0.0
        
#     unique = set(tuple(tokens[i : i + n]) for i in range(ngrams_count))
#     return 1.0 - len(unique) / ngrams_count


# @dataclass(frozen=True)
# class LengthBucket:
#     """Inclusive token boundaries for curriculum-aware sampling."""
#     name: str
#     min_tokens: int
#     max_tokens: int

#     def contains(self, token_count: int) -> bool:
#         return self.min_tokens <= token_count <= self.max_tokens


# @dataclass
# class CurriculumPhase:
#     """Optional curriculum phase weighting certain buckets or sources."""
#     start: float
#     end: float
#     weights: Mapping[str, float] = field(default_factory=dict)

#     def applies(self, progress: float) -> bool:
#         return self.start <= progress <= self.end


# @dataclass
# class SamplingItem:
#     """Container for sampler metadata."""
#     identifier: str
#     source: str
#     text_length: int
#     chinese_ratio: float
#     payload: Mapping[str, object]

#     def is_chinese(self, threshold: float) -> bool:
#         return self.chinese_ratio >= threshold


# @dataclass
# class SamplingPlan:
#     """Result bundle returned from :class:`MixedBucketSampler`."""
#     selected: List[SamplingItem]
#     rejected: List[SamplingItem]
#     stats: Mapping[str, float]


# class MixedBucketSampler:
#     """Language-aware bucket sampler with curriculum schedule support."""

#     def __init__(
#         self,
#         length_buckets: Sequence[LengthBucket],
#         target_cn_ratio: float = 0.7,
#         cn_threshold: float = 0.3,
#         allow_english_fallback: bool = False,
#         curriculum: Optional[Sequence[CurriculumPhase]] = None,
#         seed: int = 42,
#     ) -> None:
#         self.length_buckets = list(length_buckets)
#         self.target_cn_ratio = target_cn_ratio
#         self.cn_threshold = cn_threshold
#         self.allow_english_fallback = allow_english_fallback
#         self.curriculum = list(curriculum or [])
#         self._rng = random.Random(seed)

#     def plan(
#         self,
#         total_samples: int,
#         available_items: Sequence[SamplingItem],
#         *,
#         progress: float = 1.0,
#         source_weights: Optional[Mapping[str, float]] = None,
#     ) -> SamplingPlan:
#         print(f"[DEBUG] Sampler: Starting plan for {len(available_items)} items...", flush=True)
#         if total_samples <= 0 or not available_items:
#             return SamplingPlan(selected=[], rejected=list(available_items), stats={"requested": float(total_samples), "selected": 0.0})

#         print("[DEBUG] Sampler: Splitting Chinese/English pools...", flush=True)
#         chinese_pool: List[SamplingItem] = []
#         english_pool: List[SamplingItem] = []
#         for item in available_items:
#             (chinese_pool if item.is_chinese(self.cn_threshold) else english_pool).append(item)
        
#         print(f"[DEBUG] Sampler: Pools ready. CN={len(chinese_pool)}, EN={len(english_pool)}", flush=True)

#         desired_cn = int(round(total_samples * self.target_cn_ratio))
#         if not self.allow_english_fallback and len(chinese_pool) < desired_cn:
#             LOGGER.warning(
#                 "可用中文样本不足：当前=%d，目标=%d。采样量将按中文池截断",
#                 len(chinese_pool),
#                 desired_cn,
#             )

#         if not self.allow_english_fallback:
#             total_samples = min(total_samples, len(chinese_pool))

#         target_cn = int(round(total_samples * self.target_cn_ratio))
#         if not self.allow_english_fallback:
#             target_cn = min(target_cn, len(chinese_pool))

#         weights = self._merge_weights(source_weights, progress)
        
#         print(f"[DEBUG] Sampler: Selecting {target_cn} Chinese samples...", flush=True)
#         chinese_selected = self._select_weighted(chinese_pool, target_cn, weights, "CN")
#         remaining_slots = total_samples - len(chinese_selected)

#         if self.allow_english_fallback:
#             combined_pool = [item for item in available_items if item not in chinese_selected]
#         else:
#             combined_pool = [item for item in chinese_pool if item not in chinese_selected]

#         print(f"[DEBUG] Sampler: Selecting {remaining_slots} Remaining samples...", flush=True)
#         rest_selected = self._select_weighted(combined_pool, remaining_slots, weights, "REST")

#         selected = chinese_selected + rest_selected
        
#         # 优化：rejected 列表可能很大，如果不需要 debug 可以简化
#         # rejected = [item for item in available_items if item not in selected]
#         rejected = [] 

#         stats = {
#             "requested": float(total_samples),
#             "selected": float(len(selected)),
#             "chinese_selected": float(sum(item.is_chinese(self.cn_threshold) for item in selected)),
#             "english_selected": float(sum(not item.is_chinese(self.cn_threshold) for item in selected)),
#             "avg_length": float(sum(item.text_length for item in selected) / max(1, len(selected))),
#         }
#         print("[DEBUG] Sampler: Plan finished.", flush=True)
#         return SamplingPlan(selected=selected, rejected=rejected, stats=stats)

#     def _bucket_label(self, item: SamplingItem) -> str:
#         for bucket in self.length_buckets:
#             if bucket.contains(item.text_length):
#                 return bucket.name
#         return "overflow"

#     def _merge_weights(self, provided: Optional[Mapping[str, float]], progress: float) -> Dict[str, float]:
#         weights = dict(provided or {})
#         for phase in self.curriculum:
#             if phase.applies(progress):
#                 for key, value in phase.weights.items():
#                     weights[key] = weights.get(key, 1.0) * float(value)
#                 break
#         return weights

#     def _select_weighted(self, pool: Sequence[SamplingItem], k: int, weights: Mapping[str, float], debug_tag: str) -> List[SamplingItem]:
#         if k <= 0 or not pool:
#             return []
        
#         print(f"[DEBUG] Sampler({debug_tag}): Grouping {len(pool)} items into buckets...", flush=True)
#         t0 = time.time()
#         bucket_groups: MutableMapping[str, MutableMapping[str, List[SamplingItem]]] = {}
#         for item in pool:
#             bucket_groups.setdefault(self._bucket_label(item), {}).setdefault(item.source, []).append(item)
#         print(f"[DEBUG] Sampler({debug_tag}): Grouping done in {time.time()-t0:.2f}s. Shuffling...", flush=True)
        
#         for source_map in bucket_groups.values():
#             for items in source_map.values():
#                 self._rng.shuffle(items)

#         bucket_order = list(bucket_groups.keys())
#         selected: List[SamplingItem] = []
        
#         print(f"[DEBUG] Sampler({debug_tag}): Starting round-robin selection loop...", flush=True)
#         loop_cnt = 0
#         while bucket_order and len(selected) < k:
#             loop_cnt += 1
#             if loop_cnt % 5000 == 0:
#                 print(f"[DEBUG] Sampler({debug_tag}): Loop {loop_cnt}, Selected {len(selected)}/{k}", flush=True)
                
#             for bucket in list(bucket_order):
#                 source_map = bucket_groups.get(bucket, {})
#                 if not source_map:
#                     bucket_order.remove(bucket)
#                     continue
#                 for source in list(source_map.keys()):
#                     items = source_map.get(source, [])
#                     if not items:
#                         source_map.pop(source, None)
#                         continue
#                     weight = max(1e-3, weights.get(source, 1.0))
#                     steps = max(1, int(round(weight)))
#                     for _ in range(steps):
#                         if not items or len(selected) >= k:
#                             break
#                         selected.append(items.pop())
#                     if not items:
#                         source_map.pop(source, None)
#                     if len(selected) >= k:
#                         break
#                 if not source_map:
#                     bucket_order.remove(bucket)
#                 if len(selected) >= k:
#                     break
        
#         print(f"[DEBUG] Sampler({debug_tag}): Selection done. Got {len(selected)} items.", flush=True)
#         return selected


# # Alias exported for consumers
# __all__ = [
#     "LengthBucket",
#     "CurriculumPhase",
#     "SamplingItem",
#     "SamplingPlan",
#     "MixedBucketSampler",
#     "normalize_text",
#     "estimate_chinese_ratio",
#     "estimate_token_length",
#     "hash_for_text",
#     "dedupe_by_hash",
#     "compute_ngram_repetition",
#     "merge_messages",
# ]
# LOGGER = logging.getLogger(__name__)

"""Shared data utilities for the LLM fine-tuning pipeline (Performance Fix)."""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import os
from pathlib import Path

CONTROL_CHAR_PATTERN = re.compile(r"[\u0000-\u001f\u007f]")
WHITESPACE_PATTERN = re.compile(r"\s+")
HAN_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df]")
ALPHA_PATTERN = re.compile(r'[a-zA-Z]')


def get_latest_adapter(base_dir: str) -> str:
    """
    自动寻找 base_dir 下 times 子目录中时间戳最新的文件夹。
    如果找不到 times 目录，则回退到 base_dir 本身。
    """
    base_path = Path(base_dir)
    times_path = base_path / "times"
    
    # 1. 如果 times 目录不存在，说明可能还没开始用时间戳结构，直接返回原路径
    if not times_path.exists():
        print(f"[提示] 未找到 {times_path}，使用默认路径: {base_dir}")
        return base_dir
    
    # 2. 获取所有子目录
    subdirs = [d for d in times_path.iterdir() if d.is_dir()]
    
    # 3. 如果没有子目录，回退
    if not subdirs:
        return base_dir
        
    # 4. 按目录名排序（因为你的格式是 YYYYMMDD_HHMMSS，字符串排序等同于时间排序）
    # reverse=True 取第一个即为最新
    latest_dir = sorted(subdirs, key=lambda x: x.name, reverse=True)[0]
    
    print(f"[自动定位] 找到最新 Adapter: {latest_dir}")
    return str(latest_dir)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = CONTROL_CHAR_PATTERN.sub("", text.replace("\u3000", " "))
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def estimate_chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese_count = len(HAN_PATTERN.findall(text))
    alpha_count = len(ALPHA_PATTERN.findall(text))
    total = chinese_count + alpha_count
    if total == 0:
        return 0.0
    return chinese_count / total


def estimate_token_length(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def hash_for_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def merge_messages(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(msg.get("content", "") for msg in messages if isinstance(msg, Mapping))


def dedupe_by_hash(records: Iterable[Mapping[str, object]], key: str) -> List[Mapping[str, object]]:
    seen: set[str] = set()
    result: List[Mapping[str, object]] = []
    for record in records:
        digest = record.get(key)
        if not isinstance(digest, str):
            continue
        if digest in seen:
            continue
        seen.add(digest)
        result.append(record)
    return result


# 修改 scripts/utils_data.py 中的函数
def compute_ngram_repetition(text: str, n: int = 4, token_level: str = "char") -> float:
    """计算 n-gram 重复率。
    token_level="char": 按字符切分（适用于中文，识别连续4字重复）。
    token_level="word": 按空格切分（适用于英文，识别连续4词重复）。
    """
    if not text:
        return 0.0
    if len(text) > 20000:
        text = text[:20000]
    
    # 根据 token_level 选择切分方式
    if token_level == "word":
        tokens = text.split()  # 英文按空格切分单词
    else:
        tokens = list(text)    # 中文直接按字符切分
        
    if len(tokens) < n:
        return 0.0
    
    ngrams_count = len(tokens) - n + 1
    if ngrams_count <= 0:
        return 0.0
    
    unique = set(tuple(tokens[i : i + n]) for i in range(ngrams_count))
    return 1.0 - len(unique) / ngrams_count


@dataclass(frozen=True)
class LengthBucket:
    name: str
    min_tokens: int
    max_tokens: int

    def contains(self, token_count: int) -> bool:
        return self.min_tokens <= token_count <= self.max_tokens


@dataclass
class CurriculumPhase:
    start: float
    end: float
    weights: Mapping[str, float] = field(default_factory=dict)

    def applies(self, progress: float) -> bool:
        return self.start <= progress <= self.end


@dataclass
class SamplingItem:
    identifier: str
    source: str
    text_length: int
    chinese_ratio: float
    payload: Mapping[str, object]

    def is_chinese(self, threshold: float) -> bool:
        return self.chinese_ratio >= threshold


@dataclass
class SamplingPlan:
    selected: List[SamplingItem]
    rejected: List[SamplingItem]
    stats: Mapping[str, float]


class MixedBucketSampler:
    def __init__(
        self,
        length_buckets: Sequence[LengthBucket],
        target_cn_ratio: float = 0.7,
        cn_threshold: float = 0.3,
        allow_english_fallback: bool = False,
        curriculum: Optional[Sequence[CurriculumPhase]] = None,
        seed: int = 42,
    ) -> None:
        self.length_buckets = list(length_buckets)
        self.target_cn_ratio = target_cn_ratio
        self.cn_threshold = cn_threshold
        self.allow_english_fallback = allow_english_fallback
        self.curriculum = list(curriculum or [])
        self._rng = random.Random(seed)

    def plan(
        self,
        total_samples: int,
        available_items: Sequence[SamplingItem],
        *,
        progress: float = 1.0,
        source_weights: Optional[Mapping[str, float]] = None,
    ) -> SamplingPlan:
        print(f"[DEBUG] Sampler: Starting plan for {len(available_items)} items...", flush=True)
        if total_samples <= 0 or not available_items:
            return SamplingPlan(selected=[], rejected=list(available_items), stats={"requested": float(total_samples), "selected": 0.0})

        print("[DEBUG] Sampler: Splitting Chinese/English pools...", flush=True)
        chinese_pool: List[SamplingItem] = []
        english_pool: List[SamplingItem] = []
        for item in available_items:
            (chinese_pool if item.is_chinese(self.cn_threshold) else english_pool).append(item)
        
        print(f"[DEBUG] Sampler: Pools ready. CN={len(chinese_pool)}, EN={len(english_pool)}", flush=True)

        desired_cn = int(round(total_samples * self.target_cn_ratio))
        if not self.allow_english_fallback and len(chinese_pool) < desired_cn:
            LOGGER.warning("可用中文样本不足...")

        if not self.allow_english_fallback:
            total_samples = min(total_samples, len(chinese_pool))

        target_cn = int(round(total_samples * self.target_cn_ratio))
        if not self.allow_english_fallback:
            target_cn = min(target_cn, len(chinese_pool))

        weights = self._merge_weights(source_weights, progress)
        
        print(f"[DEBUG] Sampler: Selecting {target_cn} Chinese samples...", flush=True)
        chinese_selected = self._select_weighted(chinese_pool, target_cn, weights, "CN")
        remaining_slots = total_samples - len(chinese_selected)

        # === 核心优化：使用 Set 加速剔除 ===
        print(f"[DEBUG] Sampler: Excluding selected items (Optimized Set)...", flush=True)
        # 使用对象 ID 构建集合，查找速度 O(1)
        selected_ids = set(id(item) for item in chinese_selected)

        if self.allow_english_fallback:
            # O(N) 遍历一次即可
            combined_pool = [item for item in available_items if id(item) not in selected_ids]
        else:
            combined_pool = [item for item in chinese_pool if id(item) not in selected_ids]
        # ================================

        print(f"[DEBUG] Sampler: Selecting {remaining_slots} Remaining samples...", flush=True)
        rest_selected = self._select_weighted(combined_pool, remaining_slots, weights, "REST")

        selected = chinese_selected + rest_selected
        rejected = [] 

        stats = {
            "requested": float(total_samples),
            "selected": float(len(selected)),
        }
        print("[DEBUG] Sampler: Plan finished.", flush=True)
        return SamplingPlan(selected=selected, rejected=rejected, stats=stats)

    def _bucket_label(self, item: SamplingItem) -> str:
        for bucket in self.length_buckets:
            if bucket.contains(item.text_length):
                return bucket.name
        return "overflow"

    def _merge_weights(self, provided: Optional[Mapping[str, float]], progress: float) -> Dict[str, float]:
        weights = dict(provided or {})
        for phase in self.curriculum:
            if phase.applies(progress):
                for key, value in phase.weights.items():
                    weights[key] = weights.get(key, 1.0) * float(value)
                break
        return weights

    def _select_weighted(self, pool: Sequence[SamplingItem], k: int, weights: Mapping[str, float], debug_tag: str) -> List[SamplingItem]:
        if k <= 0 or not pool:
            return []
        
        print(f"[DEBUG] Sampler({debug_tag}): Grouping {len(pool)} items...", flush=True)
        bucket_groups: MutableMapping[str, MutableMapping[str, List[SamplingItem]]] = {}
        for item in pool:
            bucket_groups.setdefault(self._bucket_label(item), {}).setdefault(item.source, []).append(item)
        
        for source_map in bucket_groups.values():
            for items in source_map.values():
                self._rng.shuffle(items)

        bucket_order = list(bucket_groups.keys())
        selected: List[SamplingItem] = []
        
        # 移除详细的 Loop 打印，避免刷屏，保留首尾
        print(f"[DEBUG] Sampler({debug_tag}): Starting selection loop...", flush=True)
        while bucket_order and len(selected) < k:
            for bucket in list(bucket_order):
                source_map = bucket_groups.get(bucket, {})
                if not source_map:
                    bucket_order.remove(bucket)
                    continue
                for source in list(source_map.keys()):
                    items = source_map.get(source, [])
                    if not items:
                        source_map.pop(source, None)
                        continue
                    weight = max(1e-3, weights.get(source, 1.0))
                    steps = max(1, int(round(weight)))
                    for _ in range(steps):
                        if not items or len(selected) >= k:
                            break
                        selected.append(items.pop())
                    if not items:
                        source_map.pop(source, None)
                    if len(selected) >= k:
                        break
                if not source_map:
                    bucket_order.remove(bucket)
                if len(selected) >= k:
                    break
        
        print(f"[DEBUG] Sampler({debug_tag}): Done. Got {len(selected)} items.", flush=True)
        return selected

__all__ = [
    "LengthBucket",
    "CurriculumPhase",
    "SamplingItem",
    "SamplingPlan",
    "MixedBucketSampler",
    "normalize_text",
    "estimate_chinese_ratio",
    "estimate_token_length",
    "hash_for_text",
    "dedupe_by_hash",
    "compute_ngram_repetition",
    "merge_messages",
]
LOGGER = logging.getLogger(__name__)
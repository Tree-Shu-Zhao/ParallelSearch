"""
Generate parallel vs sequential QA splits for specific datasets and save to parquet files.

Datasets handled:
- hotpotqa (from RUC-NLPIR/FlashRAG_datasets)
- 2wikimultihopqa (from RUC-NLPIR/FlashRAG_datasets)
- MultiHopRAG (from yixuantt/MultiHopRAG, file: MultiHopRAG.json)

Output files:
- hotpotqa_parallel.parquet (type == "comparison")
- hotpotqa_sequential.parquet (type == "bridge")
- 2wikimultihopqa_parallel.parquet (type == "comparison")
- 2wikimultihopqa_sequential.parquet (type in {"inference", "compositional"})
- multihoprag_parallel.parquet (type == "comparison_query")
- multihoprag_sequential.parquet (type in {"inference", "inference_query"})
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional

import datasets

try:
    # Optional, used only if hdfs_dir is provided
    from verl.utils.hdfs_io import copy as hdfs_copy, makedirs as hdfs_makedirs
except Exception:  # pragma: no cover - optional dependency
    hdfs_copy = None
    hdfs_makedirs = None


def make_prefix(data_point: Dict[str, Any], template_type: str) -> str:
    question_text = data_point["question"]

    if template_type == "base":
        prefix = (
            """Answer the given question. """
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. "
            "If the original query is complex or involves multiple parts, you are encouraged to decompose it into smaller sub-questions, separated by ##. For example: <search> sub-question 1 ## sub-question 2 </search>. "
            "You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
            f"Question: {question_text}\n"
        )
    else:
        raise NotImplementedError(f"Unsupported template_type: {template_type}")
    return prefix


def get_eval_split(ds: datasets.DatasetDict) -> str:
    if "test" in ds:
        return "test"
    if "dev" in ds:
        return "dev"
    return "train"


def normalize_question(example: Dict[str, Any]) -> str:
    for key in ("question", "query"):
        if key in example and example[key] is not None:
            text = str(example[key]).strip()
            if text and text[-1] != "?":
                text += "?"
            return text
    raise KeyError("No question/query field found in example")


def normalize_answers(example: Dict[str, Any]) -> Any:
    for key in ("golden_answers", "answer", "answers", "final_answer"):
        if key in example and example[key] is not None:
            return example[key]
    # Fallback to empty list if not available
    return []


def normalize_type(example: Dict[str, Any]) -> Optional[str]:
    # Check metadata.type first (used by FlashRAG datasets)
    metadata = example.get("metadata")
    if metadata and metadata.get("type"):
        return str(metadata["type"])
    
    # Check direct type fields
    for key in ("type", "query_type", "category", "question_type"):
        if key in example and example[key] is not None:
            return str(example[key])
    return None


def build_data_item(
    example: Dict[str, Any],
    idx: int,
    split: str,
    data_source: str,
    template_type: str,
) -> Dict[str, Any]:
    question_text = normalize_question(example)
    golden_answers = normalize_answers(example)

    prompt_text = make_prefix({"question": question_text}, template_type=template_type)
    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": golden_answers},
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "type": normalize_type(example),
        },
    }


def filter_by_types(ds: datasets.Dataset, allowed_types: Iterable[str]) -> datasets.Dataset:
    allowed = set(allowed_types)

    def _predicate(example: Dict[str, Any]) -> bool:
        t = normalize_type(example)
        return t in allowed

    return ds.filter(_predicate)


def map_to_unified(ds: datasets.Dataset, split: str, data_source: str, template_type: str) -> datasets.Dataset:
    def _with_indices(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return build_data_item(example, idx, split=split, data_source=data_source, template_type=template_type)

    return ds.map(function=_with_indices, with_indices=True)


def save_parquet(ds: datasets.Dataset, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_parquet(output_path)
    try:
        num_rows = getattr(ds, "num_rows", None)
        rows_str = f"{num_rows:,}" if isinstance(num_rows, int) else "unknown"
        print(f"Saved parquet: {output_path} | rows={rows_str}")
    except Exception as e:  # pragma: no cover
        print(f"Saved parquet: {output_path} | rows=unknown")


def process_hotpotqa(local_dir: str, template_type: str) -> None:
    data_source = "hotpotqa"
    dsd = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)
    split = get_eval_split(dsd)
    base = dsd[split]
    
    print(f"hotpotqa: loaded {len(base)} samples from {split} split")
    
    parallel_ds = filter_by_types(base, {"comparison"})
    sequential_ds = filter_by_types(base, {"bridge"})
    
    print(f"hotpotqa: filtered to {len(parallel_ds)} comparison and {len(sequential_ds)} bridge samples")

    parallel_ds = map_to_unified(parallel_ds, split=split, data_source=data_source, template_type=template_type)
    sequential_ds = map_to_unified(sequential_ds, split=split, data_source=data_source, template_type=template_type)

    save_parquet(parallel_ds, os.path.join(local_dir, "hotpotqa_parallel.parquet"))
    save_parquet(sequential_ds, os.path.join(local_dir, "hotpotqa_sequential.parquet"))


def process_2wikimultihopqa(local_dir: str, template_type: str) -> None:
    data_source = "2wikimultihopqa"
    dsd = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)
    split = get_eval_split(dsd)
    base = dsd[split]
    
    print(f"2wikimultihopqa: loaded {len(base)} samples from {split} split")
    
    parallel_ds = filter_by_types(base, {"comparison"})
    sequential_ds = filter_by_types(base, {"inference", "compositional"})
    
    print(f"2wikimultihopqa: filtered to {len(parallel_ds)} comparison and {len(sequential_ds)} inference+compositional samples")

    parallel_ds = map_to_unified(parallel_ds, split=split, data_source=data_source, template_type=template_type)
    sequential_ds = map_to_unified(sequential_ds, split=split, data_source=data_source, template_type=template_type)

    save_parquet(parallel_ds, os.path.join(local_dir, "2wikimultihopqa_parallel.parquet"))
    save_parquet(sequential_ds, os.path.join(local_dir, "2wikimultihopqa_sequential.parquet"))


def process_multihoprag(local_dir: str, template_type: str) -> None:
    data_source = "MultiHopRAG"
    # The dataset is not part of FlashRAG_datasets; load from yixuantt/MultiHopRAG
    dsd = datasets.load_dataset("yixuantt/MultiHopRAG", data_files="MultiHopRAG.json")
    split = get_eval_split(dsd)
    base = dsd[split]

    parallel_ds = filter_by_types(base, {"comparison_query"})
    sequential_ds = filter_by_types(base, {"inference", "inference_query"})

    parallel_ds = map_to_unified(parallel_ds, split=split, data_source=data_source, template_type=template_type)
    sequential_ds = map_to_unified(sequential_ds, split=split, data_source=data_source, template_type=template_type)

    save_parquet(parallel_ds, os.path.join(local_dir, "multihoprag_parallel.parquet"))
    save_parquet(sequential_ds, os.path.join(local_dir, "multihoprag_sequential.parquet"))


def maybe_copy_to_hdfs(local_dir: str, hdfs_dir: Optional[str]) -> None:
    if not hdfs_dir:
        return
    if hdfs_makedirs is None or hdfs_copy is None:
        raise RuntimeError("HDFS utils are not available. Install verl and ensure verl.utils.hdfs_io is importable.")
    hdfs_makedirs(hdfs_dir)
    hdfs_copy(src=local_dir, dst=hdfs_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/qa_parallel_sequential")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", type=str, default="base")
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    process_hotpotqa(local_dir=local_dir, template_type=args.template_type)
    process_2wikimultihopqa(local_dir=local_dir, template_type=args.template_type)
    process_multihoprag(local_dir=local_dir, template_type=args.template_type)

    maybe_copy_to_hdfs(local_dir=local_dir, hdfs_dir=args.hdfs_dir)


if __name__ == "__main__":
    main()



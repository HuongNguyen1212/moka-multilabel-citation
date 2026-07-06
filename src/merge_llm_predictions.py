#!/usr/bin/env python3
"""
Extend merge_multicite output (step 5) by attaching LLM zero-shot and few-shot
predictions for both dynamic contexts and the full paragraph baseline.

Reads from output/merge_multicite/<group>/<exp>.json
Writes to output/merge_predictions/<group>_<slug>/<exp>.json

Usage:
  python src/merge_llm_predictions.py --group scincl --llm-model gpt-4o-mini
  python src/merge_llm_predictions.py --group specter --llm-model qwen2.5-14b
  python src/merge_llm_predictions.py --group LLM --llm-model gemma3-12b
"""

import argparse
import json
import os


EXPS_BY_GROUP = {
    "scincl": [
        "non_contiguous_acl_arc_exp1",  "non_contiguous_acl_arc_exp2",
        "non_contiguous_acl_arc_exp3",  "non_contiguous_acl_arc_exp4",
        "non_contiguous_acl_arc_exp5a", "non_contiguous_acl_arc_exp5b",
        "non_contiguous_acl_arc_exp5c", "non_contiguous_acl_arc_exp5d",
        "non_contiguous_acl_arc_exp5e", "non_contiguous_acl_arc_exp5f",
        "non_contiguous_acl_arc_exp5g", "non_contiguous_acl_arc_exp5h",
        "non_contiguous_acl_arc_exp5i", "non_contiguous_acl_arc_exp5j",
        "non_contiguous_acl_arc_exp5k", "non_contiguous_acl_arc_exp5l",
        "non_contiguous_acl_arc_exp5m", "non_contiguous_acl_arc_exp5n",
    ],
    "specter": [
        "non_contiguous_acl_arc_exp1",  "non_contiguous_acl_arc_exp2",
        "non_contiguous_acl_arc_exp3",  "non_contiguous_acl_arc_exp4",
        "non_contiguous_acl_arc_exp5a", "non_contiguous_acl_arc_exp5b",
        "non_contiguous_acl_arc_exp5c", "non_contiguous_acl_arc_exp5d",
        "non_contiguous_acl_arc_exp5e", "non_contiguous_acl_arc_exp5f",
        "non_contiguous_acl_arc_exp5g", "non_contiguous_acl_arc_exp5h",
        "non_contiguous_acl_arc_exp5i", "non_contiguous_acl_arc_exp5j",
        "non_contiguous_acl_arc_exp5k", "non_contiguous_acl_arc_exp5l",
        "non_contiguous_acl_arc_exp5m", "non_contiguous_acl_arc_exp5n",
    ],
    "LLM": [
        "non_contiguous_acl_arc_exp6",  "non_contiguous_acl_arc_exp7",
        "non_contiguous_acl_arc_exp8",  "non_contiguous_acl_arc_exp9",
        "non_contiguous_acl_arc_exp10",
    ],
}


def load_llm(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["unique_id"]: item.get("llm_prediction", []) for item in data}


def load_baseline_llm(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["unique_id"]: item.get("llm_prediction", []) for item in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["scincl", "specter", "LLM"], required=True)
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                        help="Model slug (e.g. gpt-4o-mini, qwen2.5-14b)")
    args = parser.parse_args()

    slug        = args.llm_model
    compare_dir = f"output/merge_multicite/{args.group}"
    llm_dir     = f"output/llm/zeroshot/{slug}/{args.group}"
    llm_fs_dir  = f"output/llm/fewshot/{slug}/{args.group}"
    out_dir     = f"output/merge_predictions/{args.group}_{slug}"
    os.makedirs(out_dir, exist_ok=True)

    baseline_llm_zs_map = load_baseline_llm(f"output/llm/zeroshot/{slug}/baseline/baseline.json")
    baseline_llm_fs_map = load_baseline_llm(f"output/llm/fewshot/{slug}/baseline/baseline.json")

    for exp in EXPS_BY_GROUP[args.group]:
        in_path  = os.path.join(compare_dir, f"{exp}.json")
        out_path = os.path.join(out_dir, f"{exp}.json")

        if not os.path.exists(in_path):
            print(f"  [SKIP] {exp} - merge_multicite output not found")
            continue

        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)

        llm_map    = load_llm(os.path.join(llm_dir, f"{exp}.json"))
        llm_fs_map = load_llm(os.path.join(llm_fs_dir, f"{exp}.json"))

        for entry in data:
            uid = entry["unique_id"]
            if uid in llm_map:
                entry["llm_prediction"] = llm_map[uid]
            if uid in llm_fs_map:
                entry["llm_fewshot_prediction"] = llm_fs_map[uid]
            if uid in baseline_llm_zs_map:
                entry["baseline_llm_prediction"] = baseline_llm_zs_map[uid]
            if uid in baseline_llm_fs_map:
                entry["baseline_llm_fewshot_prediction"] = baseline_llm_fs_map[uid]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        n_llm       = sum(1 for e in data if "llm_prediction" in e)
        n_llm_fs    = sum(1 for e in data if "llm_fewshot_prediction" in e)
        n_bl_llm    = sum(1 for e in data if "baseline_llm_prediction" in e)
        n_bl_llm_fs = sum(1 for e in data if "baseline_llm_fewshot_prediction" in e)
        print(f"  {exp}: llm={n_llm}, llm_fs={n_llm_fs}, bl_llm={n_bl_llm}, bl_llm_fs={n_bl_llm_fs} -> {out_path}")

    print(f"\nMerged JSONs saved to {out_dir}/")


if __name__ == "__main__":
    main()

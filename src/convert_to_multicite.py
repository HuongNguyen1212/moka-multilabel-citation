#!/usr/bin/env python3
"""
Convert CCE output to MultiCite input format (step 1).
Uses dynamic_contexts_combined by default; pass --context_col cite_context_paragraph for baseline.

Usage:
  # All experiments
  python src/convert_to_multicite.py --input_base data/moka/scincl --output_base data/converted/scincl

  # Baseline (full paragraph)
  python src/convert_to_multicite.py \
      --input_dir data/moka/scincl/non_contiguous_acl_arc_exp1 \
      --output_dir data/converted/baseline \
      --context_col cite_context_paragraph
"""

import argparse
import ast
import csv
import json
import os

DUMMY_LABEL = "background"

SPLIT_MAP_JSON = {
    "train_citation_context.json": "train.json",
    "valid_citation_context.json": "dev.json",
    "test_citation_context.json":  "test.json",
}

SPLIT_MAP_TSV = {
    "train.txt": "train.json",
    "valid.txt": "dev.json",
    "test.txt":  "test.json",
}


def _extract_citation_sentence(analysis_result):
    """Return the sentence marked as 'Citation sentence' in analysis_result."""
    for s in (analysis_result or []):
        if "citation" in s.get("role", "").lower():
            return s.get("sentence_content", "")
    return ""


def convert_llm_json(input_path, split_name):
    # Build orig_id -> unique_id map from companion test.txt
    import pathlib
    id_map = {}
    tsv_path = pathlib.Path(input_path).parent / "test.txt"
    if tsv_path.exists():
        with open(tsv_path, encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                id_map[i] = row["unique_id"]

    with open(input_path) as f:
        data = json.load(f)
    result = []
    for item in data:
        orig_idx = item.get("orig_id", item["row_id"])
        unique_id = id_map.get(orig_idx, f"{split_name}_{orig_idx}")
        context = item.get("context_paragraph", [])
        if isinstance(context, str):
            context = [context]
        citation_sent = _extract_citation_sentence(item.get("analysis_result", []))
        result.append({
            "id": unique_id,
            "orig_id": unique_id,
            "citation_sentence": citation_sent,
            "x": context,
            "y": DUMMY_LABEL,
        })
    return result


def convert_tsv(input_path, split_name, context_col=None):
    result = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if context_col:
                raw = row.get(context_col, "")
            else:
                raw = row.get("dynamic_contexts_combined") or row.get("cite_context_paragraph", "")
            try:
                context = ast.literal_eval(raw)
                if not isinstance(context, list):
                    context = [str(context)]
            except Exception:
                context = [raw]
            # citation_context column = the specific sentence with #AUTHOR_TAG
            result.append({
                "id": row.get("unique_id", f"{split_name}_{i}"),
                "orig_id": row.get("unique_id", i),
                "citation_sentence": row.get("citation_context", ""),
                "x": context,
                "y": DUMMY_LABEL,
            })
    return result


TEST_FILES_JSON = {"test_citation_context.json"}
TEST_FILES_TSV  = {"test.txt"}


def convert_experiment(input_dir, output_dir, split="test", context_col=None):
    """
    split: "test"  -> only convert test split into test.json  (default)
           "all"   -> convert each split separately
           "merge" -> merge all splits into one test.json
    context_col: column to use as context; None = auto (dynamic_contexts_combined with fallback)
    """
    os.makedirs(output_dir, exist_ok=True)
    files = set(os.listdir(input_dir))

    # TSV takes priority when test.txt exists (has unique_id + dynamic_contexts_combined)
    # JSON used only when no TSV files present
    has_tsv  = any(f in files for f in SPLIT_MAP_TSV)
    has_json = any(f in files for f in SPLIT_MAP_JSON)
    use_json = has_json and not has_tsv
    split_map = SPLIT_MAP_JSON if use_json else SPLIT_MAP_TSV
    converter  = convert_llm_json if use_json else convert_tsv
    test_files = TEST_FILES_JSON if use_json else TEST_FILES_TSV

    all_data = []
    converted = 0

    for fname, outname in split_map.items():
        if fname not in files:
            continue
        if split == "test" and fname not in test_files:
            continue

        split_name = outname.replace(".json", "")
        kwargs = {} if has_json else {"context_col": context_col}
        data = converter(os.path.join(input_dir, fname), split_name, **kwargs)
        converted += 1

        if split == "merge":
            all_data.extend(data)
            print(f"  {fname}  ({len(data)} examples) -> merged")
        else:
            out_path = os.path.join(output_dir, outname)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  {fname} -> {out_path}  ({len(data)} examples)")

    if split == "merge" and all_data:
        out_path = os.path.join(output_dir, "test.json")
        with open(out_path, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"  => test.json  ({len(all_data)} examples total)")

    if converted == 0:
        print(f"  [WARN] No recognized files in {input_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert processed_data_results to MultiCite format")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Single experiment input directory")
    group.add_argument("--input_base", help="Base directory containing multiple experiment subdirs")

    parser.add_argument("--output_dir", help="Output directory (required with --input_dir)")
    parser.add_argument("--output_base", help="Output base directory (required with --input_base)")
    parser.add_argument("--split", choices=["test", "all", "merge"], default="test",
                        help="test (default): only test split; all: keep splits separate; merge: merge all into test.json")
    parser.add_argument("--context_col", default=None,
                        help="TSV column to use as context (default: dynamic_contexts_combined with fallback to cite_context_paragraph). "
                             "Use 'cite_context_paragraph' for baseline.")
    args = parser.parse_args()

    if args.input_dir:
        if not args.output_dir:
            parser.error("--output_dir required with --input_dir")
        print(f"Converting: {args.input_dir}")
        convert_experiment(args.input_dir, args.output_dir, split=args.split, context_col=args.context_col)
    else:
        if not args.output_base:
            parser.error("--output_base required with --input_base")
        experiments = sorted(os.listdir(args.input_base))
        for exp in experiments:
            exp_in = os.path.join(args.input_base, exp)
            if not os.path.isdir(exp_in):
                continue
            exp_out = os.path.join(args.output_base, exp)
            print(f"Converting: {exp}")
            convert_experiment(exp_in, exp_out, split=args.split, context_col=args.context_col)

    print("Done.")


if __name__ == "__main__":
    main()

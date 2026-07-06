#!/usr/bin/env python3
"""
Merge ACL-ARC ground truth, MultiCite predictions, and gold human annotations
into one unified JSON file per experiment (step 4).

Usage:
  python src/merge_multicite.py \
      --moka_test            data/moka/scincl/non_contiguous_acl_arc_exp1/test.txt \
      --converted            data/converted/scincl/non_contiguous_acl_arc_exp1/test.json \
      --predictions          output/multicite/scincl/non_contiguous_acl_arc_exp1/predictions.txt \
      --baseline_predictions output/multicite/baseline/predictions.txt \
      --human_annotations    data/annotations/annotations_gold.json \
      --output               output/merge_multicite/scincl/non_contiguous_acl_arc_exp1.json
"""

import argparse
import csv
import json

ACL_ARC_LABEL_NAMES = {
    "0": "BACKGROUND",
    "1": "COMPARES_CONTRASTS",
    "2": "EXTENSION",
    "3": "FUTURE",
    "4": "MOTIVATION",
    "5": "USES",
}


def load_moka_data(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["unique_id"]: (row["citation_class_label"], row["citation_context"]) for row in reader}


def load_converted_ids(path):
    with open(path) as f:
        data = json.load(f)
    return [item["id"] for item in data]


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_human_annotations(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ann_list = data.get("annotations", data) if isinstance(data, dict) else data
    return {
        item["id"]: {"labels": item.get("labels", []), "unclear": item.get("unclear", False)}
        for item in ann_list
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moka_test",   required=True)
    parser.add_argument("--converted",   required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--baseline_predictions", default=None,
                        help="Predictions from MultiCite on full cite_context_paragraph (baseline).")
    parser.add_argument("--human_annotations", default=None,
                        help="Exported JSON from GitHub Pages annotation tool.")
    parser.add_argument("--output",      default=None,
                        help="Output JSON file. Prints to stdout if omitted.")
    args = parser.parse_args()

    moka_data      = load_moka_data(args.moka_test)
    ids            = load_converted_ids(args.converted)
    preds          = load_predictions(args.predictions)
    baseline_preds = load_predictions(args.baseline_predictions) if args.baseline_predictions else None
    human_anns     = load_human_annotations(args.human_annotations) if args.human_annotations else None

    if len(ids) != len(preds):
        raise ValueError(f"ID count ({len(ids)}) != prediction count ({len(preds)})")
    if baseline_preds is not None and len(ids) != len(baseline_preds):
        raise ValueError(f"ID count ({len(ids)}) != baseline prediction count ({len(baseline_preds)})")

    results = []
    for i, (uid, pred) in enumerate(zip(ids, preds)):
        moka_num, citation_context = moka_data.get(uid, ("?", ""))
        entry = {
            "unique_id":            uid,
            "citation_context":     citation_context,
            "acl_arc_label":         ACL_ARC_LABEL_NAMES.get(moka_num, moka_num),
            "multicite_prediction": pred.split() if pred else [],
        }
        if baseline_preds is not None:
            entry["baseline_prediction"] = baseline_preds[i].split() if baseline_preds[i] else []
        if human_anns is not None:
            ann = human_anns.get(uid)
            if ann is not None:
                labels = [] if ann.get("unclear") else ann.get("labels", [])
                entry["human_annotation"] = {"labels": labels}
        results.append(entry)

    if args.output:
        import os; os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} entries to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

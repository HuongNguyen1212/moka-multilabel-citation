#!/usr/bin/env python3
"""
Generate docs/data.json for the GitHub Pages annotation tool.

Reads cite_context_paragraph and citation_context from any one experiment's test.txt
(all experiments share the same paragraph/sentence columns).

Usage:
  python prepare_annotation_data.py
  python prepare_annotation_data.py --input data/moka/scincl/non_contiguous_acl_arc_exp1/test.txt
"""

import argparse
import ast
import csv
import json
import os

ACL_ARC_LABEL_NAMES = {
    "0": "BACKGROUND", "1": "COMPARES_CONTRASTS", "2": "EXTENSION",
    "3": "FUTURE",     "4": "MOTIVATION",          "5": "USES",
}

DEFAULT_INPUT = "data/moka/scincl/non_contiguous_acl_arc_exp1/test.txt"
OUTPUT        = "docs/data.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    args = parser.parse_args()

    os.makedirs("docs", exist_ok=True)

    data = []
    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw = row.get("cite_context_paragraph", "")
            try:
                paragraph = ast.literal_eval(raw)
                if not isinstance(paragraph, list):
                    paragraph = [str(paragraph)]
            except Exception:
                paragraph = [raw]

            data.append({
                "id":               row["unique_id"],
                "citation_sentence": row.get("citation_context", ""),
                "paragraph":         paragraph,
                # stored for post-hoc analysis, NOT displayed in annotation UI
                "acl_arc_label":    ACL_ARC_LABEL_NAMES.get(
                                        row.get("citation_class_label", ""), "?"
                                    ),
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} samples → {OUTPUT}")


if __name__ == "__main__":
    main()

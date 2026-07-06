#!/usr/bin/env python3
"""
Compute gold standard annotation via majority vote from multiple annotation files (step 3b).
Also computes per-label Fleiss' kappa and macro average (excl. high-prevalence labels).

Usage:
  python3 src/compute_gold_standard.py \
      --inputs data/annotations/annotations_Annotator_1.json \
               data/annotations/annotations_Annotator_2.json \
               data/annotations/annotations_Annotator_3.json \
      --output data/annotations/annotations_gold.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

MULTICITE_LABELS = [
    "background", "uses", "motivation", "extends",
    "similarities", "differences", "future_work",
]

# Labels excluded from macro kappa due to high prevalence (>85%) making kappa unreliable
HIGH_PREVALENCE_LABELS = {"background"}


def majority_vote(annotator_files: list[Path], threshold: int = None) -> tuple[list[dict], dict]:
    n = len(annotator_files)
    if threshold is None:
        threshold = n // 2 + 1

    by_uid: dict[str, list[set]] = {}
    annotator_names = []
    for path in annotator_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        annotator_names.append(data.get("annotator", path.name))
        for item in data["annotations"]:
            by_uid.setdefault(item["id"], []).append(set(item.get("labels", [])))

    gold = []
    stats = {"total": 0, "full_agree": 0, "partial_agree": 0, "all_empty": 0}
    for uid, label_sets in by_uid.items():
        votes: dict[str, int] = {}
        for ls in label_sets:
            for l in ls:
                votes[l] = votes.get(l, 0) + 1
        gold_labels = sorted(l for l, v in votes.items() if v >= threshold)
        gold.append({"id": uid, "labels": gold_labels})
        stats["total"] += 1
        if all(ls == set(gold_labels) for ls in label_sets):
            stats["full_agree"] += 1
        elif not gold_labels:
            stats["all_empty"] += 1
        else:
            stats["partial_agree"] += 1

    stats["annotators"] = annotator_names
    stats["threshold"]  = f"{threshold}/{n}"
    return gold, stats


def fleiss_kappa_binary(n_i1, n_raters):
    """Fleiss' kappa for one binary label. n_i1[i] = number of raters assigning label to item i."""
    import math
    N = len(n_i1)
    n = n_raters
    n_i0 = [n - x for x in n_i1]

    P_i = [(x*(x-1) + y*(y-1)) / (n*(n-1)) for x, y in zip(n_i1, n_i0)]
    P_bar = sum(P_i) / N

    p1 = sum(n_i1) / (N * n)
    p0 = 1 - p1
    P_e = p1**2 + p0**2

    if (1 - P_e) == 0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def compute_fleiss_kappa(annotator_files: list[Path]) -> dict:
    ann_maps = []
    for path in annotator_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ann_maps.append({a["id"]: set(a.get("labels", [])) for a in data["annotations"]})

    uids = list(ann_maps[0].keys())
    n = len(annotator_files)
    N = len(uids)

    kappas = {}
    prevalences = {}
    for label in MULTICITE_LABELS:
        n_i1 = [sum(1 for am in ann_maps if label in am.get(uid, set())) for uid in uids]
        kappas[label] = round(fleiss_kappa_binary(n_i1, n), 4)
        prevalences[label] = round(sum(n_i1) / (N * n) * 100, 1)

    included = {l: k for l, k in kappas.items() if l not in HIGH_PREVALENCE_LABELS}
    macro_all  = round(sum(kappas.values()) / len(kappas), 4)
    macro_excl = round(sum(included.values()) / len(included), 4)

    return {
        "per_label": kappas,
        "prevalence": prevalences,
        "macro_all": macro_all,
        "macro_excl_high_prevalence": macro_excl,
        "excluded_labels": list(HIGH_PREVALENCE_LABELS),
    }


def interpret_kappa(k: float) -> str:
    if k < 0.00: return "poor"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="data/annotations/annotations_gold.json")
    parser.add_argument("--verify", default=None)
    parser.add_argument("--threshold", type=int, default=None)
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            print(f"[ERROR] Not found: {p}"); return

    n = len(paths)
    threshold = args.threshold or (n // 2 + 1)
    print(f"Computing majority vote from {n} annotators (threshold: {threshold}/{n})\n")

    gold_annotations, stats = majority_vote(paths, threshold=threshold)

    print(f"Annotators:        {stats['annotators']}")
    print(f"Total samples:     {stats['total']}")
    print(f"Full agreement:    {stats['full_agree']}")
    print(f"Partial agreement: {stats['partial_agree']}")
    print(f"All empty:         {stats['all_empty']}")

    # ── Fleiss' kappa ──────────────────────────────────────────────────────────
    print("\n── Fleiss' kappa (per label) ──")
    kappa_result = compute_fleiss_kappa(paths)
    print(f"  {'Label':<20} {'Prevalence':>12} {'κ':>8}")
    print(f"  {'-'*42}")
    for label in MULTICITE_LABELS:
        k    = kappa_result["per_label"][label]
        prev = kappa_result["prevalence"][label]
        excl = " *" if label in HIGH_PREVALENCE_LABELS else ""
        print(f"  {label:<20} {prev:>10.1f}%  {k:>8.4f}{excl}")

    print(f"\n  Macro κ (all labels):              {kappa_result['macro_all']:.4f}  [{interpret_kappa(kappa_result['macro_all'])}]")
    print(f"  Macro κ (excl. high-prevalence*):  {kappa_result['macro_excl_high_prevalence']:.4f}  [{interpret_kappa(kappa_result['macro_excl_high_prevalence'])}]")
    print(f"  * Excluded: {kappa_result['excluded_labels']} (prevalence >85%, kappa unreliable)")
    print(f"\n  → For paper: κ = {kappa_result['macro_excl_high_prevalence']:.2f} ({interpret_kappa(kappa_result['macro_excl_high_prevalence'])} agreement, Landis & Koch 1977)")

    # ── Save gold ──────────────────────────────────────────────────────────────
    out = {
        "annotator":   "gold_majority_vote",
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") +
                       f"{datetime.now().microsecond // 1000:03d}Z",
        "source":      f"Majority vote (threshold={threshold}/{n}) from: " +
                       ", ".join(p.name for p in paths),
        "fleiss_kappa": kappa_result,
        "annotations": gold_annotations,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")

    # ── Sync to docs/annotations/ ─────────────────────────────────────────────
    import shutil
    docs_ann = Path("docs/annotations")
    docs_ann.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, docs_ann / p.name)
    shutil.copy2(out_path, docs_ann / out_path.name)
    manifest = {"annotators": [p.name for p in paths]}
    (docs_ann / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"Synced {len(paths)} annotation file(s) + gold + manifest → {docs_ann}/")

    # ── Verify ─────────────────────────────────────────────────────────────────
    if args.verify:
        with open(args.verify, encoding="utf-8") as f:
            ref = json.load(f)
        ref_map = {a["id"]: set(a.get("labels", [])) for a in ref["annotations"]}
        mv_map  = {a["id"]: set(a["labels"]) for a in gold_annotations}
        matches   = sum(1 for uid, rl in ref_map.items() if rl == mv_map.get(uid, set()))
        mismatches = len(ref_map) - matches
        print(f"\nVerification vs {Path(args.verify).name}:")
        print(f"  Match: {matches}/{len(ref_map)} ({matches/len(ref_map)*100:.1f}%)")
        if mismatches == 0:
            print("  ✓ Perfectly matches reference file")
        else:
            for uid, rl in ref_map.items():
                if rl != mv_map.get(uid, set()):
                    print(f"  [DIFF] {uid}: ref={sorted(rl)} mv={sorted(mv_map.get(uid,set()))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Per-experiment axis metrics (step 6a): containment, preservation, and human Jaccard
for both MultiCite and LLM classifiers (ZS and FS) across all CCE experiments.
Writes summary.csv per model.

Usage:
  python src/compute_axis_metrics.py --group scincl
  python src/compute_axis_metrics.py --group specter
  python src/compute_axis_metrics.py --group LLM
"""

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--group", choices=["scincl", "specter", "LLM"], default="scincl")
parser.add_argument("--merged", action="store_true",
                    help="Use merged JSONs (with LLM predictions)")
parser.add_argument("--llm-model", default="gpt-4o-mini",
                    help="Model slug used in merge step (e.g. gpt-4o-mini, qwen2.5-14b)")
args = parser.parse_args()

slug    = args.llm_model
folder  = f"{args.group}_{slug}" if args.merged else args.group
BASE    = f"output/merge_predictions/{folder}" if args.merged else f"output/merge_multicite/{folder}"
OUT_DIR = os.path.join("output/analysis", args.group, slug if args.merged else "multicite")
os.makedirs(OUT_DIR, exist_ok=True)

ACL_ARC_TO_MULTICITE = {
    "BACKGROUND":          ["background"],
    "COMPARES_CONTRASTS":  ["similarities", "differences"],
    "EXTENSION":           ["extends"],
    "FUTURE":              ["future_work"],
    "MOTIVATION":          ["motivation"],
    "USES":                ["uses"],
}

EXPS = sorted(
    f.replace(".json", "")
    for f in os.listdir(BASE)
    if f.endswith(".json")
)
EXP_SHORT = [e.replace("non_contiguous_acl_arc_", "") for e in EXPS]

sns.set_theme(style="whitegrid", palette="muted")


# ── Metric functions ─────────────────────────────────────────────────────────

def load(exp):
    with open(os.path.join(BASE, f"{exp}.json")) as f:
        return json.load(f)


def containment(entry, pred_field="multicite_prediction"):
    preds = entry.get(pred_field)
    if not preds:
        return False
    valid = ACL_ARC_TO_MULTICITE.get(entry["acl_arc_label"], [])
    return any(v in preds for v in valid)


def preservation_mc(entry):
    """Fraction of baseline MultiCite labels retained in MOKA+MultiCite predictions."""
    baseline = entry.get("baseline_prediction")
    if not baseline:
        return None
    moka = set(entry.get("multicite_prediction", []))
    return len(set(baseline) & moka) / len(set(baseline))


def preservation_llm(entry):
    """Fraction of baseline LLM zero-shot labels retained in MOKA+LLM zero-shot predictions."""
    baseline = entry.get("baseline_llm_prediction")
    if not baseline:
        return None
    moka = set(entry.get("llm_prediction", []))
    return len(set(baseline) & moka) / len(set(baseline))


def preservation_llm_fewshot(entry):
    """Fraction of baseline LLM few-shot labels retained in MOKA+LLM few-shot predictions."""
    baseline = entry.get("baseline_llm_fewshot_prediction")
    if not baseline:
        return None
    moka = set(entry.get("llm_fewshot_prediction", []))
    return len(set(baseline) & moka) / len(set(baseline))


def human_jaccard(entry, pred_field="multicite_prediction"):
    """Jaccard similarity between predictions and human annotation labels."""
    ann = entry.get("human_annotation")
    if not ann:
        return None
    human = set(ann.get("labels", []))
    if not human:
        return None
    pred = set(entry.get(pred_field, []))
    union = human | pred
    if not union:
        return None
    return len(human & pred) / len(union)


def human_recall(entry, pred_field="multicite_prediction"):
    """Fraction of human labels captured by predictions (Recall@human)."""
    ann = entry.get("human_annotation")
    if not ann:
        return None
    human = set(ann.get("labels", []))
    if not human:
        return None
    pred = set(entry.get(pred_field, []))
    return len(human & pred) / len(human)


def mean_safe(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals) * 100, 1) if vals else None


# ── Build summary DataFrame ──────────────────────────────────────────────────
rows = []
for exp, short in zip(EXPS, EXP_SHORT):
    data = load(exp)
    n = len(data)
    n0 = sum(1 for d in data if len(d.get("multicite_prediction", [])) == 0)
    n1 = sum(1 for d in data if len(d.get("multicite_prediction", [])) == 1)
    n2 = sum(1 for d in data if len(d.get("multicite_prediction", [])) >= 2)

    has_llm       = any("llm_prediction" in d for d in data)
    has_llm_fs    = any("llm_fewshot_prediction" in d for d in data)
    has_bl_llm    = any("baseline_llm_prediction" in d for d in data)
    has_bl_llm_fs = any("baseline_llm_fewshot_prediction" in d for d in data)
    has_human     = any(d.get("human_annotation") for d in data)

    rows.append({
        "exp":   short,
        "total": n,
        "0_label": n0, "1_label": n1, "2_label": n2,
        # Containment vs ACL-ARC
        "containment_multicite":   round(sum(containment(d) for d in data) / n * 100, 1),
        "containment_llm":         round(sum(containment(d, "llm_prediction") for d in data) / n * 100, 1) if has_llm else None,
        "containment_llm_fewshot": round(sum(containment(d, "llm_fewshot_prediction") for d in data) / n * 100, 1) if has_llm_fs else None,
        # Preservation vs baseline
        "preservation_multicite":  mean_safe([preservation_mc(d) for d in data]),
        "preservation_llm":        mean_safe([preservation_llm(d) for d in data]) if has_bl_llm else None,
        "preservation_llm_fewshot": mean_safe([preservation_llm_fewshot(d) for d in data]) if has_bl_llm_fs else None,
        # Human agreement (Jaccard)
        "human_jaccard_multicite":    mean_safe([human_jaccard(d, "multicite_prediction") for d in data]) if has_human else None,
        "human_jaccard_llm":          mean_safe([human_jaccard(d, "llm_prediction") for d in data]) if has_human and has_llm else None,
        "human_jaccard_llm_fewshot":  mean_safe([human_jaccard(d, "llm_fewshot_prediction") for d in data]) if has_human and has_llm_fs else None,
        # Human agreement (Recall)
        "human_recall_multicite":     mean_safe([human_recall(d, "multicite_prediction") for d in data]) if has_human else None,
        "human_recall_llm":           mean_safe([human_recall(d, "llm_prediction") for d in data]) if has_human and has_llm else None,
        "human_recall_llm_fewshot":   mean_safe([human_recall(d, "llm_fewshot_prediction") for d in data]) if has_human and has_llm_fs else None,
    })

df = pd.DataFrame(rows)

# Print summary
print_cols = ["exp", "containment_multicite"]
for col in ["containment_llm", "containment_llm_fewshot",
            "preservation_multicite", "preservation_llm", "preservation_llm_fewshot",
            "human_jaccard_multicite", "human_jaccard_llm", "human_jaccard_llm_fewshot",
            "human_recall_multicite", "human_recall_llm", "human_recall_llm_fewshot"]:
    if df[col].notna().any():
        print_cols.append(col)

print(df[print_cols].to_string(index=False))
df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)


# ── Plot 1: Containment rate — MultiCite vs LLM per experiment ──────────────
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(df))
w = 0.25
ax.bar(x - w, df["containment_multicite"], w, label="MultiCite", color="#3498db")
if df["containment_llm"].notna().any():
    ax.bar(x,     df["containment_llm"],        w, label="LLM zero-shot", color="#e67e22")
if df["containment_llm_fewshot"].notna().any():
    ax.bar(x + w, df["containment_llm_fewshot"], w, label="LLM few-shot", color="#9b59b6")
ax.set_xticks(x)
ax.set_xticklabels(df["exp"], rotation=45, ha="right")
ax.set_ylim(0, 105)
ax.set_ylabel("Containment rate (%)")
ax.set_title(f"Containment rate vs ACL-ARC ground truth [{args.group}]")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "containment_rate.png"), dpi=150)
plt.close()


# ── Plot 2: Label count distribution (stacked bar) ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(df))
w = 0.6
ax.bar(x, df["0_label"], w, label="0 labels (none)", color="#e74c3c")
ax.bar(x, df["1_label"], w, bottom=df["0_label"], label="1 label", color="#3498db")
ax.bar(x, df["2_label"], w, bottom=df["0_label"] + df["1_label"], label="2+ labels", color="#2ecc71")
ax.set_xticks(x)
ax.set_xticklabels(df["exp"], rotation=45, ha="right")
ax.set_ylabel("Number of examples")
ax.set_title(f"MultiCite prediction label count distribution [{args.group}]")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "label_count_distribution.png"), dpi=150)
plt.close()


# ── Plot 3: Containment rate per MOKA class (MultiCite) ─────────────────────
class_hits = {c: [] for c in ACL_ARC_TO_MULTICITE}
for exp in EXPS:
    data = load(exp)
    by_class = {c: [] for c in ACL_ARC_TO_MULTICITE}
    for d in data:
        lbl = d["acl_arc_label"]
        if lbl in by_class:
            by_class[lbl].append(containment(d))
    for c in ACL_ARC_TO_MULTICITE:
        vals = by_class[c]
        class_hits[c].append(sum(vals) / len(vals) * 100 if vals else 0)

class_df = pd.DataFrame(class_hits, index=EXP_SHORT)
fig, ax = plt.subplots(figsize=(13, 6))
class_df.plot(kind="bar", ax=ax, edgecolor="white")
ax.set_ylabel("Containment rate (%)")
ax.set_title(f"MultiCite containment rate per ACL-ARC label class [{args.group}]")
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, 110)
ax.legend(title="ACL-ARC label", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "containment_by_class.png"), dpi=150)
plt.close()


# ── Plot 4: Predicted label frequency ───────────────────────────────────────
all_preds = []
for exp in EXPS:
    for d in load(exp):
        all_preds.extend(d.get("multicite_prediction", []))
pred_counter = Counter(all_preds)
labels_sorted = sorted(pred_counter, key=pred_counter.get, reverse=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(labels_sorted, [pred_counter[l] for l in labels_sorted],
       color=sns.color_palette("muted", len(labels_sorted)))
ax.set_ylabel("Total predictions (all experiments combined)")
ax.set_title(f"MultiCite predicted label frequency [{args.group}]")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "multicite_label_frequency.png"), dpi=150)
plt.close()


# ── Plot 5: Preservation rate — MultiCite vs LLM (zero-shot + few-shot) ─────
has_pres_mc    = df["preservation_multicite"].notna().any()
has_pres_llm   = df["preservation_llm"].notna().any()
has_pres_llm_fs = df["preservation_llm_fewshot"].notna().any()

bars = [(c, l, col) for c, l, col in [
    ("preservation_multicite",  "Preservation MultiCite",      "#3498db"),
    ("preservation_llm",        "Preservation LLM zero-shot",  "#e67e22"),
    ("preservation_llm_fewshot","Preservation LLM few-shot",   "#9b59b6"),
] if df[c].notna().any()]

if bars:
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(df))
    n = len(bars)
    w = 0.8 / n
    for i, (col, label, color) in enumerate(bars):
        offset = (i - (n - 1) / 2) * w
        ax.bar(x + offset, df[col], w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(df["exp"], rotation=45, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Preservation rate (%)")
    ax.set_title(f"Preservation rate vs baseline (fraction of baseline labels retained) [{args.group}]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "preservation_rate.png"), dpi=150)
    plt.close()


# ── Plot 6: Human agreement (Jaccard) — MultiCite vs LLM ────────────────────
has_hj_mc  = df["human_jaccard_multicite"].notna().any()
has_hj_llm = df["human_jaccard_llm"].notna().any()
has_hj_fs  = df["human_jaccard_llm_fewshot"].notna().any()

if has_hj_mc or has_hj_llm:
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(df))
    n_bars = sum([has_hj_mc, has_hj_llm, has_hj_fs])
    w = 0.25
    offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * w
    i = 0
    if has_hj_mc:
        ax.bar(x + offsets[i], df["human_jaccard_multicite"],   w, label="MultiCite",     color="#3498db"); i+=1
    if has_hj_llm:
        ax.bar(x + offsets[i], df["human_jaccard_llm"],         w, label="LLM zero-shot", color="#e67e22"); i+=1
    if has_hj_fs:
        ax.bar(x + offsets[i], df["human_jaccard_llm_fewshot"], w, label="LLM few-shot",  color="#9b59b6")
    ax.set_xticks(x)
    ax.set_xticklabels(df["exp"], rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Jaccard similarity (%) with human annotation")
    ax.set_title(f"Human agreement (Jaccard) [{args.group}]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "human_agreement_jaccard.png"), dpi=150)
    plt.close()


# ── Plot 7: Human recall — MultiCite vs LLM ─────────────────────────────────
has_hr_mc  = df["human_recall_multicite"].notna().any()
has_hr_llm = df["human_recall_llm"].notna().any()
has_hr_fs  = df["human_recall_llm_fewshot"].notna().any()

if has_hr_mc or has_hr_llm:
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(df))
    n_bars = sum([has_hr_mc, has_hr_llm, has_hr_fs])
    w = 0.25
    offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * w
    i = 0
    if has_hr_mc:
        ax.bar(x + offsets[i], df["human_recall_multicite"],   w, label="MultiCite",     color="#3498db"); i+=1
    if has_hr_llm:
        ax.bar(x + offsets[i], df["human_recall_llm"],         w, label="LLM zero-shot", color="#e67e22"); i+=1
    if has_hr_fs:
        ax.bar(x + offsets[i], df["human_recall_llm_fewshot"], w, label="LLM few-shot",  color="#9b59b6")
    ax.set_xticks(x)
    ax.set_xticklabels(df["exp"], rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Recall@human (%)")
    ax.set_title(f"Human recall (% of human labels captured) [{args.group}]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "human_agreement_recall.png"), dpi=150)
    plt.close()


# ── Best experiment per classifier ──────────────────────────────────────────
classifier_cols = {
    "MultiCite":     "containment_multicite",
    "LLM zero-shot": "containment_llm",
    "LLM few-shot":  "containment_llm_fewshot",
}

print("\n── Best experiments per classifier ──")
rank_rows = []
for clf_name, col in classifier_cols.items():
    if col not in df.columns or df[col].isna().all():
        continue
    ranked = df[["exp", col]].dropna().sort_values(col, ascending=False).reset_index(drop=True)
    ranked.index += 1
    ranked.columns = ["exp", "containment (%)"]
    print(f"\n{clf_name} (top 5):")
    print(ranked.head(5).to_string())
    for i, row in ranked.iterrows():
        rank_rows.append({"classifier": clf_name, "rank": i, "exp": row["exp"], "containment": row["containment (%)"]})

rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(os.path.join(OUT_DIR, "best_exp_per_classifier.csv"), index=False)

# ── Plot 8: Containment ranking line plot (exp on x ordered by MultiCite) ───
if not rank_df.empty and len(classifier_cols) > 1:
    mc_order = (
        df[["exp", "containment_multicite"]].dropna()
        .sort_values("containment_multicite", ascending=False)["exp"].tolist()
    )
    pivot = df.set_index("exp").reindex(mc_order)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"MultiCite": "#3498db", "LLM zero-shot": "#e67e22", "LLM few-shot": "#9b59b6"}
    for clf_name, col in classifier_cols.items():
        if col in pivot.columns and pivot[col].notna().any():
            ax.plot(range(len(mc_order)), pivot[col].values, marker="o",
                    label=clf_name, color=colors[clf_name], linewidth=1.5, markersize=4)
    ax.set_xticks(range(len(mc_order)))
    ax.set_xticklabels(mc_order, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Containment rate (%)")
    ax.set_title(f"Containment rate per experiment (sorted by MultiCite) [{args.group}]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "containment_ranking.png"), dpi=150)
    plt.close()

print(f"\nPlots saved to {OUT_DIR}/")

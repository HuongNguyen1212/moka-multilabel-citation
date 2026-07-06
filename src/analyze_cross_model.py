#!/usr/bin/env python3
"""
Aggregate results across all models and experiments (step 6b).

Outputs to output/analysis/cross_model/:
  preservation_by_group.csv    (Table 3)
  cce_jaccard_gains.csv        (Table 4)
  baseline_jaccard.csv         (Table 5)
  best_combination_3axis.csv   (Table 6)
  baseline_vs_cce.csv          (Table: baseline vs best CCE per classifier)
  avg_labels_per_model.png/csv (Fig 2)
  containment_by_label.png/csv (Fig 3)
"""

import json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

BASE = Path(__file__).parent.parent

VALID = ["background", "uses", "motivation", "extends",
         "similarities", "differences", "future_work"]

MODELS = [
    ("GPT-4o-mini",   "gpt-4o-mini"),
    ("Gemma2-9b",     "gemma2-9b"),
    ("Qwen2.5-14b",   "qwen2.5-14b"),
    ("LLaMA3.1-8b",   "llama3.1-8b"),
    ("Mistral-7b",    "mistral-7b"),
    ("Phi4-14b",      "phi4-14b"),
    ("Gemma3-4b",     "gemma3-4b"),
    ("Gemma3-12b",    "gemma3-12b"),
    ("Mistral-NeMo",  "mistral-nemo-12b"),
]

MODEL_GROUP = {
    "GPT-4o-mini":  "Proprietary API",
    "Gemma2-9b":    "Open-source",
    "Qwen2.5-14b":  "Open-source",
    "LLaMA3.1-8b":  "Open-source",
    "Mistral-7b":   "Open-source",
    "Phi4-14b":     "Open-source",
    "Gemma3-4b":    "Open-source",
    "Gemma3-12b":   "Open-source",
    "Mistral-NeMo": "Open-source",
}

EXPS_SCINCL  = [f"non_contiguous_acl_arc_exp{e}" for e in
    ["1","2","3","4","5a","5b","5c","5d","5e","5f","5g","5h","5i","5j","5k","5l","5m","5n"]]
EXPS_SPECTER = EXPS_SCINCL
EXPS_LLM     = [f"non_contiguous_acl_arc_exp{e}" for e in ["6","7","8","9","10"]]

MERGE_MULTICITE_DIR   = BASE / "output/merge_multicite"
MERGE_PREDICTIONS_DIR = BASE / "output/merge_predictions"
ANALYSIS_DIR          = BASE / "output/analysis"

ACL_ARC_TO_MULTICITE = {
    "BACKGROUND":          ["background"],
    "COMPARES_CONTRASTS":  ["similarities", "differences"],
    "EXTENSION":           ["extends"],
    "FUTURE":              ["future_work"],
    "MOTIVATION":          ["motivation"],
    "USES":                ["uses"],
}

MULTICITE_BASELINE_JACCARD    = None
MULTICITE_BASELINE_AVG_LABELS = None


def _compute_multicite_baseline_jaccard():
    f = MERGE_PREDICTIONS_DIR / "scincl_gemma3-12b" / "non_contiguous_acl_arc_exp1.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    def jac(a, b):
        sa, sb = set(a), set(b)
        if not sa and not sb: return 1.0
        return len(sa & sb) / len(sa | sb)
    scores = [jac(d.get("baseline_prediction", []),
                  d.get("human_annotation", {}).get("labels", [])) for d in data]
    return round(100 * sum(scores) / len(scores), 1)


def _compute_multicite_baseline_avg_labels():
    f = MERGE_PREDICTIONS_DIR / "scincl_gemma3-12b" / "non_contiguous_acl_arc_exp1.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    counts = [len(d.get("baseline_prediction", [])) for d in data]
    return round(sum(counts) / len(counts), 2)


def _contains(pred, acl_label):
    return any(v in (pred or []) for v in ACL_ARC_TO_MULTICITE.get(acl_label, []))


def load_gold():
    with open(BASE / "data/annotations/annotations_gold.json") as f:
        data = json.load(f)
    return {x["id"]: set(x["labels"]) for x in data["annotations"]}


def jaccard(pred, gold):
    p = set(pred) & set(VALID)
    g = set(gold) & set(VALID)
    if not p and not g: return 1.0
    if not p or  not g: return 0.0
    return len(p & g) / len(p | g)


def eval_file(path, gold):
    path = Path(path)
    if not path.exists(): return None
    with open(path) as f: data = json.load(f)
    items = [x for x in data if gold.get(x["unique_id"])]
    if not items: return None
    scores     = [jaccard(x.get("llm_prediction", []), gold[x["unique_id"]]) for x in items]
    avg_labels = sum(len(x.get("llm_prediction", [])) for x in data) / len(data)
    empty      = sum(1 for x in data if not x.get("llm_prediction"))
    return {
        "jaccard":    round(sum(scores) / len(scores) * 100, 2),
        "avg_labels": round(avg_labels, 2),
        "empty":      empty,
        "n":          len(data),
        "data":       data,
    }


def _get_ha(x):
    ha = x.get("human_annotation")
    if not ha: return None
    if isinstance(ha, list): return ha
    if isinstance(ha, dict): return ha.get("labels")
    return None


def load_all_results(gold):
    results = {}
    for name, slug in MODELS:
        results[name] = {}
        for mode, short in [("zeroshot", "ZS"), ("fewshot", "FS")]:
            r = eval_file(BASE / f"output/llm/{mode}/{slug}/baseline/baseline.json", gold)
            if r: results[name][f"baseline_{short}"] = r
            for group, exps in [("scincl", EXPS_SCINCL), ("specter", EXPS_SPECTER), ("LLM", EXPS_LLM)]:
                for exp in exps:
                    lbl = exp.replace("non_contiguous_acl_arc_", "")
                    r = eval_file(BASE / f"output/llm/{mode}/{slug}/{group}/{exp}.json", gold)
                    if r: results[name][f"{group}_{lbl}_{short}"] = r
    return results


def load_preservation_multicite():
    out = {}
    for group, exps in [("scincl", EXPS_SCINCL), ("specter", EXPS_SPECTER), ("LLM", EXPS_LLM)]:
        per_exp = {}
        for exp in exps:
            f = MERGE_MULTICITE_DIR / group / f"{exp}.json"
            if not f.exists(): continue
            data = json.load(open(f))
            vals = []
            for e in data:
                bl = e.get("baseline_prediction")
                if not bl: continue
                moka = set(e.get("multicite_prediction", []))
                vals.append(len(set(bl) & moka) / len(set(bl)))
            if vals:
                per_exp[exp.replace("non_contiguous_acl_arc_", "")] = round(100 * sum(vals) / len(vals), 1)
        out[group] = per_exp
    return out


def load_preservation_llm():
    out = {}
    for name, slug in MODELS:
        out[name] = {}
        for group in ["scincl", "specter", "LLM"]:
            f = ANALYSIS_DIR / group / slug / "summary.csv"
            if not f.exists(): continue
            df = pd.read_csv(f)
            if "preservation_llm" in df.columns and df["preservation_llm"].notna().any():
                out[name][group] = round(df["preservation_llm"].mean(), 1)
    return out


def table1b_preservation_overview(outdir):
    mc  = load_preservation_multicite()
    llm = load_preservation_llm()
    rows = []
    for group in ["scincl", "specter", "LLM"]:
        mc_vals  = list(mc.get(group, {}).values())
        llm_vals = [llm[name][group] for name, _ in MODELS if group in llm.get(name, {})]
        rows.append({
            "Group":                  group,
            "Preservation_MultiCite": round(np.mean(mc_vals), 1)  if mc_vals  else np.nan,
            "Preservation_LLM_avg":   round(np.mean(llm_vals), 1) if llm_vals else np.nan,
            "Preservation_LLM_min":   round(min(llm_vals), 1)     if llm_vals else np.nan,
            "Preservation_LLM_max":   round(max(llm_vals), 1)     if llm_vals else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "preservation_by_group.csv", index=False)
    print("\n[Table 3] Preservation by group saved.")
    print(df.to_string(index=False))
    return df


def table1_cce_overview(results, outdir):
    rows = []
    for group, exps in [("scincl", EXPS_SCINCL), ("specter", EXPS_SPECTER), ("LLM", EXPS_LLM)]:
        for short in ["ZS", "FS"]:
            gains, beats = [], []
            for name, _ in MODELS:
                bl = results[name].get(f"baseline_{short}", {}).get("jaccard", 0)
                if not bl: continue
                lbls = [e.replace("non_contiguous_acl_arc_", "") for e in exps]
                exp_scores = [results[name].get(f"{group}_{l}_{short}", {}).get("jaccard")
                              for l in lbls]
                exp_scores = [s for s in exp_scores if s is not None]
                if not exp_scores: continue
                best = max(exp_scores)
                gains.append(best - bl)
                beats.append(sum(1 for s in exp_scores if s > bl))
            if not gains: continue
            rows.append({
                "Group":           group,
                "Mode":            short,
                "Avg_best_gain":   round(np.mean(gains), 2),
                "Pct_models_beat": round(100 * sum(1 for g in gains if g > 0) / len(gains), 1),
                "Avg_exps_beat":   round(np.mean(beats), 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "cce_jaccard_gains.csv", index=False)
    print("\n[Table 4] CCE gains saved.")
    print(df.to_string(index=False))
    return df


def table2_baseline(results, outdir):
    mc_avg_lbl = MULTICITE_BASELINE_AVG_LABELS or 0.96
    rows = [{"Model": "MultiCite", "Group": "Supervised",
             "ZS": MULTICITE_BASELINE_JACCARD, "FS": "-",
             "avg_labels": mc_avg_lbl, "Best": MULTICITE_BASELINE_JACCARD}]
    for name, _ in MODELS:
        zs = results[name].get("baseline_ZS", {}).get("jaccard", 0)
        fs = results[name].get("baseline_FS", {}).get("jaccard", 0)
        al = results[name].get("baseline_ZS", {}).get("avg_labels", 0)
        rows.append({"Model": name, "Group": MODEL_GROUP[name],
                     "ZS": zs, "FS": fs, "avg_labels": al, "Best": round(max(zs, fs), 1)})
    df = pd.DataFrame(rows).sort_values("Best", ascending=False)
    df.to_csv(outdir / "baseline_jaccard.csv", index=False)
    print("\n[Table 5] Baseline Jaccard saved.")
    print(df[["Model", "Group", "ZS", "FS", "Best", "avg_labels"]].to_string(index=False))
    return df


def table4_best_combination(results, outdir):
    rows = []
    for name, _ in MODELS:
        bl_zs = results[name].get("baseline_ZS", {}).get("jaccard", 0)
        bl_fs = results[name].get("baseline_FS", {}).get("jaccard", 0)
        best_j, best_key = 0, ""
        for mode, short in [("zeroshot", "ZS"), ("fewshot", "FS")]:
            for group, exps in [("scincl", EXPS_SCINCL), ("specter", EXPS_SPECTER), ("LLM", EXPS_LLM)]:
                for exp in exps:
                    lbl = exp.replace("non_contiguous_acl_arc_", "")
                    j = results[name].get(f"{group}_{lbl}_{short}", {}).get("jaccard", 0)
                    if j > best_j:
                        best_j   = j
                        best_key = f"{group}/{lbl} ({short})"
        bl_best = max(bl_zs, bl_fs)
        rows.append({
            "Model":         name,
            "Baseline_best": round(bl_best, 1),
            "Best_CCE":      round(best_j, 1),
            "CCE_gain":      round(best_j - bl_best, 1),
            "Best_exp":      best_key,
        })

    mc_baseline = MULTICITE_BASELINE_JACCARD or 48.3
    mc_best_j, mc_best_key = 0, ""
    for group, exps in [("scincl", EXPS_SCINCL), ("specter", EXPS_SPECTER), ("LLM", EXPS_LLM)]:
        f = ANALYSIS_DIR / group / "gemma3-12b" / "summary.csv"
        if not f.exists(): continue
        sdf = pd.read_csv(f)
        for _, row in sdf.iterrows():
            j = row.get("human_jaccard_multicite", 0)
            if j > mc_best_j:
                mc_best_j   = j
                mc_best_key = f"{group}/{row['exp']}"
    rows.append({
        "Model":         "MultiCite",
        "Baseline_best": round(mc_baseline, 1),
        "Best_CCE":      round(mc_best_j, 1),
        "CCE_gain":      round(mc_best_j - mc_baseline, 1),
        "Best_exp":      mc_best_key,
    })

    df = pd.DataFrame(rows).sort_values("Best_CCE", ascending=False)
    df.to_csv(outdir / "best_combination_per_model.csv", index=False)
    print("\n[Table 6 intermediate] Best combination per model saved.")
    print(df.to_string(index=False))
    return df


def table_baseline_vs_cce(t4_df, outdir, results=None, gold=None):
    slug_of  = {name: slug for name, slug in MODELS}
    type_map = {n: ("Proprietary" if MODEL_GROUP[n] == "Proprietary API" else "Open-source")
                for n, _ in MODELS}
    type_map["MultiCite"] = "Supervised"

    avg_map = {}
    if results:
        for name, _ in MODELS:
            vals = []
            for _mode, short in [("zeroshot", "ZS"), ("fewshot", "FS")]:
                for group, exps in [("scincl", EXPS_SCINCL), ("specter", EXPS_SPECTER), ("LLM", EXPS_LLM)]:
                    for exp in exps:
                        lbl = exp.replace("non_contiguous_acl_arc_", "")
                        j = results[name].get(f"{group}_{lbl}_{short}", {}).get("jaccard", None)
                        if j is not None:
                            vals.append(j)
            avg_map[name] = round(sum(vals) / len(vals), 1) if vals else None

    mc_vals = []
    for group, exps in [("scincl", EXPS_SCINCL), ("specter", EXPS_SPECTER), ("LLM", EXPS_LLM)]:
        f = ANALYSIS_DIR / group / "gemma3-12b" / "summary.csv"
        if not f.exists():
            continue
        sdf = pd.read_csv(f)
        mc_vals.extend(sdf["human_jaccard_multicite"].dropna().tolist())
    avg_map["MultiCite"] = round(sum(mc_vals) / len(mc_vals), 1) if mc_vals else None

    rows = []
    for _, r in t4_df.iterrows():
        name     = r["Model"]
        best_exp = r["Best_exp"]
        sig, pval = "n/a", None
        if gold:
            if name == "MultiCite":
                try:
                    _, lbl = best_exp.split("/")
                except ValueError:
                    lbl = None
                if lbl:
                    bl_s, cce_s = _per_item_jaccard_multicite(gold, lbl)
                    if bl_s:
                        sig, pval = _wilcoxon_sig(bl_s, cce_s)
            else:
                try:
                    group_lbl, short = best_exp.split(" (")
                    group, lbl = group_lbl.split("/")
                    short = short.rstrip(")")
                except (ValueError, AttributeError):
                    group = lbl = short = None
                if group and lbl and short:
                    bl_s, cce_s = _per_item_jaccard_llm(gold, slug_of[name], group, lbl, short)
                    if bl_s:
                        sig, pval = _wilcoxon_sig(bl_s, cce_s)
        rows.append({
            "Model":    name,
            "Sig.":     sig,
            "p_value":  pval,
        })

    sig_df = pd.DataFrame(rows).set_index("Model")

    df = t4_df.copy().set_index("Model")
    df["Type"]                = pd.Series(type_map)
    df["Avg CCE Jaccard (%)"] = pd.Series(avg_map)
    df["Sig."]                = sig_df["Sig."]
    df["p_value"]             = sig_df["p_value"]
    df = df.reset_index().rename(columns={
        "Model":         "Classifier",
        "Baseline_best": "Baseline Jaccard (%)",
        "Best_CCE":      "Best CCE Jaccard (%)",
        "CCE_gain":      "Δ Best (pp)",
        "Best_exp":      "Best CCE Config",
    })
    df["Δ Avg (pp)"] = (df["Avg CCE Jaccard (%)"] - df["Baseline Jaccard (%)"]).round(1)
    df = df[["Classifier", "Type", "Baseline Jaccard (%)", "Best CCE Jaccard (%)", "Δ Best (pp)",
             "Avg CCE Jaccard (%)", "Δ Avg (pp)", "Sig.", "p_value", "Best CCE Config"]]
    df = df.sort_values("Best CCE Jaccard (%)", ascending=False)
    df.to_csv(outdir / "baseline_vs_cce.csv", index=False)
    print("\n[Table] Baseline vs CCE saved.")
    print(df.to_string(index=False))


def _jac(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    return len(sa & sb) / len(sa | sb)


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _wilcoxon_sig(baseline_scores, cce_scores):
    diffs = [c - b for c, b in zip(cce_scores, baseline_scores)]
    if all(d == 0 for d in diffs):
        return "ns", 1.0
    try:
        _, p = wilcoxon(cce_scores, baseline_scores, alternative="greater", zero_method="wilcox")
        return _sig_stars(p), round(p, 4)
    except Exception:
        return "ns", None


def _per_item_jaccard_llm(gold, slug, group, exp, short):
    field    = "llm_prediction"              if short == "ZS" else "llm_fewshot_prediction"
    bl_field = "baseline_llm_prediction"     if short == "ZS" else "baseline_llm_fewshot_prediction"
    p = MERGE_PREDICTIONS_DIR / f"{group}_{slug}" / f"non_contiguous_acl_arc_{exp}.json"
    if not p.exists(): return None, None
    data  = json.loads(p.read_text())
    by_id = {x["unique_id"]: x for x in data}
    cce_scores, bl_scores = [], []
    for uid, g in gold.items():
        item = by_id.get(uid)
        if not item: continue
        cce_scores.append(_jac(item.get(field, []), g))
        bl_scores.append(_jac(item.get(bl_field, []), g))
    return bl_scores, cce_scores


def _per_item_jaccard_multicite(gold, exp):
    p = MERGE_MULTICITE_DIR / "scincl" / f"non_contiguous_acl_arc_{exp}.json"
    if not p.exists(): return None, None
    data  = json.loads(p.read_text())
    by_id = {x["unique_id"]: x for x in data}
    cce_scores, bl_scores = [], []
    for uid, g in gold.items():
        item = by_id.get(uid)
        if not item: continue
        cce_scores.append(_jac(item.get("multicite_prediction", []), g))
        bl_scores.append(_jac(item.get("baseline_prediction", []), g))
    return bl_scores, cce_scores


def table4b_combined_recommendation(table4_df, outdir):
    slug_of = {name: slug for name, slug in MODELS}
    pres_mc = load_preservation_multicite()

    rows = []
    for _, r in table4_df.iterrows():
        name = r["Model"]

        if name == "MultiCite":
            best_exp = r["Best_exp"]
            try:
                group, lbl = best_exp.split("/")
            except (ValueError, AttributeError):
                group, lbl = None, None
            containment = preservation_mc = preservation_llm = None
            if group and lbl:
                sdf_path = ANALYSIS_DIR / group / "gemma3-12b" / "summary.csv"
                if sdf_path.exists():
                    sdf     = pd.read_csv(sdf_path)
                    row_mc  = sdf[sdf.exp == lbl]
                    if not row_mc.empty:
                        containment     = round(row_mc["containment_multicite"].values[0], 1)
                        preservation_mc = round(row_mc["preservation_multicite"].values[0], 1)
                        pres_llm_vals = []
                        for _, slug in MODELS:
                            f_m = ANALYSIS_DIR / group / slug / "summary.csv"
                            if not f_m.exists(): continue
                            df_m  = pd.read_csv(f_m)
                            row_m = df_m[df_m.exp == lbl]
                            if not row_m.empty:
                                pres_llm_vals.append(row_m["preservation_llm"].values[0])
                        preservation_llm = round(np.mean(pres_llm_vals), 1) if pres_llm_vals else None
            rows.append({
                "Model":                        name,
                "Best_config":                  best_exp,
                "Jaccard_human (Axis3)":         r["Best_CCE"],
                "Containment_ACLARC (Axis1)":    containment,
                "Preservation_MultiCite (Axis2)": preservation_mc,
                "Preservation_LLM (Axis2)":      preservation_llm,
            })
            continue

        slug = slug_of[name]
        try:
            group_lbl, short = r["Best_exp"].split(" (")
            group, lbl = group_lbl.split("/")
            short = short.rstrip(")")
        except (ValueError, AttributeError):
            group, lbl, short = None, None, None

        containment = preservation_mc = preservation_llm = None
        if group:
            preservation_mc = pres_mc.get(group, {}).get(lbl)
            pres_col = "preservation_llm"        if short == "ZS" else "preservation_llm_fewshot"
            cont_col = "containment_llm"         if short == "ZS" else "containment_llm_fewshot"
            f = ANALYSIS_DIR / group / slug / "summary.csv"
            if f.exists():
                sdf = pd.read_csv(f)
                row = sdf[sdf.exp == lbl]
                if not row.empty:
                    if pres_col in row.columns:
                        preservation_llm = round(row[pres_col].values[0], 1)
                    if cont_col in row.columns:
                        containment = round(row[cont_col].values[0], 1)

        rows.append({
            "Model":                        name,
            "Best_config":                  r["Best_exp"],
            "Jaccard_human (Axis3)":         r["Best_CCE"],
            "Containment_ACLARC (Axis1)":    containment,
            "Preservation_MultiCite (Axis2)": preservation_mc,
            "Preservation_LLM (Axis2)":      preservation_llm,
        })

    df = pd.DataFrame(rows)
    llm_rows = df[df["Model"] != "MultiCite"].sort_values("Jaccard_human (Axis3)", ascending=False)
    mc_rows  = df[df["Model"] == "MultiCite"]
    df = pd.concat([llm_rows, mc_rows], ignore_index=True)
    df.to_csv(outdir / "best_combination_3axis.csv", index=False)
    print("\n[Table 6] Combined 3-axis recommendation saved.")
    print(df.to_string(index=False))
    return df


def fig_per_label_containment(outdir):
    acl_labels = list(ACL_ARC_TO_MULTICITE.keys())
    mc_counts  = {l: [] for l in acl_labels}
    llm_zs     = {l: [] for l in acl_labels}
    llm_fs     = {l: [] for l in acl_labels}

    all_exps = (
        [("scincl",   e) for e in EXPS_SCINCL] +
        [("specter",  e) for e in EXPS_SPECTER] +
        [("LLM",      e) for e in EXPS_LLM]
    )

    # MultiCite: average multicite_prediction across all 41 CCE experiments
    for group, exp in all_exps:
        f = MERGE_MULTICITE_DIR / group / f"{exp}.json"
        if not f.exists():
            continue
        for e in json.load(open(f)):
            al = e.get("acl_arc_label")
            if al not in acl_labels:
                continue
            mc_counts[al].append(_contains(e.get("multicite_prediction"), al))

    # LLM: average llm_prediction/llm_fewshot_prediction across all 41 exps and 9 models
    for _name, slug in MODELS:
        for group, exp in all_exps:
            f = MERGE_PREDICTIONS_DIR / f"{group}_{slug}" / f"{exp}.json"
            if not f.exists():
                continue
            for e in json.load(open(f)):
                al = e.get("acl_arc_label")
                if al not in acl_labels:
                    continue
                llm_zs[al].append(_contains(e.get("llm_prediction"), al))
                llm_fs[al].append(_contains(e.get("llm_fewshot_prediction"), al))

    mc_rates = [100 * np.mean(mc_counts[l]) if mc_counts[l] else 0 for l in acl_labels]
    zs_rates = [100 * np.mean(llm_zs[l])    if llm_zs[l]   else 0 for l in acl_labels]
    fs_rates = [100 * np.mean(llm_fs[l])    if llm_fs[l]   else 0 for l in acl_labels]

    pd.DataFrame({
        "ACL_ARC_label": acl_labels,
        "MultiCite":     [round(r, 1) for r in mc_rates],
        "LLM_avg_ZS":    [round(r, 1) for r in zs_rates],
        "LLM_avg_FS":    [round(r, 1) for r in fs_rates],
    }).to_csv(outdir / "containment_by_label.csv", index=False)

    x, w = np.arange(len(acl_labels)), 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, mc_rates, w, label="MultiCite",    color="#009E73", alpha=0.88, edgecolor="black", lw=0.5)
    ax.bar(x,     zs_rates, w, label="LLM avg (ZS)", color="#0072B2", alpha=0.88, edgecolor="black", lw=0.5)
    ax.bar(x + w, fs_rates, w, label="LLM avg (FS)", color="#E69F00", alpha=0.88, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in acl_labels], fontsize=9)
    ax.set_ylabel("Containment rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "containment_by_label.png", dpi=150)
    plt.close()
    print("[Fig 3] Per-label containment saved.")


def fig_multilabel_rate(results, outdir):
    model_names = [n for n, _ in MODELS]
    zs_vals = [results[n].get("baseline_ZS", {}).get("avg_labels", 0) for n in model_names]
    fs_vals = [results[n].get("baseline_FS", {}).get("avg_labels", 0) for n in model_names]
    order   = sorted(range(len(model_names)), key=lambda i: max(zs_vals[i], fs_vals[i]), reverse=True)
    names_s = [model_names[i] for i in order]
    zs_s    = [zs_vals[i]     for i in order]
    fs_s    = [fs_vals[i]     for i in order]

    pd.DataFrame({
        "Model":                names_s,
        "ZS_avg_labels":        [round(v, 2) for v in zs_s],
        "FS_avg_labels":        [round(v, 2) for v in fs_s],
        "MultiCite_avg_labels": MULTICITE_BASELINE_AVG_LABELS or 0.96,
    }).to_csv(outdir / "avg_labels_per_model.csv", index=False)

    mc_avg = MULTICITE_BASELINE_AVG_LABELS or 0.96
    x, w = np.arange(len(names_s)), 0.35
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w/2, zs_s, w, label="Zero-shot", color="#0072B2", alpha=0.85, edgecolor="black", lw=0.5)
    ax.bar(x + w/2, fs_s, w, label="Few-shot",  color="#E69F00", alpha=0.85, edgecolor="black", lw=0.5)
    ax.axhline(mc_avg, color="#009E73", linestyle="--", lw=1.5, label=f"MultiCite avg ({mc_avg})")
    ax.set_xticks(x)
    ax.set_xticklabels(names_s, rotation=25, ha="right")
    ax.set_ylabel("Avg labels per prediction")
    ax.set_ylim(0, 4.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "avg_labels_per_model.png", dpi=150)
    plt.close()
    print("[Fig 2] Avg labels per model saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/analysis/cross_model")
    args   = parser.parse_args()
    outdir = BASE / args.output
    outdir.mkdir(parents=True, exist_ok=True)

    global MULTICITE_BASELINE_JACCARD, MULTICITE_BASELINE_AVG_LABELS
    MULTICITE_BASELINE_JACCARD    = _compute_multicite_baseline_jaccard()
    MULTICITE_BASELINE_AVG_LABELS = _compute_multicite_baseline_avg_labels()
    print(f"MultiCite baseline Jaccard: {MULTICITE_BASELINE_JACCARD}%")
    print(f"MultiCite baseline avg labels: {MULTICITE_BASELINE_AVG_LABELS}")

    gold    = load_gold()
    results = load_all_results(gold)

    table1b_preservation_overview(outdir)
    table1_cce_overview(results, outdir)
    table2_baseline(results, outdir)
    t4_df = table4_best_combination(results, outdir)
    table4b_combined_recommendation(t4_df, outdir)
    table_baseline_vs_cce(t4_df, outdir, results=results, gold=gold)
    fig_multilabel_rate(results, outdir)
    fig_per_label_containment(outdir)

    print(f"\nAll done → {outdir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Classify CCE citation contexts using the OpenAI API (step 3a, multi-label).

Usage:
  python llm_classify.py --group scincl --exp non_contiguous_acl_arc_exp1
  python llm_classify.py --group scincl --all
  python llm_classify.py --group specter --all
  python llm_classify.py --group LLM --all
  python llm_classify.py --group scincl --all --few-shot
"""

import argparse
import ast
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

DEFAULT_MODEL = "openai/gpt-4o-mini"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def model_slug(model: str) -> str:
    """openai/gpt-4o-mini → gpt-4o-mini, qwen2.5:14b → qwen2.5-14b, models/gemini-2.5-flash → gemini-2.5-flash"""
    return model.split("/")[-1].replace(":", "-")

MULTICITE_LABELS = [
    "background", "uses", "motivation", "extends",
    "similarities", "differences", "future_work",
]

ACL_ARC_LABEL_NAMES = {
    "0": "BACKGROUND", "1": "COMPARES_CONTRASTS", "2": "EXTENSION",
    "3": "FUTURE",     "4": "MOTIVATION",         "5": "USES",
}

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

SYSTEM_PROMPT = """You are an expert in scientific citation classification.

Given a citation context from an academic paper, your task is to classify the citation intent of the work marked by #AUTHOR_TAG to understand why the author cites this work. The citation may serve multiple purposes. Assign one or more of the following labels:

- background: The cited work is presented as prior or related work, providing context for the field. Only assign this if the citation explicitly serves as background, not simply because the surrounding paragraph discusses the field in general.
- uses: The citing paper uses the methodology, tools, or data created by the cited paper.
- motivation: The citing paper is directly motivated or inspired by the cited paper.
- extends: The citing paper extends, improves, or builds upon the methods or findings of the cited paper.
- similarities: The citing paper expresses similarities or comparable results to the cited paper.
- differences: The citing paper expresses differences from, contrasts with, or disagrees with the cited paper.
- future_work: The cited paper is suggested as a potential avenue for future work.

Respond ONLY with a JSON object in this exact format:
{"labels": ["label1", "label2"]}

Use only the labels listed above."""

FEW_SHOT_EXAMPLES = [
    {
        "citation_sentence": "Several approaches have been proposed for machine translation, including phrase-based systems ( #AUTHOR_TAG ) and neural models .",
        "context": "Machine translation has evolved significantly over the past decade. Several approaches have been proposed for machine translation, including phrase-based systems ( #AUTHOR_TAG ) and neural models . Our work builds on this body of literature to address low-resource translation scenarios.",
        "labels": ["background"],
    },
    {
        "citation_sentence": "The findings of #AUTHOR_TAG directly motivated our work ; we adopt their evaluation framework and extend it to the multi-label setting .",
        "context": "Prior studies have highlighted the limitations of single-label citation classification. The findings of #AUTHOR_TAG directly motivated our work ; we adopt their evaluation framework and extend it to the multi-label setting . Their analysis showed that citation behavior is inherently multi-faceted.",
        "labels": ["motivation", "uses", "extends"],
    },
    {
        "citation_sentence": "Similar to #AUTHOR_TAG , our model uses contextual embeddings , but we show it generalises better to out-of-domain data .",
        "context": "Contextual representations have become the standard for text classification tasks. Similar to #AUTHOR_TAG , our model uses contextual embeddings , but we show it generalises better to out-of-domain data . This suggests that our training strategy leads to more robust representations.",
        "labels": ["similarities", "differences"],
    },
    {
        "citation_sentence": "We leave the adaptation of #AUTHOR_TAG 's framework to low-resource languages as an avenue for future investigation .",
        "context": "Our study focuses on high-resource languages where sufficient annotated data is available. We leave the adaptation of #AUTHOR_TAG 's framework to low-resource languages as an avenue for future investigation . Addressing this gap would significantly broaden the applicability of citation analysis tools.",
        "labels": ["future_work"],
    },
    {
        "citation_sentence": "We adopt the annotation guidelines of #AUTHOR_TAG and use their publicly available dataset to train and evaluate our classifier .",
        "context": "Reproducibility and comparability require the use of standardised annotation schemes. We adopt the annotation guidelines of #AUTHOR_TAG and use their publicly available dataset to train and evaluate our classifier . This choice enables direct comparison with previously reported results.",
        "labels": ["background", "uses"],
    },
]


def load_baseline_test() -> list[dict]:
    """Load test data using cite_context_paragraph (full paragraph) as context."""
    path = Path(__file__).parent.parent / "data/moka/scincl/non_contiguous_acl_arc_exp1/test.txt"
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw = row.get("cite_context_paragraph", "")
            try:
                context_sentences = ast.literal_eval(raw)
                if not isinstance(context_sentences, list):
                    context_sentences = [str(context_sentences)]
            except Exception:
                context_sentences = [raw]
            rows.append({
                "unique_id":        row["unique_id"],
                "citation_context": row["citation_context"],
                "context_para":     context_sentences,
                "moka_label":       ACL_ARC_LABEL_NAMES.get(row["citation_class_label"], row["citation_class_label"]),
            })
    return rows


def load_moka_test(group: str, exp: str) -> list[dict]:
    base = Path(__file__).parent.parent / "data/moka" / group / exp
    rows = []

    # LLM group uses JSON format (orig_id = row index into test.txt)
    json_path = base / "test_citation_context.json"
    if json_path.exists():
        # Build index: row position -> unique_id and moka_label from test.txt
        id_map = {}
        tsv_path = base / "test.txt"
        if tsv_path.exists():
            with open(tsv_path, encoding="utf-8") as f:
                for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                    id_map[i] = {
                        "unique_id":  row["unique_id"],
                        "moka_label": ACL_ARC_LABEL_NAMES.get(row["citation_class_label"], "?"),
                        "citation_context": row.get("citation_context", ""),
                    }

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            idx = item.get("orig_id", item["row_id"])
            meta = id_map.get(idx, {})
            context = item.get("context_paragraph", [])
            if isinstance(context, str):
                context = [context]
            citation_sent = meta.get("citation_context", "")
            if not citation_sent:
                for s in (item.get("analysis_result") or []):
                    if "citation" in s.get("role", "").lower():
                        citation_sent = s.get("sentence_content", "")
                        break
            rows.append({
                "unique_id":        meta.get("unique_id", str(idx)),
                "citation_context": citation_sent,
                "context_para":     context,
                "moka_label":       meta.get("moka_label", "?"),
            })
        return rows

    # scincl / specter use TSV format
    tsv_path = base / "test.txt"
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw = row.get("dynamic_contexts_combined") or row.get("cite_context_paragraph", "")
            try:
                context_sentences = ast.literal_eval(raw)
                if not isinstance(context_sentences, list):
                    context_sentences = [str(context_sentences)]
            except Exception:
                context_sentences = [raw]
            rows.append({
                "unique_id":        row["unique_id"],
                "citation_context": row["citation_context"],
                "context_para":     context_sentences,
                "moka_label":       ACL_ARC_LABEL_NAMES.get(row["citation_class_label"], row["citation_class_label"]),
            })
    return rows


def build_user_msg(row: dict) -> str:
    context_text = " ".join(row["context_para"])
    return (
        f"Citation sentence: {row['citation_context']}\n\n"
        f"Surrounding context:\n{context_text}\n\n"
        f"Classify this citation. Assign one or more labels."
    )


def parse_labels(text: str) -> list[str]:
    """Robustly extract labels from LLM response.

    Handles: <think>...</think> prefix (deepseek-r1), markdown ```json blocks,
    plain JSON, and fallback keyword scan.
    """
    import re
    # Strip deepseek-r1 thinking block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Extract from markdown code block if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Try JSON parse
    try:
        data = json.loads(text)
        labels = data.get("labels", [])
        return [l for l in labels if l in MULTICITE_LABELS]
    except Exception:
        pass
    # Fallback: scan for any valid label names in the text
    found = [l for l in MULTICITE_LABELS if l in text]
    return found


def classify_one(client: OpenAI, row: dict, model: str, few_shot: bool = False) -> list[str]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    examples = FEW_SHOT_EXAMPLES if few_shot else []
    if examples:
        for ex in examples:
            ex_msg = (
                f"Citation sentence: {ex['citation_sentence']}\n\n"
                f"Surrounding context:\n{ex['context']}\n\n"
                f"Classify this citation. Assign one or more labels."
            )
            messages.append({"role": "user",      "content": ex_msg})
            messages.append({"role": "assistant",  "content": json.dumps({"labels": ex["labels"]})})

    messages.append({"role": "user", "content": build_user_msg(row)})

    kwargs = dict(model=model, messages=messages, max_tokens=512, temperature=0)
    # json_object mode: supported by OpenRouter, Ollama, and Gemini OpenAI-compatible endpoint.
    # Skip for deepseek-r1 (thinking model).
    if "deepseek" not in model.lower():
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content.strip()
    return parse_labels(text)


def run_experiment(group: str, exp: str, model: str, base_url: str = "https://api.openai.com/v1",
                   resume: bool = True, few_shot: bool = False, baseline: bool = False,
                   retry_empty: bool = False):
    mode_dir = "fewshot" if few_shot else "zeroshot"
    slug = model_slug(model)

    if baseline:
        out_dir = Path(f"output/llm/{mode_dir}/{slug}/baseline")
        out_path = out_dir / "baseline.json"
    else:
        out_dir = Path(f"output/llm/{mode_dir}/{slug}/{group}")
        out_path = out_dir / f"{exp}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = {}
    if resume and out_path.exists():
        with open(out_path) as f:
            for item in json.load(f):
                existing[item["unique_id"]] = item

    rows   = load_baseline_test() if baseline else load_moka_test(group, exp)

    if "generativelanguage" in base_url:
        api_key = os.getenv("GOOGLE_API_KEY", "")
    elif "openrouter" in base_url:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
    elif "openai.com" in base_url:
        api_key = os.getenv("OPENAI_API_KEY", "")
    else:
        api_key = "ollama"

    client = OpenAI(api_key=api_key, base_url=base_url)
    results = []
    n_cached = n_new = 0

    mode = "few-shot" if few_shot else "zero-shot"
    label = "baseline" if baseline else f"{group} / {exp}"
    print(f"\n{label}  ({len(rows)} examples, model={model}, {mode})")

    for i, row in enumerate(rows):
        uid = row["unique_id"]

        if uid in existing:
            if retry_empty and existing[uid].get("llm_prediction") == []:
                pass  # retry this sample
            else:
                results.append(existing[uid])
                n_cached += 1
                continue

        try:
            pred_labels = classify_one(client, row, model, few_shot=few_shot)
        except Exception as e:
            print(f"  [ERROR] {uid}: {e}")
            pred_labels = []
            time.sleep(5)

        results.append({
            "unique_id":        uid,
            "citation_context": row["citation_context"],
            "llm_prediction":   pred_labels,
        })
        n_new += 1

        if n_new % 20 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{i+1}/{len(rows)}] saved ({n_new} new, {n_cached} cached)")

        time.sleep(0.1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Done: {n_new} new + {n_cached} cached → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["scincl", "specter", "LLM"],
                        help="Experiment group to run (not needed with --baseline)")
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--exp",      help="Single experiment name")
    mutex.add_argument("--all",      action="store_true", help="Run all experiments in the group")
    mutex.add_argument("--baseline", action="store_true", help="Run on full cite_context_paragraph (baseline)")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="https://api.openai.com/v1",
                        help="OpenAI-compatible API base URL (e.g. http://localhost:11434/v1 for Ollama)")
    parser.add_argument("--few-shot", action="store_true",
                        help="5 diverse examples in prompt (mix of single and multi-label)")
    parser.add_argument("--no-resume",    action="store_true")
    parser.add_argument("--retry-empty",  action="store_true",
                        help="Retry samples that previously returned empty predictions (e.g. after a 500 error)")
    args = parser.parse_args()

    if args.baseline:
        run_experiment("baseline", "baseline", model=args.model, base_url=args.base_url,
                       resume=not args.no_resume, few_shot=args.few_shot, baseline=True,
                       retry_empty=args.retry_empty)
    else:
        if not args.group:
            parser.error("--group is required when not using --baseline")
        exps = EXPS_BY_GROUP[args.group] if args.all else [args.exp]
        for exp in exps:
            run_experiment(args.group, exp, model=args.model, base_url=args.base_url,
                           resume=not args.no_resume, few_shot=args.few_shot,
                           retry_empty=args.retry_empty)
    print("\nAll done.")


if __name__ == "__main__":
    main()

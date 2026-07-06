#!/usr/bin/env bash
# Run multicite inference on baseline + all acl_arc experiments
# Groups: scincl (exp1-5n), specter (exp1-5n), LLM (exp6-10)
# Uses pretrained allenai/multicite-multilabel-scibert from HuggingFace

set -euo pipefail

MODEL="${1:-allenai/multicite-multilabel-scibert}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EXPS_SCINCL=(
  non_contiguous_acl_arc_exp1  non_contiguous_acl_arc_exp2
  non_contiguous_acl_arc_exp3  non_contiguous_acl_arc_exp4
  non_contiguous_acl_arc_exp5a non_contiguous_acl_arc_exp5b
  non_contiguous_acl_arc_exp5c non_contiguous_acl_arc_exp5d
  non_contiguous_acl_arc_exp5e non_contiguous_acl_arc_exp5f
  non_contiguous_acl_arc_exp5g non_contiguous_acl_arc_exp5h
  non_contiguous_acl_arc_exp5i non_contiguous_acl_arc_exp5j
  non_contiguous_acl_arc_exp5k non_contiguous_acl_arc_exp5l
  non_contiguous_acl_arc_exp5m non_contiguous_acl_arc_exp5n
)

EXPS_SPECTER=(
  non_contiguous_acl_arc_exp1  non_contiguous_acl_arc_exp2
  non_contiguous_acl_arc_exp3  non_contiguous_acl_arc_exp4
  non_contiguous_acl_arc_exp5a non_contiguous_acl_arc_exp5b
  non_contiguous_acl_arc_exp5c non_contiguous_acl_arc_exp5d
  non_contiguous_acl_arc_exp5e non_contiguous_acl_arc_exp5f
  non_contiguous_acl_arc_exp5g non_contiguous_acl_arc_exp5h
  non_contiguous_acl_arc_exp5i non_contiguous_acl_arc_exp5j
  non_contiguous_acl_arc_exp5k non_contiguous_acl_arc_exp5l
  non_contiguous_acl_arc_exp5m non_contiguous_acl_arc_exp5n
)

EXPS_LLM=(
  non_contiguous_acl_arc_exp6  non_contiguous_acl_arc_exp7
  non_contiguous_acl_arc_exp8  non_contiguous_acl_arc_exp9
  non_contiguous_acl_arc_exp10
)

TOTAL=$(( 1 + ${#EXPS_SCINCL[@]} + ${#EXPS_SPECTER[@]} + ${#EXPS_LLM[@]} ))

run_inference() {
    local DATA_DIR="$1"
    local OUTPUT_DIR="$2"
    mkdir -p "${OUTPUT_DIR}"
    echo "  data : ${DATA_DIR}"
    echo "  out  : ${OUTPUT_DIR}"
    PYTHONPATH="${ROOT_DIR}/external" python3 external/classification/run_citation_classification.py \
        --model_name_or_path "${MODEL}" \
        --model_type bert \
        --task_name ours \
        --do_test \
        --data_dir "${DATA_DIR}" \
        --max_seq_length 512 \
        --per_gpu_train_batch_size 8 \
        --output_dir "${OUTPUT_DIR}" \
        --classification_type multilabel \
        --overwrite_cache \
        --overwrite_output_dir \
        --save_steps -1
}

echo "Model : ${MODEL}"
echo "Total : 1 baseline + ${TOTAL} experiments"
echo "=========================================="

# Baseline: full cite_context_paragraph (upper bound reference)
echo ""
echo "baseline (cite_context_paragraph)"
run_inference "data/converted/baseline" "output/multicite/baseline"

# scincl experiments
for EXP in "${EXPS_SCINCL[@]}"; do
    echo ""
    echo "scincl / ${EXP}"
    run_inference "data/converted/scincl/${EXP}" "output/multicite/scincl/${EXP}"
done

# specter experiments
for EXP in "${EXPS_SPECTER[@]}"; do
    echo ""
    echo "specter / ${EXP}"
    run_inference "data/converted/specter/${EXP}" "output/multicite/specter/${EXP}"
done

# LLM experiments
for EXP in "${EXPS_LLM[@]}"; do
    echo ""
    echo "LLM / ${EXP}"
    run_inference "data/converted/LLM/${EXP}" "output/multicite/LLM/${EXP}"
done

echo ""
echo "=========================================="
echo "Done. Results in output/multicite/"

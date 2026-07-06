#!/bin/bash
# Run all LLM classification from scratch (new prompt)
# Order: gpt-4o-mini → gemma2 → qwen2.5 → llama → mistral → phi4 → gemma3 → gemini-2.0-flash
# deepseek-r1:8b excluded: 32.7% empty predictions due to Ollama context length crashes
# Gemini: requires GOOGLE_API_KEY in .env (free tier: 1,500 req/day via aistudio.google.com)

set -e
cd /home/thu-huong-nguyen/WORK/MSCA/moka-multilabel-citation
source .venv/bin/activate

OLLAMA="http://localhost:11434/v1"
GEMINI="https://generativelanguage.googleapis.com/v1beta/openai/"

run_model() {
    local MODEL=$1
    local BASE_URL=$2
    echo "=========================================="
    echo "MODEL: $MODEL"
    echo "=========================================="
    python3 src/llm_classify.py --baseline              --model $MODEL --base-url $BASE_URL
    python3 src/llm_classify.py --baseline --few-shot   --model $MODEL --base-url $BASE_URL
    for GROUP in scincl specter LLM; do
        python3 src/llm_classify.py --group $GROUP --all            --model $MODEL --base-url $BASE_URL
        python3 src/llm_classify.py --group $GROUP --all --few-shot --model $MODEL --base-url $BASE_URL
    done
    echo "DONE: $MODEL"
}

run_model "openai/gpt-4o-mini"  "https://openrouter.ai/api/v1"
run_model "gemma2:9b"           "$OLLAMA"
run_model "qwen2.5:14b"         "$OLLAMA"
run_model "llama3.1:8b"         "$OLLAMA"
run_model "mistral:7b"      "$OLLAMA"
# run_model "deepseek-r1:8b"      "$OLLAMA"  # excluded: too many empty predictions
run_model "phi4:14b"            "$OLLAMA"
run_model "gemma3:12b"          "$OLLAMA"
run_model "models/gemini-2.5-flash" "$GEMINI"

echo "ALL DONE"

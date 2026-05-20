#!/bin/bash
# Run all 6 LongMemEval-S categories through a cloud judge.
# Usage: run_cloud_judge.sh <model-name> <output-suffix>
#   e.g. run_cloud_judge.sh deepseek-v3.1:671b-cloud deepseek
set -u

MODEL="${1:?need model}"
SUFFIX="${2:?need suffix}"
URL="${3:-http://192.168.1.242:11434/v1}"
RESULTS_DIR="${HOME}/heurchain-bench/results"
SUMMARY="${RESULTS_DIR}/cloud_judge_${SUFFIX}_summary.txt"
LOG="${RESULTS_DIR}/cloud_judge_${SUFFIX}.log"

CATEGORIES=(
  single-session-assistant
  temporal-reasoning
  single-session-user
  single-session-preference
  knowledge-update
  multi-session
)

echo "Cloud-judge re-run: model=${MODEL} started at $(date)" | tee "$SUMMARY"

for cat in "${CATEGORIES[@]}"; do
  IN="${RESULTS_DIR}/facts_v2_${cat}_max30.json"
  OUT="${RESULTS_DIR}/facts_v2cloud-${SUFFIX}_${cat}_max30.json"
  echo "" | tee -a "$SUMMARY"
  echo "=== ${cat} ===" | tee -a "$SUMMARY"
  echo "  start: $(date)" | tee -a "$SUMMARY"
  python3 "${HOME}/heurchain-bench/judge_rerun_cloud.py" \
    --input "$IN" \
    --output "$OUT" \
    --model "$MODEL" \
    --base-url "$URL" 2>&1 | tee -a "$LOG" | tail -2 | head -1 | tee -a "$SUMMARY"
  # extract QA acc from the JSON we just wrote
  qa=$(python3 -c "import json; d=json.load(open('${OUT}')); print(f'{d[\"overall_qa_acc\"]*100:.2f}%')" 2>/dev/null || echo "ERR")
  echo "  QA Accuracy: ${qa}" | tee -a "$SUMMARY"
  echo "  done:  $(date)" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "All ${SUFFIX} categories complete at $(date)" | tee -a "$SUMMARY"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_SCRIPT="$SCRIPT_DIR/run_remove_norm.py"

MODEL="${MODEL:-gemini-3.1-flash-lite}"
MODEL_SLUG="${MODEL//./_}"
MODEL_SLUG="${MODEL_SLUG//-/_}"
NORM_COUNT="${NORM_COUNT:-5}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/batch_${MODEL_SLUG}_${RUN_TIMESTAMP}}"
LOG_DIR="$OUTPUT_ROOT/logs"
MANIFEST_FILE="$OUTPUT_ROOT/shell_run_manifest.json"
SAMPLES="${SAMPLES:-}"

mkdir -p "$LOG_DIR"

init_manifest() {
  "$PYTHON_BIN" - "$MANIFEST_FILE" "$MODEL" "$OUTPUT_ROOT" "$RUN_TIMESTAMP" "$NORM_COUNT" <<'PY'
import json
import sys

manifest_path, model, output_root, run_timestamp, norm_count = sys.argv[1:]
payload = {
    "model": model,
    "run_timestamp": run_timestamp,
    "output_root": output_root,
    "norm_count": int(norm_count),
    "runs": [],
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

append_manifest() {
  "$PYTHON_BIN" - \
    "$MANIFEST_FILE" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" <<'PY'
import json
import sys

(
    manifest_path,
    norm_index,
    run_id,
    started_at,
    finished_at,
    exit_code,
    output_dir,
    log_file,
    summary_csv,
    summary_json,
    prompt_json,
    command_text,
) = sys.argv[1:]

with open(manifest_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

payload["runs"].append(
    {
        "norm_index": int(norm_index),
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": int(exit_code),
        "output_dir": output_dir,
        "log_file": log_file,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "prompt_metadata_json": prompt_json,
        "command": command_text,
    }
)

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

init_manifest

for norm_index in $(seq 1 "$NORM_COUNT"); do
  run_id="drop_norm_${norm_index}_${RUN_TIMESTAMP}"
  variant_output_dir="$OUTPUT_ROOT/drop_norm_${norm_index}"
  log_file="$LOG_DIR/drop_norm_${norm_index}_${RUN_TIMESTAMP}.log"
  summary_csv="$variant_output_dir/summary_${MODEL_SLUG}_${run_id}.csv"
  summary_json="$variant_output_dir/summary_${MODEL_SLUG}_${run_id}.json"
  prompt_json="$variant_output_dir/prompt_variants_${MODEL_SLUG}_${run_id}.json"

  mkdir -p "$variant_output_dir"

  cmd=(
    "$PYTHON_BIN"
    -u
    "$PY_SCRIPT"
    --model "$MODEL"
    --skip-base
    --drop-index "$norm_index"
    --output-dir "$variant_output_dir"
    --run-id "$run_id"
  )

  if [[ -n "$SAMPLES" ]]; then
    cmd+=(--samples "$SAMPLES")
  fi

  if [[ $# -gt 0 ]]; then
    cmd+=("$@")
  fi

  command_text="$(printf '%q ' "${cmd[@]}")"
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"

  append_manifest \
    "$norm_index" \
    "$run_id" \
    "$started_at" \
    "$finished_at" \
    "$exit_code" \
    "$variant_output_dir" \
    "$log_file" \
    "$summary_csv" \
    "$summary_json" \
    "$prompt_json" \
    "$command_text"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "Run failed for norm $norm_index. See $log_file"
    echo "Manifest: $MANIFEST_FILE"
    exit "$exit_code"
  fi
done

echo "Completed all norm-drop runs."
echo "Root output directory: $OUTPUT_ROOT"
echo "Manifest: $MANIFEST_FILE"

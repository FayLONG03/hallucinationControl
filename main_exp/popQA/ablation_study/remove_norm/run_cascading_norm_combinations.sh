#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_SCRIPT="$SCRIPT_DIR/run_remove_norm.py"

MODEL="${MODEL:-gemini-3.1-flash-lite}"
MODEL_SLUG="${MODEL//./_}"
MODEL_SLUG="${MODEL_SLUG//-/_}"
NORM_COUNT="${NORM_COUNT:-5}"
KEEP_SIZES="${KEEP_SIZES:-1 2 3}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/cascade_${MODEL_SLUG}_${RUN_TIMESTAMP}}"
LOG_DIR="$OUTPUT_ROOT/logs"
MANIFEST_FILE="$OUTPUT_ROOT/cascade_run_manifest.json"
SAMPLES="${SAMPLES:-}"

mkdir -p "$LOG_DIR"

init_manifest() {
  "$PYTHON_BIN" - "$MANIFEST_FILE" "$MODEL" "$OUTPUT_ROOT" "$RUN_TIMESTAMP" "$NORM_COUNT" "$KEEP_SIZES" <<'PY'
import json
import sys

manifest_path, model, output_root, run_timestamp, norm_count, keep_sizes = sys.argv[1:]
payload = {
    "model": model,
    "run_timestamp": run_timestamp,
    "output_root": output_root,
    "norm_count": int(norm_count),
    "keep_sizes": keep_sizes.split(),
    "runs": [],
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

append_manifest() {
  "$PYTHON_BIN" - \
    "$MANIFEST_FILE" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" <<'PY'
import json
import sys

(
    manifest_path,
    variant_name,
    kept_indexes,
    started_at,
    finished_at,
    exit_code,
    output_dir,
    log_file,
    summary_csv,
    summary_json,
    prompt_json,
) = sys.argv[1:]

with open(manifest_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

kept_list = [int(part) for part in kept_indexes.split(",") if part]

payload["runs"].append(
    {
        "variant_name": variant_name,
        "kept_indexes": kept_list,
        "keep_size": len(kept_list),
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": int(exit_code),
        "output_dir": output_dir,
        "log_file": log_file,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "prompt_metadata_json": prompt_json,
    }
)

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

generate_combinations() {
  "$PYTHON_BIN" - "$NORM_COUNT" "$KEEP_SIZES" <<'PY'
import itertools
import sys

norm_count = int(sys.argv[1])
keep_sizes = [int(size) for size in sys.argv[2].split() if size.strip()]

for keep_size in keep_sizes:
    if keep_size < 1 or keep_size > norm_count:
        raise SystemExit(f"Invalid keep size: {keep_size}. Valid range is 1-{norm_count}.")
    for combo in itertools.combinations(range(1, norm_count + 1), keep_size):
        print(",".join(str(index) for index in combo))
PY
}

init_manifest

while IFS= read -r combo; do
  [[ -z "$combo" ]] && continue

  variant_suffix="${combo//,/_}"
  variant_name="keep_norms_${variant_suffix}"
  run_id="${variant_name}_${RUN_TIMESTAMP}"
  variant_output_dir="$OUTPUT_ROOT/$variant_name"
  log_file="$LOG_DIR/${variant_name}_${RUN_TIMESTAMP}.log"
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
    --variant-name "$variant_name"
    --output-dir "$variant_output_dir"
    --run-id "$run_id"
  )

  IFS=',' read -r -a kept_indexes <<< "$combo"
  for kept_index in "${kept_indexes[@]}"; do
    cmd+=(--keep-index "$kept_index")
  done

  if [[ -n "$SAMPLES" ]]; then
    cmd+=(--samples "$SAMPLES")
  fi

  if [[ $# -gt 0 ]]; then
    cmd+=("$@")
  fi

  started_at="$(date '+%Y-%m-%d %H:%M:%S')"

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"

  append_manifest \
    "$variant_name" \
    "$combo" \
    "$started_at" \
    "$finished_at" \
    "$exit_code" \
    "$variant_output_dir" \
    "$log_file" \
    "$summary_csv" \
    "$summary_json" \
    "$prompt_json"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "Run failed for $variant_name. See $log_file"
    echo "Manifest: $MANIFEST_FILE"
    exit "$exit_code"
  fi
done < <(generate_combinations)

echo "Completed cascading norm-combination runs."
echo "Root output directory: $OUTPUT_ROOT"
echo "Manifest: $MANIFEST_FILE"

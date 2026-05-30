#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: repowise_sync_after_pull.sh [--skip-pull] [--dry-run] [repo ...]

Pull canonical repos, refresh the Repowise workspace index, and write receipts
under .onex_state/repowise-sync/.
EOF
}

skip_pull=false
dry_run=false
repos=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-pull)
      skip_pull=true
      shift
      ;;
    --dry-run)
      dry_run=true
      skip_pull=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      repos+=("$@")
      break
      ;;
    -*)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      repos+=("$1")
      shift
      ;;
  esac
done

omni_home="${OMNI_HOME:-$(pwd)}"
workspace_config="$omni_home/.repowise-workspace.yaml"
state_root="$omni_home/.onex_state/repowise-sync"
run_id="repowise-sync-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$state_root/$run_id"
receipt_file="$run_dir/receipt.json"
freshness_receipt_file="$run_dir/freshness-receipt.json"

mkdir -p "$run_dir"

exec > >(tee -a "$run_dir/run.log") 2>&1

if [[ ! -f "$workspace_config" ]]; then
  echo "ERROR: missing workspace config: $workspace_config" >&2
  exit 1
fi

if ! command -v repowise >/dev/null 2>&1; then
  echo "ERROR: repowise CLI not found on PATH" >&2
  exit 1
fi

if [[ "$skip_pull" == false ]]; then
  if [[ "${#repos[@]}" -gt 0 ]]; then
    OMNI_HOME="$omni_home" bash "$omni_home/omnibase_infra/scripts/pull-all.sh" "${repos[@]}" \
      > "$run_dir/pull-all.log" 2>&1
  else
    OMNI_HOME="$omni_home" bash "$omni_home/omnibase_infra/scripts/pull-all.sh" \
      > "$run_dir/pull-all.log" 2>&1
  fi
else
  : > "$run_dir/pull-all.log"
fi

if [[ "$dry_run" == true ]]; then
  repowise update --workspace --index-only --dry-run "$omni_home" \
    > "$run_dir/repowise-update.log" 2>&1
else
  repowise update --workspace --index-only "$omni_home" \
    > "$run_dir/repowise-update.log" 2>&1
fi

repowise status --workspace "$omni_home" > "$run_dir/repowise-status.txt" 2>&1 || true
repowise doctor --workspace "$omni_home" > "$run_dir/repowise-doctor.txt" 2>&1 || true

python3 "$omni_home/omnibase_infra/scripts/emit_repowise_freshness_receipt.py" \
  --omni-home "$omni_home" \
  --out "$freshness_receipt_file" \
  > "$run_dir/freshness-receipt.log" 2>&1

jq -n \
  --arg run_id "$run_id" \
  --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg omni_home "$omni_home" \
  --arg receipt "$freshness_receipt_file" \
  --arg log "$run_dir/run.log" \
  --arg pull_all "$run_dir/pull-all.log" \
  --arg update "$run_dir/repowise-update.log" \
  --arg status_file "$run_dir/repowise-status.txt" \
  --arg doctor "$run_dir/repowise-doctor.txt" \
  --argjson dry_run "$dry_run" \
  --argjson skip_pull "$skip_pull" \
  '{
    run_id: $run_id,
    generated_at: $generated_at,
    omni_home: $omni_home,
    options: {dry_run: $dry_run, skip_pull: $skip_pull},
    files: {
      log: $log,
      pull_all: $pull_all,
      repowise_update: $update,
      repowise_status: $status_file,
      repowise_doctor: $doctor,
      repowise_freshness: $receipt
    }
  }' > "$receipt_file"

ln -sfn "$run_dir" "$state_root/latest"
echo "receipt: $receipt_file"
echo "freshness_receipt: $freshness_receipt_file"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# validate-state.sh — State file schema validation for headless pipeline [OMN-6988]
#
# Validates that state files produced by headless pipeline stages conform to
# the expected JSON schema. Called after each stage completes and before
# dependent stages consume the output.
#
# Usage:
#   ./scripts/headless-pipeline/validate-state.sh <state-file> [--stage <stage-name>]
#   ./scripts/headless-pipeline/validate-state.sh <state-dir> --all
#
# Exit codes:
#   0  All validations passed
#   1  Validation failed (schema mismatch or missing required fields)
#   2  File not found or not valid JSON

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Required fields per stage ---
# Each stage must produce at minimum: schema_version, stage, repo, status
COMMON_FIELDS=("schema_version" "stage" "repo" "status")

declare -A STAGE_FIELDS
STAGE_FIELDS[merge-sweep]="prs_merged prs_skipped prs_failed"
STAGE_FIELDS[release]="version tag commits_since_last_tag"
STAGE_FIELDS[redeploy]="images_rebuilt health_checks"
STAGE_FIELDS[ticket-close]="tickets_closed tickets_skipped"
STAGE_FIELDS[integration-sweep]="checks"

declare -A VALID_STATUSES
VALID_STATUSES[merge-sweep]="success partial failed dry-run"
VALID_STATUSES[release]="success skipped failed dry-run"
VALID_STATUSES[redeploy]="success skipped failed dry-run"
VALID_STATUSES[ticket-close]="success partial failed dry-run"
VALID_STATUSES[integration-sweep]="pass warn fail dry-run"

# --- Parse arguments ---
STATE_FILE=""
STAGE_NAME=""
VALIDATE_ALL=false
QUIET=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE_NAME="$2"; shift 2 ;;
    --all) VALIDATE_ALL=true; shift ;;
    --quiet) QUIET=true; shift ;;
    -*) echo "Unknown flag: $1" >&2; exit 2 ;;
    *) STATE_FILE="$1"; shift ;;
  esac
done

if [[ -z "$STATE_FILE" ]]; then
  echo "Usage: validate-state.sh <state-file|state-dir> [--stage <name>] [--all] [--quiet]" >&2
  exit 2
fi

# --- Logging ---
log() {
  if [[ "$QUIET" != "true" ]]; then
    echo "$@"
  fi
}

log_error() {
  echo "ERROR: $*" >&2
}

# --- Validate a single state file ---
validate_file() {
  local file="$1"
  local expected_stage="${2:-}"
  local errors=0

  # Check file exists
  if [[ ! -f "$file" ]]; then
    log_error "State file not found: $file"
    return 2
  fi

  # Check valid JSON
  if ! jq empty "$file" 2>/dev/null; then
    log_error "Invalid JSON: $file"
    return 2
  fi

  # Check common required fields
  for field in "${COMMON_FIELDS[@]}"; do
    local value
    value=$(jq -r ".${field} // empty" "$file" 2>/dev/null)
    if [[ -z "$value" ]]; then
      log_error "Missing required field '${field}' in $file"
      ((errors++))
    fi
  done

  # Extract stage from file if not provided
  local stage
  stage=$(jq -r '.stage // empty' "$file" 2>/dev/null)
  if [[ -n "$expected_stage" && -n "$stage" && "$stage" != "$expected_stage" ]]; then
    log_error "Stage mismatch: expected '${expected_stage}', got '${stage}' in $file"
    ((errors++))
  fi

  # Use discovered stage for field validation
  local check_stage="${expected_stage:-$stage}"

  # Validate status value
  if [[ -n "$check_stage" && -n "${VALID_STATUSES[$check_stage]+x}" ]]; then
    local status
    status=$(jq -r '.status // empty' "$file" 2>/dev/null)
    local valid=false
    for valid_status in ${VALID_STATUSES[$check_stage]}; do
      if [[ "$status" == "$valid_status" ]]; then
        valid=true
        break
      fi
    done
    if [[ "$valid" != "true" && -n "$status" ]]; then
      log_error "Invalid status '${status}' for stage '${check_stage}' in $file (valid: ${VALID_STATUSES[$check_stage]})"
      ((errors++))
    fi
  fi

  # Check stage-specific fields (warn, don't fail -- they may be optional)
  if [[ -n "$check_stage" && -n "${STAGE_FIELDS[$check_stage]+x}" ]]; then
    for field in ${STAGE_FIELDS[$check_stage]}; do
      local has_field
      has_field=$(jq "has(\"${field}\")" "$file" 2>/dev/null)
      if [[ "$has_field" != "true" ]]; then
        log "WARN: Missing expected field '${field}' for stage '${check_stage}' in $file"
      fi
    done
  fi

  # Validate schema_version is "1.0"
  local schema_version
  schema_version=$(jq -r '.schema_version // empty' "$file" 2>/dev/null)
  if [[ -n "$schema_version" && "$schema_version" != "1.0" ]]; then
    log_error "Unsupported schema_version '${schema_version}' in $file (expected: 1.0)"
    ((errors++))
  fi

  if [[ $errors -gt 0 ]]; then
    return 1
  fi

  log "PASS: $file"
  return 0
}

# --- Main ---
total_errors=0

if [[ "$VALIDATE_ALL" == "true" ]]; then
  # Validate all .json files in the state directory
  if [[ ! -d "$STATE_FILE" ]]; then
    log_error "Not a directory: $STATE_FILE"
    exit 2
  fi

  found=0
  while IFS= read -r -d '' json_file; do
    found=1
    # Skip manifest.json and summary.json (different schema)
    basename=$(basename "$json_file")
    if [[ "$basename" == "manifest.json" || "$basename" == "summary.json" ]]; then
      log "SKIP: $json_file (manifest/summary)"
      continue
    fi

    # Infer stage from filename
    inferred_stage="${basename%.json}"
    validate_file "$json_file" "$inferred_stage" || ((total_errors++))
  done < <(find "$STATE_FILE" -name "*.json" -not -name "manifest.json" -not -name "summary.json" -print0 | sort -z)

  if [[ $found -eq 0 ]]; then
    log "No state files found in $STATE_FILE"
  fi
else
  # Validate a single file
  validate_file "$STATE_FILE" "$STAGE_NAME" || ((total_errors++))
fi

if [[ $total_errors -gt 0 ]]; then
  log_error "${total_errors} validation error(s)"
  exit 1
fi

log "All validations passed"
exit 0

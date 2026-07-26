#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# docker-volume-gc.sh — conservative repo-owned Docker volume cleanup for .201.
#
# Removes only volumes that are not mounted by any container and match one of the
# explicitly allowed disposable classes:
#   - anonymous Docker volumes (64 hex chars)
#   - ephemeral runtime boot Redpanda volumes (omnibase-infra-boot-*--redpanda-data)
#   - legacy standalone redpanda_data volumes that are no longer mounted
#
# It never removes named lane/runtime volumes such as prod, stability-test, judge,
# or dev Redpanda data while they are linked to live containers.

set -euo pipefail

EXECUTE=false
EMIT_JSON=false
LOG_FILE="${HOME}/.local/log/onex/docker-volume-gc.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true; shift ;;
    --json) EMIT_JSON=true; shift ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [docker-volume-gc] $*" | tee -a "$LOG_FILE" >&2; }

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 3; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 3; }

log "Starting ($( [[ "$EXECUTE" == true ]] && echo EXECUTE || echo DRY-RUN ))"

VOLUME_NAMES="$(docker volume ls -q 2>/dev/null || true)"
CONTAINER_IDS="$(docker ps -aq 2>/dev/null || true)"
USED_VOLUME_NAMES=""
if [[ -n "${CONTAINER_IDS}" ]]; then
  USED_VOLUME_NAMES="$(docker inspect --format '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}}{{"\n"}}{{end}}{{end}}' ${CONTAINER_IDS} 2>/dev/null | sort -u || true)"
fi
PLAN_JSON="$(
  VOLUME_NAMES="${VOLUME_NAMES}" USED_VOLUME_NAMES="${USED_VOLUME_NAMES}" python3 - <<'PY'
import json
import os
import re

anonymous = re.compile(r"^[0-9a-f]{64}$")
boot_redpanda = re.compile(r"^omnibase-infra-boot-[0-9]+-[0-9]+--redpanda-data$")
legacy_redpanda = re.compile(r"^redpanda_data$")
used = {line.strip() for line in os.environ.get("USED_VOLUME_NAMES", "").splitlines() if line.strip()}

plan = []
for name in [line.strip() for line in os.environ.get("VOLUME_NAMES", "").splitlines() if line.strip()]:
    if name in used:
        continue
    if anonymous.match(name):
        reason = "unused-anonymous-volume"
    elif boot_redpanda.match(name):
        reason = "unused-ephemeral-boot-redpanda-volume"
    elif legacy_redpanda.match(name):
        reason = "unused-legacy-redpanda-volume"
    else:
        continue
    plan.append({"name": name, "reason": reason})

print(json.dumps({"remove_volumes": plan}, sort_keys=True))
PY
)"

if [[ "$EMIT_JSON" == true ]]; then
  echo "$PLAN_JSON"
fi

VOLUMES="$(echo "$PLAN_JSON" | python3 -c 'import json,sys;[print(v["name"]) for v in json.load(sys.stdin)["remove_volumes"]]')"
COUNT="$(echo "$VOLUMES" | grep -c . || true)"
log "Plan: ${COUNT} unused disposable volume(s)"

if [[ "$EXECUTE" != true ]]; then
  log "DRY-RUN — would remove the above. Re-run with --execute to act."
  [[ -n "$VOLUMES" ]] && { echo "VOLUMES TO REMOVE:"; echo "$VOLUMES"; } >&2
  exit 0
fi

if [[ -n "$VOLUMES" ]]; then
  while IFS= read -r volume; do
    [[ -z "$volume" ]] && continue
    if docker volume rm "$volume" >>"$LOG_FILE" 2>&1; then
      log "removed volume $volume"
    else
      log "kept/failed volume $volume (likely became referenced)"
    fi
  done <<< "$VOLUMES"
fi

log "Done."

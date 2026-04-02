#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# caia-watchdog.sh — CAIA session watchdog
#
# Monitors for session checkpoints written by the checkpoint skill and
# relaunches `claude -p` when the reset time arrives.
#
# Checkpoint file: ~/.onex_state/orchestrator/checkpoint.yaml
# Log file:        ~/.onex_state/orchestrator/watchdog.log
#
# Managed by launchd plist ai.omninode.caia-watchdog.
# The script runs as a long-lived process; launchd restarts it on exit.

set -euo pipefail

CHECKPOINT_DIR="${HOME}/.onex_state/orchestrator"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/checkpoint.yaml"
LOG_FILE="${CHECKPOINT_DIR}/watchdog.log"
ARCHIVE_DIR="${CHECKPOINT_DIR}/archive"
POLL_INTERVAL="${CAIA_WATCHDOG_POLL_SECONDS:-300}"  # 5 minutes default

mkdir -p "$CHECKPOINT_DIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG_FILE"; }

log "Watchdog started. Polling ${CHECKPOINT_FILE} every ${POLL_INTERVAL}s"

while true; do
    if [[ -f "$CHECKPOINT_FILE" ]]; then
        log "Checkpoint found"

        # Parse YAML fields using python3 (available on macOS)
        RESET_AT=$(python3 -c "
import yaml, sys
try:
    d = yaml.safe_load(open('$CHECKPOINT_FILE'))
    print(d.get('reset_at', '') or '')
except Exception as e:
    print('', file=sys.stderr)
    print('')
" 2>>"$LOG_FILE")

        RESUME_PROMPT=$(python3 -c "
import yaml, sys
try:
    d = yaml.safe_load(open('$CHECKPOINT_FILE'))
    print(d.get('resume_prompt', '') or '')
except Exception as e:
    print('', file=sys.stderr)
    print('')
" 2>>"$LOG_FILE")

        # Validate that we got a resume prompt
        if [[ -z "$RESUME_PROMPT" ]]; then
            log "ERROR: checkpoint has no resume_prompt, removing"
            rm -f "$CHECKPOINT_FILE"
            sleep "$POLL_INTERVAL"
            continue
        fi

        # If reset_at is set, sleep until that time
        if [[ -n "$RESET_AT" ]]; then
            # Parse ISO-8601 timestamp to epoch (macOS date -j)
            # Strip fractional seconds and timezone suffix for date parsing
            CLEAN_TS=$(echo "$RESET_AT" | sed 's/\.[0-9]*//; s/Z$//; s/+00:00$//')
            RESET_EPOCH=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$CLEAN_TS" "+%s" 2>/dev/null || echo 0)
            NOW_EPOCH=$(date "+%s")
            SLEEP_SECS=$(( RESET_EPOCH - NOW_EPOCH ))

            if [[ $SLEEP_SECS -gt 0 ]]; then
                log "Sleeping ${SLEEP_SECS}s until reset at ${RESET_AT}"
                sleep "$SLEEP_SECS"
            else
                log "Reset time already passed (${RESET_AT}), launching immediately"
            fi
        fi

        # Archive checkpoint before launch
        mkdir -p "$ARCHIVE_DIR"
        cp "$CHECKPOINT_FILE" "${ARCHIVE_DIR}/checkpoint-$(date +%s).yaml"
        rm -f "$CHECKPOINT_FILE"

        log "Launching claude -p with resume prompt"

        # Launch claude -p — run in foreground so launchd tracks the process
        ONEX_RUN_ID="watchdog-$(date +%s)" \
        claude -p "$RESUME_PROMPT" \
            --allowedTools "Bash,Read,Write,Edit,Glob,Grep,mcp__linear-server__*" \
            >> "$LOG_FILE" 2>&1 || log "claude -p exited with code $?"

        log "claude -p session completed"
    fi

    sleep "$POLL_INTERVAL"
done

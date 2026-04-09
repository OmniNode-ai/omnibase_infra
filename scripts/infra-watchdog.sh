#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# infra-watchdog.sh — ONEX infra stack watchdog
#
# Checks if core infra containers (postgres, redpanda, valkey) are running.
# If any are not running (Created, Exited, or absent), restarts the infra stack.
#
# Runs on .201 as a systemd timer every 5 minutes.
# Log: /var/log/onex/infra-watchdog.log
# Unit: deploy/infra-watchdog.service + deploy/infra-watchdog.timer
#
# Install: see deploy/infra-watchdog.service (systemd) or README in this file.

set -euo pipefail

COMPOSE_FILE="${OMNI_HOME:-/home/jonah/Code/omni_home}/omnibase_infra/docker/docker-compose.infra.yml"
LOG_FILE="/var/log/onex/infra-watchdog.log"
INFRA_CONTAINERS=("omnibase-infra-postgres" "omnibase-infra-redpanda" "omnibase-infra-valkey")

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [infra-watchdog] $*" | tee -a "$LOG_FILE"
}

# Check if a container is running (not just present)
container_is_running() {
    local name="$1"
    local state
    state=$(docker inspect --format='{{.State.Status}}' "$name" 2>/dev/null || echo "absent")
    [[ "$state" == "running" ]]
}

log "Watchdog check starting"

DOWN=()
for container in "${INFRA_CONTAINERS[@]}"; do
    if ! container_is_running "$container"; then
        state=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "absent")
        log "UNHEALTHY: $container is in state '$state'"
        DOWN+=("$container")
    else
        log "OK: $container is running"
    fi
done

if [[ ${#DOWN[@]} -gt 0 ]]; then
    log "ALERT: ${#DOWN[@]} infra container(s) down: ${DOWN[*]}"
    log "Restarting infra stack via docker compose..."

    # Bring up the core infra (postgres, redpanda, valkey — no profiles needed)
    docker compose -f "$COMPOSE_FILE" up -d --no-build 2>&1 | tee -a "$LOG_FILE"

    # Verify containers came back up
    sleep 10
    STILL_DOWN=()
    for container in "${DOWN[@]}"; do
        if ! container_is_running "$container"; then
            STILL_DOWN+=("$container")
        fi
    done

    if [[ ${#STILL_DOWN[@]} -gt 0 ]]; then
        log "ERROR: Restart did not recover: ${STILL_DOWN[*]}"
        exit 1
    else
        log "Recovery succeeded — all containers running"
    fi
else
    log "All infra containers healthy — no action needed"
fi

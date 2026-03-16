#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# omnidash-lifecycle.sh — start, stop, restart, and status for the local omnidash dev server.
#
# Usage:
#   omnidash-lifecycle.sh start   [--bus local|cloud]  # Start omnidash (kills any prior instance first)
#   omnidash-lifecycle.sh stop                          # Stop omnidash gracefully
#   omnidash-lifecycle.sh restart [--bus local|cloud]  # Stop then start
#   omnidash-lifecycle.sh status                        # Check if omnidash is running and healthy
#
# Environment:
#   OMNIDASH_DIR    Override omnidash repo path (default: sibling of OMNIBASE_INFRA_DIR or /Volumes/PRO-G40/Code/omni_home/omnidash)
#   OMNIDASH_PORT   Override port (default: 3000)
#
# The script sources ~/.omnibase/.env for platform config. Bus mode can be
# overridden with --bus local|cloud (maps to npm run dev:local / dev:cloud / dev).
#
# [OMN-5142]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve omnidash directory
_resolve_omnidash_dir() {
  if [[ -n "${OMNIDASH_DIR:-}" ]] && [[ -d "${OMNIDASH_DIR}" ]]; then
    echo "${OMNIDASH_DIR}"
    return 0
  fi

  # Derive from OMNIBASE_INFRA_DIR (sibling repo)
  if [[ -n "${OMNIBASE_INFRA_DIR:-}" ]]; then
    local candidate
    candidate="$(dirname "${OMNIBASE_INFRA_DIR}")/omnidash"
    if [[ -d "${candidate}/package.json" ]] || [[ -f "${candidate}/package.json" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi

  # Hardcoded fallback
  local fallback="/Volumes/PRO-G40/Code/omni_home/omnidash"
  if [[ -f "${fallback}/package.json" ]]; then
    echo "${fallback}"
    return 0
  fi

  echo "[omnidash] ERROR: Cannot find omnidash directory. Set OMNIDASH_DIR." >&2
  return 1
}

OMNIDASH_PORT="${OMNIDASH_PORT:-3000}"

cmd_start() {
  local bus_mode=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --bus) bus_mode="$2"; shift 2 ;;
      *) echo "[omnidash] Unknown option: $1" >&2; return 1 ;;
    esac
  done

  local omnidash_dir
  omnidash_dir="$(_resolve_omnidash_dir)" || return 1

  echo "[omnidash] Starting omnidash from: ${omnidash_dir}"

  # Kill any existing instance first (using omnidash's own kill-server.sh)
  if [[ -f "${omnidash_dir}/scripts/kill-server.sh" ]]; then
    PORT="${OMNIDASH_PORT}" bash "${omnidash_dir}/scripts/kill-server.sh" 2>/dev/null || true
  fi

  # Determine npm script based on bus mode
  local npm_script="dev"
  case "${bus_mode}" in
    local)  npm_script="dev:local" ;;
    cloud)  npm_script="dev:cloud" ;;
    "")     npm_script="dev" ;;
    *)      echo "[omnidash] Unknown bus mode: ${bus_mode}. Use 'local' or 'cloud'." >&2; return 1 ;;
  esac

  echo "[omnidash] npm run ${npm_script} (port ${OMNIDASH_PORT})"

  # Start in background, redirect output to log file
  local log_file="${omnidash_dir}/.omnidash-server.log"
  cd "${omnidash_dir}"
  nohup npm run "${npm_script}" > "${log_file}" 2>&1 &
  local server_pid=$!
  echo "[omnidash] Server starting (PID ${server_pid})"

  # Wait for server to become responsive (up to 30s)
  local max_wait=30
  local elapsed=0
  while ! curl -sf "http://localhost:${OMNIDASH_PORT}/api/health-probe" > /dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [[ ${elapsed} -ge ${max_wait} ]]; then
      echo "[omnidash] WARNING: Server not responding after ${max_wait}s. Check ${log_file}" >&2
      echo "[omnidash] PID ${server_pid} may still be starting up." >&2
      return 0  # Non-fatal — server may still be loading
    fi
  done

  echo "[omnidash] Server healthy on port ${OMNIDASH_PORT} (startup took ~${elapsed}s)"
}

cmd_stop() {
  local omnidash_dir
  omnidash_dir="$(_resolve_omnidash_dir)" || return 1

  echo "[omnidash] Stopping omnidash..."

  if [[ -f "${omnidash_dir}/scripts/kill-server.sh" ]]; then
    PORT="${OMNIDASH_PORT}" bash "${omnidash_dir}/scripts/kill-server.sh" 2>/dev/null || true
  else
    # Fallback: kill anything on the port
    local port_pids
    port_pids=$(lsof -ti:"${OMNIDASH_PORT}" 2>/dev/null || true)
    if [[ -n "${port_pids}" ]]; then
      echo "${port_pids}" | xargs kill -TERM 2>/dev/null || true
      sleep 2
      echo "${port_pids}" | xargs kill -9 2>/dev/null || true
    fi
  fi

  echo "[omnidash] Stopped."
}

cmd_restart() {
  cmd_stop
  cmd_start "$@"
}

cmd_status() {
  local omnidash_dir
  omnidash_dir="$(_resolve_omnidash_dir)" || return 1

  # Check if port is in use
  local port_pids
  port_pids=$(lsof -ti:"${OMNIDASH_PORT}" 2>/dev/null || true)

  if [[ -z "${port_pids}" ]]; then
    echo "[omnidash] Status: NOT RUNNING (port ${OMNIDASH_PORT} not in use)"
    return 1
  fi

  # Check health endpoint
  local health_response
  if health_response=$(curl -sf "http://localhost:${OMNIDASH_PORT}/api/health-probe" 2>/dev/null); then
    echo "[omnidash] Status: RUNNING on port ${OMNIDASH_PORT}"
    echo "[omnidash] Health: ${health_response}"

    # Check build info if available
    local build_info
    if build_info=$(curl -sf "http://localhost:${OMNIDASH_PORT}/api/build-info" 2>/dev/null); then
      echo "[omnidash] Build:  ${build_info}"
    fi
    return 0
  else
    echo "[omnidash] Status: PORT IN USE (port ${OMNIDASH_PORT}) but health probe failed"
    echo "[omnidash] PIDs on port: ${port_pids}"
    return 2
  fi
}

# Dispatch subcommand
case "${1:-}" in
  start)   shift; cmd_start "$@" ;;
  stop)    cmd_stop ;;
  restart) shift; cmd_restart "$@" ;;
  status)  cmd_status ;;
  *)
    echo "Usage: omnidash-lifecycle.sh {start|stop|restart|status} [--bus local|cloud]"
    exit 1
    ;;
esac

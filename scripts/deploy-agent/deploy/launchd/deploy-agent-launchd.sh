#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-19543: the launchd entry point of a deploy-agent instance on a macOS lab
# host. launchd has no ExecStartPre, so this runs the fail-fast port preflight
# the systemd units run there (OMN-16939) and then execs the one launcher every
# unit uses, deploy-agent-launch.sh, which sources the env file and execs the
# agent. Nothing else: every name comes from the plist's EnvironmentVariables.
set -euo pipefail
[ -n "${DEPLOY_AGENT_PORT:-}" ] || {
	echo "[deploy-agent-launchd] ERROR: DEPLOY_AGENT_PORT is required" >&2
	exit 2
}
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$here/../preflight_port_free.sh" "$DEPLOY_AGENT_PORT"
exec "$here/../deploy-agent-launch.sh" "$@"

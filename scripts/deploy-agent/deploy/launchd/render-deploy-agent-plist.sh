#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-19543: render the LaunchAgent plist of one deploy-agent instance to stdout.
#
#   render-deploy-agent-plist.sh <instance> <health-port> <lane-root>
#
# <instance>   a name in config/deploy_lane_routing.yaml with a lane: block
#              (dev-200, ...); the default instance runs under systemd on .201.
# <health-port> the agent's health port, inside the instance's port block and
#              published by no container of its lane.
# <lane-root>  the instance's data dir: it holds the agent clone
#              <lane-root>/omnibase_infra, the build workspace
#              <lane-root>/omni_home, state/ and logs/. Never the omni_home
#              canonical tree.
#
# HOME is read from the caller's environment, since the plist is per user.
# Refuses rather than guessing on any missing or malformed argument.
set -euo pipefail
die() {
	echo "render-deploy-agent-plist: $1" >&2
	exit 2
}
[ "$#" -eq 3 ] || die "usage: $0 <instance> <health-port> <lane-root>"
instance="$1"
port="$2"
lane_root="$3"
[[ "$instance" =~ ^dev-[a-z0-9]+$ ]] || die "instance must look like dev-NNN, got '$instance'"
[[ "$port" =~ ^[0-9]+$ ]] && [ "$port" -ge 1024 ] && [ "$port" -le 49151 ] ||
	die "health port must be 1024-49151 (below the macOS ephemeral range), got '$port'"
[[ "$lane_root" == /* ]] || die "lane root must be absolute, got '$lane_root'"
[ -n "${HOME:-}" ] || die "HOME is unset"
for value in "$lane_root" "$HOME"; do
	case "$value" in
	*[\|\&\<\>]*) die "path contains a character the plist or sed cannot carry: $value" ;;
	esac
done
template="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/deploy-agent-instance.plist.template"
sed -e "s|@INSTANCE@|${instance}|g" \
	-e "s|@PORT@|${port}|g" \
	-e "s|@LANE_ROOT@|${lane_root%/}|g" \
	-e "s|@HOME@|${HOME%/}|g" \
	"$template"

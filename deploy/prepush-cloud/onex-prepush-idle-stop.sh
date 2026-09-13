#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Idle auto-stop for the cloud pre-push host (OMN-16634).
#
# Invoked by onex-prepush-idle-stop.timer every 10 minutes. The host is IDLE
# only when ALL of the following hold:
#   1. no heavy-suite LOCK dir exists under the workroot (LOCK, LOCK.<k>)
#   2. no pytest / `uv sync` / prepush wrapper process is running
#   3. no interactive session is logged in (`who`)
#   4. nothing under <workroot>/runs was modified in the last 30 minutes
#   5. uptime exceeds 30 minutes (boot/bootstrap grace)
#
# Two CONSECUTIVE idle observations (>= 10 minutes apart) are required before
# the stop fires, via a stamp under /run (cleared on boot and on any activity),
# so the effective idle-to-stop window is ~20-30 minutes.
#
# The stop is `shutdown -h now`: this instance is EBS-backed with
# InstanceInitiatedShutdownBehavior=stop, so an OS halt lands it in the EC2
# STOPPED state ($0 compute, EBS + EIP only) with the warm caches intact. No
# AWS credentials exist or are needed on this host. Restart is lab-side:
#   aws ec2 start-instances --instance-ids <id>   (see README.md)
#
# FAIL-ACTIVE: any signal that cannot be read counts as activity. The wrong
# stop kills a governed suite mid-run; the wrong keep-alive costs cents.
#
# Picker behavior while stopped: the load probe's `ssh -o ConnectTimeout=3`
# fails, the row logs `hcloud=unreachable`, and placement falls through to the
# lab rows -- SKIP, never "assumed fit". Fail-closed semantics are untouched.
set -uo pipefail

# Bound by the unit file at install time (bootstrap.sh substitutes the
# installing user's workroot). Fail-fast rather than guessing a default: a
# wrong workroot reads as "no activity" and stops a host mid-suite.
WORKROOT="${ONEX_PREPUSH_WORKROOT:?ONEX_PREPUSH_WORKROOT must be set by the unit file}"
STAMP="/run/onex-prepush-idle.stamp"
GRACE_S=1800
RECENT_MIN=30

active() { rm -f "$STAMP" 2> /dev/null || true; exit 0; }

# 5. boot grace
up_s="$(cut -d. -f1 /proc/uptime 2> /dev/null || echo 0)"
[ "$up_s" -ge "$GRACE_S" ] 2> /dev/null || active

# 1. heavy-suite slot locks
if [ -d "$WORKROOT" ]; then
  if find "$WORKROOT" -maxdepth 1 -type d \( -name 'LOCK' -o -name 'LOCK.*' \) 2> /dev/null | grep -q .; then
    active
  fi
else
  # workroot missing entirely -- unreadable state, stay up
  active
fi

# 2. suite / sync / wrapper processes
if pgrep -f 'pytest|uv sync|prepush_smart_tests' > /dev/null 2>&1; then
  active
fi

# 3. logged-in sessions
if [ -n "$(who 2> /dev/null)" ]; then
  active
fi

# 4. recent run activity
if find "$WORKROOT/runs" -mindepth 1 -mmin "-${RECENT_MIN}" 2> /dev/null | grep -q .; then
  active
fi

# Idle. Require a second consecutive observation before stopping.
if [ ! -f "$STAMP" ]; then
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$STAMP" 2> /dev/null || exit 0
  exit 0
fi

logger -t onex-prepush-idle-stop "idle across two consecutive checks (first: $(cat "$STAMP" 2> /dev/null)); stopping instance"
/usr/sbin/shutdown -h now "onex-prepush idle auto-stop (OMN-16634)"

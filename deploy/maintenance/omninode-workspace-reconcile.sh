#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# omninode-workspace-reconcile.sh — the `.201` scheduler entry point for the
# workspace reconciler (OMN-17311).
#
# WHAT THIS IS
#   The thin host-side wrapper that /etc/cron.d/omninode-workspace-reconcile
#   invokes as root. It does three things and nothing else: load the alert
#   credentials, point OMNI_HOME at the deploy-source tree, and exec
#   `scripts/reconcile-host.sh` from that tree.
#
#   All reconciliation logic and all movement verification live in
#   `scripts/reconcile-host.sh` (OMN-17307), which is the SAME script the Mac
#   runs from its plugin tick. There is one reconciler; this file is a
#   scheduler adapter, and it must stay that way — the moment it grows a repair
#   step, the two hosts stop being reconciled by the same code and the whole
#   point is gone.
#
# WHY IT RUNS THE SCRIPT FROM THE CLONE, NOT FROM /data/maintenance/bin
#   `reconcile-host.sh` resolves its collaborators relative to its own location:
#   the clone manifest, the movement verifier, the clone delegate, the venv
#   delegate. Installed flat into /data/maintenance/bin it would find none of
#   them. Running it from `${OMNI_HOME}/omnibase_infra/scripts/` keeps that tree
#   internally consistent.
#
#   The obvious objection is real and is accepted deliberately: a stale clone
#   runs a stale reconciler. It is bounded — the reconciler's first act is to
#   advance the clones, so the next tick runs the current code — and the
#   alternative (a hand-copied second copy of five files) is the OMN-15525 drift
#   this maintenance path exists to prevent. What must NOT drift is this wrapper
#   and its cron unit, and both are in the MANIFEST of
#   `omninode-host-maintenance-sync.sh`, which reddens on divergence from
#   origin/dev.
#
# WHY IT SOURCES THE ALERT ENV FILE
#   `reconcile-host.sh` alerts on an unprovable surface via SLACK_BOT_TOKEN /
#   SLACK_CHANNEL_ID. cron starts with almost no environment, so without this
#   the failure would be detected, exit non-zero, and be seen by nobody — the
#   quiet half of the failure mode this whole epic is about.
#
# LANE BOUNDARIES
#   This touches deploy-source CLONES and host VENVS under OMNI_HOME. It never
#   restarts a container, never rebuilds an image, and never touches the prod or
#   judge lanes, which are read-only.
#
# Env (all overridable; nothing is discovered by guessing):
#   OMNI_HOME                    deploy-source root (default: /data/omninode/omni_home)
#   OMNINODE_ALERT_ENV_FILE      env file carrying the Slack credentials
#   RECONCILE_BRANCH             tracked branch (default: dev)
#
# Exit codes are `reconcile-host.sh`'s, unchanged: 0 proven, 2 a surface could
# not be proven at target, 3 indeterminate configuration.
set -uo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

OMNI_HOME="${OMNI_HOME:-/data/omninode/omni_home}"
ALERT_ENV_FILE="${OMNINODE_ALERT_ENV_FILE:-/data/omninode/omnibase_infra/.env}"
RECONCILER="${OMNI_HOME}/omnibase_infra/scripts/reconcile-host.sh"

if [[ -r "$ALERT_ENV_FILE" ]]; then
  # xtrace is suppressed across the source and restored afterwards. Under
  # `bash -x` — which is what anyone debugging a cron job reaches for first —
  # sourcing an env file echoes every assignment, so the Slack bot token would
  # land in a root-owned log verbatim and outlive the run. Suppressing it here
  # means the leak is impossible rather than merely unlikely.
  _xtrace_was_on=0
  case "$-" in *x*) _xtrace_was_on=1; set +x ;; esac
  # `set -a` so the sourced assignments are exported to the child.
  set -a
  # shellcheck disable=SC1090
  . "$ALERT_ENV_FILE"
  set +a
  [[ "$_xtrace_was_on" -eq 1 ]] && set -x
  unset _xtrace_was_on
fi

if [[ ! -f "$RECONCILER" ]]; then
  echo "[workspace-reconcile] FATAL: no reconciler at $RECONCILER" >&2
  echo "[workspace-reconcile]   OMNI_HOME=$OMNI_HOME — is the deploy-source clone present?" >&2
  exit 3
fi

exec env OMNI_HOME="$OMNI_HOME" bash "$RECONCILER" \
  --omni-home "$OMNI_HOME" \
  --branch "${RECONCILE_BRANCH:-dev}"

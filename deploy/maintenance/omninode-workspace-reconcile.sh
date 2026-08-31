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

ALERT_ENV_FILE="${OMNINODE_ALERT_ENV_FILE:-/data/omninode/omnibase_infra/.env}"

# OMN-17365: the DEFAULT is applied here, but RECONCILER is resolved AFTER the
# sourcing below. The two halves land on opposite sides of that block on purpose,
# and getting either one wrong has already broken this host once.
#
# Why the default must come BEFORE the sourcing:
#   the `.201` env file both ASSIGNS OMNI_HOME and later REFERENCES it (line 154
#   expands "$OMNI_HOME"). This script runs under `set -u`, so with no value in
#   scope the source aborts on `OMNI_HOME: unbound variable` and the whole
#   reconcile exits non-zero having done nothing -- which is exactly what the
#   first cut of this fix caused, live, by moving the default below the block.
#
# Why RECONCILER must come AFTER it:
#   the sourcing is `set -a` and it overwrites OMNI_HOME. Resolving the path up
#   here took the DEFAULT root while `--omni-home` below took the SOURCED one, so
#   the script that ran and the tree it reconciled were two different checkouts:
#
#     executed:    /data/omninode/omni_home/omnibase_infra/scripts/  (nothing advances this)
#     reconciled:  /data/omninode/omnibase_infra/                    (the clone loop's target)
#
#   That silently voids the bound this design rests on. The header above accepts
#   "a stale clone runs a stale reconciler" because "the reconciler's first act
#   is to advance the clones, so the next tick runs the current code" -- true
#   only when the tree it runs from is a tree it advances. It was not, so every
#   tick re-ran identical stale code and no merged fix could reach the host.
#
# Net: the default seeds a value the env file can read; the env file may then
# override it; the path is derived from whatever survives.
OMNI_HOME="${OMNI_HOME:-/data/omninode/omni_home}"

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

RECONCILER="${OMNI_HOME}/omnibase_infra/scripts/reconcile-host.sh"

if [[ ! -f "$RECONCILER" ]]; then
  echo "[workspace-reconcile] FATAL: no reconciler at $RECONCILER" >&2
  echo "[workspace-reconcile]   OMNI_HOME=$OMNI_HOME — is the deploy-source clone present?" >&2
  exit 3
fi

# No runtime assertion that these two agree: with the single assignment above
# they are the same expression, so any such check would be tautological -- dead
# code that reads like a safety net. The property is enforced statically instead,
# by tests/scripts/test_workspace_reconcile_wrapper_omn17365.py, which parses
# this file and fails if RECONCILER is ever assigned before the sourcing block
# again. That is the check that can actually go red.

exec env OMNI_HOME="$OMNI_HOME" bash "$RECONCILER" \
  --omni-home "$OMNI_HOME" \
  --branch "${RECONCILE_BRANCH:-dev}"

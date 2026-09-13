#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# =============================================================================
# Cloud pre-push capacity host bootstrap (OMN-16634)
# =============================================================================
# Provisions an Ubuntu 24.04 x86_64 EC2 instance as a governed pre-push
# capacity target for the OMN-16991 picker (scripts/hooks/prepush_dispatch.sh)
# AND as a runtime-capable ONEX target: the same toolchain (git + python3.12 +
# uv) is what the OMN-17226 per-host runner EFFECT node needs to execute on
# this box when the ONEX-native pre-push path lands. This host is designated
# on OMN-17226 as the FIRST cloud runner-EFFECT target; this script is the
# provisioning seam that the OMN-17226 runner image/manifest retires.
#
# NO hand-built installs (memory: feedback_no_hand_built_installs): this file
# is the committed record of everything on the host. Re-running it is
# idempotent. Anything installed on the host outside this script is drift.
#
# WHAT IS DELIBERATELY ABSENT:
#   * docker           -- the remote leg runs `pytest <targets>
#                         --ignore=tests/integration`; the unit+ci suites are
#                         proven hermetic on hosts with no docker in the run
#                         path (h101/h105 promotions, OMN-17161/OMN-16991).
#                         A test that needs a lab service on the private LAN
#                         here is a hermeticity DEFECT to be ticketed, never
#                         a reason to install docker or punch network paths.
#   * tailscale        -- operator directive: this host is reached by plain
#                         SSH through its locked security group. It is NOT a
#                         bus-attached LAN peer; platform-facing traffic goes
#                         through the platform gateway as a verified external
#                         actor (unified-external-ingress doctrine).
#   * GitHub credentials -- the picker transport is git-bundle + scp; the
#                         warm cache (warm_cache.sh) is bundle-fed too. Zero
#                         repo credentials ever land on this host.
#   * AWS credentials  -- idle auto-stop uses OS shutdown (the instance's
#                         shutdown behavior is `stop`), so the host needs no
#                         IAM role and holds no cloud API credentials.
#
# Usage, from a lab host (the only secretless path -- everything travels over
# the SSH trust already established for the picker):
#   scp deploy/prepush-cloud/bootstrap.sh \
#       deploy/prepush-cloud/onex-prepush-idle-stop.sh \
#       deploy/prepush-cloud/onex-prepush-idle-stop.service \
#       deploy/prepush-cloud/onex-prepush-idle-stop.timer \
#       <ssh-target>:/tmp/prepush-cloud/
#   ssh <ssh-target> 'bash /tmp/prepush-cloud/bootstrap.sh'
# Then warm the caches:
#   deploy/prepush-cloud/warm_cache.sh <ssh-target>
#
# -----------------------------------------------------------------------------
# WHERE THE INSTANCE IDENTITY LIVES: NOT HERE.
# -----------------------------------------------------------------------------
# This repository is PUBLIC. The instance id, address, security-group id and
# account are held in the private placement overlay, exactly as the `hcloud`
# row in scripts/hooks/prepush_hosts.tsv already states. Everything in this
# directory takes the host as an argument so no deployment identity is ever
# committed. Read the identity from the overlay:
#
#   CLOUD_SSH_TARGET="<user>@<address-from-the-private-placement-overlay>"
#   CLOUD_INSTANCE_ID="<instance-id-from-the-private-placement-overlay>"
#
# The SHAPE of the instance is not identifying and is worth recording: a
# 16-vCPU / 32-GiB compute instance, ON-DEMAND, 100 GiB gp3 root volume with
# DeleteOnTermination=true, InstanceInitiatedShutdownBehavior=stop, and a
# security group permitting inbound TCP/22 only from allowlisted CIDRs.
#
# On-demand rather than spot, deliberately: stop-on-idle is the cost mechanism
# and a one-time spot instance TERMINATES on OS shutdown, destroying the warm
# caches, while a spot interruption kills a governed suite mid-run. At the
# observed volume (~20-40 heavy escalations/month, <=1h each) the saving is a
# few dollars and the failure modes are not worth it.
#
# -----------------------------------------------------------------------------
# LIFECYCLE AND COST CONTROL
# -----------------------------------------------------------------------------
#   * Idle auto-stop: onex-prepush-idle-stop.timer, 10-minute cadence, stops
#     the instance after two consecutive idle observations (~20-30 min idle).
#     Idle = no heavy-suite LOCK, no pytest / `uv sync` / wrapper process, no
#     interactive session, no run-dir activity in 30 min, uptime > 30 min.
#     FAIL-ACTIVE: an unreadable signal keeps the box up. Never kill a governed
#     run to save cents.
#   * Restart on demand, lab side, needs AWS credentials:
#       aws ec2 start-instances --instance-ids "$CLOUD_INSTANCE_ID"
#     ~40s to SSH-reachable; the address is static so the host table needs no
#     edit.
#   * Picker behavior while stopped: the 3-second SSH probe fails, the row logs
#     `hcloud=unreachable`, and placement falls through to the lab rows. This is
#     what preserves lab-first placement -- the cloud row only competes when
#     someone has deliberately started the instance. Fail-closed semantics are
#     untouched: SKIP, never "assumed fit".
#   * Cost at observed volume: ~20-40 runs/month * <=1h * $0.8211/h, about
#     $16-33/month compute, plus $8/month EBS and ~$3.6/month static address
#     while stopped. A stopped month costs ~$12.
#
# This script substitutes the installing user's workroot into the idle-stop
# unit at install time. The COMMITTED unit carries the @ONEX_PREPUSH_WORKROOT@
# placeholder and the idle-stop script fails fast when the variable is unset,
# so no machine-specific path is checked in and a mis-installed unit cannot
# silently read "no activity" and stop a host mid-suite.
#
# -----------------------------------------------------------------------------
# COLLABORATOR ACCESS
# -----------------------------------------------------------------------------
# An authorized collaborator's public key is installed for the same user the
# picker dispatches as. Interactive use is an ordinary `ssh "$CLOUD_SSH_TARGET"`;
# a governed pre-push needs nothing special, because once their checkout carries
# the `hcloud` table row their hook's picker probes and dispatches here exactly
# as it does for lab hosts. Collaborators who reach the lab over the tailnet do
# NOT reach this host that way -- it deliberately does not join it -- so their
# public egress address is added to the security group once:
#
#   aws ec2 authorize-security-group-ingress \
#     --group-id "<security-group-id-from-the-private-placement-overlay>" \
#     --ip-permissions 'IpProtocol=tcp,FromPort=22,ToPort=22,\
# IpRanges=[{CidrIp=<address>/32,Description=<label>}]'
#
# -----------------------------------------------------------------------------
# HOST-TABLE ROW
# -----------------------------------------------------------------------------
# scripts/hooks/prepush_hosts.tsv, row `hcloud`. Memory+load probed like every
# other row; ranking unchanged (ascending load ratio among slot-free hosts).
# Per-repo tables in omnibase_core and omnimarket carry the row only with that
# repo's own transport proof, per the OMN-17159 / OMN-17435 discipline.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKROOT="${HOME}/onex-prepush"
UV_BIN="${HOME}/.local/bin/uv"
UV_FLOOR="0.11.0"

log() { printf 'bootstrap: %s\n' "$*"; }

# --- base toolchain ----------------------------------------------------------
# The suite shells out to tools by BARE NAME over the picker's non-interactive
# ssh PATH: `uv` (tests/unit/infra/test_catalog_cli.py), `shellcheck` (the
# shell-hygiene gate tests). build-essential covers any sdist native builds
# during `uv sync`. python3 is 3.12 on Ubuntu 24.04 (the repos' floor); uv
# fetches its own interpreter builds where a lockfile needs a different one.
log "apt packages"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
  git curl ca-certificates build-essential shellcheck python3-venv gh
# `gh` is the BINARY ONLY, never authenticated: the omnimarket dod_verify
# command-shape guards (tests/unit/nodes/node_dod_verify/test_omn_15597_*,
# test_omn_15391_*) resolve `gh` as an executable to distinguish a command
# from prose -- measured 2026-09-01: 13 of 16 residual reds on this host were
# exactly "first token 'gh' is not a resolvable executable". No `gh auth`
# is ever run here; the zero-credential posture of this host is unchanged.

# --- uv ----------------------------------------------------------------------
# Official installer into ~/.local/bin (the uv_abs_path the host-table row
# declares -- uv is on NO host's non-interactive PATH anywhere in the fleet,
# so the picker always invokes the absolute path).
if [ ! -x "$UV_BIN" ]; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv_version="$("$UV_BIN" --version | awk '{print $2}')"
if [ "$(printf '%s\n%s\n' "$UV_FLOOR" "$uv_version" | sort -V | head -1)" != "$UV_FLOOR" ]; then
  log "uv ${uv_version} below floor ${UV_FLOOR} -- self-updating"
  "$UV_BIN" self update
  uv_version="$("$UV_BIN" --version | awk '{print $2}')"
fi

# --- workroot ----------------------------------------------------------------
# Same layout every capacity row declares: bundles/, trees/, runs/, and the
# exclusive heavy-suite LOCK dir the remote wrapper takes at run time.
log "workroot ${WORKROOT}"
mkdir -p "${WORKROOT}/bundles" "${WORKROOT}/trees" "${WORKROOT}/runs" "${WORKROOT}/warm"

# --- idle auto-stop ----------------------------------------------------------
# Cost control: a systemd timer stops the instance when it is provably idle.
# See onex-prepush-idle-stop.sh for the idle definition (fail-ACTIVE: any
# unreadable signal keeps the box up rather than killing a governed run).
log "idle-stop units"
sudo install -m 0755 "${HERE}/onex-prepush-idle-stop.sh" /usr/local/bin/onex-prepush-idle-stop.sh
# The idle-stop unit runs as root, so it cannot infer the picker user's
# workroot. Bind it at install time from the workroot this run just created --
# the committed unit carries a placeholder, never a machine-specific path.
sed "s|@ONEX_PREPUSH_WORKROOT@|${WORKROOT}|" "${HERE}/onex-prepush-idle-stop.service" \
  | sudo tee /etc/systemd/system/onex-prepush-idle-stop.service > /dev/null
sudo chmod 0644 /etc/systemd/system/onex-prepush-idle-stop.service
sudo install -m 0644 "${HERE}/onex-prepush-idle-stop.timer" /etc/systemd/system/onex-prepush-idle-stop.timer
sudo systemctl daemon-reload
sudo systemctl enable --now onex-prepush-idle-stop.timer

# --- receipt -----------------------------------------------------------------
log "receipt"
printf 'host=%s\n' "$(hostname -s)"
printf 'kernel=%s arch=%s\n' "$(uname -r)" "$(uname -m)"
printf 'nproc=%s\n' "$(nproc)"
printf 'python3=%s\n' "$(python3 --version 2>&1)"
printf 'git=%s\n' "$(git --version)"
printf 'uv=%s (at %s)\n' "$uv_version" "$UV_BIN"
printf 'shellcheck=%s\n' "$(shellcheck --version | sed -n 's/^version: //p')"
printf 'idle_stop_timer=%s\n' "$(systemctl is-enabled onex-prepush-idle-stop.timer)"
log "done"

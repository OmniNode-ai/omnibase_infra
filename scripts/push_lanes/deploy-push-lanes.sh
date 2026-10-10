#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Install the versioned .201 push-lane queue scripts onto the queue host
# (OMN-17392, closing OMN-17221 DoD4).
#
# WHY THIS EXISTS: until this landed, the entire governed .201 pre-push queue --
# a 491-line durable FIFO runner with fcntl locking, an atomic journal, owner-
# only mode enforcement and a contract validator -- lived ONLY at
# ~/push-lanes/ on one machine. No history, no review, no backup, and no way to
# tell whether the thing running today is the thing anyone last reasoned about.
# Two separate tickets (OMN-16968, OMN-17221) have had to reconstruct its
# behavior by SSHing in and reading it.
#
# DIRECTION OF TRUTH: this repo is now the source, the host is the copy.
# Do not hand-edit ~/push-lanes/*.py on .201; edit here, land, then deploy.
#
# SAFE WHILE LANES ARE IN FLIGHT: install is via a temp file + `mv` on the same
# filesystem, and CPython reads a script fully at process start, so replacing
# the file cannot affect a queue-runner already running. The next invocation
# picks up the new copy.
#
# It deliberately does NOT touch ~/push-lanes/QUEUE, the journal, lanes/, or any
# in-flight process -- those are queue STATE, not code, and this script has no
# business in them (the OMN-17221 scoping constraint).

set -euo pipefail

HOST="${1:-jonah@192.168.86.201}"  # onex-allow-internal-ip OMN-17392 reason="the queue host is a fixed lab machine; the queue is host-local by construction"
DEST="push-lanes"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Every file this directory versions, and the mode it must land with. The queue
# artifacts are owner-only (0700 dir, 0600 files) by design -- queue-runner.py
# REFUSES to run against a queue whose parent dir or artifacts are group- or
# world-accessible, so a deploy that widened these modes would break the runner
# rather than merely loosen it.
#
# README-QUEUE.md was deployed here until OMN-16607 (epic OMN-16602). It was the
# only prose in the set -- the operator runbook for the queue, never read by the
# runner -- and it now lives at
# knowledge-base-internal:runbooks/omnibase-infra-heavy-prepush-queue.md. This
# array is the deploy contract, so dropping the file without dropping the row
# would make every deploy exit 1 on "versioned file missing".
FILES=(
  "queue-runner.py:700"
  "queue-runner.sh:700"
  "queue-contract-validator.py:700"
  "detect_foreign_prepush.py:600"
)

echo "[deploy-push-lanes] target: ${HOST}:~/${DEST}"

for entry in "${FILES[@]}"; do
  name="${entry%%:*}"
  mode="${entry##*:}"
  src="${HERE}/${name}"
  if [ ! -f "$src" ]; then
    echo "[deploy-push-lanes] FATAL: versioned file missing: ${src}" >&2
    exit 1
  fi
  # scp to a temp name then mv, so a partial transfer can never leave a
  # truncated runner in place.
  scp -q "$src" "${HOST}:${DEST}/.${name}.incoming"
  ssh -n "$HOST" "chmod ${mode} ${DEST}/.${name}.incoming && mv ${DEST}/.${name}.incoming ${DEST}/${name}"
  echo "[deploy-push-lanes] installed ${name} (mode ${mode})"
done

echo "[deploy-push-lanes] verifying deployed content matches this tree..."
rc=0
for entry in "${FILES[@]}"; do
  name="${entry%%:*}"
  local_sum="$(shasum -a 256 "${HERE}/${name}" | cut -d' ' -f1)"
  remote_sum="$(ssh -n "$HOST" "sha256sum ${DEST}/${name}" | cut -d' ' -f1)"
  if [ "$local_sum" != "$remote_sum" ]; then
    echo "[deploy-push-lanes] MISMATCH ${name}: local ${local_sum} remote ${remote_sum}" >&2
    rc=1
  fi
done
[ "$rc" -eq 0 ] || exit "$rc"

echo "[deploy-push-lanes] OK -- every versioned file is byte-identical on ${HOST}"

# OMN-20805: the foreign-prepush detector is scheduled by a user systemd timer,
# not a crontab line. A cron line records a start and no end; the journal of a
# oneshot service records every run's start, end and exit status. The timer is
# enabled and confirmed active BEFORE the legacy crontab line is removed, so a
# failed enable leaves the old schedule running rather than none.
UNIT_SRC="${HERE}/../../deploy/push-lanes"
UNIT_DIR=".config/systemd/user"
for unit in onex-foreign-prepush-detect.service onex-foreign-prepush-detect.timer; do
  [ -f "${UNIT_SRC}/${unit}" ] || { echo "[deploy-push-lanes] FATAL: versioned unit missing: ${UNIT_SRC}/${unit}" >&2; exit 1; }
  ssh -n "$HOST" "mkdir -p ${UNIT_DIR}"
  scp -q "${UNIT_SRC}/${unit}" "${HOST}:${UNIT_DIR}/${unit}"
  local_sum="$(shasum -a 256 "${UNIT_SRC}/${unit}" | cut -d' ' -f1)"
  remote_sum="$(ssh -n "$HOST" "sha256sum ${UNIT_DIR}/${unit}" | cut -d' ' -f1)"
  [ "$local_sum" = "$remote_sum" ] || { echo "[deploy-push-lanes] MISMATCH ${unit}: local ${local_sum} remote ${remote_sum}" >&2; exit 1; }
  echo "[deploy-push-lanes] installed ${unit}"
done
ssh -n "$HOST" "systemctl --user daemon-reload && systemctl --user enable --now onex-foreign-prepush-detect.timer && systemctl --user is-active --quiet onex-foreign-prepush-detect.timer"
ssh -n "$HOST" "crontab -l 2>/dev/null | grep -v 'omn16968-foreign-prepush-detect' | crontab -"
echo "[deploy-push-lanes] OK -- onex-foreign-prepush-detect.timer is active and the legacy crontab line is removed"

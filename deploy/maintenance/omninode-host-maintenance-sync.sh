#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# omninode-host-maintenance-sync.sh — install/verify the `.201` host maintenance
# artifacts that live in this repo but execute from outside any container.
#
# WHY THIS EXISTS (OMN-15525)
#   `deploy/maintenance/omninode-system-slack-report.sh` runs as root from
#   /data/maintenance/bin/ under /etc/cron.d/. No sanctioned deploy path covered
#   that directory: `deploy-runtime.sh` deploys containers, not host files. So
#   the script was hand-copied once, drifted for weeks, and the divergence was
#   invisible — OMN-15509 fixed the repo copy and changed nothing about what the
#   platform actually alarmed on, because nothing installs or checks the host
#   copy. `omnibase_infra#2572` merged and the monitor stayed blind.
#
#   The same shape already bit a second artifact (OMN-15521: the gateway
#   forwarder hand-deployed to root-owned /opt/omninode/gateway). Two host
#   artifacts sharing one structural gap is the argument for one install path
#   rather than another one-off copy, which is what this is.
#
#   Per CLAUDE.md rule 5 and `feedback_a_rule_is_not_a_mechanism`: a runbook step
#   saying "remember to copy the file" is not enforcement. `--check` runs on a
#   schedule and FAILS — non-zero exit, and a Slack alert with --slack — when an
#   installed artifact does not match `origin/dev`.
#
# WHAT IS COMPARED
#   The installed file's sha256 against the sha256 of the blob at
#   `origin/dev`, read with `git cat-file` after a fetch. Deliberately NOT the
#   clone's working tree: /data/omninode/omnibase_infra on .201 sat 40+ commits
#   behind `dev` while this was written, so a working-tree comparison would have
#   reported "in sync" against a stale checkout — a false green in the checker
#   built to catch false greens.
#
# FAIL-CLOSED
#   Missing host file, missing repo blob, failed fetch, unresolvable ref, or an
#   unreadable path is CRITICAL. "Could not determine" is never "fine".

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
INFRA_REPO_ROOT=${OMNINODE_INFRA_REPO_ROOT:-/data/omninode/omnibase_infra}
SYNC_REF=${OMNINODE_MAINTENANCE_SYNC_REF:-origin/dev}
SYNC_REMOTE=${OMNINODE_MAINTENANCE_SYNC_REMOTE:-origin}
SYNC_BRANCH=${OMNINODE_MAINTENANCE_SYNC_BRANCH:-dev}
ENV_FILE=${OMNINODE_ALERT_ENV_FILE:-/data/omninode/omnibase_infra/.env}
# Skip the network round-trip (tests, and any caller that already fetched).
SKIP_FETCH=${OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH:-0}

# repo-relative path | installed path | mode
#
# Adding a host artifact here is what makes it governed. An artifact absent from
# this manifest is exactly the OMN-15525 condition and will not be checked.
MANIFEST=(
  "deploy/maintenance/omninode-system-slack-report.sh|/data/maintenance/bin/omninode-system-slack-report.sh|0755"
  # OMN-15550. The reporter shells out to this probe from `collect()`, so an
  # un-synced copy is a silently blind detector -- exactly the OMN-15525
  # condition (merged, never deployed, nothing alarms) that this manifest
  # exists to make impossible.
  "scripts/omninode-ci-required-context-probe.py|/data/maintenance/bin/omninode-ci-required-context-probe.py|0755"
  "deploy/maintenance/cron.d/omninode-system-slack-report|/etc/cron.d/omninode-system-slack-report|0644"
  "deploy/maintenance/omninode-host-maintenance-sync.sh|/data/maintenance/bin/omninode-host-maintenance-sync.sh|0755"
  "deploy/maintenance/cron.d/omninode-host-maintenance-sync|/etc/cron.d/omninode-host-maintenance-sync|0644"
)

# Optional manifest override: a file of `relpath|hostpath|mode` lines, blank and
# `#` lines ignored. This exists so the detector can be exercised against
# scratch paths — both by the hermetic tests and by the OMN-15525 AC5 proof that
# it actually reddens — WITHOUT pointing a `--install` run at, or otherwise
# touching, the live root-owned artifacts. Never set in the cron unit.
MANIFEST_FILE=${OMNINODE_MAINTENANCE_SYNC_MANIFEST:-}
if [[ -n "$MANIFEST_FILE" ]]; then
  if [[ ! -r "$MANIFEST_FILE" ]]; then
    echo "FATAL: manifest $MANIFEST_FILE is unreadable" >&2
    exit 2
  fi
  MANIFEST=()
  while IFS= read -r line; do
    [[ -n "$line" && "$line" != \#* ]] || continue
    MANIFEST+=("$line")
  done <"$MANIFEST_FILE"
  (( ${#MANIFEST[@]} > 0 )) || {
    echo "FATAL: manifest $MANIFEST_FILE declares no artifacts" >&2
    exit 2
  }
fi

MODE=check
SLACK=0
for arg in "$@"; do
  case "$arg" in
    --check)   MODE=check ;;
    --install) MODE=install ;;
    --slack)   SLACK=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

die() {
  echo "FATAL: $*" >&2
  exit 2
}

[[ -d "$INFRA_REPO_ROOT/.git" ]] || die "no git clone at $INFRA_REPO_ROOT (set OMNINODE_INFRA_REPO_ROOT)"

if [[ "$SKIP_FETCH" != "1" ]]; then
  git -C "$INFRA_REPO_ROOT" fetch --quiet "$SYNC_REMOTE" \
    "+refs/heads/${SYNC_BRANCH}:refs/remotes/${SYNC_REMOTE}/${SYNC_BRANCH}" \
    || die "fetch of ${SYNC_REMOTE}/${SYNC_BRANCH} failed; cannot compare against $SYNC_REF"
fi

REF_SHA=$(git -C "$INFRA_REPO_ROOT" rev-parse --verify "$SYNC_REF" 2>/dev/null) \
  || die "cannot resolve $SYNC_REF in $INFRA_REPO_ROOT"

sha_of_stdin() { sha256sum | awk '{print $1}'; }

# sha256 of a path as it exists at $SYNC_REF, or empty when the blob is absent.
ref_blob_sha() {
  local relpath="$1"
  git -C "$INFRA_REPO_ROOT" cat-file blob "${SYNC_REF}:${relpath}" 2>/dev/null | sha_of_stdin
}

ref_blob_exists() {
  git -C "$INFRA_REPO_ROOT" cat-file -e "${SYNC_REF}:$1" 2>/dev/null
}

installed_sha() {
  local path="$1"
  [[ -r "$path" ]] || return 1
  sha256sum "$path" | awk '{print $1}'
}

drift_count=0
missing_count=0
report_lines=()

for entry in "${MANIFEST[@]}"; do
  IFS='|' read -r relpath hostpath mode <<<"$entry"

  if ! ref_blob_exists "$relpath"; then
    report_lines+=("CRITICAL|$hostpath|blob ${relpath} absent at ${SYNC_REF}")
    drift_count=$((drift_count + 1))
    continue
  fi
  want=$(ref_blob_sha "$relpath")
  if [[ -z "$want" ]]; then
    report_lines+=("CRITICAL|$hostpath|could not read ${relpath} at ${SYNC_REF}")
    drift_count=$((drift_count + 1))
    continue
  fi

  if [[ "$MODE" == "install" ]]; then
    tmp=$(mktemp)
    git -C "$INFRA_REPO_ROOT" cat-file blob "${SYNC_REF}:${relpath}" >"$tmp" \
      || die "failed to extract ${relpath} at ${SYNC_REF}"
    install -m "$mode" "$tmp" "${hostpath}.omn-sync.tmp" \
      || die "cannot write ${hostpath}.omn-sync.tmp (root required?)"
    # Rename is atomic: a cron run reading the old inode is never handed a
    # half-written script.
    mv -f "${hostpath}.omn-sync.tmp" "$hostpath" || die "cannot replace $hostpath"
    rm -f "$tmp"
  fi

  if ! have=$(installed_sha "$hostpath"); then
    report_lines+=("CRITICAL|$hostpath|NOT INSTALLED or unreadable (want ${want:0:12})")
    missing_count=$((missing_count + 1))
    drift_count=$((drift_count + 1))
    continue
  fi

  if [[ "$have" == "$want" ]]; then
    report_lines+=("OK|$hostpath|${have:0:12} matches ${SYNC_REF}")
  else
    report_lines+=("CRITICAL|$hostpath|DRIFT installed=${have:0:12} ${SYNC_REF}=${want:0:12}")
    drift_count=$((drift_count + 1))
  fi
done

echo "omninode host maintenance sync — mode=$MODE ref=$SYNC_REF (${REF_SHA:0:12}) repo=$INFRA_REPO_ROOT"
printf '%s\n' "${report_lines[@]}"
echo "drifted=$drift_count missing=$missing_count checked=${#MANIFEST[@]}"

if (( drift_count > 0 )) && (( SLACK == 1 )); then
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    set +u
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set -u
    set +a
  fi
  channel="${SLACK_CHANNEL_ID:-${SLACK_DEFAULT_CHANNEL:-}}"
  if [[ -n "${SLACK_BOT_TOKEN:-}" && -n "$channel" ]]; then
    text=$(printf '*OmniNode host maintenance drift*\nHost: %s\n%s host artifact(s) do not match `%s`.\n```\n%s\n```' \
      "$(hostname)" "$drift_count" "$SYNC_REF" "$(printf '%s\n' "${report_lines[@]}")")
    payload=$(jq -n --arg channel "$channel" --arg text "$text" \
      '{channel:$channel,text:$text,attachments:[{color:"danger",text:$text,mrkdwn_in:["text"]}]}')
    curl -fsS --retry 2 --max-time 10 \
      -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
      -H 'Content-Type: application/json; charset=utf-8' \
      -d "$payload" https://slack.com/api/chat.postMessage \
      | jq -e '.ok == true' >/dev/null || echo "WARNING: Slack post failed" >&2
  else
    echo "WARNING: --slack requested but no SLACK_BOT_TOKEN/channel in $ENV_FILE" >&2
  fi
fi

# Non-zero on drift is the enforcement: cron reddens, and any caller that gates
# on this script fails rather than logging a line nobody reads.
(( drift_count == 0 )) || exit 1

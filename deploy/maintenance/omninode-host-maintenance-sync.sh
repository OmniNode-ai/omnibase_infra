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
#
# THE FETCH BELOW WRITES INTO SOMEONE ELSE'S CLONE (OMN-17443)
#   The cron unit runs this as ROOT at :37. `$INFRA_REPO_ROOT` on `.201` is
#   owned by the OPERATOR, and `git fetch` writes objects, refs and reflogs --
#   so every hourly tick deposited root-owned paths into an operator-owned
#   clone, after which a plain `git pull` dies with `unpack-objects failed`.
#   Measured: 30 minutes after OMN-17366 repaired 2010 such paths, 32 were back,
#   all under `omnibase_infra/.git/objects`, all stamped `:37`.
#
#   That is OMN-17366's defect in a different job, so it takes OMN-17366's
#   guard, not a second one: `scripts/reconcile_privilege_lib.sh` decides who
#   the write runs as (`rp_plan_privileges`) and becomes them (`as_owner`).
#   Two implementations of a privilege rule drift, and the half that drifts is
#   the half nobody is watching.
#
#   The library is sourced FROM THE CLONE. This script is installed flat into
#   /data/maintenance/bin with no repo beside it, and the clone it syncs is the
#   same repository the library ships in. Sourcing from the tree under
#   verification is a real objection and is bounded the same way
#   omninode-workspace-reconcile.sh bounds it: this file and its cron unit are
#   both in the MANIFEST below, so a divergence between host copy and
#   `origin/dev` reddens this very check.
#
#   Only the FETCH is covered. `--install` writes /data/maintenance/bin and
#   /etc/cron.d, which are root-owned by design; dropping privilege there would
#   break the install and those are not the paths that broke.
#
# DETECTION IS NOT ENFORCEMENT -- WHY `--converge` EXISTS (OMN-17898)
#   The three modes are not interchangeable:
#
#     --check     compare and report. Writes nothing. Reddens on drift.
#     --install   write EVERY manifest entry unconditionally. The operator's
#                 bring-up verb; no before/after receipt, so it is not a thing
#                 to run on a timer.
#     --converge  compare, then write ONLY the entries that differ, read each
#                 written file back, and receipt every entry with its before
#                 and after sha256-12. This is the scheduled verb.
#
#   The cron unit ran `--check --slack` from OMN-15525 until OMN-17898. That is
#   a detector with no repair attached: drift was found hourly, alarmed on
#   hourly, and corrected only when a human happened to type `--install`.
#   Measured 2026-09-16: `omnibase_infra#3629` (26e734f6) fixed a false
#   `runtime-prod-28085` CRITICAL in the system reporter, merged, and the live
#   host went on posting the false alert because the installed copy was never
#   refreshed -- `--check` on `.201` read `drifted=1 missing=0 checked=7`. Per
#   CLAUDE.md rule 5, a detector that is not wired to a repair is advisory and
#   gets ignored.
#
#   The objection the earlier revision recorded -- "--install from cron would
#   silently overwrite host state on every tick" -- was correct about
#   `--install` and is what `--converge` answers rather than waives. An in-sync
#   tick writes nothing and does not even replace the inode; every write names
#   the bytes it replaced and the bytes it wrote; and a write that does not
#   read back as the ref's bytes is a FAILURE, not a success with a warning.
#
#   WHAT THIS ACCEPTS, SAID PLAINLY: `origin/dev` now reaches these seven host
#   paths without a human in the loop. That is the same trust boundary the
#   deploy agent and the workspace reconciler already run on for this host, and
#   the files are CI-gated on the way to `dev`. The alternative on offer was
#   not "a human reviews each change" -- it was "nobody installs it at all",
#   which is the condition above.
#
#   ALERTING: only a converge that FAILS pages. A successful self-heal is the
#   mechanism working, and a channel that fires on every success is a channel
#   nobody reads when the real failure arrives.
#
#   KNOWN, BOUNDED, NOT CLOSED HERE (OMN-15580): bash parses MANIFEST at
#   process start, so a tick that converges a NEW version of this script is
#   still running the old manifest and will not install an entry that version
#   added. Scheduling the converge changes that window from unbounded (it
#   needed a human) to one hour (the next tick). Closing it needs the re-exec
#   in OMN-15580 and is deliberately not attempted here.

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
INFRA_REPO_ROOT=${OMNINODE_INFRA_REPO_ROOT:-/data/omninode/omnibase_infra}
SYNC_REF=${OMNINODE_MAINTENANCE_SYNC_REF:-origin/dev}
SYNC_REMOTE=${OMNINODE_MAINTENANCE_SYNC_REMOTE:-origin}
SYNC_BRANCH=${OMNINODE_MAINTENANCE_SYNC_BRANCH:-dev}
ENV_FILE=${OMNINODE_ALERT_ENV_FILE:-/data/omninode/omnibase_infra/.env}
# Skip the network round-trip (tests, and any caller that already fetched).
SKIP_FETCH=${OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH:-0}
# The one privilege rule, sourced rather than re-implemented (OMN-17443).
PRIVILEGE_LIB="$INFRA_REPO_ROOT/scripts/reconcile_privilege_lib.sh"

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
  # OMN-17311. The workspace reconciler's scheduler adapter and its cron unit.
  # These are in the manifest for the reason the manifest exists: an artifact
  # that is merged but never installed, with nothing alarming, is the OMN-15525
  # condition. It bit twice before this file was written, and a reconciler that
  # is not actually running is exactly as useless as a monitor that cannot see —
  # with the extra harm that the workspace looks governed while it drifts.
  #
  # The reconciler ITSELF is deliberately not listed: it is executed from the
  # deploy-source clone (${OMNI_HOME}/omnibase_infra/scripts/reconcile-host.sh)
  # so its collaborators resolve, and it is kept current by the reconcile it
  # performs. Only the two host-resident files need this guard.
  "deploy/maintenance/omninode-workspace-reconcile.sh|/data/maintenance/bin/omninode-workspace-reconcile.sh|0755"
  "deploy/maintenance/cron.d/omninode-workspace-reconcile|/etc/cron.d/omninode-workspace-reconcile|0644"
  # OMN-18567. The deploy runner's private clone tree converger and its cron
  # unit. Listing them here IS the installation: the hourly --converge writes
  # every entry that differs from origin/dev and reads it back, so the tick
  # reaches the host by merging to `dev` rather than by anyone editing a
  # crontab. Both files are host-resident and neither resolves a collaborator
  # relative to itself, which is why both are listed -- unlike the workspace
  # reconciler proper, which is executed from the clone so its collaborators
  # resolve and is therefore deliberately absent.
  "deploy/maintenance/omninode-runner-tree-converge.sh|/data/maintenance/bin/omninode-runner-tree-converge.sh|0755"
  "deploy/maintenance/cron.d/omninode-runner-tree-converge|/etc/cron.d/omninode-runner-tree-converge|0644"
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
    --check)    MODE=check ;;
    --install)  MODE=install ;;
    --converge) MODE=converge ;;
    --slack)    SLACK=1 ;;
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
  # The privilege plan is required only on the path that WRITES. With
  # SKIP_FETCH the caller already fetched and this run touches no clone, so
  # demanding a plan there would be ceremony -- and ceremony is what gets
  # loosened later in the one script that must stay fail-closed.
  [[ -f "$PRIVILEGE_LIB" ]] || die "privilege library missing at $PRIVILEGE_LIB; without it there is no way to know who owns $INFRA_REPO_ROOT, and fetching as whoever this process happens to be is the OMN-17443 defect (set OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH=1 only if the caller already fetched)"
  # shellcheck source=../../scripts/reconcile_privilege_lib.sh
  source "$PRIVILEGE_LIB"

  plan_rc=0
  rp_plan_privileges "$INFRA_REPO_ROOT" || plan_rc=$?
  case "$plan_rc" in
    0) ;;
    1) die "cannot read the owner of $INFRA_REPO_ROOT; the fetch below writes objects, refs and reflogs into it, and writing as the wrong user leaves a clone its owner can no longer fetch into (OMN-17443)" ;;
    2) die "$INFRA_REPO_ROOT is owned by '${RP_OWNER}' and this process (${CURRENT_USER}) cannot become that user, so the fetch is REFUSED rather than performed as the wrong user (OMN-17443). Run this as ${RP_OWNER}, or as root on a host with runuser" ;;
    3) die "$INFRA_REPO_ROOT is owned by '${RP_OWNER}', whose home directory cannot be resolved. Dropping to them without HOME would make git read root's .gitconfig and credentials -- a confusing failure two layers from its cause" ;;
  esac
  if [[ -n "$RP_OWNER" && "$RP_OWNER" != "$CURRENT_USER" ]]; then
    # Announced, because a silent privilege drop cannot be audited from a cron log.
    echo "writing as $RP_OWNER (owner of $INFRA_REPO_ROOT)" >&2
  fi

  as_owner git -C "$INFRA_REPO_ROOT" fetch --quiet "$SYNC_REMOTE" \
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

# Write one manifest entry from the ref to its host path. Returns non-zero on
# any failure instead of calling die(): in --converge the run must finish and
# receipt EVERY artifact, because "it failed" and "nobody ran it" have to stay
# distinguishable from the output alone.
install_blob() {
  local relpath="$1" hostpath="$2" mode="$3"
  local tmp staged
  staged="${hostpath}.omn-sync.tmp"
  tmp=$(mktemp) || return 1
  if ! git -C "$INFRA_REPO_ROOT" cat-file blob "${SYNC_REF}:${relpath}" >"$tmp" 2>/dev/null; then
    rm -f "$tmp"
    return 1
  fi
  if ! install -m "$mode" "$tmp" "$staged" 2>/dev/null; then
    rm -f "$tmp" "$staged"
    return 1
  fi
  rm -f "$tmp"
  # Rename is atomic: a cron run reading the old inode is never handed a
  # half-written script.
  if ! mv -f "$staged" "$hostpath" 2>/dev/null; then
    rm -f "$staged"
    return 1
  fi
  return 0
}

drift_count=0
missing_count=0
converged_count=0
failed_count=0
in_sync_count=0
report_lines=()

for entry in "${MANIFEST[@]}"; do
  IFS='|' read -r relpath hostpath mode <<<"$entry"

  # Unknown known-good bytes. Fail-closed, and in --converge that also means
  # the host copy is left exactly as found: there is nothing to converge TO,
  # and writing something else would be worse than the drift.
  if ! ref_blob_exists "$relpath"; then
    report_lines+=("CRITICAL|$hostpath|blob ${relpath} absent at ${SYNC_REF}")
    drift_count=$((drift_count + 1))
    if [[ "$MODE" == "converge" ]]; then failed_count=$((failed_count + 1)); fi
    continue
  fi
  want=$(ref_blob_sha "$relpath")
  if [[ -z "$want" ]]; then
    report_lines+=("CRITICAL|$hostpath|could not read ${relpath} at ${SYNC_REF}")
    drift_count=$((drift_count + 1))
    if [[ "$MODE" == "converge" ]]; then failed_count=$((failed_count + 1)); fi
    continue
  fi

  if [[ "$MODE" == "converge" ]]; then
    have=$(installed_sha "$hostpath") || have=""
    if [[ -n "$have" && "$have" == "$want" ]]; then
      report_lines+=("OK|$hostpath|${have:0:12} matches ${SYNC_REF}")
      in_sync_count=$((in_sync_count + 1))
      continue
    fi

    if [[ -n "$have" ]]; then before="${have:0:12}"; else before="absent"; fi

    if install_blob "$relpath" "$hostpath" "$mode"; then
      # Read the file back rather than trusting the write. An install that
      # reports success and leaves the wrong bytes is the failure mode this
      # whole surface exists to make impossible.
      after=$(installed_sha "$hostpath") || after=""
      if [[ "$after" == "$want" ]]; then
        report_lines+=("CONVERGED|$hostpath|before=${before} after=${after:0:12} matches ${SYNC_REF}")
        converged_count=$((converged_count + 1))
      else
        if [[ -n "$after" ]]; then after_short="${after:0:12}"; else after_short="unreadable"; fi
        report_lines+=("CRITICAL|$hostpath|CONVERGE FAILED before=${before} after=${after_short} want=${want:0:12} (readback does not match ${SYNC_REF})")
        failed_count=$((failed_count + 1))
      fi
    else
      report_lines+=("CRITICAL|$hostpath|CONVERGE FAILED before=${before} after=${before} want=${want:0:12} (write refused -- unwritable path, or not running as root?)")
      failed_count=$((failed_count + 1))
    fi
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
if [[ "$MODE" == "converge" ]]; then
  echo "converged=$converged_count failed=$failed_count in_sync=$in_sync_count checked=${#MANIFEST[@]}"
else
  echo "drifted=$drift_count missing=$missing_count checked=${#MANIFEST[@]}"
fi

# What pages. In --check, drift is the finding. In --converge, drift is the
# ordinary case the mechanism exists to absorb -- only a converge that could
# NOT repair is worth a human's attention, and alerting on the successful
# self-heals is how the channel stops being read.
alert=0
if [[ "$MODE" == "converge" ]]; then
  if (( failed_count > 0 )); then alert=1; fi
else
  if (( drift_count > 0 )); then alert=1; fi
fi

if (( alert == 1 )) && (( SLACK == 1 )); then
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
    if [[ "$MODE" == "converge" ]]; then
      headline='*OmniNode host maintenance CONVERGE FAILED*'
      detail="$failed_count host artifact(s) could not be converged onto \`$SYNC_REF\`. The host is still running the old bytes."
    else
      headline='*OmniNode host maintenance drift*'
      detail="$drift_count host artifact(s) do not match \`$SYNC_REF\`."
    fi
    text=$(printf '%s\nHost: %s\n%s\n```\n%s\n```' \
      "$headline" "$(hostname)" "$detail" "$(printf '%s\n' "${report_lines[@]}")")
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

# Non-zero is the enforcement: cron reddens, and any caller that gates on this
# script fails rather than logging a line nobody reads. In --check the failing
# condition is drift; in --converge it is drift that could not be REPAIRED,
# because a repaired artifact is the mechanism succeeding, not a fault.
if [[ "$MODE" == "converge" ]]; then
  (( failed_count == 0 )) || exit 1
else
  (( drift_count == 0 )) || exit 1
fi
exit 0

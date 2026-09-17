#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# omninode-runner-tree-converge.sh -- converge the deploy runner's PRIVATE
# OMNI_HOME clone tree on a schedule (OMN-18567).
#
# WHY THIS EXISTS
#   `/data/omninode/runner_omni_home` is the deploy runner's own OMNI_HOME:
#   supplied host-side as DEPLOY_RUNNER_OMNI_HOME, bind-mounted into
#   `omninode-deploy-runner` at the identical path, owned by the runner uid, and
#   deliberately NOT the shared operator clone tree. It is the build source the
#   dev-lane refresh, the stability-lane refresh and the release-train tag cut
#   all read.
#
#   Nothing converged it. On 2026-09-17 five of its six clones were on detached
#   HEADs and every one of the six was behind `origin/dev` -- `omnibase_infra`
#   by 294 commits, `onex_change_control` by 1120. Two mechanisms already run on
#   this host and neither reaches this tree by design: the host maintenance sync
#   converges host-resident FILES and touches no git clone, and the workspace
#   reconciler is scoped to the shared operator-owned tree. The tree's own
#   provisioning helper (`scripts/runtime_build/ensure_runner_clones.sh`) clones
#   when a clone is MISSING and never advances one that already exists, so the
#   tree moved only as an incidental side effect of refresh runs fetching for
#   their own purposes.
#
#   The one-time repair is done. Per CLAUDE.md rule 5 a repair nobody schedules
#   is advisory, so this is the recurrence half.
#
# WHAT IT DOES
#   For every direct child of the tree that is a git clone: fetch, then delegate
#   the repair to `converge-canonical-clone.sh` -- the ONE sanctioned converger,
#   which preserves before it resets (per-area patches, the full diff vs HEAD,
#   untracked copies, reflog, sha256 manifest), re-attaches a detached HEAD to a
#   DERIVED branch, and returns a clone to its own branch when it has been left
#   on the wrong one. This file re-implements none of that. It decides WHEN the
#   converge may run, WHICH clones it covers, and WHAT is receipted.
#
# THE CLONE SET IS THE TREE ITSELF, NOT A LIST
#   Discovery is "every direct child with a .git DIRECTORY" (a worktree carries
#   a .git FILE and is never converged here). A hand-maintained repo list is the
#   OMN-15137 defect -- `omnibase_spi` was added to one sibling list, forgotten
#   in the other, and the gap surfaced three deploy hops later. It would also be
#   wrong today: `sibling_clone_manifest.sh` names five repos and this tree holds
#   six, because `onex_change_control` arrives by a different path. The tree is
#   the only list that cannot drift from the tree.
#
# WHEN IT MAY RUN -- THE BUSY GATE, AND WHY IT IS SHAPED THIS WAY
#   A `reset --hard` underneath a running deploy job corrupts that job's build
#   source. Three host-local signals decide, and they do not have equal standing:
#
#     IDLE PROOF (required, must read as a clean zero)
#       `docker top omninode-deploy-runner | grep -c Runner.Worker`. That
#       container is the ONLY one with this tree mounted -- the verify runner
#       shares the runner group and is deliberately built without the bind -- so
#       no job worker in it means nothing inside the container can be touching
#       the tree. `docker top` is daemon-side: it creates no process in the
#       container, which matters because a bare `docker exec` here defaults to
#       ROOT and has already left root-owned files under a job workspace
#       (documented in `docker/runners/runner-job-started.sh`).
#
#       An EMPTY answer is UNKNOWN, never zero. `... | grep -c X || true` exits
#       0 with no output when the container is gone or the daemon errors, and
#       reading that as "zero workers" is the fail-open shape that killed a live
#       job once already (`scripts/deploy-runners.sh:1141`).
#
#     VETOES (a hit refuses; an absence proves nothing on its own)
#       * a process whose cwd or open fd resolves inside the tree. This is the
#         only signal that sees an OUT-OF-BAND writer -- anyone hand-running a
#         lane refresh or the tag cut on `.201` with OMNI_HOME pointed here is
#         invisible to the container probe. The container/host bind is
#         path-identical, so a container process's cwd readlinks to the same
#         host path string.
#       * an in-flight git operation in any clone. `Runner.Worker` can be gone
#         while a git child it spawned is mid-checkout. The marker set is the one
#         `scripts/git-gc-auto.sh:76` already uses for "do not touch this object
#         store right now".
#
#   THE GITHUB `busy` FLAG IS DELIBERATELY NOT CONSULTED, and that is a
#   departure from `scripts/deploy-runners.sh`, which requires it. Root on `.201`
#   holds no credential with that scope: `gh` as root is unauthenticated, and the
#   `GH_PAT` in the operator env file returns 403 on
#   `/orgs/OmniNode-ai/actions/runners` (verified 2026-09-17). The only working
#   credential is the operator's personal oauth token, and a scheduled mechanism
#   that depends on one person's PAT fails silently the day it expires. Requiring
#   a signal that cannot be obtained would make every tick refuse forever, which
#   is AC1 unreachable dressed as caution. The worker-process probe answers the
#   question that actually matters here -- "is something RUNNING in the container
#   that holds the tree" -- while GitHub's flag answers "is a job ASSIGNED"; the
#   gap between them is the assignment-to-spawn window, covered below.
#
#   THE IRREDUCIBLE RACE, STATED. The runner can accept a queued job at any
#   instant and nothing on this host can hold it off. `deploy-runners.sh`
#   documents the same window even with both signals. It is bounded here by
#   re-running the gate immediately before EACH clone and aborting the rest of
#   the run the moment it changes, so at most one clone is exposed rather than
#   six -- and by scheduling hourly rather than by the minute, so the exposure
#   is rare as well as small.
#
# WHO IT WRITES AS -- AC3
#   cron runs this as root; the tree is owned by the runner uid. A root converge
#   rewrites working-tree files as root and leaves a tree the runner can no
#   longer write, turning a stale clone into a broken one -- strictly worse than
#   the drift. So every command that touches the tree goes through `as_owner`
#   from `scripts/reconcile_privilege_lib.sh`, the ONE privilege rule (OMN-17366,
#   OMN-17443). It is sourced, never re-implemented: two copies of a privilege
#   rule drift, and the half that drifts is the half nobody is watching.
#   `scripts/check_reconciler_privilege.py` enforces that for the bare `git`
#   writes below; the delegation to the converge script -- where every reset
#   actually happens -- is pinned by
#   `tests/scripts/test_runner_tree_converge_omn18567.py`.
#
# THE CONVERGER IS RUN FROM THE CLONE, AND ADVANCED FIRST
#   `converge-canonical-clone.sh` lives in omniclaude, which this repo's host
#   manifest cannot install (the manifest reads blobs out of THIS repository).
#   It is therefore executed from the operator deploy tree, and nothing else on
#   this host advances that clone -- the workspace reconciler's manifest names
#   five python siblings and omniclaude is not among them. An un-advanced
#   dependency is the OMN-15525 condition, so this tick fast-forwards that clone
#   itself, as its owner, before using it. Fast-forward only: a clone that cannot
#   fast-forward is reported in the receipt and the run proceeds on the bytes it
#   has, because a stale converger is bounded and a skipped convergence is not.
#
# WHAT IT RECEIPTS
#   One verdict line, to stdout and to $STATE_DIR/runner-tree-converge.status:
#
#     runner-tree-converge|<VERDICT>|ts=|tree=|clones=|in_sync=|converged=|failed=|skipped=|reason=|last_success=|detail=
#
#   VERDICT is one of IN_SYNC, CONVERGED, DRIFTED (--check only), REFUSED,
#   FAILED, PRECONDITION. `detail` names every clone as
#   `<name>:<before12>-><after12>`, so what moved is auditable after the fact.
#   `omninode-system-slack-report.sh` reads that file, which is what makes a
#   tick that FAILS -- or one that silently stopped running -- reach a human.
#
#   `last_success` advances ONLY on a completed run with nothing failed and
#   nothing skipped. That is what keeps a permanently busy runner from looking
#   identical to a permanently converged tree, and it is why exiting 0 on a
#   refusal is safe: a refusal is the guard working, and reddening cron on every
#   busy tick is how a channel stops being read.
#
# movement-proof: every clone's HEAD is re-read from git AFTER the converge
#   delegate returns, and the before/after pair is receipted per clone. A clone
#   whose HEAD did not reach its upstream counts as FAILED even when the
#   delegate exited 0.
#
# Exit: 0 IN_SYNC / CONVERGED / DRIFTED / REFUSED; 1 FAILED; 2 precondition.

set -uo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

TREE=${OMNINODE_RUNNER_TREE:-/data/omninode/runner_omni_home}
DEPLOY_OMNI_HOME=${OMNINODE_DEPLOY_OMNI_HOME:-/data/omninode/omni_home}
CONVERGE_SCRIPT=${OMNINODE_RUNNER_TREE_CONVERGE_SCRIPT:-$DEPLOY_OMNI_HOME/omniclaude/scripts/converge-canonical-clone.sh}
INFRA_REPO_ROOT=${OMNINODE_INFRA_REPO_ROOT:-/data/omninode/omnibase_infra}
STATE_DIR=${OMNINODE_RUNNER_TREE_STATE_DIR:-/data/maintenance/state}
RUNNER_CONTAINER=${OMNINODE_RUNNER_TREE_CONTAINER:-omninode-deploy-runner}
TICKET=${OMNINODE_RUNNER_TREE_TICKET:-OMN-18567}
LANE=${OMNINODE_RUNNER_TREE_LANE:-runner-tree-converge}
SKIP_SCRIPT_REFRESH=${OMNINODE_RUNNER_TREE_SKIP_SCRIPT_REFRESH:-0}
PRIVILEGE_LIB="$INFRA_REPO_ROOT/scripts/reconcile_privilege_lib.sh"
STATUS_FILE="$STATE_DIR/runner-tree-converge.status"

# The two host-only probes, as overridable COMMANDS. Each one is a seam the
# hermetic tests drive, because neither a docker daemon nor a procfs full of the
# right processes can be constructed in a test. The DEFAULTS are the real
# probes and are pinned by `test_default_probes_are_host_local_and_named`, so
# the seam cannot quietly become the implementation.
#
# WORKER_PROBE must print a COUNT on stdout and exit 0. Empty output or a
# non-zero exit is UNKNOWN. `|| true` is deliberate and is why empty has to be
# handled separately from "0": grep -c exits 1 on no match, and conflating those
# two is the fail-open bug.
WORKER_PROBE_CMD=${OMNINODE_RUNNER_TREE_WORKER_PROBE_CMD:-"docker top $RUNNER_CONTAINER 2>/dev/null | grep -c 'Runner.Worker' || true"}
# PROC_SCAN must print one line per process found inside the tree, and exit 0
# even when it finds none. `-lname` matches a symlink's TARGET without a fork
# per descriptor, which keeps a whole-procfs sweep near a second on a busy host.
PROC_SCAN_CMD=${OMNINODE_RUNNER_TREE_PROC_SCAN_CMD:-"find /proc -mindepth 2 -maxdepth 3 \\( -name cwd -o -path '/proc/[0-9]*/fd/*' \\) -lname '$TREE/*' 2>/dev/null"}

# The marker set from scripts/git-gc-auto.sh:76 -- the existing precedent in
# this repo for "a git operation is in flight, leave this object store alone".
GIT_OP_MARKERS="index.lock rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG"

MODE=check
for arg in "$@"; do
  case "$arg" in
    --check)    MODE=check ;;
    --converge) MODE=converge ;;
    -h|--help)  sed -n '2,60p' "$0"; exit 0 ;;
    *)          echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

say() { echo "[runner-tree-converge] $*"; }

# Read the previous run's last_success so a refusal cannot erase the fact that
# the tree WAS converged at some point. Absent or unparseable reads as never,
# which ages immediately and is the safe direction.
last_success="never"
if [[ -r "$STATUS_FILE" ]]; then
  prior="$(sed -n 's/.*|last_success=\([^|]*\)|.*/\1/p' "$STATUS_FILE" 2>/dev/null | tail -n1)"
  [[ -n "$prior" ]] && last_success="$prior"
fi

emit_verdict() {
  local verdict="$1" reason="$2" clones="$3" in_sync="$4" converged="$5" failed="$6" skipped="$7" detail="$8"
  local line
  line="runner-tree-converge|${verdict}|ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)|tree=${TREE}|clones=${clones}|in_sync=${in_sync}|converged=${converged}|failed=${failed}|skipped=${skipped}|reason=${reason}|last_success=${last_success}|detail=${detail}"
  echo "$line"
  if mkdir -p "$STATE_DIR" 2>/dev/null; then
    printf '%s\n' "$line" >"$STATUS_FILE" 2>/dev/null \
      || echo "WARNING: could not write $STATUS_FILE; the Slack reporter will age this tick as missing" >&2
  else
    echo "WARNING: could not create $STATE_DIR; the Slack reporter will age this tick as missing" >&2
  fi
}

precondition() {
  say "PRECONDITION: $*"
  emit_verdict PRECONDITION "$1" 0 0 0 0 0 ""
  exit 2
}

# --------------------------------------------------------------------------- #
# preconditions
# --------------------------------------------------------------------------- #
[[ -d "$TREE" ]] || precondition "no tree at $TREE (set OMNINODE_RUNNER_TREE)"
[[ -f "$CONVERGE_SCRIPT" ]] || precondition "no converger at $CONVERGE_SCRIPT; this tick delegates every repair to it and re-implements none of it"
[[ -f "$PRIVILEGE_LIB" ]] || precondition "privilege library missing at $PRIVILEGE_LIB; without it there is no way to know who owns $TREE, and converging as whoever this process happens to be is the OMN-17443 defect"

# shellcheck source=../../scripts/reconcile_privilege_lib.sh
source "$PRIVILEGE_LIB"

plan_rc=0
rp_plan_privileges "$TREE" || plan_rc=$?
case "$plan_rc" in
  0) ;;
  1) precondition "cannot read the owner of $TREE; every command below rewrites working-tree files, and running them as the wrong user leaves a tree the runner cannot write" ;;
  2) precondition "$TREE is owned by '${RP_OWNER}' and this process (${CURRENT_USER}) cannot become that user, so the converge is REFUSED rather than performed as the wrong user. Run this as ${RP_OWNER}, or as root on a host with runuser" ;;
  3) precondition "$TREE is owned by '${RP_OWNER}', whose home directory cannot be resolved. Dropping to them without HOME would make git read root's .gitconfig and credentials" ;;
esac
if [[ -n "$RP_OWNER" && "$RP_OWNER" != "$CURRENT_USER" ]]; then
  # Announced, because a silent privilege drop cannot be audited from a cron log.
  say "writing as $RP_OWNER (owner of $TREE)"
fi

# --------------------------------------------------------------------------- #
# the busy gate
# --------------------------------------------------------------------------- #
# Echoes "idle", "busy:<why>" or "unknown:<why>". Never exits; the caller
# decides. Both non-idle answers refuse -- the distinction is kept only so the
# receipt says which, because "a job was running" and "we could not look" are
# different operational facts even though they lead to the same action.
busy_state() {
  local workers
  if ! workers="$(eval "$WORKER_PROBE_CMD" 2>/dev/null)"; then
    echo "unknown:worker_probe_failed"
    return 0
  fi
  workers="${workers//[$'\r\n\t ']/}"
  if [[ -z "$workers" ]]; then
    echo "unknown:worker_probe_empty"
    return 0
  fi
  if [[ ! "$workers" =~ ^[0-9]+$ ]]; then
    echo "unknown:worker_probe_unparseable"
    return 0
  fi
  if (( workers > 0 )); then
    echo "busy:job_worker_running"
    return 0
  fi

  local hits
  if ! hits="$(eval "$PROC_SCAN_CMD" 2>/dev/null)"; then
    echo "unknown:proc_scan_failed"
    return 0
  fi
  if [[ -n "${hits//[$'\r\n\t ']/}" ]]; then
    echo "busy:in_tree_process"
    return 0
  fi

  local clone marker
  for clone in "$TREE"/*/; do
    [[ -d "${clone}.git" ]] || continue
    for marker in $GIT_OP_MARKERS; do
      if [[ -e "${clone}.git/${marker}" ]]; then
        echo "busy:git_operation_in_flight"
        return 0
      fi
    done
  done

  echo "idle"
}

# --------------------------------------------------------------------------- #
# clone discovery -- the tree is the list
# --------------------------------------------------------------------------- #
clones=()
for candidate in "$TREE"/*/; do
  [[ -d "${candidate}.git" ]] || continue
  clones+=("$(basename "$candidate")")
done
clone_count=${#clones[@]}
if (( clone_count == 0 )); then
  precondition "no git clones found directly under $TREE; a tree with nothing in it is a provisioning failure, not a converged tree"
fi

# --------------------------------------------------------------------------- #
# advance the converger before running it
# --------------------------------------------------------------------------- #
script_note=""
if [[ "$SKIP_SCRIPT_REFRESH" != "1" ]]; then
  script_clone="$(cd "$(dirname "$CONVERGE_SCRIPT")/.." && pwd -P)"
  if [[ -d "$script_clone/.git" ]]; then
    script_plan_rc=0
    # The converger's own clone has its own owner, which need not be the tree's.
    rp_plan_privileges "$script_clone" || script_plan_rc=$?
    if (( script_plan_rc == 0 )); then
      as_owner git -C "$script_clone" fetch --quiet origin 2>/dev/null \
        || script_note="converger_fetch_failed"
      if [[ -z "$script_note" ]]; then
        as_owner git -C "$script_clone" merge --ff-only --quiet '@{u}' 2>/dev/null \
          || script_note="converger_not_fast_forwardable"
      fi
    else
      script_note="converger_owner_unresolved"
    fi
    say "converger at $(git -C "$script_clone" rev-parse --short HEAD 2>/dev/null || echo unknown)${script_note:+ (${script_note})}"
    # Re-plan for the tree: rp_plan_privileges above overwrote RUN_AS with the
    # converger clone's owner, and every command from here on touches the TREE.
    rp_plan_privileges "$TREE" || precondition "the tree's privilege plan could not be re-established after refreshing the converger"
  fi
fi

# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
gate="$(busy_state)"
if [[ "$gate" != "idle" ]]; then
  say "REFUSED: ${gate}"
  emit_verdict REFUSED "${gate}" "$clone_count" 0 0 0 "$clone_count" ""
  exit 0
fi

in_sync=0
converged=0
failed=0
skipped=0
details=()
abort_reason=""

# Is this clone at its upstream, attached to the branch the remote publishes,
# and clean? Anything else is drift the converger is asked to repair.
clone_is_converged() {
  local clone="$1" branch upstream head target default_branch probe_remote
  branch="$(git -C "$clone" symbolic-ref -q --short HEAD 2>/dev/null)" || return 1
  [[ -n "$branch" ]] || return 1
  upstream="$(git -C "$clone" rev-parse --abbrev-ref --symbolic-full-name "${branch}@{u}" 2>/dev/null)" || return 1
  probe_remote="${upstream%%/*}"
  default_branch="$(git -C "$clone" symbolic-ref -q --short "refs/remotes/${probe_remote}/HEAD" 2>/dev/null)"
  default_branch="${default_branch#"${probe_remote}/"}"
  # A clone parked on a feature branch has a perfectly good upstream of its own,
  # which is how OMN-16497 read as a no-op SUCCESS for two days. When the remote
  # publishes a default branch, being on a different one IS drift.
  if [[ -n "$default_branch" && "$default_branch" != "$branch" ]]; then
    return 1
  fi
  head="$(git -C "$clone" rev-parse HEAD 2>/dev/null)" || return 1
  target="$(git -C "$clone" rev-parse "$upstream" 2>/dev/null)" || return 1
  [[ "$head" == "$target" ]] || return 1
  [[ -z "$(git -C "$clone" status --porcelain 2>/dev/null)" ]] || return 1
  return 0
}

for name in "${clones[@]}"; do
  clone="$TREE/$name"

  if [[ -n "$abort_reason" ]]; then
    skipped=$((skipped + 1))
    details+=("${name}:skipped")
    continue
  fi

  # Re-check immediately before touching this clone: the runner can accept a
  # queued job at any instant, and the selection pass above is already history.
  gate="$(busy_state)"
  if [[ "$gate" != "idle" ]]; then
    abort_reason="$gate"
    skipped=$((skipped + 1))
    details+=("${name}:skipped")
    continue
  fi

  before="$(git -C "$clone" rev-parse HEAD 2>/dev/null)" || before=""
  if [[ -z "$before" ]]; then
    failed=$((failed + 1))
    details+=("${name}:unreadable->unreadable")
    say "FAILED ${name}: HEAD is unreadable"
    continue
  fi

  # Establish the target before judging drift. Without this a clone whose
  # remote-tracking refs are stale reads as converged against a checkout nobody
  # has advanced -- a false green in the checker built to catch false greens.
  as_owner git -C "$clone" fetch --quiet --prune origin 2>/dev/null || true

  if clone_is_converged "$clone"; then
    in_sync=$((in_sync + 1))
    details+=("${name}:${before:0:12}->${before:0:12}")
    say "IN_SYNC ${name} at ${before:0:12}"
    continue
  fi

  if [[ "$MODE" == "check" ]]; then
    details+=("${name}:${before:0:12}->${before:0:12}")
    say "DRIFTED ${name} at ${before:0:12} (check mode; nothing written)"
    continue
  fi

  converge_rc=0
  as_owner env "OMNI_HOME=$TREE" bash "$CONVERGE_SCRIPT" "$name" \
    --execute --ticket "$TICKET" --lane "$LANE" || converge_rc=$?

  after="$(git -C "$clone" rev-parse HEAD 2>/dev/null)" || after=""
  # The delegate's exit code is not the proof. Read the surface back: a clone
  # that did not reach its upstream FAILED even when the converger exited 0.
  if (( converge_rc == 0 )) && [[ -n "$after" ]] && clone_is_converged "$clone"; then
    converged=$((converged + 1))
    details+=("${name}:${before:0:12}->${after:0:12}")
    say "CONVERGED ${name} ${before:0:12} -> ${after:0:12}"
  else
    failed=$((failed + 1))
    details+=("${name}:${before:0:12}->${after:0:12}")
    say "FAILED ${name} (converger exit ${converge_rc}; HEAD ${before:0:12} -> ${after:0:12})"
  fi
done

detail_joined="$(IFS=';'; echo "${details[*]}")"

if (( failed > 0 )); then
  emit_verdict FAILED "${script_note:-clone_converge_failed}" "$clone_count" "$in_sync" "$converged" "$failed" "$skipped" "$detail_joined"
  exit 1
fi

if [[ -n "$abort_reason" ]]; then
  emit_verdict REFUSED "$abort_reason" "$clone_count" "$in_sync" "$converged" "$failed" "$skipped" "$detail_joined"
  exit 0
fi

if [[ "$MODE" == "check" ]]; then
  if (( in_sync == clone_count )); then
    last_success="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    emit_verdict IN_SYNC "${script_note:-none}" "$clone_count" "$in_sync" 0 0 0 "$detail_joined"
  else
    emit_verdict DRIFTED "${script_note:-none}" "$clone_count" "$in_sync" 0 0 0 "$detail_joined"
  fi
  exit 0
fi

# Only a COMPLETED, clean run advances last_success. That is the field the
# Slack reporter ages, and it is what keeps a permanently busy runner from
# reading as a permanently converged tree.
last_success="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if (( converged > 0 )); then
  emit_verdict CONVERGED "${script_note:-none}" "$clone_count" "$in_sync" "$converged" 0 0 "$detail_joined"
else
  emit_verdict IN_SYNC "${script_note:-none}" "$clone_count" "$in_sync" 0 0 0 "$detail_joined"
fi
exit 0

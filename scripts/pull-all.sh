#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# pull-all.sh — Pull all omni_home canonical repos to latest dev and main
#
# Usage:
#   ./pull-all.sh           # pull all repos
#   ./pull-all.sh omniclaude omnibase_core   # pull specific repos

set -euo pipefail

OMNI_HOME="${OMNI_HOME:-/Volumes/PRO-G40/Code/omni_home}"

REPOS=(
  omniclaude
  omnibase_compat
  omnibase_core
  omnibase_infra
  omnibase_spi
  omnidash
  omnidash-v2
  omnigemini
  omniintelligence
  omnimarket
  omnimemory
  omninode_infra
  omniweb
  onex_change_control
)

# Allow caller to override which repos to pull
if [[ $# -gt 0 ]]; then
  REPOS=("$@")
fi


RESULTS_DIR=$(mktemp -d)

# === Stage tracking + terminal completion signal (OMN-15590) ===
# Field failure this closes (remote-gate-readiness run wf_c69db51c-74d,
# 2026-07-31, host stickybeatz-studio): the drift-repair stage below ran
# unbounded, overran the caller's 3-minute timeout, orphaned three bash
# processes, and never reached the plugin-cache / pre-commit / summary stages.
# A caller that runs the documented "sync first" step and sees no error got a
# PARTIALLY-EXECUTED sync with nothing surfaced. Independently, the stage's
# failure path printed a banner and fell through -- a clean drift-repair
# FAILURE also produced a green-looking, exit-0 run.
#
# Every stage now carries an explicit status, the summary is emitted from an
# EXIT trap (so it appears on every exit path, including early aborts), and one
# machine-parseable line lets a caller distinguish a complete run from a
# stopped-or-failed one without reading prose.
STAGE_REPOS="PENDING"
STAGE_DRIFT_REPAIR="PENDING"
STAGE_PLUGIN_CACHE="PENDING"
STAGE_PRECOMMIT_HOOKS="PENDING"
STAGE_FAILURES=()
SUMMARY_EMITTED=0
ROOT_BASHPID="${BASHPID:-$$}"

# Aggregates are declared here (not at the aggregation step) so the EXIT-trap
# summary can read them even when the script aborts before that step.
OK=0
FAILED=()
WARNED=()

# The bound for the drift-repair stage, in seconds. Explicit and declared;
# overridable per-invocation for tests and for hosts with a slower canonical
# venv. 300s is ~150x the measured healthy cost of the repair's two uv steps on
# the gate host (1.93s resolve + 43ms leaf check, measured 2026-08-02), so it
# only fires on a genuine stall, never on a slow-but-progressing install.
DRIFT_REPAIR_TIMEOUT_SECONDS="${PULL_ALL_DRIFT_REPAIR_TIMEOUT_SECONDS:-300}"

# Same bound, applied to the other unbounded network call on this script: the
# best-effort `pre-commit install-hooks` env pre-build. Non-fatal either way
# (the hook script is already written by then), but it must not be able to
# stall the run past its terminal summary.
HOOK_ENV_TIMEOUT_SECONDS="${PULL_ALL_HOOK_ENV_TIMEOUT_SECONDS:-300}"

# Same bound for the plugin-cache content hash, which walks a 53k-file tree on
# the gate host. See _plugin_content_hash for the process-per-file defect that
# made this stage the SECOND unbounded stall found while proving OMN-15590.
PLUGIN_HASH_TIMEOUT_SECONDS="${PULL_ALL_PLUGIN_HASH_TIMEOUT_SECONDS:-300}"

_emit_summary() {
  [[ "$SUMMARY_EMITTED" -eq 1 ]] && return 0
  SUMMARY_EMITTED=1

  local overall
  if [[ "$STAGE_REPOS" == "PENDING" ]]; then
    overall="ABORTED"
  elif [[ ${#FAILED[@]} -gt 0 || ${#STAGE_FAILURES[@]} -gt 0 ]]; then
    overall="FAILED"
  elif [[ "$STAGE_PLUGIN_CACHE" == "PENDING" || "$STAGE_PRECOMMIT_HOOKS" == "PENDING" ]]; then
    overall="INCOMPLETE"
  else
    overall="OK"
  fi

  echo ""
  if [[ ${#WARNED[@]} -gt 0 ]]; then
    echo "WARN: ${#WARNED[@]} repo(s) not found locally and were skipped:"
    local w
    for w in "${WARNED[@]}"; do
      echo "  WARN     $w"
    done
  fi
  echo "${OK} repo(s) up to date. ${#FAILED[@]} failed. ${#WARNED[@]} absent (skipped)."

  echo ""
  echo "== pull-all.sh stage summary =="
  echo "  repos                  : $STAGE_REPOS"
  echo "  omnimarket-drift-repair: $STAGE_DRIFT_REPAIR"
  echo "  plugin-cache-refresh   : $STAGE_PLUGIN_CACHE"
  echo "  pre-commit-hooks       : $STAGE_PRECOMMIT_HOOKS"
  if [[ ${#STAGE_FAILURES[@]} -gt 0 ]]; then
    local f
    for f in "${STAGE_FAILURES[@]}"; do
      echo "  STAGE FAILED           : $f"
    done
  fi

  # Single machine-parseable completion signal. A caller checks this ONE line
  # instead of parsing prose; its absence means the run did not reach the end.
  echo "PULL-ALL-RESULT: overall=${overall} repos_ok=${OK} repos_failed=${#FAILED[@]} repos_absent=${#WARNED[@]} drift_repair=${STAGE_DRIFT_REPAIR} plugin_cache=${STAGE_PLUGIN_CACHE} precommit_hooks=${STAGE_PRECOMMIT_HOOKS}"
}

_on_exit() {
  local rc=$?
  [[ "${BASHPID:-$$}" == "$ROOT_BASHPID" ]] || return "$rc"
  _emit_summary || true
  rm -rf "$RESULTS_DIR" || true
  return "$rc"
}
trap _on_exit EXIT

# Run a command under a hard wall-clock bound, in its OWN process group, so a
# timeout kills the entire descendant tree. Returns the command's exit status,
# or 124 on timeout (GNU timeout's convention).
#
# Why not `timeout(1)`: macOS ships neither coreutils `timeout`/`gtimeout` nor
# `setsid` (verified on the .200 gate host, 2026-08-02), and GNU `timeout`
# signals only its direct child by default -- which on this stage would leave
# the git/uv/python grandchildren running. Those grandchildren ARE the field
# symptom (orphaned PIDs 61908/62077/62078). Enabling job control (`set -m`)
# makes the backgrounded job a process-group leader, so `kill -- -PGID` reaches
# the whole tree.
_run_bounded() {
  local bound="$1"
  shift

  local marker cmd_pid watchdog_pid rc waited
  marker=$(mktemp)

  set -m
  "$@" &
  cmd_pid=$!
  set +m

  (
    waited=0
    while [[ "$waited" -lt "$bound" ]]; do
      sleep 1
      waited=$((waited + 1))
      kill -0 "$cmd_pid" 2>/dev/null || exit 0
    done
    echo "timeout" > "$marker"
    kill -TERM -"$cmd_pid" 2>/dev/null || kill -TERM "$cmd_pid" 2>/dev/null || true
    sleep 5
    kill -KILL -"$cmd_pid" 2>/dev/null || kill -KILL "$cmd_pid" 2>/dev/null || true
  ) &
  watchdog_pid=$!

  rc=0
  wait "$cmd_pid" || rc=$?

  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true

  # Sweep the group unconditionally: a command can exit while leaving a
  # backgrounded descendant behind, which is the same orphan class.
  kill -KILL -"$cmd_pid" 2>/dev/null || true

  if [[ -s "$marker" ]]; then
    rc=124
  fi
  rm -f "$marker"
  return "$rc"
}
# === End stage tracking ===

# === Pre-pull validation: detect bare repo corruption (OMN-7600) ===
# If core.bare=true, git pull updates refs but NOT the working tree, causing
# stale files. This is corruption in omni_home — repos must be non-bare clones.
BARE_REPOS=()
for repo in "${REPOS[@]}"; do
  dir="$OMNI_HOME/$repo"
  [[ -d "$dir" ]] || continue
  is_bare=$(git -C "$dir" rev-parse --is-bare-repository 2>/dev/null || echo "unknown")
  if [[ "$is_bare" == "true" ]]; then
    BARE_REPOS+=("$repo")
  fi
done

if [[ ${#BARE_REPOS[@]} -gt 0 ]]; then
  echo ""
  echo "ERROR: Bare repo corruption detected in omni_home!"
  echo ""
  echo "The following repos have core.bare=true, which means git pull"
  echo "updates refs but NOT the working tree — files go stale silently."
  echo ""
  for repo in "${BARE_REPOS[@]}"; do
    echo "  CORRUPT  $repo"
    echo "           Fix: git -C $OMNI_HOME/$repo config core.bare false"
    echo "           Then: git -C $OMNI_HOME/$repo reset --hard HEAD"
  done
  echo ""
  echo "Fix all corrupted repos above, then re-run pull-all.sh."
  exit 1
fi
# === End bare repo validation ===

# === Converge-script snapshot (OMN-16500 race fix) ===
# _pull_one routes a non-fast-forwardable main through the sanctioned
# convergence script in the canonical omniclaude clone -- but pull-all ITSELF
# switches that clone between main (the release pointer, which can predate or
# lack the script) and dev while the parallel pulls run. Reading the script
# through the omniclaude working tree mid-run is therefore a race against our
# own branch switching: the 2026-08-24 proof run lost omnimemory and
# omnibase_spi to "script missing" precisely this way, while six sibling
# repos converged fine. Snapshot the script ONCE, before any repo starts
# moving, and invoke only the snapshot. A missing source at startup is
# reported per-repo at the point of need, naming the source path.
CONVERGE_SCRIPT_SOURCE="$OMNI_HOME/omniclaude/scripts/converge-canonical-clone.sh"
CONVERGE_SCRIPT="$RESULTS_DIR/converge-canonical-clone.sh"
if [[ -f "$CONVERGE_SCRIPT_SOURCE" ]]; then
  cp "$CONVERGE_SCRIPT_SOURCE" "$CONVERGE_SCRIPT"
  chmod +x "$CONVERGE_SCRIPT"
fi
# === End converge-script snapshot ===

# Switch to a branch, creating it from origin/<branch> when needed, then
# fast-forward it to the fetched remote branch.
_checkout_and_ff() {
  local dir="$1"
  local branch="$2"

  if git -C "$dir" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$dir" switch "$branch"
  else
    git -C "$dir" switch --track -c "$branch" "origin/$branch"
  fi

  git -C "$dir" merge --ff-only "origin/$branch"
}

_branch_summary() {
  local dir="$1"
  local branch="$2"
  local before="$3"
  local after

  after=$(git -C "$dir" rev-parse --verify --quiet "refs/heads/$branch" || true)
  if [[ -z "$before" ]]; then
    echo "$branch created"
  elif [[ "$before" == "$after" ]]; then
    echo "$branch already up to date"
  else
    local commits
    commits=$(git -C "$dir" rev-list --count "$before..$after" 2>/dev/null | tr -d ' ')
    echo "$branch +${commits} commit(s)"
  fi
}

_leave_on_dev() {
  local dir="$1"

  if git -C "$dir" show-ref --verify --quiet "refs/heads/dev"; then
    git -C "$dir" switch dev >/dev/null 2>&1
  elif git -C "$dir" show-ref --verify --quiet "refs/remotes/origin/dev"; then
    git -C "$dir" switch --track -c dev origin/dev >/dev/null 2>&1
  else
    return 1
  fi
}

# Pull a single repo — writes result to a temp file for aggregation.
_pull_one() {
  local repo="$1"
  local dir="$OMNI_HOME/$repo"
  local result_file="$RESULTS_DIR/$repo"

  if [[ ! -d "$dir" ]]; then
    echo "  MISSING  $repo"
    echo "MISSING" > "$result_file"
    return
  fi

  local branch
  branch=$(git -C "$dir" branch --show-current 2>/dev/null)
  if [[ "$branch" != "main" && "$branch" != "dev" ]]; then
    echo "  SKIPPED  $repo (on branch: $branch)"
    echo "SKIPPED" > "$result_file"
    return
  fi

  local dirty
  dirty=$(git -C "$dir" status --porcelain)
  if [[ -n "$dirty" ]]; then
    echo "  FAILED   $repo (dirty worktree; refusing to switch branches)"
    echo "           Commit, stash, or remove local changes before re-running."
    echo "FAILED" > "$result_file"
    return
  fi

  local before_main before_dev output main_converged=0
  before_main=$(git -C "$dir" rev-parse --verify --quiet refs/heads/main || true)
  before_dev=$(git -C "$dir" rev-parse --verify --quiet refs/heads/dev || true)

  # OMN-16502: `git fetch origin main dev` fails WHOLESALE when either ref is
  # absent on origin. Most registry repos carry both branches, but a few
  # (e.g. omnigemini) are main-only -- probe before fetching so a genuinely
  # main-only repo is fetched/ff'd on main alone and reported OK, instead of
  # failing the fetch stage on every run.
  local dev_on_remote=1
  if ! git -C "$dir" ls-remote --exit-code --heads origin dev >/dev/null 2>&1; then
    dev_on_remote=0
  fi

  if [[ "$dev_on_remote" == "1" ]]; then
    if ! output=$(git -C "$dir" fetch --prune origin main dev 2>&1); then
      echo "  FAILED   $repo (fetch main/dev)"
      echo "           $output"
      echo "FAILED" > "$result_file"
      return
    fi
  else
    if ! output=$(git -C "$dir" fetch --prune origin main 2>&1); then
      echo "  FAILED   $repo (fetch main)"
      echo "           $output"
      echo "FAILED" > "$result_file"
      return
    fi
  fi

  if ! output=$(_checkout_and_ff "$dir" main 2>&1); then
    # main is a release-pointer branch (release-synced-main policy): release.yml
    # rewrites origin/main to the release tag, so a local main still carrying
    # the pre-rewrite promotion commits can NEVER fast-forward again -- and main
    # is never worked on locally, so converging it to origin/main is always
    # correct (OMN-16500; 9 of 12 registry clones were stuck here 2026-08-24).
    # The ref move MUST go through the sanctioned convergence script
    # (canonical-clone guard, OMN-16496): it preserves the orphaned commits as
    # evidence and appends a ledger row before the branch -f. dev below stays
    # strictly ff-only -- a dev that cannot fast-forward is a real problem.
    if ! _leave_on_dev "$dir"; then
      echo "  FAILED   $repo (fast-forward main; could not return to dev to converge)"
      echo "           $output"
      echo "FAILED" > "$result_file"
      return
    fi
    if [[ ! -f "$CONVERGE_SCRIPT" ]]; then
      echo "  FAILED   $repo (fast-forward main; sanctioned converge script missing: $CONVERGE_SCRIPT_SOURCE)"
      echo "           $output"
      echo "FAILED" > "$result_file"
      return
    fi
    local converge_out
    if ! converge_out=$(env OMNI_HOME="$OMNI_HOME" \
        bash "$CONVERGE_SCRIPT" "$repo" --branch main --execute --lane pull-all 2>&1); then
      echo "  FAILED   $repo (fast-forward main; converge-canonical-clone.sh --branch main failed)"
      echo "           $converge_out"
      echo "FAILED" > "$result_file"
      return
    fi
    main_converged=1
  fi

  # OMN-16502: origin has no dev ref for this repo -- there is no dev leg to
  # pull or switch to. Report main's outcome and leave the repo on main.
  if [[ "$dev_on_remote" == "0" ]]; then
    local main_summary
    if [[ "$main_converged" == "1" ]]; then
      main_summary="main converged to origin/main (non-ff; orphaned commits preserved)"
    else
      main_summary=$(_branch_summary "$dir" main "$before_main")
    fi
    echo "  OK       $repo ($main_summary; no dev branch on origin -- main-only repo; left on main)"
    echo "OK" > "$result_file"
    return
  fi

  if output=$(_checkout_and_ff "$dir" dev 2>&1); then
    local main_summary dev_summary
    if [[ "$main_converged" == "1" ]]; then
      main_summary="main converged to origin/main (non-ff; orphaned commits preserved)"
    else
      main_summary=$(_branch_summary "$dir" main "$before_main")
    fi
    dev_summary=$(_branch_summary "$dir" dev "$before_dev")
    echo "  OK       $repo ($main_summary; $dev_summary; left on dev)"
    echo "OK" > "$result_file"
  else
    echo "  FAILED   $repo (fast-forward dev)"
    echo "           $output"
    echo "FAILED" > "$result_file"
  fi
}

# Launch all fetches in parallel
for repo in "${REPOS[@]}"; do
  _pull_one "$repo" &
done

wait

# Aggregate results (OK / FAILED / WARNED are declared with the stage tracking
# block above so the EXIT-trap summary can read them on any exit path).
for repo in "${REPOS[@]}"; do
  result_file="$RESULTS_DIR/$repo"
  if [[ -f "$result_file" ]]; then
    status=$(cat "$result_file")
    case "$status" in
      OK) (( OK++ )) || true ;;
      FAILED) FAILED+=("$repo") ;;
      MISSING) WARNED+=("$repo (not cloned — skipped)") ;;
      # SKIPPED — don't count
    esac
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  STAGE_REPOS="FAILED"
else
  STAGE_REPOS="OK"
fi

# === Omnimarket venv drift auto-repair (OMN-15242) ===
# The OMN-14060 pre-flight guard (src/omnibase_infra/cli/omnimarket_drift_guard.py)
# detects when the canonical omnibase_infra venv's installed omnimarket has
# fallen behind the just-advanced $OMNI_HOME/omnimarket clone -- but it only
# detects and instructs, it never repairs. A canonical omnimarket pull (right
# here, above) is the EXACT event that creates that drift. Two same-day
# 2026-07-27 incidents (13:04Z and 19:23Z) bricked every onex CLI/skill
# dispatch on this Mac until a human ran the repair by hand.
#
# INTERACTIVE-SESSION SCOPE ONLY. Preregistered battery runs use the frozen
# execution-environment mechanism (OMN-15265) and must NEVER be auto-repaired
# mid-run -- that would change the delegation stack version between seeds and
# contaminate the run. This hook lives only in pull-all.sh (the interactive/
# session sync entrypoint), never in a battery driver, so that boundary holds
# structurally rather than by convention.
#
# Design guarantees:
#   * Only triggers when omnimarket was part of THIS run and its pull result
#     was OK -- a FAILED/MISSING/SKIPPED/not-requested omnimarket is untouched.
#   * Skip-guarded -- a missing local omnibase_infra clone, missing drift
#     script, or missing canonical venv is a clean no-op (nothing to repair
#     against).
#   * BOUNDED and FATAL (OMN-15590, revising OMN-15242's
#     "fail-loud-but-not-fatal") -- the invocation runs under an explicit
#     wall-clock bound in its own process group, and BOTH a timeout and a
#     non-zero exit are carried into the terminal summary AND the exit code.
#     The original non-fatal design meant a caller could not distinguish
#     "sync completed" from "sync stopped or failed partway" -- and the
#     unbounded call meant the run never reached the later stages at all.
#   * Attributable -- the check/repair invocation and outcome are echoed
#     inline in pull-all.sh's own stdout, the same log surface as every other
#     step here.
#
# ROOT CAUSE of the observed hang (OMN-15590 AC6), established by controlled
# experiment on stickybeatz-studio 2026-08-02, not inferred:
#   The repair chain ends in `uv pip install --python <canonical infra venv>`.
#   uv takes an EXCLUSIVE flock on `<venv>/.lock` for the duration of an
#   install, has no lock-acquisition timeout, and prints nothing at default
#   verbosity while it waits. Holding that lock from a second process made an
#   otherwise 2-second install sit silent past 30s and then finish in
#   milliseconds the instant the lock was released. `.200` runs many parallel
#   sessions against ONE shared canonical venv -- including the readiness probe
#   that itself runs pull-all.sh -- so a peer's uv operation blocks this stage
#   for the peer's entire duration, unbounded. Resolve/network was excluded by
#   measurement on the same host: step-1 git+HTTPS resolve 1.93s, step-2 leaf
#   resolve 43ms, `git ls-remote` 0.19s.
_omnimarket_result_file="$RESULTS_DIR/omnimarket"
if [[ ! -f "$_omnimarket_result_file" || "$(cat "$_omnimarket_result_file")" != "OK" ]]; then
  STAGE_DRIFT_REPAIR="SKIPPED"  # omnimarket not in this run, or its pull did not succeed
else
  _infra_dir="$OMNI_HOME/omnibase_infra"
  _drift_script="$_infra_dir/scripts/check-omnimarket-venv-drift.sh"
  _infra_venv_python="$_infra_dir/.venv/bin/python"

  if [[ ! -d "$_infra_dir" || ! -x "$_drift_script" ]]; then
    STAGE_DRIFT_REPAIR="SKIPPED"  # no local omnibase_infra clone (or drift script)
  elif [[ ! -x "$_infra_venv_python" ]]; then
    STAGE_DRIFT_REPAIR="SKIPPED"  # no canonical omnibase_infra venv to repair
  else
    echo ""
    echo "== checking omnimarket venv drift against canonical omnibase_infra venv =="
    echo "   (bounded at ${DRIFT_REPAIR_TIMEOUT_SECONDS}s -- OMN-15590; override with PULL_ALL_DRIFT_REPAIR_TIMEOUT_SECONDS)"
    _drift_rc=0
    _run_bounded "$DRIFT_REPAIR_TIMEOUT_SECONDS" \
      env OMNI_HOME="$OMNI_HOME" bash "$_drift_script" --repair "$_infra_venv_python" \
      || _drift_rc=$?

    if [[ "$_drift_rc" -eq 0 ]]; then
      STAGE_DRIFT_REPAIR="OK"
      echo "  DRIFT-REPAIR omnimarket venv OK (canonical omnibase_infra venv)"
    else
      if [[ "$_drift_rc" -eq 124 ]]; then
        STAGE_DRIFT_REPAIR="TIMEOUT"
        STAGE_FAILURES+=("omnimarket-drift-repair timed out after ${DRIFT_REPAIR_TIMEOUT_SECONDS}s")
      else
        STAGE_DRIFT_REPAIR="FAILED"
        STAGE_FAILURES+=("omnimarket-drift-repair exited ${_drift_rc}")
      fi
      echo ""
      echo "############################################################"
      if [[ "$STAGE_DRIFT_REPAIR" == "TIMEOUT" ]]; then
        echo "# OMN-15590: omnimarket venv drift-repair TIMED OUT after ${DRIFT_REPAIR_TIMEOUT_SECONDS}s"
        echo "#"
        echo "# The stage was killed (whole process group) and this run is a"
        echo "# FAILURE. Most likely cause: another process holds the exclusive"
        echo "# uv lock on the canonical venv"
        echo "#   $_infra_dir/.venv/.lock"
        echo "# uv waits on that lock forever and prints nothing while waiting."
        echo "# Check for a concurrent uv/pull-all/session on this host, then"
        echo "# re-run. Raise PULL_ALL_DRIFT_REPAIR_TIMEOUT_SECONDS only if the"
        echo "# repair is genuinely slow rather than blocked."
      else
        echo "# OMN-15242: omnimarket venv drift-repair FAILED (exit ${_drift_rc})"
      fi
      echo "#"
      echo "# pull-all.sh just advanced the canonical omnimarket clone, but"
      echo "# could not repair the canonical omnibase_infra venv against it."
      echo "# Every onex CLI / skill dispatch command is now at risk of the"
      echo "# OMN-14060 OmnimarketDriftError until this is fixed BY HAND:"
      echo "#"
      echo "#   OMNI_HOME=$OMNI_HOME bash $_drift_script --repair $_infra_venv_python"
      echo "#"
      echo "# See OMN-15590 / OMN-15242 / OMN-14060 for context."
      echo "############################################################"
      echo ""
      echo "  (continuing to the remaining stages -- their disposition is"
      echo "   reported in the terminal summary; nothing is silently skipped)"
    fi
  fi
fi
# === End omnimarket venv drift auto-repair ===

# === Plugin cache refresh (Layer 2, OMN-7369) ===
# When omniclaude was updated, refresh the Claude Code plugin cache.
#
# The cache lives at a versioned path:
#   ~/.claude/plugins/cache/omninode-tools/onex/<version>/
# Locate it by searching for the .deployed-commit marker file (more specific
# than matching a generic 'skills' directory — which also appears under
# other plugins like claude-plugins-official).
_omniclaude_dir="$OMNI_HOME/omniclaude"
_plugin_cache="${CLAUDE_PLUGIN_ROOT:-}"
if [[ -z "${_plugin_cache}" ]]; then
  # Search for the .deployed-commit marker inside the onex plugin cache tree.
  # maxdepth 5 covers: cache/omninode-tools/onex/<version>/.deployed-commit.
  # The `|| true` suffix stops `set -eo pipefail` from treating a missing
  # cache directory (find exits non-zero when the root does not exist) as
  # a fatal script error — a missing cache must be a clean no-op.
  _marker=$(find "${HOME}/.claude/plugins/cache" -maxdepth 5 -path "*/omninode-tools/onex/*" -name ".deployed-commit" -type f 2>/dev/null | head -1 || true)
  if [[ -n "${_marker}" ]]; then
    _plugin_cache=$(dirname "${_marker}")
  else
    # Fallback: marker absent (first deploy). Search for a versioned onex dir.
    _plugin_cache=$(find "${HOME}/.claude/plugins/cache" -maxdepth 4 -path "*/omninode-tools/onex/*" -type d 2>/dev/null | head -1 || true)
  fi
fi

# Compute a content hash of all plugin files under a directory.
# Excludes __pycache__, .pyc, and the marker files themselves so the hash
# is stable regardless of which directory it is computed against.
#
# The hash is computed against RELATIVE paths so a repo-side and cache-side
# computation of the same plugin tree yield the same hash (shasum emits
# `hash  path` — absolute paths would otherwise break comparability).
#
# `-exec ... +` NOT `-exec ... \;` (OMN-15590). The per-file form spawned ONE
# `shasum` (a perl script) process per file. Measured on the gate host
# 2026-08-02: the live plugin cache holds 53,057 files, so the per-file form
# forks ~53k perl interpreters and the stage ran >10 minutes without finishing
# on a loaded host -- a second unbounded stall in this same script, hit while
# proving the drift-repair bound. The batched form produces byte-identical
# output (same `hash  ./path` lines, re-sorted downstream anyway) and completes
# the same tree in 7.9s.
_plugin_content_hash() {
  local root="$1"
  ( cd "${root}" && find . -type f \
      ! -name "*.pyc" \
      ! -path "*/__pycache__/*" \
      ! -name ".deployed-commit" \
      ! -name ".content-hash" \
      -exec shasum {} + 2>/dev/null | sort | shasum | cut -d' ' -f1 )
}

STAGE_PLUGIN_CACHE="SKIPPED"  # no cache and/or no omniclaude clone -- nothing to refresh
if [[ -n "${_plugin_cache}" && -d "${_omniclaude_dir}" && -d "${_plugin_cache}" ]]; then
  STAGE_PLUGIN_CACHE="UP-TO-DATE"
  _current=$(git -C "${_omniclaude_dir}" rev-parse HEAD 2>/dev/null)
  _deployed=""
  [[ -f "${_plugin_cache}/.deployed-commit" ]] && _deployed=$(cat "${_plugin_cache}/.deployed-commit" 2>/dev/null)

  # Compare against repo content hash as a second signal beyond commit SHA.
  _repo_hash=""
  if [[ -d "${_omniclaude_dir}/plugins/onex" ]]; then
    # Bounded for the same reason as the drift stage: a stall here used to hang
    # the run past its terminal summary. An empty hash on timeout fails toward
    # "cache looks stale" (a refresh), never toward a silent skip.
    _repo_hash=$(_run_bounded "$PLUGIN_HASH_TIMEOUT_SECONDS" _plugin_content_hash "${_omniclaude_dir}/plugins/onex") || {
      echo "WARN: plugin content hash (repo) exceeded ${PLUGIN_HASH_TIMEOUT_SECONDS}s -- treating cache as stale."
      STAGE_PLUGIN_CACHE="WARN"
      _repo_hash=""
    }
  fi
  _cache_hash=""
  [[ -f "${_plugin_cache}/.content-hash" ]] && _cache_hash=$(cat "${_plugin_cache}/.content-hash" 2>/dev/null)

  if [[ -n "${_current}" ]] && { [[ "${_current}" != "${_deployed}" ]] || [[ "${_repo_hash}" != "${_cache_hash}" ]]; }; then
    echo ""
    echo "Refreshing Claude Code plugin cache (${_deployed:-none} → ${_current:0:8})..."
    _tmpdir=$(mktemp -d)
    # Refresh the entire plugins/onex/ tree (hooks, skills, lib, agents,
    # runtime, scripts, docs, prompts, models, _bin, tests). Copying only
    # skills/ leaves stale code in sibling directories that schema changes
    # silently drop — the exact failure this fix prevents.
    if git -C "${_omniclaude_dir}" archive HEAD plugins/onex/ 2>/dev/null | tar -x -C "${_tmpdir}" 2>/dev/null; then
      if [[ -d "${_tmpdir}/plugins/onex" ]]; then
        # rsync with --delete would remove files the cache added (e.g.
        # __pycache__), so use cp -R of the contents to update in place.
        cp -R "${_tmpdir}/plugins/onex/." "${_plugin_cache}/"
        echo "${_current}" > "${_plugin_cache}/.deployed-commit"
        # Recompute hash against the cache after refresh and persist.
        _new_hash=$(_run_bounded "$PLUGIN_HASH_TIMEOUT_SECONDS" _plugin_content_hash "${_plugin_cache}") || _new_hash=""
        echo "${_new_hash}" > "${_plugin_cache}/.content-hash"
        echo "Plugin cache refreshed (content hash ${_new_hash:0:8})."
        STAGE_PLUGIN_CACHE="REFRESHED"
      else
        echo "WARN: Plugin cache refresh failed (archive missing plugins/onex/)."
        STAGE_PLUGIN_CACHE="WARN"
      fi
    else
      echo "WARN: Plugin cache refresh failed (git archive error)."
      STAGE_PLUGIN_CACHE="WARN"
    fi
    rm -rf "${_tmpdir}"
  fi
fi
# === End plugin cache refresh ===

# === Local pre-commit hook installation (OMN-14099) ===
# Root cause of the "defects caught at the most expensive layer (CI/review/merge)
# instead of the cheapest" leak: the pre-commit git hook was never installed in
# the canonical clones, so every repo's .pre-commit-config.yaml was pure
# decoration -- commits silently skipped ALL local enforcement (hardcoded
# IPs/topics, banned constructs, URL/model-literal authority, skip tokens, ...)
# and CI became the first catch point. No `--no-verify` needed; the hook simply
# never ran. Installing it here -- in the sync entrypoint every session already
# runs before ticket work -- activates the already-written, correctly-scoped
# hooks at commit time. This is NOT a new pattern check; it closes a bypass leak.
#
# Design guarantees:
#   * Idempotent -- skips any repo whose hook is already pre-commit-managed, so
#     steady-state runs are a fast no-op.
#   * Scoped -- only touches repos that actually ship a .pre-commit-config.yaml.
#   * Fail-soft -- a hook-install problem NEVER fails the pull (hook install is a
#     convenience layer, not the sync's core job) and is left out of FAILED.
#   * Offline-safe -- the load-bearing step is writing the hook script (no
#     network); environment pre-build is best-effort so an offline machine still
#     gets commit-time enforcement (envs then install lazily at first commit).
if ! command -v pre-commit >/dev/null 2>&1; then
  echo ""
  echo "WARN: 'pre-commit' not found on PATH -- local git hooks were NOT installed."
  echo "      Install it (e.g. 'brew install pre-commit') so pattern/static"
  echo "      defects fail at commit time instead of first failing in CI."
  STAGE_PRECOMMIT_HOOKS="UNAVAILABLE"
else
  STAGE_PRECOMMIT_HOOKS="OK"
  for repo in "${REPOS[@]}"; do
    _pc_dir="$OMNI_HOME/$repo"
    [[ -d "$_pc_dir" ]] || continue
    [[ -f "$_pc_dir/.pre-commit-config.yaml" ]] || continue
    # The install runs in a subshell (it `cd`s), so a failure inside it cannot
    # assign to STAGE_PRECOMMIT_HOOKS directly. It signals with exit 3 and the
    # parent downgrades the stage -- otherwise the summary would report
    # `precommit_hooks=OK` on a run that printed a screenful of WARN lines,
    # which is the same "green-looking incomplete run" defect this ticket
    # exists to close, just moved one stage over. (OMN-15590)
    if ! (
      cd "$_pc_dir" || exit 0
      # `git rev-parse --git-path hooks` resolves to the SHARED hooks dir (the
      # common git dir), so installing in the canonical clone covers all of its
      # linked worktrees too -- worktrees do not get their own pre-commit hook.
      _hook="$(git rev-parse --git-path hooks 2>/dev/null)/pre-commit"
      if [[ -f "$_hook" ]] && grep -q "File generated by pre-commit" "$_hook" 2>/dev/null; then
        exit 0  # already pre-commit-managed -- idempotent no-op
      fi
      # Guard-managed chain (OMN-16500): on this fleet, canonical clones set
      # core.hooksPath to the canonical-clone guard, which CHAINS to the
      # installed hook or invokes `pre-commit hook-impl` itself (OMN-15071) --
      # commit-time enforcement is ACTIVE. `pre-commit install` refuses to
      # write hook files while core.hooksPath is set, so attempting it here
      # produced a false "commit-time enforcement inactive" WARN. Any OTHER
      # core.hooksPath value falls through to the install attempt and keeps
      # the WARN -- there, enforcement genuinely is inactive.
      if [[ -f "$_hook" ]] && grep -q "Canonical-clone worktree-discipline guard" "$_hook" 2>/dev/null; then
        exit 0  # guard chains the real hook chain -- nothing to install
      fi
      # Load-bearing: write the hook script (fast, no network). This alone
      # closes the leak; hook environments install lazily at first commit.
      if pre-commit install >/dev/null 2>&1; then
        # Best-effort env pre-build so the first real commit is not slow. Runs
        # at most once per repo (guarded above); a failure here is non-fatal
        # because the hook script is already written and will still fire.
        #
        # BOUNDED (OMN-15590): this is a network-bound call with no client
        # timeout -- the same unbounded class as the drift-repair stage above,
        # on the same script. It is bounded with the same mechanism so a hung
        # hook-env build cannot stall the sync past its terminal summary.
        _run_bounded "$HOOK_ENV_TIMEOUT_SECONDS" \
          bash -c 'pre-commit install-hooks >/dev/null 2>&1' || true
        echo "  HOOK     $repo (pre-commit git hook installed)"
      else
        echo "  WARN     $repo (pre-commit install failed -- commit-time enforcement inactive)"
        exit 3
      fi
    ); then
      STAGE_PRECOMMIT_HOOKS="WARN"
    fi
  done
fi
# === End pre-commit hook installation ===

# The terminal summary is emitted by the EXIT trap (_on_exit -> _emit_summary)
# so it appears on EVERY exit path, including the early bare-repo abort and any
# unexpected `set -e` failure -- not only on this happy path. All that remains
# here is the exit code.
#
# A repo failure OR a failed/timed-out stage is a failed run (OMN-15590): a
# caller running the documented "sync first" step must not read exit 0 off a
# sync that stopped or failed partway.
if [[ ${#FAILED[@]} -gt 0 || ${#STAGE_FAILURES[@]} -gt 0 ]]; then
  exit 1
fi
exit 0

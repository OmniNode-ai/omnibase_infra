#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Pre-push governed impacted-test selector (OMN-13973 / WS7 OMN-14655 fan-out).
#
# Runs the FAST LOCAL IMPACTED SUBSET of the unit suite once per `git push`,
# using the SAME governed selector CI uses -- scripts/ci/detect_test_paths.py +
# scripts/ci/test_selection_adjacency.yaml -- NOT a hand-typed `-k`. The selector
# is fail-closed: it escalates to the full unit suite whenever it cannot prove
# narrowing is safe (a shared module -- models/enums/runtime/errors/nodes/topics --
# a test-infrastructure change (pyproject.toml, tests/conftest.py, tests/fixtures/,
# tests/helpers/, pytest.ini), a >=6-module change, or the main branch). See root
# CLAUDE.md Rule #4.
#
# This hook is deliberately NOT byte-parity with an enforced CI context. On CI the
# selector is gated behind ENABLE_SMART_TESTS (off by default during rollout) and
# the enforced merge gate is the FULL suite. This hook is *net-new, fast local
# impacted-subset enforcement* -- a fast local mirror that is ADVISORY of the full
# CI suite, run before the push leaves the machine. It retires the "run the whole
# unit suite by hand before every push" default (CLAUDE.md Rule #4: "until
# OMN-13973 lands, the full local suite remains the fail-closed default").
#
# PER-REPO SEAM-MATCH (this is an adaptation of the omnibase_core#1451 canary,
# NOT a paste):
#   * omnibase_infra's scripts/ci/detect_test_paths.py hardcodes
#     SRC_PREFIX = "src/omnibase_infra/" and its own adjacency map; the selector
#     is invoked with the SAME flags CI uses (.github/workflows/ci.yml).
#   * infra's detect_test_paths.py main() does NOT accept a `--base-ref` argument
#     (core's does), and infra CI does not pass one either. So this wrapper
#     computes the merge-base locally for `git diff` but MUST NOT forward
#     `--base-ref` to the selector -- doing so would make the selector argparse
#     hard-error on every push. DRY parity is with infra's OWN CI invocation.
#   * The full-suite escalation runs the full UNIT suite (tests/unit/), not all of
#     tests/. Infra's tests/chaos, tests/replay, tests/performance and MOST of
#     tests/integration need a live runtime (Docker/Postgres/Kafka) and stay a
#     CI-only concern. The one enumerated exception (OMN-16825) is
#     tests/integration/chains/, which runs on EventBusInmemory and needs no
#     live service; see filter_prepush_runnable_paths below for the allowlist
#     and its fail-closed default.
#
# WORKFLOW-DIFF RULING (OMN-16745) -- what the selector proves for a
# `.github/workflows`-only diff, and why that is not the unit suite:
#
#   The necessary and sufficient proof is the CI-CONTRACT CLASS -- tests/ci/,
#   the workflow-shape / required-context / gate-wiring tests that read
#   .github/workflows/** off disk and assert its contents -- plus, when the
#   diff also touches a test module, that module itself. No test under
#   tests/unit/ has an outcome a workflow YAML edit can change, so escalating
#   this class to the full unit suite is cost without proof: OMN-16346 sat
#   through ~20 refused pushes (zero bypasses) waiting for a host with headroom
#   for a suite that could not have falsified the diff.
#
#   Selecting NOTHING is equally wrong and is NOT what this does. Workflow
#   files break the ENFORCEMENT of tests rather than the tests themselves --
#   OMN-15541 is the live counterexample, where ci.yml hardcoded
#   `pytest src/omnibase_compat/tests/` while the selector and pyproject named
#   different roots, so full-suite escalation collected ZERO of the top-level
#   tests/ tree: a fail-OPEN safety net produced by a workflow edit. The class
#   is therefore positively named, always non-empty, and asserted to be
#   populated and workflow-aware by a test.
#
#   Fail-closed is untouched: a workflow file alongside a shared module still
#   escalates, alongside test infrastructure still escalates, and alongside an
#   ordinary source file rides additively with that file's own narrowing.
#   There is no new env knob -- see the "Env overrides" list above, which is
#   unchanged by OMN-16745.
#
# FAIL-LOUD (CLAUDE.md Rule #8): if the diff base, the selector, or its adjacency
# config cannot resolve, this hook HARD-ERRORS with a remediation message and a
# non-zero exit. It never degrades to a green skip -- a gate that cannot run must
# be indistinguishable from a failing gate. (Verified by the landed
# precommit-fail-loud-meta-gate, which scans this script.)
#
# Env overrides (all optional):
#   PREPUSH_BASE_REF     git ref to diff against            (default: origin/dev)
#   PREPUSH_ADJACENCY    adjacency yaml path            (default: selector built-in)
#   PREPUSH_PYTEST_ARGS  extra args appended to the pytest invocation
#   ENABLE_SMART_TESTS   set false/0/off to force the FULL suite (parity with the
#                        CI var name); default here is smart selection ON, because
#                        the whole point of the local hook is the impacted subset.
#   PREPUSH_FULL_SUITE   set non-empty to force the FULL suite.
#
# Heavy-escalation execution targets (in precedence order, see
# knowledge-base-internal:runbooks/omnibase-infra-prepush-remote-full-suite-verify.md):
#   1. the local host, when it IS a designated gate host AND is under the load
#      threshold -- `.200` (OMN-15059) or the `.201` gate-runner (OMN-16295)
#   2. a GitHub-hosted FULL-suite CI run pinned to the exact HEAD sha
#      (OMN-16688) -- consulted only when 1 is unavailable
#   3. a single-use, receipted, degraded-evidence override grant
#   4. refusal
# Target 2 adds NO env override and cannot make the gate accept less work: the
# run must be sha-pinned, green, and full-suite shaped, and an unresolvable
# check counts as no evidence, not as a pass.
#
# NOT an env override (OMN-16480): the host/capacity escape hatch. Any
# `PREPUSH_ALLOW_*` variable found in the environment is REJECTED at entry --
# see the rejection block below. The override is a single-use, repo+HEAD-scoped,
# TTL-bounded, receipted grant minted with
# `uv run python scripts/hooks/prepush_override_grant.py mint --reason '<why>'`.
# Note the asymmetry: the two knobs above can only make the hook run MORE tests,
# so they are not bypasses; an ALLOW override makes it accept WEAKER evidence,
# which is why it may not be ambient.
#
# ALSO not an env override (OMN-17441): the per-host PLACEMENT maps
# `PREPUSH_*_OVERRIDE_MAP` (LOAD / SLOT / MEM / UV / REACH), which prepush_dispatch.sh
# consults instead of the live ssh probe. They are a test-injection seam, they
# are rejected at entry outside pytest's own harness, and they can only steer
# WHERE work runs -- never whether it passed.

set -euo pipefail

log() { printf '[prepush-smart-tests] %s\n' "$1" >&2; }
die() {
  log "ERROR: $1"
  log "REMEDIATION: $2"
  exit 1
}

# =============================================================================
# Recursion guard (OMN-16489, F-01)
# =============================================================================
# This hook spawns pytest, and the spawned suite contains tests that exec THIS
# script again (tests/ci/test_prepush_hook_host_identity_guard.py and
# siblings). OMN-16425 proved one leaked override var turns that re-entry into
# a recursive full-suite launcher (~9h03m lost across 5 failed ~1h45m runs;
# friction report F-01) — and its fix covered the test sites, not the hook.
# The env scrub at the pytest invocations below closes the override-
# inheritance vector; this sentinel closes the re-entry class itself: a nested
# invocation refuses fail-closed before the selector resolves or any pytest
# spawns. The sentinel deliberately survives the override scrub — children
# must inherit it for this guard to hold. A test that intends to exercise this
# script's FIRST-entry behavior must strip ONEX_PREPUSH_HOOK_ACTIVE from the
# subprocess env it constructs.
if [ -n "${ONEX_PREPUSH_HOOK_ACTIVE:-}" ]; then
  die "nested invocation refused: this hook is already active in an ancestor process (ONEX_PREPUSH_HOOK_ACTIVE=${ONEX_PREPUSH_HOOK_ACTIVE}, this pid $$)" \
      "a pre-push hook run must never be spawned from inside another pre-push hook run (OMN-16425 recursion class). If a test means to exercise first-entry behavior, construct the subprocess env explicitly and strip ONEX_PREPUSH_HOOK_ACTIVE"
fi
export ONEX_PREPUSH_HOOK_ACTIVE="$$"

# =============================================================================
# Inheritable env-var gate overrides are REJECTED AT ENTRY (OMN-16480)
# =============================================================================
# This gate's escape hatch used to BE an environment variable
# (`PREPUSH_ALLOW_LOCAL_FULL_SUITE=1`). An environment variable is inherited by
# every descendant process, is bound to no repo/commit/run, never expires, and
# leaves no receipt -- so "permission to bypass the load gate once, for this
# push" was really "permission for every process this shell ever spawns to
# bypass this gate, silently". Same failure shape Rule 10 was hardened against
# for `[skip-*` tokens (OMN-9731 / OMN-13388), one layer down.
#
# Measured: on 2026-08-23 that variable leaked from an operator shell into a
# guard test's `env=dict(os.environ)` subprocess copy; this hook took its
# degraded-override branch and recursively launched another full 44,064-test
# suite, which reached the same test and recursed again -- ~9h03m, ~72% of all
# serialized suite wall-clock in that window (friction report F-01/F-04).
# Compliance was PERFECT that night: zero `[skip-*`, zero `--no-verify`. The
# damage came from the sanctioned escape path being used correctly.
#
# So the variable is no longer an arming signal in either direction: its
# presence is a HARD REFUSAL, not a bypass. That is what makes inheritance
# harmless -- a leaked override can no longer arm anything, and it surfaces
# immediately instead of silently disarming the gate for a whole process tree.
# The supported path is a single-use, repo+HEAD-scoped, TTL-bounded, receipted
# grant token: scripts/hooks/prepush_override_grant.py.
#
# Matched by PREFIX, not by one exact name, so a future
# `PREPUSH_ALLOW_SOMETHING_ELSE` cannot quietly reopen the class.
reject_inherited_env_overrides() {
  local leaked
  leaked="$(env | sed -n 's/^\(PREPUSH_ALLOW_[A-Za-z0-9_]*\)=..*/\1/p' | sort -u | tr '\n' ' ')"
  leaked="${leaked% }"
  [ -n "$leaked" ] || return 0
  die "inheritable gate-override environment variable(s) present: ${leaked} -- these are REJECTED, never honored (OMN-16480)" \
      "unset them in this shell (e.g. \`unset ${leaked%% *}\`), then, if this run genuinely must proceed on this host, mint a scoped single-use grant: \`uv run python scripts/hooks/prepush_override_grant.py mint --reason '<why>'\`. The grant is bound to this repo and this HEAD sha, expires in minutes, is consumed by the first guard that reads it (so no child process can reuse it), and appends a receipt line to .onex_state/prepush_override/receipts.jsonl"
}
reject_inherited_env_overrides

# =============================================================================
# Inheritable PLACEMENT-override maps are REJECTED AT ENTRY (OMN-17441)
# =============================================================================
# A second inheritable class, and deliberately a SEPARATE guard rather than a
# widened prefix on the one above -- the two differ in what they can do, and
# collapsing them would either overstate this risk or understate that one.
#
# `PREPUSH_LOAD_OVERRIDE_MAP` / `PREPUSH_SLOT_OVERRIDE_MAP` (and, alongside them
# in prepush_dispatch.sh, `PREPUSH_MEM_OVERRIDE_MAP` / `PREPUSH_UV_OVERRIDE_MAP`,
# and `PREPUSH_REACH_OVERRIDE_MAP` as of OMN-17280)
# are consulted by prepush_map_lookup INSTEAD of the live ssh probe. They exist
# purely as a test-injection seam. An inherited value cannot manufacture a PASS
# -- the verdict is still a real pytest exit bound to the tree by a completion
# marker -- but it decides WHERE the work goes: a stale fixture in an operator
# shell can hand the picker a phantom free slot or a manufactured idle host and
# route a real push onto a machine nothing probed. omnibase_infra#3091 named
# this residual in its own PR body and deferred it; this is that ticket.
#
# Matched by SHAPE (`PREPUSH_*_OVERRIDE_MAP`), so a future
# `PREPUSH_DISK_OVERRIDE_MAP` is covered without anyone remembering a list --
# the OMN-16480 lesson applied to the suffix that delimits this class.
#
# THE ONE EXEMPTION, and why it is not a hole: pytest's own PYTEST_CURRENT_TEST.
# tests/ci/_prepush_lab_isolation.py sets PREPUSH_SLOT_OVERRIDE_MAP to a map
# naming no real row, which is what stops the hook-subprocess tests in that
# directory from shipping a real git bundle to a real lab host and holding its
# exclusive slot for an hour (observed live 2026-08-30). pytest sets and clears
# that marker per test; it is not a knob this repo can widen, it is absent from
# an operator shell, and because this class cannot alter a verdict, the worst a
# leak of both could cost is routing accuracy. The scrub before each pytest
# spawn (scrub_prepush_override_env) already stops the maps inheriting DOWN into
# the suite; this stops them inheriting IN.
reject_inherited_placement_maps() {
  local maps
  [ -z "${PYTEST_CURRENT_TEST:-}" ] || return 0
  maps="$(env | sed -n 's/^\(PREPUSH_[A-Za-z0-9_]*_OVERRIDE_MAP\)=..*/\1/p' | sort -u | tr '\n' ' ')"
  maps="${maps% }"
  [ -n "$maps" ] || return 0
  die "inheritable placement-override map(s) present: ${maps} -- these are REJECTED, never honored outside the test harness (OMN-17441). They replace the LIVE per-host load/slot/memory/uv readings with whatever this shell is carrying, so the picker routes a real push to a host nothing probed" \
      "unset them in this shell (e.g. \`unset ${maps%% *}\`) and re-run. To simulate host states, call the picker's own functions directly the way tests/unit/scripts/test_prepush_host_table.py does, rather than exporting the maps into a live hook run"
}
reject_inherited_placement_maps

# consume_override_grant CONTEXT -- 0 when a valid single-use grant was claimed
# for this run, 1 otherwise. Delegates to the one implementation
# (scripts/hooks/prepush_override_grant.py) that the pytest-side guard also
# uses, so the two entry points can never drift apart on what a valid grant is.
# Routed through `uv run` per the OMN-14953 pinned-interpreter gate.
consume_override_grant() {
  uv run python "${REPO_ROOT}/scripts/hooks/prepush_override_grant.py" \
    consume --context "$1"
}

# =============================================================================
# .200-default host guard for the heavy (full-suite) escalation (OMN-15059)
# =============================================================================
# CLAUDE.md documents that pushes / heavy gate runs default to the `.200`
# execution host, not the local Mac -- but a rule stated only in a doc/prompt
# has zero enforcement force without a call-site mechanism (memory
# feedback_a_rule_is_not_a_mechanism). Evidence this is load-bearing: a
# 2026-07-24 session drove the local Mac to load ~55 / 93% swap running this
# exact full-suite escalation for 115+ minutes before .200 was invoked as a
# rescue instead of having been the execution target from the start. This
# guard fires ONLY on the heavy branch below (full-suite fail-closed
# escalation), never on the fast impacted-subset path -- gating every push
# would get this hook disabled within a week, which is worse than no guard.
#
# An UNRESOLVABLE hostname fails CLOSED (OMN-16489 defect 3, redesign plan
# 2026-08-24 §4 S0 item 3 / C2 — supersedes the earlier fail-open note here).
# Heavy runs are routed BY host identity; a host that cannot be identified
# cannot be routed, and proceeding on that silence is the same assumed-
# headroom failure class as the load-probe incidents below. The refusal is
# cheap (<1s, before any pytest) and names its remediation, consistent with
# this hook's fail-loud doctrine: a gate that cannot run must be
# indistinguishable from a failing gate.
PREPUSH_200_HOSTNAME="${PREPUSH_200_HOSTNAME:-stickybeatz-studio}"

# =============================================================================
# Live-load host selection (OMN-16295)
# =============================================================================
# Extends the host-IDENTITY guard below with a CAPACITY dimension: `.200`
# being the right host by IDENTITY does not mean it has headroom. Measured
# 2026-08-20: `.200` load average 32-34 against 24 cores (and, live during
# this same investigation, 56/24 -- 2.3x oversubscribed) driving an 89-93
# minute full-suite run with orphaned pytest processes left behind. Same
# failure class as the 2026-07-24 incident described below, recurring under
# concurrent-session load. OMN-16295 adds a second execution target -- a
# hard-capped gate-runner container on `.201`
# (docker/docker-compose.gate-runner.yml, this repo), selected ONLY when
# `.200` is over threshold, never the default.
#
# FAIL-CLOSED, unlike the host-IDENTITY guard's fail-open posture below --
# deliberately different, not inconsistent. An unresolvable HOSTNAME is
# ambiguous evidence about WHERE we are (fail open: don't lock a developer out
# of their own repo on a shaky read). An unresolvable LOAD reading is a
# failure to prove EITHER candidate host has capacity, and proceeding anyway
# on that silence is exactly how the 2026-07-24 / 2026-08-20 incidents
# happened -- assumed headroom that was not there. "Neither host reachable"
# refuses; it does not skip the check.
PREPUSH_201_GATE_RUNNER_HOSTNAME="${PREPUSH_201_GATE_RUNNER_HOSTNAME:-gate-runner-201}"
PREPUSH_200_SSH_TARGET="${PREPUSH_200_SSH_TARGET:-jonah@stickybeatz-studio.tail75df5e.ts.net}"  # onex-allow-internal-ip OMN-16295 reason="pre-push guard needs the real host target to probe live load"
PREPUSH_201_SSH_TARGET="${PREPUSH_201_SSH_TARGET:-jonah@192.168.86.201}"  # onex-allow-internal-ip OMN-16295 reason="pre-push guard needs the real host target to probe live load" # fallback-ok: real .201 host target, not a dev/local placeholder
# load1/cores at or under this ratio counts as "fit". 1.0 == "not
# oversubscribed" (a standard load-average heuristic); correctly reads the
# observed-fit `.201` snapshot (~0.4x, 2026-08-20) as fit and both observed
# `.200` snapshots above (1.33x and 2.3x) as over threshold.
PREPUSH_LOAD_THRESHOLD="${PREPUSH_LOAD_THRESHOLD:-1.0}"

# Cross-platform (Linux `.201` / macOS `.200`) load probe, printing
# "<load1> <nproc>". Deliberately interpreter-free: OMN-14953's pinned-
# interpreter gate (tests/ci/test_prepush_hook_pinned_interpreter.py) requires
# every python/python3 invocation under scripts/hooks/ to route through
# `uv run`, and the two ssh branches below cannot -- `.201` has no `uv` binary
# at all (probed 2026-08-20). Dropping the interpreter satisfies that gate
# rather than carving an exception out of it, and keeps interpreter startup
# off the pre-push critical path.
#
# Two portability constraints, both load-bearing:
#   1. Field extraction uses cut(1), NOT `set -- $(...)` word splitting.
#      `.200`'s remote login shell is zsh, which does not word-split unquoted
#      command substitution, so `set --` would collapse the whole line into $1
#      there while working fine on `.201`'s bash.
#   2. This snippet is handed to ssh(1) as the remote command and executed by
#      whatever login shell the remote user has, so it stays POSIX and carries
#      no single quotes (it is itself a single-quoted assignment here).
# shellcheck disable=SC2016  # intentionally unexpanded: evaluated by the local
# `sh -c` / the remote login shell, not by this script.
#
# THIRD FIELD: AVAILABLE MEMORY IN MiB (OMN-17392, the OMN-17271 memory
# dimension). load1 is a CPU-time proxy and says nothing about the resource
# that actually killed a suite: on 2026-08-31 an OMN-17316 landing lost hours
# to the `.201` gate-runner OOM-killing full suites at its 8 GiB cap while this
# picker -- reading CPU only -- kept ranking `.201` FIRST. Measured live while
# building this change, one second apart:
#
#   .201 HOST:      load 3.27 / 32 cores = 0.10x   mem_avail 49771 MiB
#   gate-runner:    load 3.27 / 32 cores = 0.10x   mem_avail  2562 MiB
#                   (/sys/fs/cgroup/memory.max 8589934592
#                    - memory.current 5902548992)
#
# Identical load, 19x difference in the resource that OOMs. A CPU-only probe
# cannot tell those two apart, which is exactly why the picker kept
# recommending a saturated target.
#
# CGROUP-AWARE ON PURPOSE: inside a memory-capped container the machine's
# MemAvailable is not the headroom the suite gets, so the probe reports
# min(MemAvailable, memory.max - memory.current) and a capped container
# advertises its OWN cap. Both cgroup v2 (memory.max/current) and v1
# (memory.limit_in_bytes/usage_in_bytes) are read; an uncapped v1 limit is a
# huge sentinel, hence the 1e12 guard.
#
# `-1` means COULD NOT READ, and the picker treats it as unfit -- never as
# ample. Silence is not headroom (the posture the load probe already had).
#
# awk is deliberately NOT used for the memory read even though it is used
# below: every awk program here would need single quotes, and this snippet is
# itself a single-quoted assignment. POSIX arithmetic + cut/grep/tr carry no
# single quotes and need no second quoting level. Verified live on all four
# lab hosts plus the capped container (macOS vm_stat path and Linux
# /proc/meminfo path both).
# shellcheck disable=SC2016  # intentionally unexpanded: evaluated by the local
# `sh -c` / the remote login shell, not by this script.
_PREPUSH_LOAD_PROBE_SH='n=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)
[ "$n" -gt 0 ] || exit 1
if [ -r /proc/loadavg ]; then
  l=$(cut -d" " -f1 /proc/loadavg)
else
  l=$(sysctl -n vm.loadavg 2>/dev/null | cut -d" " -f2)
fi
[ -n "$l" ] || exit 1
m=-1
if [ -r /proc/meminfo ]; then
  k=$(grep MemAvailable /proc/meminfo | tr -s " " | cut -d" " -f2)
  [ -n "$k" ] && m=$((k / 1024))
  if [ -r /sys/fs/cgroup/memory.max ] && [ -r /sys/fs/cgroup/memory.current ]; then
    x=$(cat /sys/fs/cgroup/memory.max)
    c=$(cat /sys/fs/cgroup/memory.current)
    if [ "$x" != max ] && [ -n "$c" ]; then
      h=$(((x - c) / 1048576))
      [ "$h" -lt "$m" ] && m=$h
    fi
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ] && [ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
    x=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    c=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
    if [ -n "$x" ] && [ -n "$c" ] && [ "$x" -lt 1000000000000 ]; then
      h=$(((x - c) / 1048576))
      [ "$h" -lt "$m" ] && m=$h
    fi
  fi
else
  v=$(vm_stat 2>/dev/null) || v=""
  if [ -n "$v" ]; then
    p=$(printf "%s\n" "$v" | grep "page size of" | tr -dc "0-9")
    f=$(printf "%s\n" "$v" | grep "Pages free" | tr -dc "0-9")
    i=$(printf "%s\n" "$v" | grep "Pages inactive" | tr -dc "0-9")
    s=$(printf "%s\n" "$v" | grep "Pages speculative" | tr -dc "0-9")
    u=$(printf "%s\n" "$v" | grep "Pages purgeable" | tr -dc "0-9")
    [ -n "$p" ] && [ -n "$f" ] && m=$(((f + ${i:-0} + ${s:-0} + ${u:-0}) * p / 1048576))
  fi
fi
printf "%s %s %s\n" "$l" "$n" "$m"'

# Prefer GNU coreutils timeout(1); fall back to gtimeout(1) (Homebrew name on
# macOS); fall back to no wrapper at all (ssh -o ConnectTimeout already bounds
# the connection phase, and the remote command is a single fast shell probe).
_prepush_timeout_cmd() {
  if command -v timeout > /dev/null 2>&1; then
    printf 'timeout'
  elif command -v gtimeout > /dev/null 2>&1; then
    printf 'gtimeout'
  fi
}

# `ssh -n` IS LOAD-BEARING, not hygiene (OMN-16991 verify finding 1). This probe
# is called from inside the host-table row loop in pick_capacity_host, whose
# stdin is the row list. Without -n, ssh(1) reads and discards that stdin, so
# the FIRST probe swallowed every remaining row and the picker evaluated
# exactly one host -- live, it probed h200 and never saw h201/h101/h105.
# host_load_ratio TARGET -- prints "<load1> <nproc> <ratio>" and returns 0, or
# prints nothing and returns 1 on any read/parse/timeout failure. TARGET is
# empty for "read this host directly" or an ssh(1) target string for a
# bounded remote read. Deterministic, network-free overrides for tests (each a
# "<load1> <nproc>" pair -- the ratio is still computed from it, never
# hardcoded):
#   PREPUSH_LOAD_OVERRIDE_LOCAL   overrides the direct (TARGET="") read
#   PREPUSH_LOAD_OVERRIDE_REMOTE  overrides every ssh-target read
host_load_ratio() {
  local target="$1" raw load1 ncpu memmb timeout_cmd
  # OMN-16995: REAP FIRST, MEASURE SECOND. A leaked `sh -c while :; do :; done`
  # orphan is indistinguishable from real work in load1, and 19 of them once
  # put `.200` at 1.64x-core and refused every heavy escalation in the lab. The
  # reaper is defined in prepush_dispatch.sh, which is sourced below this
  # definition and therefore resolved by the time any caller runs.
  reap_spin_loop_orphans "$target" || true
  if [ -z "$target" ]; then
    if [ -n "${PREPUSH_LOAD_OVERRIDE_LOCAL:-}" ]; then
      raw="$PREPUSH_LOAD_OVERRIDE_LOCAL"
    else
      raw="$(sh -c "$_PREPUSH_LOAD_PROBE_SH" 2> /dev/null)" || return 1
    fi
  else
    if [ -n "${PREPUSH_LOAD_OVERRIDE_REMOTE:-}" ]; then
      raw="$PREPUSH_LOAD_OVERRIDE_REMOTE"
    else
      timeout_cmd="$(_prepush_timeout_cmd)"
      if [ -n "$timeout_cmd" ]; then
        raw="$("$timeout_cmd" 6 ssh -n -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
          "$target" "$_PREPUSH_LOAD_PROBE_SH" 2> /dev/null)" || return 1
      else
        raw="$(ssh -n -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
          "$target" "$_PREPUSH_LOAD_PROBE_SH" 2> /dev/null)" || return 1
      fi
    fi
  fi
  [ -n "$raw" ] || return 1
  # shellcheck disable=SC2086
  set -- $raw
  load1="${1:-}"
  ncpu="${2:-}"
  # Third field is available MiB (OMN-17392). An override that supplies only
  # the historical "<load1> <nproc>" pair reports -1, i.e. "could not read",
  # and is therefore treated as UNFIT rather than ample -- the same posture the
  # real probe takes when /proc/meminfo and vm_stat are both unreadable. An
  # override is a test seam, not an exemption from the memory floor.
  memmb="${3:--1}"
  case "$memmb" in '' | *[!0-9-]*) memmb=-1 ;; esac
  [ -n "$load1" ] && [ -n "$ncpu" ] && [ "$ncpu" != "0" ] || return 1
  awk -v l="$load1" -v n="$ncpu" -v m="$memmb" \
    'BEGIN { if (n + 0 <= 0) exit 1; printf "%s %s %.3f %s\n", l, n, (l / n), m }'
}

# The floor, in MiB, of available memory a host must PROVE before it may take a
# heavy suite (OMN-17392). A DELIBERATE CONSTANT, not `${VAR:-4096}`: an env
# indirection here would be a one-word bypass of the exact admission control
# this adds (`PREPUSH_MIN_FREE_MEM_MB=0` restores the blind picker), and the
# operator directive is explicit that PREPUSH_* overrides stay forbidden.
# Tests drive the MEASUREMENT through PREPUSH_MEM_OVERRIDE_MAP, never the floor.
#
# 4096 is chosen against the measured OOM, not picked round: the `.201`
# gate-runner OOM-killed full suites at a 8 GiB cap (OMN-17247), and its
# headroom while running one measured 2562 MiB. A 4 GiB floor refuses that
# container while it is saturated and re-admits it once the suite drains,
# while keeping every host the lab actually uses in the fleet -- measured the
# same minute: h200 77564, h201 49771, h105 14664, h101 7459. A floor at 8192
# would have excluded h101, shrinking the fleet and pushing work back onto the
# Mac, which is the opposite of what this change is for.
PREPUSH_MIN_FREE_MEM_MB=4096

# host_is_fit TARGET -- 0 if the host proved BOTH capacity dimensions, 1 if it
# is measurably over on either, 2 if the read itself failed
# (unreachable/unresolvable/unreadable memory). Callers must not conflate 1 and
# 2 anywhere the difference is user-visible ("over capacity" vs "could not
# check"). Sets PREPUSH_LAST_FIT_DETAIL so a caller can say WHICH dimension
# refused instead of reporting a bare "unfit".
host_is_fit() {
  local target="$1" reading ratio memmb
  PREPUSH_LAST_FIT_DETAIL=""
  reading="$(host_load_ratio "$target")" || return 2
  ratio="$(printf '%s' "$reading" | awk '{print $3}')"
  memmb="$(printf '%s' "$reading" | awk '{print $4}')"
  [ -n "$ratio" ] || return 2
  if ! awk -v r="$ratio" -v thr="$PREPUSH_LOAD_THRESHOLD" 'BEGIN { exit !(r <= thr + 0) }'; then
    PREPUSH_LAST_FIT_DETAIL="load ${ratio}x > ${PREPUSH_LOAD_THRESHOLD}x"
    return 1
  fi
  # Memory is checked AFTER load and reported separately: a host that is idle
  # but memory-starved is the case the CPU-only picker got wrong, and calling
  # it "over capacity" without naming memory would send the reader hunting for
  # CPU load that is not there.
  if [ -z "$memmb" ] || [ "$memmb" = "-1" ]; then
    PREPUSH_LAST_FIT_DETAIL="memory unreadable"
    return 2
  fi
  if [ "$memmb" -lt "$PREPUSH_MIN_FREE_MEM_MB" ] 2> /dev/null; then
    PREPUSH_LAST_FIT_DETAIL="mem ${memmb}MiB < ${PREPUSH_MIN_FREE_MEM_MB}MiB"
    return 1
  fi
  PREPUSH_LAST_FIT_DETAIL="load ${ratio}x, mem ${memmb}MiB"
  return 0
}

# =============================================================================
# Lab-wide distribution helpers (OMN-16991)
# =============================================================================
# Sourced AFTER host_load_ratio/host_is_fit/_prepush_timeout_cmd, which the
# library reuses rather than reimplementing, and BEFORE guard_full_suite_host,
# which is its only caller. Located relative to this script so it resolves the
# same way whether git invokes the hook through .git/hooks or core.hooksPath.
# shellcheck source=scripts/hooks/prepush_dispatch.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepush_dispatch.sh"

# =============================================================================
# Remote full-suite verification -- third execution target (OMN-16688)
# =============================================================================
# `.200` and the `.201` gate-runner are the two LOCAL heavy targets. Both can be
# over the load threshold at the same time, and on 2026-08-26 both were: `.201`
# sat at load 74 against 32 cores (2.3x) with 50 of 53 self-hosted runners busy,
# so every heavy escalation refused and the pre-push queue stalled outright --
# not slow, stalled, because `host_is_fit` is a HOOK constraint and no amount of
# queue depth can satisfy it.
#
# Every OmniNode repo is PUBLIC, so GitHub-hosted minutes are free and
# unmetered, and `.github/workflows/ci.yml` ALREADY runs this same full sharded
# suite on `ubuntu-latest`. When neither local host can take the work, a
# GitHub-hosted run pinned to the exact sha being pushed is better evidence than
# what we would otherwise accept -- a degraded-capacity override grant running a
# contended local suite. This target is therefore an EVIDENCE UPGRADE over the
# fallback it displaces, not a discount on it.
#
# NOT A BYPASS, and shaped so it cannot become one:
#   * It is consulted ONLY on the paths that would otherwise `die` or fall back
#     to a degraded-evidence override grant. Every currently-passing path
#     behaves identically.
#   * It accepts no PREPUSH_* override and reads no local artifact. The answer
#     is re-derived live from the GitHub API each time, so there is no file on
#     disk to forge.
#   * It cannot make the gate accept LESS work: prepush_remote_verify.py
#     requires the run to be sha-pinned, green, AND full-suite shaped (all
#     `_FULL_SUITE_SPLIT_COUNT` shards green -- the constant is imported from
#     the selector itself). A selector-narrowed run is rejected.
#   * Exit 2 ("could not resolve", e.g. gh unavailable) is treated as NO
#     evidence and falls through to the existing refusal -- never as a pass.
#     Same fail-closed posture as the load probe above.
REMOTE_FULL_SUITE_VERIFIED=0

# remote_full_suite_verified HEAVY_WHAT -- 0 if a sha-pinned, green, full-suite
# CI run already exists for HEAD; 1 otherwise. Sets REMOTE_FULL_SUITE_VERIFIED
# on success so the caller can skip the local pytest invocation entirely.
remote_full_suite_verified() {
  local heavy_what="$1" head_sha rc=0 out
  head_sha="$(git rev-parse HEAD 2> /dev/null || true)"
  if [ -z "$head_sha" ]; then
    return 1
  fi
  log "checking for a GitHub-hosted FULL-suite run pinned to ${head_sha} before refusing ${heavy_what}..."
  out="$(uv run python "${REPO_ROOT}/scripts/hooks/prepush_remote_verify.py" \
    check --head-sha "$head_sha" 2>&1)" || rc=$?
  if [ "$rc" -eq 0 ]; then
    log "REMOTE FULL-SUITE PASS accepted in place of ${heavy_what}: ${out}"
    log "  evidence: the full suite ran to green on GitHub-hosted CI against this exact tree (${head_sha})."
    REMOTE_FULL_SUITE_VERIFIED=1
    return 0
  fi
  # rc 1 = resolved, no qualifying run. rc 2 = could not resolve at all. Both
  # mean "no evidence"; only the operator-facing wording differs.
  log "no remote full-suite evidence for ${head_sha}: ${out}"
  return 1
}

# REMOTE_LAB_RUN_VERDICT (OMN-16991) -- set to 1 when a designated lab host ran
# this exact tree green over the remote leg, so the caller elides the local
# pytest exactly as it does for the GitHub-hosted pass. Deliberately a DISTINCT
# sentinel from REMOTE_FULL_SUITE_VERIFIED: the two carry different evidence
# strength and different log wording, and collapsing them would make a
# lab-host run indistinguishable from a GitHub-hosted one in the transcript.
REMOTE_LAB_RUN_VERDICT=0

# dispatch_to_lab_host HEAVY_WHAT -- try to satisfy HEAVY_WHAT by running it on
# a designated lab host, placement_tier-major then cheapest-loaded within a
# tier (OMN-17485: a `last_resort` row -- `.201`, the dev-lane evidence surface
# and interactive collaborator workspace -- is reached only when no
# default-tier host is fit).
# 0 = satisfied (green), 1 = no evidence (caller falls through), and it does
# NOT return on a remote RED: a suite that genuinely failed on a designated
# host is a failing gate, so it refuses here rather than letting the caller
# fall through to the degraded-evidence override grant.
#
# It walks the RANKED candidate list rather than betting the whole escalation on
# one host (OMN-16991 verify finding 3). Only a verdict -- green or red -- ends
# the walk. "No evidence" (unreachable on arrival, no completion marker) and
# "slot taken between the probe and the run" (rc 4) are placement misses, not
# statements about the tree, so they advance to the next fit host instead of
# refusing a push that another idle lab host could have cleared.
#
# `authorizing` is passed EXPLICITLY: this is the verdict-bearing path, and a
# shadow row's verdict cannot satisfy the escalation by definition. Ranking one
# in would spend a bundle, an scp, a `uv sync` and a full suite to produce an
# answer that is then thrown away, while the authorizing host that could have
# answered goes unprobed.
dispatch_to_lab_host() {
  local heavy_what repo rc=0 idx=1 total
  heavy_what="$1"
  repo="$(basename "$REPO_ROOT")"
  if ! pick_capacity_host "$PREPUSH_LC_HOST" "$repo" authorizing; then
    log "no lab host is fit for ${heavy_what}: ${PREPUSH_PROBE_LOG:-no hosts probed}"
    return 1
  fi
  total="$(prepush_candidate_count)"
  while [ "$idx" -le "$total" ]; do
    prepush_select_candidate "$idx" || break
    if [ -z "$PREPUSH_PICK_SSH" ]; then
      # This candidate IS this host: there is nothing to DISTRIBUTE, so the
      # remote leg cannot answer for it and the ranked hosts after it still
      # can. Skipping it here is correct -- but it used to be SILENT, and that
      # silence is how OMN-17280 stayed invisible: for an actor who can reach
      # no other host, this was the only fit candidate in the lab, and the walk
      # dropped it without a word before falling through to die(). The
      # same-host route now lives in prepush_local_actor_route, one rung below
      # this call in guard_full_suite_host; naming the skip makes the transcript
      # explain how control got there.
      log "lab placement: ${PREPUSH_PICK_LABEL} IS this host, so it carries no remote leg; the same-host route is evaluated after the lab walk (OMN-17280)"
      idx=$((idx + 1))
      continue
    fi
    rc=0
    prepush_remote_run "$heavy_what" || rc=$?
    case "$rc" in
      0)
        REMOTE_LAB_RUN_VERDICT=1
        return 0
        ;;
      3)
        die "${heavy_what} FAILED on the designated lab host '${PREPUSH_PICK_HOSTNAME}' (${PREPUSH_PICK_LABEL})" \
            "the suite genuinely failed on a host we designated -- this is a red gate, not a capacity problem. Read the streamed [${PREPUSH_PICK_LABEL}] output above (the tail of that host's suite.log is printed there), fix the failing tests, then re-push. A remote red is never satisfied by minting an override grant"
        ;;
      4)
        log "lab placement: ${PREPUSH_PICK_LABEL}'s heavy-suite slot was taken on arrival; trying the next fit host"
        ;;
      *)
        log "lab placement: ${PREPUSH_PICK_LABEL} returned no usable evidence; trying the next fit host"
        ;;
    esac
    idx=$((idx + 1))
  done
  log "no fit lab host produced a verdict for ${heavy_what}: ${PREPUSH_PROBE_LOG:-no hosts probed}"
  return 1
}

# =============================================================================
# Off-box-by-default routing (OMN-17392)
# =============================================================================
# Operator directive 2026-08-31, verbatim: "we should move prepush off this box
# if possible". The box is `.200` (row h200), and the reason the directive was
# needed is a single short-circuit below: the guard ran the heavy suite LOCALLY
# the moment the local host was a designated capacity row and its load probe
# read under threshold. Lab dispatch was only ever reached once the local host
# was already over threshold -- i.e. the fleet was consulted only after this
# machine was too loaded to be worth consulting it about. Measured that day:
# load1 96.58 / 24 cores = 4.02x during landings, with h105 at 0.12x and h201
# at 0.10x sitting idle.
#
# The budget and interval are CONSTANTS, not `${VAR:-...}`. An env indirection
# would be a one-word bypass of exactly this policy (`..._BUDGET=0` collapses
# straight through to the local fallback), and the directive is explicit that
# PREPUSH_* overrides stay forbidden. Tests pass the budget positionally.
PREPUSH_OFFBOX_WAIT_BUDGET_SECONDS=900
PREPUSH_OFFBOX_WAIT_INTERVAL_SECONDS=60

# prepush_try_local_heavy_slot -- 0 when this host has PROVEN both capacity and
# an exclusive heavy-suite slot (and now holds it), 1 otherwise. Factored out of
# guard_full_suite_host unchanged in substance so the `allowed` path and the
# post-wait fallback share one implementation and cannot drift into two
# different notions of "may run here".
#
# It records WHY it said no in PREPUSH_LOCAL_HEAVY_REASON. Before this function
# existed the refusal was logged from inside an `if host_is_fit ""` branch, so
# "this host is fit but its slot is held" was necessarily true wherever it
# printed. Hoisting the check into here made that sentence reachable for an
# over-loaded or memory-starved host too, where it is measurably false and
# sends the reader hunting for a held lock that does not exist.
prepush_try_local_heavy_slot() {
  local lw lock_rc=0
  PREPUSH_LOCAL_HEAVY_REASON=""
  if ! host_is_fit ""; then
    PREPUSH_LOCAL_HEAVY_REASON="this host is not fit (${PREPUSH_LAST_FIT_DETAIL:-unmeasured})"
    return 1
  fi
  lw="$(prepush_local_workroot "$PREPUSH_LC_HOST" || true)"
  [ -n "$lw" ] || lw="${REPO_ROOT}/.onex_state/prepush_distribution"
  prepush_lock_acquire "$lw" || lock_rc=$?
  if [ "$lock_rc" -eq 0 ]; then
    # No `trap ... EXIT` here: prepush_hook_cleanup (installed once, below)
    # already releases the lock. Installing a second EXIT trap would drop the
    # temp-file cleanup this hook installed first.
    return 0
  fi
  if [ "$lock_rc" -eq 2 ]; then
    # The workroot is unusable, which says nothing about this host's capacity.
    # Proceed exactly as the hook did before this lock existed rather than
    # inventing a refusal out of an infrastructural failure.
    # OMN-17280. Before degrading to an UNSERIALIZED run, ask whether this is
    # the actor case: a workroot we cannot write is the signature of running as
    # someone other than whoever provisioned this host, and when NO capacity row
    # is reachable for that actor the same-host route is the governed answer --
    # it takes a per-actor slot under $HOME instead of running with no lock at
    # all, and it writes the receipt that names why the suite ran here. It
    # declines the moment any lab host is reachable, so an OWNER whose workroot
    # is genuinely broken still gets exactly the warning below.
    if prepush_local_actor_route "${heavy_what:-heavy fail-closed full-suite escalation}" \
      "$(prepush_identity_label "$PREPUSH_LC_HOST" || true)"; then
      return 0
    fi
    log "WARNING: could not create the heavy-suite slot lock under '${lw}' -- running unserialized on this host (pre-OMN-16991 behavior). Fix the workroot to restore serialization (OMN-16174)."
    return 0
  fi
  PREPUSH_LOCAL_HEAVY_REASON="this host is fit (${PREPUSH_LAST_FIT_DETAIL:-unmeasured}) but its heavy-suite slot is already held"
  return 1
}

# prepush_lab_has_transient_capacity -- 0 when the last probe refused at least
# one candidate for a reason that CAN resolve on its own, 1 when every refusal
# is structural.
#
# The bounded wait below exists to catch a lab slot freeing up. That premise
# holds for `busy` (a suite finishes), `over` (load drains) and `mem-over` (the
# suite holding the memory exits). It does NOT hold for `unreachable`,
# `repo-denied`, `disabled`, `uv-unfit` or `mode-*-not-eligible`: none of those
# change because a pusher waited, so spending the budget on them buys nothing
# and costs 900s of silence before the fallback the push was always going to
# reach. The concrete case is a Mac off the lab LAN -- every remote row probes
# `unreachable`, and without this gate EVERY heavy push there pays the full
# budget before running locally anyway.
#
# This can only SHORTEN a wait, never skip a gate: the caller still returns "no
# placement", and the local fallback it falls through to still has to prove
# measured capacity AND an exclusive slot. It matches on the probe-log tokens
# pick_capacity_host writes, so a new refusal reason defaults to STRUCTURAL --
# a reason we have not classified does not silently earn a 15-minute wait.
prepush_lab_has_transient_capacity() {
  case "${PREPUSH_PROBE_LOG:-}" in
    *"=busy("* | *"=over("* | *"=mem-over("*) return 0 ;;
  esac
  return 1
}

# prepush_wait_for_lab_capacity HEAVY_WHAT BUDGET INTERVAL -- retry lab
# placement until a host takes the work or BUDGET seconds elapse. 0 = a lab host
# produced a green verdict, 1 = the budget is exhausted with no placement.
#
# This is the "queue and wait" rung the directive asks for, and it is VISIBLE by
# construction: every attempt logs the probe trail it just took, how much of the
# budget it has spent, and when it will re-probe. A push that waits looks like a
# push that is waiting, not like a hung hook.
#
# It re-probes the WHOLE ranked list each round rather than re-trying one host:
# the thing being waited on is a slot freeing up ANYWHERE in the lab, and by the
# next round the ranking has usually changed.
#
# It does NOT catch a remote RED: dispatch_to_lab_host die()s on a genuine
# failure, so a red suite still refuses the push immediately instead of being
# retried until the budget runs out.
prepush_wait_for_lab_capacity() {
  local heavy_what="$1" budget="$2" interval="$3" waited=0 attempt=1
  while :; do
    if dispatch_to_lab_host "$heavy_what"; then
      return 0
    fi
    [ "$waited" -lt "$budget" ] || break
    if ! prepush_lab_has_transient_capacity; then
      log "OFF-BOX QUEUE-AND-WAIT: not waiting -- every lab refusal cannot resolve on its own."
      log "  probed: ${PREPUSH_PROBE_LOG:-none}"
      log "  No host is merely busy/over/memory-starved, so re-probing would return the same answer for the full ${budget}s. Falling through to the refusal ladder now."
      return 1
    fi
    log "OFF-BOX QUEUE-AND-WAIT (attempt ${attempt}): no lab host has headroom for ${heavy_what} yet."
    log "  probed: ${PREPUSH_PROBE_LOG:-none}"
    log "  waited ${waited}s of a ${budget}s budget; re-probing the whole ranked list in ${interval}s. Ctrl-C aborts the push."
    sleep "$interval"
    waited=$((waited + interval))
    attempt=$((attempt + 1))
  done
  log "OFF-BOX QUEUE-AND-WAIT: ${budget}s budget exhausted after ${attempt} attempt(s); no lab host took ${heavy_what}."
  return 1
}

guard_full_suite_host() {
  local host lc_host label heavy_what designated policy
  # OMN-15408: the caller names WHICH heavyweight run is being guarded, so the
  # refusal names the real cause. Default preserves the OMN-15059 wording for
  # the flag-driven escalation call sites, which pass no argument.
  heavy_what="${1:-heavy fail-closed full-suite escalation}"
  host="$(hostname -s 2>/dev/null || true)"
  if [ -z "$host" ]; then
    # Fail CLOSED (OMN-16489): see the routing note above PREPUSH_200_HOSTNAME.
    die "could not determine the local hostname while deciding where ${heavy_what} may run" \
        "heavy gate runs are routed by host identity (OMN-15059) and an unidentifiable host cannot be routed. Fix 'hostname -s' (macOS: 'sudo scutil --set HostName <name>'; Linux: 'hostnamectl set-hostname <name>'), or run the push from a designated gate host listed in ${PREPUSH_HOST_TABLE_REL}"
  fi
  lc_host="$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')"
  PREPUSH_LC_HOST="$lc_host"

  # OMN-16991: host identity now resolves against the COMMITTED host table
  # instead of the two hard-coded names this guard used to test
  # (`[ "$lc_host" = "$lc_target" ] || [ "$lc_host" = "$lc_201" ]`). That
  # literal `||` -- not policy -- was the entire structural reason .101 and
  # .105 could not be used, and it is also why `.201` only ever matched from
  # INSIDE the gate-runner container: the container sets hostname
  # gate-runner-201 while the host itself reports omninode-pc, so every push on
  # the host needed PREPUSH_201_GATE_RUNNER_HOSTNAME exported to pass. Both
  # names are now rows, so `.201` is designated intrinsically and no env var
  # has to survive a process or ssh boundary for the guard to see it.
  #
  # An UNREADABLE table fails CLOSED, on the same reasoning as the unresolvable
  # hostname above: heavy runs are routed by host identity, and identity that
  # cannot be resolved cannot be routed.
  if ! prepush_table_text > /dev/null 2>&1; then
    die "the pre-push host table (${PREPUSH_HOST_TABLE_REL}) could not be read from HEAD, so no host can be identified as a designated gate host for ${heavy_what}" \
        "the table is read from the COMMITTED tree so an uncommitted row cannot self-designate this machine as an authorizing gate host. Commit ${PREPUSH_HOST_TABLE_REL} (or, if you have edited it, commit the edit so HEAD and the working tree agree), then re-push"
  fi
  label="$(prepush_identity_label "$lc_host" || true)"
  designated="$(prepush_designated_hostnames)"

  if [ -n "$label" ]; then
    policy="$(prepush_heavy_local_policy "$lc_host" || true)"
    [ -n "$policy" ] || policy="allowed"

    if [ "$policy" = "prefer_remote" ]; then
      # OFF-BOX BY DEFAULT (OMN-17392). This host is a designated, authorizing
      # gate host and could very well be fit right now -- and that is exactly
      # the case the directive retires. Being ABLE to run the suite here is no
      # longer a reason to. The local run is still reachable, but only as the
      # LAST rung, after the lab has been asked and asked again.
      log "OFF-BOX ROUTING: '${host}' (${label}) is heavy_local=prefer_remote, so ${heavy_what} looks for a lab host BEFORE running here (OMN-17392)."
      if remote_full_suite_verified "$heavy_what"; then
        return 0
      fi
      if prepush_wait_for_lab_capacity "$heavy_what" \
        "$PREPUSH_OFFBOX_WAIT_BUDGET_SECONDS" "$PREPUSH_OFFBOX_WAIT_INTERVAL_SECONDS"; then
        return 0
      fi
      # The bounded wait is spent. Running here is now permitted -- but ONLY on
      # the same proof any other host must produce: measured capacity AND an
      # exclusive slot. A local host over threshold still refuses below exactly
      # as it did before this change, so this fallback is strictly narrower
      # than the pre-OMN-17392 behavior it replaces, never wider.
      if prepush_try_local_heavy_slot; then
        log "=============================================================================="
        log "LOCAL FALLBACK IN EFFECT -- ${heavy_what} is running ON THIS BOX ('${host}')."
        log "  This host is heavy_local=prefer_remote: off-box was tried FIRST and did not"
        log "  place. Waited the full ${PREPUSH_OFFBOX_WAIT_BUDGET_SECONDS}s off-box budget before falling back."
        log "  last probe: ${PREPUSH_PROBE_LOG:-none}"
        log "  local capacity accepted on: ${PREPUSH_LAST_FIT_DETAIL:-unknown}"
        log "  This is NOT a bypass: the full escalation runs here, unmodified. It is a"
        log "  capacity event -- if you are seeing it often, the lab is undersized or a"
        log "  host is wedged (knowledge-base-internal:runbooks/omnibase-infra-lab-prepush-host-table.md)."
        log "=============================================================================="
        return 0
      fi
      log "off-box placement failed and this host cannot take the work either -- ${PREPUSH_LOCAL_HEAVY_REASON:-no local capacity measured}; refusing rather than running a suite this host cannot support"
    else
      # OMN-16295: identity alone is not enough -- this known-good host must
      # also have capacity right now.
      #
      # OMN-16174/OMN-16991: the LOCAL heavy path took no lock of any kind
      # before that change, which is why five concurrent full suites once ran
      # on one host with one of them taking 97+ minutes. It is the busiest
      # path in the hook and was the only unserialized one. Take the same
      # exclusive slot a remote host would have to take.
      if prepush_try_local_heavy_slot; then
        return 0
      fi
      log "${PREPUSH_LOCAL_HEAVY_REASON:-this host cannot take the work}; looking for another lab host before refusing"
    fi
    # Precedence, in order of EVIDENCE STRENGTH -- not convenience:
    #   1. GitHub-hosted sha-pinned FULL-suite pass (OMN-16688). Strongest:
    #      uncontended, full-suite shaped, re-derived live from the API with no
    #      file on disk to forge. It stays FIRST; putting the lab leg ahead of
    #      it would silently demote the strongest evidence the hook has.
    #   2. A designated lab host running this exact tree (OMN-16991). Weaker
    #      than (1) -- the tree is materialized on another host -- but far
    #      stronger than (3), because a real suite actually ran on hardware we
    #      designate, bound to this sha by a completion marker.
    #   3. Single-use receipted degraded-capacity grant. Weakest: it runs a
    #      contended suite here and says so.
    #   4. die().
    #
    # OMN-17392 did NOT reorder this ladder -- it changed only WHEN a
    # `prefer_remote` host is allowed to skip it by running locally. On an
    # `allowed` host the ladder is reached exactly as before (local unfit or
    # slot held). On a `prefer_remote` host rungs 1 and 2 have already been
    # walked, and a bounded wait spent on rung 2, before control arrives here;
    # re-walking them costs two read-only probes and is worth it, because the
    # lab's state may well have changed during a 900s wait.
    if remote_full_suite_verified "$heavy_what"; then
      return 0
    fi
    if dispatch_to_lab_host "$heavy_what"; then
      return 0
    fi
    # OMN-17280 -- SAME-HOST ROUTE, above the grant on evidence strength.
    # Placed here, and only here, so it can fire ONLY after the lab has been
    # asked and answered nothing. It refuses itself the moment any capacity row
    # is reachable for this actor, which is every one of the owner's own
    # pushes, so the OMN-17392 / OMN-17485 off-box preference is untouched. It
    # is above consume_override_grant because it produces a real full suite on
    # a designated authorizing host -- strictly stronger evidence than a
    # receipted degraded-capacity grant, and it burns no grant to get there.
    if prepush_local_actor_route "$heavy_what" "$label"; then
      return 0
    fi
    if consume_override_grant "degraded-capacity: ${heavy_what} on '${host}' at/over the ${PREPUSH_LOAD_THRESHOLD}x-core load threshold"; then
      log "WARNING: DEGRADED-CAPACITY OVERRIDE IN EFFECT (single-use grant consumed) -- running ${heavy_what} on '${host}' at/over the ${PREPUSH_LOAD_THRESHOLD}x-core load threshold. Treat any evidence from this run as WEAKER than a fit-host-run gate."
      return 0
    fi
    die "${heavy_what} triggered on '${host}' (designated gate host '${label}'), but its load is at/over the ${PREPUSH_LOAD_THRESHOLD}x-core threshold and no other lab host could take the work" \
        "probed hosts: ${PREPUSH_PROBE_LOG:-none}. PREFERRED: open/refresh the PR so GitHub-hosted CI runs the FULL suite on this exact sha, then re-push -- this hook accepts that run automatically (OMN-16688; check it with 'uv run python scripts/hooks/prepush_remote_verify.py check --head-sha \$(git rev-parse HEAD)'). See knowledge-base-internal:runbooks/omnibase-infra-lab-prepush-host-table.md to add or re-enable a lab host, or mint a single-use grant to run here anyway (degraded evidence -- do not use as a routine bypass): uv run python scripts/hooks/prepush_override_grant.py mint --reason '<why>'"
  fi
  # Not a designated host. Same precedence, same ordering, same reasoning.
  if remote_full_suite_verified "$heavy_what"; then
    return 0
  fi
  if dispatch_to_lab_host "$heavy_what"; then
    return 0
  fi
  if consume_override_grant "degraded-host: ${heavy_what} on '${host}', not a designated gate host"; then
    log "WARNING: DEGRADED-HOST OVERRIDE IN EFFECT (single-use grant consumed) -- running ${heavy_what} on '${host}', NOT a designated gate host (${designated}). This host has weaker isolation/headroom; treat any evidence from this run as WEAKER than a designated-host gate. See knowledge-base-internal:runbooks/omnibase-infra-lab-prepush-host-table.md."
    return 0
  fi
  die "${heavy_what} triggered on host '${host}', not the designated .200 build host ('${PREPUSH_200_HOSTNAME}') nor any other designated gate host (${designated})" \
      "probed lab hosts: ${PREPUSH_PROBE_LOG:-none}. Push from a designated host, OR let GitHub-hosted CI run the FULL suite on this exact sha and re-push -- this hook accepts a sha-pinned green full-suite run automatically (OMN-16688; check it with 'uv run python scripts/hooks/prepush_remote_verify.py check --head-sha \$(git rev-parse HEAD)'), OR see knowledge-base-internal:runbooks/omnibase-infra-lab-prepush-host-table.md to add/enable a lab host, OR mint a single-use override grant to run the full suite on this host anyway (visible, receipted, degraded-evidence override -- do not use as a routine bypass): uv run python scripts/hooks/prepush_override_grant.py mint --reason '<why>'"
}

# -----------------------------------------------------------------------------
# Heavyweight-SELECTION predicate (OMN-15408)
# -----------------------------------------------------------------------------
# The OMN-15059 guard above was wired to fire on the selector's `is_full_suite`
# FLAG. That is the wrong key: the selector routinely emits
# `is_full_suite=False` with a whole-suite path set -- the entire suite
# arriving as an "impacted subset" -- and those runs sail straight past the
# guard. Measured on host `omnibook` through real `git push` runs on
# 2026-07-29: this repo selected `is_full_suite=False` and executed 2,429 tests
# in 245s locally with the guard never invoked, and omnimarket (identical
# predicate, `paths=[ tests/ ]`) executed 13,898 tests in 506s the same way.
# The SAME selected work forced via `PREPUSH_FULL_SUITE=1` (`is_full_suite=True
# reason=feature_flag_off paths=[ tests/ ]`) WAS refused, in this repo,
# verbatim. Identical cost, opposite outcome, decided by a flag.
#
# SEAM -- what "heavyweight selection" means, exactly: the selection is
# heavyweight when the paths pytest is about to be handed COVER THE ENTIRE
# full-suite target this hook would run on a fail-closed escalation
# (`$FULL_SUITE_TARGET` -- `tests/unit/` in this repo, since infra's escalation
# is unit-scoped by design; defined next to the pytest invocation below so the
# predicate and the actual run can never drift apart). Concretely: some
# selected path is `$FULL_SUITE_TARGET` itself or a directory ANCESTOR of it
# (so a bare `tests/` selection, which is strictly MORE than the escalation
# target, is caught too). That is "the selection failed to be a proper
# narrowing" expressed against the selector's own output -- NOT a parallel cost
# model, no test counting, no timing heuristic, nothing this hook does not
# already parse.
#
# The predicate is evaluated against the RUNNABLE path set (post
# `filter_prepush_runnable_paths`), i.e. exactly the argv pytest receives --
# never the pre-filter selection, which would let a deferred integration path
# influence a local-cost decision it has no bearing on.
#
# A genuine narrow selection (`tests/unit/scripts/`, a single test module) is
# strictly below the target and stays runnable locally -- the guard must not
# brick every push from a developer's machine, only the ones that are the
# full-suite run wearing a different label.
#
# Keep this function self-contained (target passed in, no globals): it is
# extracted and EXECUTED directly by
# tests/ci/test_prepush_hook_host_identity_guard.py.
selection_is_whole_suite() {
  local target normalized_target p normalized
  target="$1"
  shift
  [ -n "$target" ] || return 1
  normalized_target="${target%/}/"
  for p in "$@"; do
    [ -n "$p" ] || continue
    normalized="${p%/}/"
    case "$normalized_target" in
      "$normalized"*) return 0 ;;
    esac
  done
  return 1
}

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || die "not inside a git worktree" \
         "run 'git push' from within the omnibase_infra repository"
cd "$REPO_ROOT"

# =============================================================================
# bash>=5 canary (OMN-15617)
# =============================================================================
# On stickybeatz-studio (.200, the rule-11a default gate host), non-interactive
# ssh sessions resolve `bash` to the system 3.2.57 shell even though a modern
# bash 5.x sits at /opt/homebrew/bin/bash -- it just is not first on PATH for
# that session class. runner-monitor.sh (exercised end-to-end by
# tests/unit/observability/runner_health/test_runner_monitor_*.py) uses
# `declare -A`, which bash 3.2 does not support, so those tests fail SILENTLY
# on every push from this host class -- a bash syntax error deep inside a
# subprocess, not a resolvable "wrong interpreter" diagnostic.
#
# Resolve a bash>=5 interpreter EXPLICITLY here, independent of PATH order,
# via the same resolver the pytest harness uses (single source of truth --
# scripts/ci/resolve_modern_bash.sh -- so the two can never drift apart), and
# export it so the harness does not have to re-discover it. Fail LOUD with a
# pointed remediation message if none is resolvable anywhere -- never a quiet
# skip, never a silent fallback to whatever "bash" happens to resolve first.
MODERN_BASH="$(bash "${REPO_ROOT}/scripts/ci/resolve_modern_bash.sh")" \
  || die "no bash>=5 interpreter resolvable on this host" \
         "install a modern bash (e.g. 'brew install bash') and/or set OMNIBASE_INFRA_BASH_BIN to its absolute path; see scripts/ci/resolve_modern_bash.sh"
export OMNIBASE_INFRA_BASH_BIN="$MODERN_BASH"
log "bash>=5 canary: resolved ${MODERN_BASH}"

BASE_REF="${PREPUSH_BASE_REF:-origin/dev}"

# Deterministic diff base: fetch the base ref best-effort so an online push gets
# an up-to-date merge-base, then REQUIRE it to resolve. Offline is tolerated ONLY
# when the ref already exists locally; an entirely unresolvable base HARD-ERRORS
# rather than silently diffing against nothing.
git fetch --quiet origin "${BASE_REF#origin/}" 2>/dev/null || true
if ! git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null; then
  die "base ref '${BASE_REF}' could not be resolved" \
      "fetch it ('git fetch origin ${BASE_REF#origin/}') or set PREPUSH_BASE_REF to a resolvable ref"
fi

BASE_SHA="$(git merge-base "${BASE_REF}" HEAD 2>/dev/null)" \
  || die "no common ancestor between '${BASE_REF}' and HEAD" \
         "rebase your branch onto ${BASE_REF} so a merge-base exists"

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD)"

CHANGED_FILE="$(mktemp)"
SELECTION_FILE="$(mktemp)"
SELECTION_ERR="$(mktemp)"

# ONE exit trap for the whole hook. bash keeps exactly one EXIT trap per shell,
# so the later `trap prepush_lock_release EXIT` that guard_full_suite_host used
# to install silently REPLACED the temp-file cleanup and leaked three mktemp
# files on every heavy run that took the host slot. Both jobs live in one
# handler instead, so neither can displace the other.
prepush_hook_cleanup() {
  rm -f "${CHANGED_FILE:-}" "${SELECTION_FILE:-}" "${SELECTION_ERR:-}" 2> /dev/null || true
  prepush_lock_release
}
trap prepush_hook_cleanup EXIT

git diff --name-only "${BASE_SHA}" HEAD > "$CHANGED_FILE"

# Feature-flag: default ON (impacted subset). Honor the CI var name and an
# explicit full-suite override. Neither knob is a silent bypass -- forcing OFF
# runs MORE tests (the whole suite), never fewer.
FLAG="on"
case "${ENABLE_SMART_TESTS:-}" in
  false | False | FALSE | 0 | off | OFF) FLAG="off" ;;
esac
if [ -n "${PREPUSH_FULL_SUITE:-}" ]; then
  FLAG="off"
fi

# DRY: invoke the EXACT module CI runs (scripts.ci.detect_test_paths) with the
# SAME flags .github/workflows/ci.yml passes -- crucially WITHOUT `--base-ref`,
# which infra's selector does not accept. Split on the optional adjacency
# override to avoid empty-array expansion under `set -u` on bash 3.2 (macOS
# system bash).
run_selector() {
  if [ -n "${PREPUSH_ADJACENCY:-}" ]; then
    uv run python -m scripts.ci.detect_test_paths \
      --changed-files-from "$CHANGED_FILE" \
      --ref-name "$BRANCH" \
      --event-name pull_request \
      --feature-flag "$FLAG" \
      --adjacency "$PREPUSH_ADJACENCY"
  else
    uv run python -m scripts.ci.detect_test_paths \
      --changed-files-from "$CHANGED_FILE" \
      --ref-name "$BRANCH" \
      --event-name pull_request \
      --feature-flag "$FLAG"
  fi
}

if ! run_selector > "$SELECTION_FILE" 2> "$SELECTION_ERR"; then
  log "selector stderr follows:"
  cat "$SELECTION_ERR" >&2 || true
  die "governed test selector failed to resolve a selection" \
      "verify scripts/ci/detect_test_paths.py + scripts/ci/test_selection_adjacency.yaml resolve under 'uv run' in this worktree"
fi

# Parse the selection with stdlib json -- fail loud on any parse error.
read_sel() {
  uv run python - "$SELECTION_FILE" "$1" << 'PY'
import json
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)
val = data[sys.argv[2]]
if isinstance(val, list):
    print("\n".join(val))
else:
    print(val)
PY
}

IS_FULL="$(read_sel is_full_suite)" \
  || die "could not parse selector output (is_full_suite)" \
         "the selector emitted non-JSON; inspect $SELECTION_FILE"
REASON="$(read_sel full_suite_reason 2> /dev/null || true)"

# OMN-15245 SEAM: the selector now emits changed tests/integration/ paths -- a
# changed test module is never dropped by narrowing (fail-closed invariant).
# Most of that tree needs a live service (Postgres, a broker, a running lane)
# that a developer's machine does not have, so those paths are deferred to CI,
# which runs them. Filter them out here, visibly, rather than handing pytest a
# selection it cannot execute. Keep this function self-contained (no globals):
# it is extracted and EXECUTED by
# tests/unit/scripts/test_prepush_smart_tests_seam.py.
#
# OMN-16825 NARROWING -- the classifier, not an override. "Lives under
# tests/integration/" was doing duty for "needs a live service", and those are
# different sets. tests/integration/chains/ runs the chain gates entirely on
# EventBusInmemory: no Postgres, no Kafka, no lane endpoint, nothing this hook
# cannot provide. Worse, that subtree is collected wholesale by the REQUIRED
# `Event Chain Gate` job (ci_summary_gate.py::STRICT_GATE_JOBS fail-closes on
# it), so the blanket path heuristic made the local selector structurally blind
# to a load-bearing merge gate: a chain regression was discoverable only after
# the push. Recognising it here is the fix; forcing a full suite with
# PREPUSH_FULL_SUITE / ENABLE_SMART_TESTS=off is NOT (CLAUDE.md Rule #4 forbids
# both, and neither would have run tests/integration/chains/ anyway -- the
# escalation target is tests/unit/).
#
# The allowlist is a POSITIVE, enumerated exception. Everything else under
# tests/integration/ keeps the old default and is deferred, so an unrecognised
# or brand-new subtree fails closed (deferred, never silently included). The
# entries are matched on a path SEGMENT boundary -- `chains_experimental/` and
# `chain/` do NOT match. The allowlist's premise (these suites declare no
# live-service marker) is itself asserted by the seam test, so a Postgres-backed
# test dropped into chains/ reddens CI instead of every developer's pre-push.
filter_prepush_runnable_paths() {
  local p prefix keep
  # Integration subtrees proven service-free and therefore locally runnable.
  local -a locally_runnable_integration_prefixes=(
    "tests/integration/chains/"
  )
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    case "$p" in
      tests/integration/*)
        keep=0
        for prefix in "${locally_runnable_integration_prefixes[@]}"; do
          # Append '/' to $p so a bare directory selection ("…/chains") matches
          # the prefix on a segment boundary exactly as "…/chains/" does.
          case "${p}/" in
            "$prefix"*) keep=1 ;;
          esac
        done
        [ "$keep" -eq 1 ] || continue
        ;;
    esac
    printf '%s\n' "$p"
  done
}

ALL_PATHS=()
while IFS= read -r p; do
  if [ -n "$p" ]; then
    ALL_PATHS+=("$p")
  fi
done < <(read_sel selected_paths)

PATHS=()
PATHS_STR=""
DEFERRED_STR=""
# OMN-16825: the subset of PATHS that lives under tests/integration/ -- i.e. the
# allowlisted, service-free integration suites. Tracked separately because the
# fail-closed FULL-suite escalation runs a fixed target (tests/unit/) that does
# NOT contain them; without appending these, escalating would run FEWER of the
# impacted tests than the narrow selection did. An escalation must never be a
# coverage downgrade.
RUNNABLE_INTEGRATION_PATHS=()
RUNNABLE_INTEGRATION_STR=""
# Guard the array expansions: bash 3.2 (macOS system bash) errors on
# "${arr[@]}" for an empty array under `set -u`.
if [ "${#ALL_PATHS[@]}" -gt 0 ]; then
  while IFS= read -r p; do
    if [ -n "$p" ]; then
      PATHS+=("$p")
      PATHS_STR="${PATHS_STR}${p} "
      case "$p" in
        tests/integration/*)
          RUNNABLE_INTEGRATION_PATHS+=("$p")
          RUNNABLE_INTEGRATION_STR="${RUNNABLE_INTEGRATION_STR}${p} "
          ;;
      esac
    fi
  done < <(printf '%s\n' "${ALL_PATHS[@]}" | filter_prepush_runnable_paths)
  for p in "${ALL_PATHS[@]}"; do
    case " $PATHS_STR " in
      *" $p "*) ;;
      *) DEFERRED_STR="${DEFERRED_STR}${p} " ;;
    esac
  done
fi

log "selection: is_full_suite=${IS_FULL} reason=${REASON:-none} paths=[ ${PATHS_STR}] (feature-flag=${FLAG})"
if [ -n "$DEFERRED_STR" ]; then
  log "deferred to CI (needs live services this hook cannot provide): [ ${DEFERRED_STR}]"
fi
if [ "${#RUNNABLE_INTEGRATION_PATHS[@]}" -gt 0 ]; then
  log "running locally (service-free integration suite, OMN-16825): [ ${RUNNABLE_INTEGRATION_STR}]"
fi

# Assemble the pytest target set. tests/integration is always ignored -- it needs
# real services and stays a CI-only concern. On a fail-closed escalation we run
# the full UNIT suite (tests/unit/), NOT all of tests/, so the pre-push hook stays
# unit-scoped and service-free (infra seam-match).
RC=0
# SINGLE SOURCE OF TRUTH for "what the heavy run is" (OMN-15408): the
# fail-closed escalation runs exactly this target, and `selection_is_whole_suite`
# measures the impacted-subset selection against this same value. Changing the
# escalation target automatically moves the guard predicate with it.
FULL_SUITE_TARGET="tests/unit/"

# OMN-15071: git EXPORTS repo-scoping variables into hook processes -- a live
# `git push` from a worktree hands this hook
# `GIT_DIR=<common>/worktrees/<name>` -- and those variables OVERRIDE both `-C`
# and the cwd for every descendant `git` call (memory
# `reference_git_env_vars_override_c_and_cwd`). Tests that build throwaway
# repositories under `tmp_path` and commit into them therefore operate on THIS
# worktree instead, and fail at setup. Unset them for the test run only: the
# hook has already resolved everything it needs from git, and pytest must
# rediscover the repository from its own cwd like any ordinary invocation.
#
# This is not a latent nicety. Until OMN-15071 chained the canonical-clone
# guard into the real hook chain, `core.hooksPath` on `.200` meant this hook
# never executed there at all, so the leak had no observable effect on the
# documented default gate host. Turning the hook on without this unset would
# hand every `.200` push a fail-closed pre-push that cannot pass.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_COMMON_DIR GIT_PREFIX

# =============================================================================
# Override-inheritance sanitization (OMN-16489, F-04)
# =============================================================================
# PREPUSH_* overrides (and ENABLE_SMART_TESTS) are honored at THIS hook's
# entry only. They must never inherit into the pytest subprocess tree: a test
# down there that re-invokes this script would receive the OUTER push's bypass
# grants -- the exact mechanism that turned one sanctioned override into a
# recursive 44k-test full-suite launcher (friction report F-01/F-04, ~9h03m).
# Called inside the subshell wrapping each pytest invocation, after the
# command's own knobs have been captured into non-PREPUSH names, so the parent
# hook's variables are untouched. Only EXPORTED names can inherit, so only
# those are scrubbed. ONEX_PREPUSH_HOOK_ACTIVE deliberately survives -- the
# recursion guard above depends on children inheriting it. This stops
# inheritance ONLY. OMN-16480 (landed) separately rejects PREPUSH_ALLOW_* at
# hook entry; this scrub covers the whole PREPUSH_* prefix class one layer
# deeper, at the child boundary.
scrub_prepush_override_env() {
  local v
  for v in $(compgen -A export PREPUSH_ || true); do
    unset "$v" || true
  done
  unset ENABLE_SMART_TESTS || true
}

if [ "$IS_FULL" = "True" ] || [ "$IS_FULL" = "true" ]; then
  guard_full_suite_host
  if [ "$REMOTE_FULL_SUITE_VERIFIED" -eq 1 ] || [ "$REMOTE_LAB_RUN_VERDICT" -eq 1 ]; then
    # OMN-16688: the escalation is SATISFIED, not skipped -- the full suite ran
    # to green on GitHub-hosted CI against this exact sha. Re-running it locally
    # would re-execute the identical tests on the identical tree, which is why
    # the local invocation is elided rather than merely deferred.
    if [ "$REMOTE_LAB_RUN_VERDICT" -eq 1 ]; then
      # OMN-16991: satisfied by a designated LAB host, not by GitHub-hosted CI.
      # The two are logged distinctly on purpose -- they are different evidence.
      log "FULL unit suite satisfied by the remote LAB-host run on '${PREPUSH_PICK_HOSTNAME}' (${PREPUSH_PICK_LABEL}); not re-running it locally."
    else
      log "FULL unit suite satisfied by the remote GitHub-hosted full-suite pass; not re-running it locally."
    fi
  else
    # OMN-16825: $FULL_SUITE_TARGET is tests/unit/, which does NOT contain the
    # allowlisted service-free integration suites. Append them so the
    # fail-closed escalation stays a strict SUPERSET of the narrow selection it
    # replaces -- an escalation that ran FEWER of the impacted tests than the
    # narrowing would be a coverage downgrade wearing the word "full". The
    # escalation still runs $FULL_SUITE_TARGET itself (single-sourced with
    # selection_is_whole_suite above); this only ADDS to it. bash 3.2 under
    # `set -u` errors on "${arr[@]}" for an empty array, hence the ${arr[@]+...}
    # guard rather than a bare expansion.
    log "running FULL unit suite (fail-closed escalation): uv run pytest ${FULL_SUITE_TARGET} ${RUNNABLE_INTEGRATION_STR}--ignore=tests/integration ${PREPUSH_PYTEST_ARGS:-}"
    (
      _pytest_extra_args="${PREPUSH_PYTEST_ARGS:-}"
      scrub_prepush_override_env
      # shellcheck disable=SC2086
      exec uv run pytest "${FULL_SUITE_TARGET}" ${RUNNABLE_INTEGRATION_PATHS[@]+"${RUNNABLE_INTEGRATION_PATHS[@]}"} --ignore=tests/integration --tb=short ${_pytest_extra_args}
    ) || RC=$?
  fi
elif [ "${#PATHS[@]}" -gt 0 ]; then
  # OMN-15408: guard on the SELECTED WORK, not the is_full_suite flag. A
  # selection that covers the whole full-suite target is the heavy run under
  # another name and must be routed to .200 exactly as the flagged escalation is.
  if selection_is_whole_suite "$FULL_SUITE_TARGET" "${PATHS[@]}"; then
    guard_full_suite_host "whole-suite-equivalent impacted selection (is_full_suite=${IS_FULL}, selected paths [ ${PATHS_STR}] cover the entire '${FULL_SUITE_TARGET}' escalation target)"
  fi
  if [ "$REMOTE_FULL_SUITE_VERIFIED" -eq 1 ] || [ "$REMOTE_LAB_RUN_VERDICT" -eq 1 ]; then
    # Only reachable when the selection was whole-suite-equivalent (the guard
    # above is the sole setter), so the remote FULL suite strictly covers this
    # selection -- it ran MORE tests than this invocation would have.
    if [ "$REMOTE_LAB_RUN_VERDICT" -eq 1 ]; then
      log "impacted selection is whole-suite-equivalent and was run on the designated lab host '${PREPUSH_PICK_HOSTNAME}' (${PREPUSH_PICK_LABEL}); not re-running it locally."
    else
      log "impacted selection is whole-suite-equivalent and is covered by the remote GitHub-hosted full-suite pass; not re-running it locally."
    fi
  else
    log "running impacted subset: uv run pytest ${PATHS_STR}--ignore=tests/integration ${PREPUSH_PYTEST_ARGS:-}"
    (
      _pytest_extra_args="${PREPUSH_PYTEST_ARGS:-}"
      scrub_prepush_override_env
      # shellcheck disable=SC2086
      exec uv run pytest "${PATHS[@]}" --ignore=tests/integration --tb=short ${_pytest_extra_args}
    ) || RC=$?
  fi
else
  log "no impacted unit tests mapped for this push (no source/test change contributed a target); nothing to run."
fi

if [ "$RC" -ne 0 ]; then
  log "ERROR: impacted tests failed (pytest exit ${RC})"
  log "REMEDIATION: fix the failing tests, then re-push. Reproduce with: uv run pytest ${PATHS_STR:-tests/unit/} --ignore=tests/integration"
  exit "$RC"
fi

log "impacted tests passed; allowing push."
exit "$RC"

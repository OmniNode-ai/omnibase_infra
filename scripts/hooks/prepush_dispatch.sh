#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# =============================================================================
# Lab-wide pre-push distribution (OMN-16991) -- sourced helper library
# =============================================================================
# Sourced by scripts/hooks/prepush_smart_tests.sh. Adds three things the hook
# has never had:
#
#   1. A host TABLE replacing the two-hostname literal that was the structural
#      reason .101/.105 could not be used (they were absent by a literal `||`,
#      not by policy).
#   2. SLOT-AWARE placement. Measured 2026-08-30T05:0xZ: .201 read load1
#      14.08/32 = 0.44x -- the FITTEST ratio in the lab -- while running three
#      concurrent prepush suites behind a 10-deep queue. load1 is a CPU-time
#      proxy; the scarce resource is an exclusive heavy-suite slot, so a host
#      with a held slot is UNFIT (rc 3), not merely low-ranked.
#   3. A real remote EXECUTION leg (bundle transplant + identical argv +
#      completion-marker readback), where before the hook only ever probed the
#      other host and interpolated the answer into a refusal string.
#
# NON-NEGOTIABLES, all preserved here:
#   * Nothing in this file can make the gate accept LESS work. Every path
#     either produces a real green suite run on a designated host, or returns
#     "no evidence" and lets the caller fall through to the pre-existing
#     precedence (GitHub-hosted verify -> grant -> die).
#   * A remote RED is a REFUSAL, never a fall-through to the override grant.
#   * Unreachable / unreadable / below-floor / busy all mean SKIP, never
#     "assumed fit" -- the same fail-closed posture as the load probe.
#   * bash 3.2 compatible (macOS system bash): no associative arrays, no
#     `${var,,}`, no `{fd}` redirection, guarded empty-array expansion.

# -----------------------------------------------------------------------------
# Table access -- COMMITTED tree only
# -----------------------------------------------------------------------------
PREPUSH_HOST_TABLE_REL="scripts/hooks/prepush_hosts.tsv"

# prepush_table_text -- prints the committed table, or returns 1 with a reason
# on stderr. Reading from HEAD (not the working tree) is what stops an
# uncommitted row from self-designating this machine as an authorizing gate
# host; the working-tree divergence check stops the inverse trick of editing
# the file after a commit that CI already saw.
prepush_table_text() {
  local head_copy work_copy
  if ! head_copy="$(git -C "$REPO_ROOT" show "HEAD:${PREPUSH_HOST_TABLE_REL}" 2> /dev/null)"; then
    printf 'host table absent at HEAD (%s)\n' "$PREPUSH_HOST_TABLE_REL" >&2
    return 1
  fi
  if [ -f "${REPO_ROOT}/${PREPUSH_HOST_TABLE_REL}" ]; then
    work_copy="$(cat "${REPO_ROOT}/${PREPUSH_HOST_TABLE_REL}")"
    if [ "$work_copy" != "$head_copy" ]; then
      printf 'host table differs between the working tree and HEAD\n' >&2
      return 1
    fi
  fi
  printf '%s\n' "$head_copy"
}

# prepush_table_rows -- data rows only (comments and blanks dropped).
prepush_table_rows() {
  prepush_table_text | sed -e 's/#.*$//' -e '/^[[:space:]]*$/d'
}

# prepush_field ROW N -- Nth tab-separated field of ROW.
prepush_field() {
  printf '%s' "$1" | cut -d'	' -f"$2"
}

# prepush_override_var LABEL -- the env var name that REPLACES this row's
# hostname. An override REPLACES the row it names; it never ADDS a name to the
# designated set. That distinction is load-bearing: under a table that lists
# several hosts, an override that merely appended a name could no longer
# DE-designate the local machine, silently inverting the OMN-15059 guard (and
# with it test_guard_refuses_full_suite_escalation_on_non_200_host, which
# proves the refusal by forcing a nonsense hostname).
prepush_override_var() {
  printf 'PREPUSH_HOST_OVERRIDE_%s' "$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9' '_')"
}

# prepush_row_hostname ROW -- the row's effective hostname, lowercased, after
# applying its override. Two legacy aliases are still honored so no existing
# invocation or test breaks: PREPUSH_200_HOSTNAME replaces row h200 and
# PREPUSH_201_GATE_RUNNER_HOSTNAME replaces row h201c (the CONTAINER row --
# that variable always named the container, never the .201 host itself).
prepush_row_hostname() {
  local row label name var val
  row="$1"
  label="$(prepush_field "$row" 1)"
  name="$(prepush_field "$row" 3)"
  case "$label" in
    h200) [ -n "${PREPUSH_200_HOSTNAME:-}" ] && name="$PREPUSH_200_HOSTNAME" ;;
    h201c) [ -n "${PREPUSH_201_GATE_RUNNER_HOSTNAME:-}" ] && name="$PREPUSH_201_GATE_RUNNER_HOSTNAME" ;;
  esac
  var="$(prepush_override_var "$label")"
  eval "val=\${$var:-}"
  [ -n "$val" ] && name="$val"
  printf '%s' "$name" | tr '[:upper:]' '[:lower:]'
}

# prepush_identity_label LC_HOST -- prints the label of the AUTHORIZING row
# this host is, or nothing. Only mode=authorizing rows confer identity: a
# `shadow` host is a placement target whose verdict may not satisfy the
# escalation, so it must not be treated as a designated gate host either --
# otherwise the identity guard would start passing on a host still in
# shadow, which is the exact inversion this table is meant to prevent.
prepush_identity_label() {
  local lc_host row
  lc_host="$1"
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    [ "$(prepush_field "$row" 11)" = "authorizing" ] || continue
    if [ "$(prepush_row_hostname "$row")" = "$lc_host" ]; then
      prepush_field "$row" 1
      return 0
    fi
  done <<EOF
$(prepush_table_rows)
EOF
  return 1
}

# prepush_designated_hostnames -- every authorizing hostname, for messages.
prepush_designated_hostnames() {
  local row out=""
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    [ "$(prepush_field "$row" 11)" = "authorizing" ] || continue
    out="${out}'$(prepush_row_hostname "$row")' "
  done <<EOF
$(prepush_table_rows)
EOF
  printf '%s' "${out% }"
}

# -----------------------------------------------------------------------------
# Slot state -- the dimension load1 is blind to
# -----------------------------------------------------------------------------
# A host is BUSY when a heavy pre-push is already executing there or is queued
# behind one. Returns 0 free / 2 unknown / 3 busy. `unknown` is NOT free: a
# host we cannot prove idle is skipped exactly like one we cannot reach.
#
# The probe counts live prepush_smart_tests.sh processes because that is the
# only signal that sees FOREIGN detached runs -- the ones .201's queue can
# neither observe nor preempt (OMN-16968). A lock that only counts its own
# holders reproduces that defect one host wider.
_PREPUSH_SLOT_PROBE_SH='q=0
if [ -r "$HOME/push-lanes/QUEUE" ]; then q=$(grep -c . "$HOME/push-lanes/QUEUE" 2>/dev/null || echo 0); fi
p=$(ps ax 2>/dev/null | grep prepush_smart_tests.sh | grep -v grep | grep -c . || true)
[ -n "$p" ] || p=0
l=0
if [ -n "$PREPUSH_WORKROOT" ] && [ -d "$PREPUSH_WORKROOT/LOCK" ]; then l=1; fi
printf "%s %s %s\n" "$q" "$p" "$l"'

# prepush_slot_state TARGET WORKROOT SELF_PIDS -- SELF_PIDS is how many
# prepush_smart_tests.sh processes are expected to be OUR OWN on that host
# (1 when probing the local host -- this very hook -- else 0).
prepush_slot_state() {
  local target workroot self raw q p l tcmd
  target="$1"; workroot="$2"; self="$3"
  if [ -n "${PREPUSH_SLOT_OVERRIDE:-}" ]; then
    raw="$PREPUSH_SLOT_OVERRIDE"
  elif [ -z "$target" ]; then
    raw="$(PREPUSH_WORKROOT="$workroot" sh -c "$_PREPUSH_SLOT_PROBE_SH" 2> /dev/null)" || return 2
  else
    tcmd="$(_prepush_timeout_cmd)"
    if [ -n "$tcmd" ]; then
      raw="$("$tcmd" 12 ssh -o ConnectTimeout=4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "$target" "PREPUSH_WORKROOT='${workroot}'; $_PREPUSH_SLOT_PROBE_SH" 2> /dev/null)" || return 2
    else
      raw="$(ssh -o ConnectTimeout=4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "$target" "PREPUSH_WORKROOT='${workroot}'; $_PREPUSH_SLOT_PROBE_SH" 2> /dev/null)" || return 2
    fi
  fi
  [ -n "$raw" ] || return 2
  # shellcheck disable=SC2086
  set -- $raw
  q="${1:-}"; p="${2:-}"; l="${3:-}"
  [ -n "$q" ] && [ -n "$p" ] && [ -n "$l" ] || return 2
  PREPUSH_SLOT_DETAIL="queue=${q} heavy_pids=${p} lock=${l}"
  [ "$l" -eq 0 ] || return 3
  [ "$q" -eq 0 ] || return 3
  [ "$p" -le "$self" ] || return 3
  return 0
}

# -----------------------------------------------------------------------------
# uv floor -- presence is not enough
# -----------------------------------------------------------------------------
# Verified by VERSION, not by path existence: the live fleet spread is 0.8.3
# (.101, 13 months old) to 0.11.32 (.200) against a lockfile at revision 3.
# Below the floor, or unreadable, means SKIP.
prepush_uv_version_ok() {
  local target uv floor out tcmd
  target="$1"; uv="$2"; floor="$3"
  [ -n "$uv" ] && [ "$uv" != "-" ] || return 2
  if [ -n "${PREPUSH_UV_VERSION_OVERRIDE:-}" ]; then
    out="$PREPUSH_UV_VERSION_OVERRIDE"
  elif [ -z "$target" ]; then
    out="$("$uv" --version 2> /dev/null)" || return 2
  else
    tcmd="$(_prepush_timeout_cmd)"
    if [ -n "$tcmd" ]; then
      out="$("$tcmd" 12 ssh -o ConnectTimeout=4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "$target" "'${uv}' --version" 2> /dev/null)" || return 2
    else
      out="$(ssh -o ConnectTimeout=4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "$target" "'${uv}' --version" 2> /dev/null)" || return 2
    fi
  fi
  out="$(printf '%s' "$out" | sed -n 's/^uv \([0-9][0-9.]*\).*/\1/p')"
  [ -n "$out" ] || return 2
  PREPUSH_UV_VERSION_SEEN="$out"
  awk -v have="$out" -v want="$floor" 'BEGIN {
    nh = split(have, h, "."); nw = split(want, w, ".");
    n = (nh > nw ? nh : nw);
    for (i = 1; i <= n; i++) {
      a = (i <= nh ? h[i] + 0 : 0); b = (i <= nw ? w[i] + 0 : 0);
      if (a > b) exit 0;
      if (a < b) exit 1;
    }
    exit 0
  }'
}

# -----------------------------------------------------------------------------
# Deterministic, network-free per-host overrides (tests only)
# -----------------------------------------------------------------------------
# The pre-existing PREPUSH_LOAD_OVERRIDE_LOCAL/_REMOTE pair collapses EVERY ssh
# target to one value, which cannot express "host A is fit, host B is busy" --
# the only interesting input to a multi-host picker. These maps are keyed by
# row LABEL so a test can drive the real picker with no network at all.
#
# Same risk profile as the two overrides already shipped: a forged value can
# only change WHERE work is routed, never whether it passed. The verdict still
# comes from a real pytest exit code bound to the tree by a completion marker,
# so no map value can turn a red suite green.
#
# prepush_map_lookup MAP LABEL -- value for LABEL in a "a=1,b=2" map, or empty.
prepush_map_lookup() {
  printf '%s' "$1" | tr ',' '\n' | sed -n "s/^${2}=//p" | head -1
}

# prepush_probe_ratio LABEL TARGET -- prints the load ratio or returns 1.
prepush_probe_ratio() {
  local v
  if [ -n "${PREPUSH_LOAD_OVERRIDE_MAP:-}" ]; then
    v="$(prepush_map_lookup "$PREPUSH_LOAD_OVERRIDE_MAP" "$1")"
    [ -n "$v" ] || return 1
    printf '%s' "$v"
    return 0
  fi
  host_load_ratio "$2" | awk '{print $3}'
}

# prepush_probe_slot LABEL TARGET WORKROOT SELF -- 0 free / 2 unknown / 3 busy.
prepush_probe_slot() {
  local v
  if [ -n "${PREPUSH_SLOT_OVERRIDE_MAP:-}" ]; then
    v="$(prepush_map_lookup "$PREPUSH_SLOT_OVERRIDE_MAP" "$1")"
    case "$v" in
      free) PREPUSH_SLOT_DETAIL="override=free"; return 0 ;;
      busy) PREPUSH_SLOT_DETAIL="override=busy"; return 3 ;;
      *) PREPUSH_SLOT_DETAIL="override=unknown"; return 2 ;;
    esac
  fi
  prepush_slot_state "$2" "$3" "$4"
}

# prepush_probe_uv LABEL TARGET UV FLOOR -- 0 ok / 1 below floor / 2 unreadable.
prepush_probe_uv() {
  local v
  if [ -n "${PREPUSH_UV_OVERRIDE_MAP:-}" ]; then
    v="$(prepush_map_lookup "$PREPUSH_UV_OVERRIDE_MAP" "$1")"
    [ -n "$v" ] || return 2
    PREPUSH_UV_VERSION_SEEN="$v"
    PREPUSH_UV_VERSION_OVERRIDE="uv $v" prepush_uv_version_ok "" "$3" "$4"
    return $?
  fi
  prepush_uv_version_ok "$2" "$3" "$4"
}

# -----------------------------------------------------------------------------
# Placement
# -----------------------------------------------------------------------------
# pick_capacity_host LC_HOST REPO -- chooses the least-loaded host that has
# PROVEN a free slot, or returns 1. Sets, on success:
#   PREPUSH_PICK_LABEL / _HOSTNAME / _SSH / _UV / _WORKROOT / _SLOTMODE
#   PREPUSH_PICK_RATIO / _MODE
# and always sets PREPUSH_PROBE_LOG (a "label=verdict" trail for the receipt
# and the refusal message -- every probed host is on the record, so a refusal
# can be audited rather than believed).
#
# Order of elimination is deliberate: cheap local facts first (disabled, repo
# denial), then the slot (the scarce resource), then load, then the toolchain.
# load1 ranks only among hosts already proven to hold a free slot -- it is a
# tiebreaker, not the placement key.
pick_capacity_host() {
  local lc_host repo row label role name ssh_t uv floor workroot slotmode denied mode
  local self ratio rc best_ratio=""
  lc_host="$1"; repo="$2"
  PREPUSH_PROBE_LOG=""
  PREPUSH_PICK_LABEL=""
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    label="$(prepush_field "$row" 1)"
    role="$(prepush_field "$row" 2)"
    mode="$(prepush_field "$row" 11)"
    [ "$role" = "capacity" ] || continue
    if [ "$mode" = "disabled" ]; then
      PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG}${label}=disabled "
      continue
    fi
    denied="$(prepush_field "$row" 10)"
    case ",${denied}," in
      *",${repo},"*)
        PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG}${label}=repo-denied "
        continue
        ;;
    esac
    name="$(prepush_row_hostname "$row")"
    ssh_t="$(prepush_field "$row" 4)"
    uv="$(prepush_field "$row" 6)"
    floor="$(prepush_field "$row" 7)"
    workroot="$(prepush_field "$row" 8)"
    slotmode="$(prepush_field "$row" 9)"
    self=0
    if [ "$name" = "$lc_host" ]; then
      # This host: probe it directly, and expect to see OUR OWN hook process.
      ssh_t=""
      self=1
    fi

    rc=0
    prepush_probe_slot "$label" "$ssh_t" "$workroot" "$self" || rc=$?
    case "$rc" in
      3)
        PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG}${label}=busy(${PREPUSH_SLOT_DETAIL:-}) "
        continue
        ;;
      2)
        PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG}${label}=slot-unknown "
        continue
        ;;
    esac

    ratio="$(prepush_probe_ratio "$label" "$ssh_t")" || ratio=""
    if [ -z "$ratio" ]; then
      PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG}${label}=unreachable "
      continue
    fi
    if ! awk -v r="$ratio" -v thr="$PREPUSH_LOAD_THRESHOLD" 'BEGIN { exit !(r <= thr + 0) }'; then
      PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG}${label}=over(${ratio}) "
      continue
    fi

    rc=0
    prepush_probe_uv "$label" "$ssh_t" "$uv" "$floor" || rc=$?
    if [ "$rc" -ne 0 ]; then
      PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG}${label}=uv-unfit(${PREPUSH_UV_VERSION_SEEN:-unreadable}<${floor}) "
      continue
    fi

    PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG}${label}=fit(${ratio},${mode}) "
    if [ -z "$best_ratio" ] || awk -v a="$ratio" -v b="$best_ratio" 'BEGIN { exit !(a < b) }'; then
      best_ratio="$ratio"
      PREPUSH_PICK_LABEL="$label"
      PREPUSH_PICK_HOSTNAME="$name"
      PREPUSH_PICK_SSH="$ssh_t"
      PREPUSH_PICK_UV="$uv"
      PREPUSH_PICK_WORKROOT="$workroot"
      PREPUSH_PICK_SLOTMODE="$slotmode"
      PREPUSH_PICK_RATIO="$ratio"
      PREPUSH_PICK_MODE="$mode"
    fi
  done <<EOF
$(prepush_table_rows)
EOF
  PREPUSH_PROBE_LOG="${PREPUSH_PROBE_LOG% }"
  [ -n "$PREPUSH_PICK_LABEL" ]
}

# prepush_local_workroot LC_HOST -- the workroot of the capacity row that IS
# this host, or empty. The heavy-suite slot is a property of the HOST, not of a
# repo: two different repos pushing from the same machine must contend for the
# same lock, so the lock lives under the host's workroot rather than inside any
# one checkout.
prepush_local_workroot() {
  local lc_host row
  lc_host="$1"
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    [ "$(prepush_field "$row" 2)" = "capacity" ] || continue
    if [ "$(prepush_row_hostname "$row")" = "$lc_host" ]; then
      prepush_field "$row" 8
      return 0
    fi
  done <<EOF
$(prepush_table_rows)
EOF
  return 1
}

# -----------------------------------------------------------------------------
# Exclusive slot
# -----------------------------------------------------------------------------
# mkdir(2) is the lock primitive on every host, deliberately, rather than
# flock(1): flock is ABSENT on both Macs (probed live -- .101 and .105 have no
# flock and no gtimeout), and its fd-holding idiom needs `exec {fd}<>` which
# macOS system bash 3.2 cannot parse. mkdir is atomic on every POSIX
# filesystem and works in bash 3.2, so the fleet gets ONE lock implementation
# instead of a Linux path and a Mac path that can drift.
#
# What mkdir lacks versus flock is automatic release when the holder dies, so
# the holder's pid is recorded and a lock whose holder is gone is reclaimed --
# without that, one killed run (OMN-16713: the selector gets SIGTERMed from
# outside) would wedge a host permanently.
PREPUSH_HELD_LOCK=""

prepush_lock_acquire() {
  local workroot lockdir holder
  workroot="$1"
  lockdir="${workroot}/LOCK"
  mkdir -p "$workroot" 2> /dev/null || return 1
  if mkdir "$lockdir" 2> /dev/null; then
    printf '%s %s %s\n' "$$" "$(hostname -s 2> /dev/null || echo unknown)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      > "${lockdir}/holder" 2> /dev/null || true
    PREPUSH_HELD_LOCK="$lockdir"
    return 0
  fi
  # Occupied. Reclaim only if the recorded holder is provably gone AND it was
  # this same machine (a pid from another host says nothing about ours).
  holder="$(cut -d' ' -f1 "${lockdir}/holder" 2> /dev/null || true)"
  if [ -n "$holder" ] && [ "$(cut -d' ' -f2 "${lockdir}/holder" 2> /dev/null || true)" = "$(hostname -s 2> /dev/null || echo unknown)" ] \
    && ! kill -0 "$holder" 2> /dev/null; then
    rm -rf "$lockdir" 2> /dev/null || true
    if mkdir "$lockdir" 2> /dev/null; then
      printf '%s %s %s\n' "$$" "$(hostname -s 2> /dev/null || echo unknown)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        > "${lockdir}/holder" 2> /dev/null || true
      PREPUSH_HELD_LOCK="$lockdir"
      return 0
    fi
  fi
  return 1
}

prepush_lock_release() {
  [ -n "$PREPUSH_HELD_LOCK" ] || return 0
  rm -rf "$PREPUSH_HELD_LOCK" 2> /dev/null || true
  PREPUSH_HELD_LOCK=""
}

# -----------------------------------------------------------------------------
# Receipts
# -----------------------------------------------------------------------------
prepush_emit_receipt() {
  local dir
  dir="${REPO_ROOT}/.onex_state/prepush_distribution"
  mkdir -p "$dir" 2> /dev/null || return 0
  printf '%s\n' "$1" >> "${dir}/receipts.jsonl" 2> /dev/null || true
}

prepush_json_escape() {
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr -d '\n'
}

# -----------------------------------------------------------------------------
# Remote execution leg
# -----------------------------------------------------------------------------
# git bundle transplant -> scp -> clone -> uv sync -> the IDENTICAL pytest argv
# -> completion marker read back. This is the leg the hook has never had; until
# now the "other host" was probed and the answer interpolated into a refusal
# string (the old L427-433), so `.201` was reachable only by a human reading
# the die() text and hand-driving a recipe.
#
# Bundle transplant is not new machinery here: ~/push-lanes on .201 is already
# full of *.bundle files from exactly this recipe. What is new is that the HOOK
# drives it instead of a person.
#
# WHY A COMPLETION MARKER AND NOT THE SSH EXIT CODE: ssh returns 255 for a
# transport failure, which is indistinguishable from a test failure, and any
# backgrounding/nohup/tee wrapper returns 0 with nothing having run -- a
# fail-OPEN shape. The verdict is therefore a marker file written on the remote
# host carrying {head_sha, argv_sha, exit, collected, log_sha256}; absence or
# mismatch is NO EVIDENCE and falls through to refusal, never to a pass.
_prepush_sha256_sh='if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d" " -f1; else shasum -a 256 "$1" | cut -d" " -f1; fi'

prepush_sha256_file() {
  sh -c "$_prepush_sha256_sh" _ "$1" 2> /dev/null
}

# prepush_remote_argv -- the EXACT pytest argv this call site would have run
# locally, one item per line. The two local call sites carry DIFFERENT argv and
# conflating them would be a silent coverage downgrade: the heavy site runs
# $FULL_SUITE_TARGET **plus** ${RUNNABLE_INTEGRATION_PATHS[@]} to satisfy
# OMN-16825's "an escalation must never run FEWER of the impacted tests than
# the narrowing it replaces" invariant, while the whole-suite-equivalent narrow
# site runs ${PATHS[@]}. Shipping only tests/unit/ would silently drop
# tests/integration/chains/, a required Event Chain Gate surface, with no test
# firing.
prepush_remote_argv() {
  if [ "${IS_FULL:-}" = "True" ] || [ "${IS_FULL:-}" = "true" ]; then
    printf '%s\n' "$FULL_SUITE_TARGET"
    if [ "${#RUNNABLE_INTEGRATION_PATHS[@]}" -gt 0 ]; then
      printf '%s\n' "${RUNNABLE_INTEGRATION_PATHS[@]}"
    fi
  else
    if [ "${#PATHS[@]}" -gt 0 ]; then
      printf '%s\n' "${PATHS[@]}"
    fi
  fi
}

# prepush_remote_run -- executes the suite on the picked host.
# Returns 0 = GREEN (verdict may be used), 1 = NO EVIDENCE (fall through),
# 3 = RED (the suite genuinely failed on a designated host; the caller MUST
# refuse the push rather than fall through to an override grant -- a remote red
# falling through to a grant would be a bypass wearing the word "fallback").
prepush_remote_run() {
  local heavy_what repo head_sha runid workroot ssh_t uv label rundir
  local bundle argvfile runner localdir marker rc=0 argv_sha log_sha
  local m_exit m_head m_argv m_log m_collected started ended dur
  heavy_what="$1"
  repo="$(basename "$REPO_ROOT")"
  head_sha="$(git -C "$REPO_ROOT" rev-parse HEAD 2> /dev/null || true)"
  [ -n "$head_sha" ] || return 1
  label="$PREPUSH_PICK_LABEL"
  ssh_t="$PREPUSH_PICK_SSH"
  uv="$PREPUSH_PICK_UV"
  workroot="$PREPUSH_PICK_WORKROOT"
  [ -n "$ssh_t" ] || return 1
  runid="${repo}-$(printf '%s' "$head_sha" | cut -c1-12)-$$"
  rundir="${workroot}/runs/${runid}"

  localdir="$(mktemp -d 2> /dev/null)" || return 1
  bundle="${localdir}/tree.bundle"
  argvfile="${localdir}/argv.txt"
  runner="${localdir}/prepush_smart_tests.sh"

  if ! git -C "$REPO_ROOT" bundle create "$bundle" HEAD > /dev/null 2>&1; then
    log "remote leg: could not create a git bundle for ${head_sha}"
    rm -rf "$localdir"
    return 1
  fi
  prepush_remote_argv > "$argvfile"
  if [ ! -s "$argvfile" ]; then
    rm -rf "$localdir"
    return 1
  fi
  argv_sha="$(prepush_sha256_file "$argvfile")"

  # The remote wrapper is NAMED prepush_smart_tests.sh on purpose. .201's queue
  # runner gates every lane on `ps ax | grep prepush_smart_tests.sh` ("no other
  # heavy prepush running host-wide, covers foreign runs not launched through
  # this queue"). Matching that name makes THIS run visible to the queue's own
  # existing enforcement surface, so the queue and this leg share one mutex
  # instead of the leg becoming another foreign detached run -- the exact
  # defect class OMN-16968 is open against. It also makes the run visible to
  # prepush_slot_state above, so a second dispatcher sees the host as busy.
  cat > "$runner" <<'REMOTE'
#!/usr/bin/env bash
set -uo pipefail
RUNDIR="$1"; UV="$2"; HEAD_SHA="$3"; ARGV_SHA="$4"; ORIGIN="$5"
cd "$RUNDIR" || exit 90
# Re-arm BOTH guards explicitly. ssh forwards neither, so without this the
# remote repo's own suite -- which subprocesses this very hook from
# tests/ci/test_prepush_hook_host_identity_guard.py and siblings -- would take
# FIRST-entry behavior on the remote host, resolve the selector, pick a host
# and ship another bundle: an unbounded DISTRIBUTED variant of the
# OMN-16425/OMN-16489 F-01 recursion (~9h03m, 44,064 tests) the sentinel exists
# to stop.
for v in $(env | sed -n 's/^\(PREPUSH_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v" || true; done
unset ENABLE_SMART_TESTS || true
export ONEX_PREPUSH_HOOK_ACTIVE="remote-leg:${ORIGIN}"
ARGV=()
while IFS= read -r line; do [ -n "$line" ] && ARGV+=("$line"); done < "$RUNDIR/argv.txt"
[ "${#ARGV[@]}" -gt 0 ] || exit 91
cd "$RUNDIR/tree" || exit 92
"$UV" sync --all-extras > "$RUNDIR/sync.log" 2>&1 || { echo "UV_SYNC_FAILED" >&2; exit 93; }
"$UV" run pytest "${ARGV[@]}" --ignore=tests/integration --tb=short > "$RUNDIR/suite.log" 2>&1
rc=$?
if command -v sha256sum > /dev/null 2>&1; then
  LOGSHA=$(sha256sum "$RUNDIR/suite.log" | cut -d" " -f1)
else
  LOGSHA=$(shasum -a 256 "$RUNDIR/suite.log" | cut -d" " -f1)
fi
COLLECTED=$(sed -n 's/^collected \([0-9][0-9]*\) item.*/\1/p' "$RUNDIR/suite.log" | tail -1)
[ -n "$COLLECTED" ] || COLLECTED=0
{
  echo "head_sha=$HEAD_SHA"
  echo "argv_sha=$ARGV_SHA"
  echo "exit=$rc"
  echo "collected=$COLLECTED"
  echo "log_sha256=$LOGSHA"
  echo "host=$(hostname)"
} > "$RUNDIR/MARKER"
exit "$rc"
REMOTE

  log "remote leg: dispatching ${heavy_what} to ${label} (${PREPUSH_PICK_HOSTNAME}, ratio ${PREPUSH_PICK_RATIO}, mode ${PREPUSH_PICK_MODE})"
  log "remote leg: probed -> ${PREPUSH_PROBE_LOG}"
  started="$(date -u '+%s')"

  if ! ssh -o ConnectTimeout=6 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ssh_t" \
    "mkdir -p '${rundir}'" > /dev/null 2>&1; then
    log "remote leg: could not create ${rundir} on ${label}"
    rm -rf "$localdir"
    return 1
  fi
  if ! scp -q -o ConnectTimeout=6 -o BatchMode=yes "$bundle" "$argvfile" "$runner" "${ssh_t}:${rundir}/" > /dev/null 2>&1; then
    log "remote leg: transfer to ${label} failed"
    rm -rf "$localdir"
    return 1
  fi

  # Stream the remote suite back as it runs, prefixed, so a distributed run is
  # no less observable than a local one.
  ssh -o ConnectTimeout=6 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ssh_t" \
    "set -e; cd '${rundir}'; rm -rf tree; git clone -q tree.bundle tree > /dev/null 2>&1; cd tree; git checkout -q '${head_sha}' 2>/dev/null || true; cd '${rundir}'; chmod +x prepush_smart_tests.sh; ./prepush_smart_tests.sh '${rundir}' '${uv}' '${head_sha}' '${argv_sha}' '$(hostname -s 2> /dev/null || echo unknown):$$'; echo REMOTE_WRAPPER_EXIT=\$?" 2>&1 |
    sed "s/^/[${label}] /" >&2 || true

  marker="$(ssh -o ConnectTimeout=6 -o BatchMode=yes "$ssh_t" "cat '${rundir}/MARKER' 2>/dev/null" 2> /dev/null || true)"
  ended="$(date -u '+%s')"
  dur=$((ended - started))
  rm -rf "$localdir"

  if [ -z "$marker" ]; then
    log "remote leg: NO completion marker from ${label} -- treating as NO EVIDENCE (not a pass, not a failure)"
    return 1
  fi
  m_head="$(printf '%s\n' "$marker" | sed -n 's/^head_sha=//p')"
  m_argv="$(printf '%s\n' "$marker" | sed -n 's/^argv_sha=//p')"
  m_exit="$(printf '%s\n' "$marker" | sed -n 's/^exit=//p')"
  m_collected="$(printf '%s\n' "$marker" | sed -n 's/^collected=//p')"
  m_log="$(printf '%s\n' "$marker" | sed -n 's/^log_sha256=//p')"
  if [ "$m_head" != "$head_sha" ] || [ "$m_argv" != "$argv_sha" ] || [ -z "$m_exit" ] || [ -z "$m_log" ]; then
    log "remote leg: marker from ${label} does not bind to this tree/argv -- NO EVIDENCE"
    return 1
  fi
  log_sha="$m_log"

  prepush_emit_receipt "{\"ts\":\"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\",\"repo\":\"$(prepush_json_escape "$repo")\",\"head_sha\":\"${head_sha}\",\"chosen_host\":\"$(prepush_json_escape "$PREPUSH_PICK_HOSTNAME")\",\"chosen_label\":\"${label}\",\"host_mode\":\"${PREPUSH_PICK_MODE}\",\"host_load_ratio\":\"${PREPUSH_PICK_RATIO}\",\"all_probed_ratios\":\"$(prepush_json_escape "$PREPUSH_PROBE_LOG")\",\"selection_paths\":\"$(prepush_json_escape "$(prepush_remote_argv | tr '\n' ' ')")\",\"pytest_exit\":${m_exit},\"collected\":${m_collected:-0},\"duration_s\":${dur},\"suite_log_sha256\":\"${log_sha}\"}"

  if [ "$PREPUSH_PICK_MODE" = "shadow" ]; then
    log "remote leg: ${label} is in SHADOW -- ran ${m_collected} tests, exit ${m_exit}, but a shadow host NEVER authorizes. Receipt written; falling through to the normal precedence."
    return 1
  fi
  if [ "$m_exit" -eq 0 ]; then
    log "REMOTE LAB RUN PASS accepted in place of ${heavy_what}: ${label} ran ${m_collected} tests green on ${head_sha} (suite log sha256 ${log_sha}, ${dur}s)"
    return 0
  fi
  log "remote leg: ${label} ran ${m_collected} tests and FAILED (pytest exit ${m_exit}) on ${head_sha}"
  return 3
}

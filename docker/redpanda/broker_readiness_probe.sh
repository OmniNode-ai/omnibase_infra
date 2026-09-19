#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Dev-lane broker readiness probe (OMN-18789).
#
# WHAT THIS REPLACES, AND WHY
# ---------------------------
# The dev lane's broker healthcheck was:
#
#     rpk cluster health | grep -q 'Healthy:.*true' || exit 1
#
# That is an ADMIN-API call. It never touches the Kafka data path. On
# 2026-09-18 the broker left partition reconciliation stuck for seventeen
# minutes and logged 5921 `error obtaining latest start offset -
# { error_code: not_leader_for_partition [6] }` lines in a five-minute slice,
# while that command stayed green, `docker ps` reported `healthy` with
# RestartCount 0, and `/v1/cluster/health_overview` returned `is_healthy true`
# with `leaderless_count 0` throughout. Ninety-seven minutes of total
# data-plane outage, invisible on every surface anyone watches.
#
# `leaderless_count: 0` was not even a lie. On a single-node cluster every
# partition nominally has node 0 as its leader. It answers a question no client
# asks. The question a client asks is "does a read return?", so that is the
# question this probe asks.
#
# THE TWO CHECKS
# --------------
# (1) A PARTITION READ. `rpk topic describe -p` against the declared probe
#     topic must return numeric offsets for every partition. This is the same
#     ListOffsets path that failed during the outage.
#
# (2) CONSUMER-GROUP SYNC. A group that is `Stable`, holds at least one member,
#     is BEHIND, has committed at least once, and has not moved its committed
#     offset for longer than the declared window is NOT READY. That is the
#     exact shape the outage left behind: `STATE Stable MEMBERS 1` at
#     CURRENT-OFFSET 7473 against LOG-END-OFFSET 7478, byte-identical across
#     two samples twenty-two seconds apart, for an hour.
#
# WHY "HAS COMMITTED AT LEAST ONCE" IS PART OF THE CONDITION AND NOT AN
# OVERSIGHT. Measured live on the dev lane 2026-09-19: the
# `local.omnimarket-projection-api.snapshot-cache...` group tails its topic
# without ever committing, so its lag grows without bound by design and every
# CURRENT-OFFSET reads `-`. It has no committed offset that could freeze. A
# probe that counted it would report a healthy lane broken permanently, which
# is a worse failure than the one being fixed. RESIDUAL, STATED: a group that
# wedges BEFORE its first commit is therefore invisible here. That class is the
# zero-member / never-joined shape the runtime health monitor's
# `empty_consumer_groups` dimension already carries, and OMN-18640 owns the
# consumer-side wedge.
#
# WHY A BASH SCRIPT. It runs inside `redpandadata/redpanda:v24.2.7` as that
# container's healthcheck. Measured in-container 2026-09-19: bash, sh, awk,
# sed, grep, curl, tr, cut and rpk are present; python3 is NOT. Nothing here
# imports anything.
#
# BLAST RADIUS. The `redpanda` service carries no `autoheal=true` label, so a
# NOT READY verdict makes `docker ps` say `unhealthy` and restarts nothing.
# This probe reports; it does not act.
#
# STATE. One line per group in the state file, which lives in the container's
# `/tmp` and is therefore discarded on recreate -- a rebuilt broker starts its
# observation window fresh, which is correct.
#
# EXIT: 0 READY, 1 NOT READY. The reason token is printed on both streams so it
# survives `docker inspect`'s healthcheck log either way.

set -uo pipefail

DECLARATION_PATH="/etc/onex/broker_readiness_declaration.conf"
STATE_PATH="/tmp/onex-broker-readiness.state"
NOW=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --declaration) DECLARATION_PATH="$2"; shift 2 ;;
    --state)       STATE_PATH="$2";       shift 2 ;;
    --now)         NOW="$2";              shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

not_ready() {
  echo "NOT_READY [$1] $2"
  echo "NOT_READY [$1] $2" >&2
  exit 1
}

ready() {
  echo "READY [$1] $2"
  exit 0
}

# --- the declaration is the only source of every bound (Operating Rule 8) ----
[ -r "$DECLARATION_PATH" ] || not_ready declaration_unreadable \
  "no readable declaration at ${DECLARATION_PATH}"

sync_window_seconds=""
probe_topic=""
broker=""
sasl_mechanism=""
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  key="${line%%=*}"
  value="${line#*=}"
  case "$key" in
    sync_window_seconds) sync_window_seconds="$value" ;;
    probe_topic)         probe_topic="$value" ;;
    broker)              broker="$value" ;;
    sasl_mechanism)      sasl_mechanism="$value" ;;
  esac
done < "$DECLARATION_PATH"

for required in sync_window_seconds probe_topic broker sasl_mechanism; do
  eval "declared=\${$required}"
  [ -n "$declared" ] || not_ready declaration_missing_key \
    "${required} is not declared in ${DECLARATION_PATH}"
done
case "$sync_window_seconds" in
  ''|*[!0-9]*) not_ready declaration_missing_key \
    "sync_window_seconds must be a whole number of seconds" ;;
esac

[ -n "$NOW" ] || NOW="$(date +%s)"

# --- resolve how this broker will talk to us, authenticated or not -----------
# The dev lane runs SASL after the OMN-18012 flip and plaintext before it (and
# on a fresh volume, before the principal exists). Try the authenticated read
# first, fall back to the unauthenticated one, and be NOT READY only when
# neither answers. Credentials travel in the environment, never in argv.
#
# `-i` on the topic listing is load-bearing, not tidiness: `rpk topic list`
# omits INTERNAL topics, so `__consumer_offsets` -- the declared probe
# topic, and the one whose partitions the 2026-09-18 outage actually left
# stuck -- is invisible without it. Measured on the dev lane 2026-09-19:
# the bare listing matched it 0 times and `-i` matched it once, and the
# first draft of this probe silently read `_schemas` instead.
export RPK_BROKERS="$broker"
topics=""
mode=""
if [ -n "${DEV_KAFKA_SASL_USERNAME:-}" ] && [ -n "${DEV_KAFKA_SASL_PASSWORD:-}" ]; then
  export RPK_USER="$DEV_KAFKA_SASL_USERNAME"
  export RPK_PASS="$DEV_KAFKA_SASL_PASSWORD"
  export RPK_SASL_MECHANISM="$sasl_mechanism"
  if topics="$(rpk topic list -i 2>/dev/null)"; then
    mode="sasl"
  fi
fi
if [ -z "$mode" ]; then
  unset RPK_USER RPK_PASS RPK_SASL_MECHANISM
  if topics="$(rpk topic list -i 2>/dev/null)"; then
    mode="plaintext"
  fi
fi
[ -n "$mode" ] || not_ready broker_unreachable \
  "no metadata read succeeded against ${broker}, authenticated or not"

# --- check 1: a partition read must actually return numbers ------------------
topic_names="$(printf '%s\n' "$topics" | awk 'NR>1 && NF>0 {print $1}')"
if [ -z "$topic_names" ]; then
  # A cluster that has formed but carries no topics has no partition to read,
  # so nothing here is proven broken. `redpanda-scram-user` gates on
  # `service_healthy`, so failing closed here would make a fresh volume
  # undeployable by the only sanctioned path to the lane.
  ready no_topics "broker answered over ${mode}; cluster carries no topics yet"
fi

target=""
for name in $topic_names; do
  if [ "$name" = "$probe_topic" ]; then target="$name"; break; fi
done
[ -n "$target" ] || target="$(printf '%s\n' "$topic_names" | head -n 1)"

describe="$(rpk topic describe -p "$target" 2>&1)"
describe_status=$?
if [ "$describe_status" -ne 0 ]; then
  not_ready partition_read_failed \
    "offset read on ${target} exited ${describe_status}: $(printf '%s' "$describe" | head -n 1)"
fi

# The verdict is the NUMBERS, not the exit code: rpk reports a per-partition
# error in an extra column and can still exit 0. Read the HIGH-WATERMARK column
# by its header position and require every data row to carry an integer there.
offsets_verdict="$(printf '%s\n' "$describe" | awk '
  /HIGH-WATERMARK/ && col == 0 {
    for (i = 1; i <= NF; i++) if ($i == "HIGH-WATERMARK") col = i
    next
  }
  col > 0 && NF > 0 {
    rows++
    if ($col !~ /^[0-9]+$/) bad++
  }
  END {
    if (col == 0)  { print "no_header"; exit }
    if (rows == 0) { print "no_rows";   exit }
    if (bad > 0)   { print "non_numeric"; exit }
    print "ok"
  }
')"
if [ "$offsets_verdict" != "ok" ]; then
  not_ready partition_read_failed \
    "offset read on ${target} returned ${offsets_verdict}: $(printf '%s' "$describe" | head -n 1)"
fi

# --- check 2: a group that has committed before must still be committing -----
# `rpk group describe` exit status is deliberately not consulted: the broker has
# just proved it serves offsets, and a lane whose only fault is that one group
# listing errored should not be declared down. What is consulted is whatever it
# printed.
groups="$(rpk group describe -r '.*' 2>/dev/null)"

new_state="${STATE_PATH}.new"
stale="$(printf '%s\n' "$groups" | awk \
  -v now="$NOW" -v window="$sync_window_seconds" \
  -v statefile="$STATE_PATH" -v newstate="$new_state" '
  function flush() {
    if (g != "" && state == "Stable" && members > 0 && lag > 0 && committed > 0) {
      cur[g] = progress
    }
  }
  BEGIN {
    while ((getline line < statefile) > 0) {
      n = split(line, a, "\t")
      if (n == 3) { prev[a[1]] = a[2]; seen[a[1]] = a[3] }
    }
    close(statefile)
  }
  /^GROUP[ \t]/     { flush(); g = $2; state = ""; members = 0; lag = 0; progress = 0; committed = 0; next }
  /^STATE[ \t]/     { state = $2; next }
  /^MEMBERS[ \t]/   { members = $2 + 0; next }
  /^TOTAL-LAG[ \t]/ { lag = $2 + 0; next }
  NF >= 6 && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ { progress += $3 + 0; committed++; next }
  END {
    flush()
    for (k in cur) {
      if (k in prev && prev[k] == cur[k]) { first = seen[k] } else { first = now }
      print k "\t" cur[k] "\t" first > newstate
      if (now - first > window) print k "@" cur[k] "/" (now - first) "s"
    }
    close(newstate)
  }
')"

if [ -f "$new_state" ]; then
  mv -f "$new_state" "$STATE_PATH" 2>/dev/null
else
  : > "$STATE_PATH" 2>/dev/null
fi

if [ -n "$stale" ]; then
  not_ready group_not_synced \
    "committed offset unchanged for more than ${sync_window_seconds}s while behind: $(printf '%s' "$stale" | tr '\n' ' ')"
fi

ready synced \
  "offset read on ${target} returned over ${mode}; every behind group that has committed is still committing"

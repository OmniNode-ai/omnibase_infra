#!/usr/bin/env sh
# unbound entrypoint for the OMN-15736 runner-fleet DNS cache.
#
# Starts unbound in the foreground bound to all interfaces (see unbound.conf),
# and background-loops a periodic hit/miss stats emission to stdout so the
# container logs carry a scrapeable hit-rate signal (AC1) without adding a
# separate metrics sidecar.
set -eu

CONF="/etc/unbound/unbound.conf"

# Generate a fresh control-channel key pair only if remote-control needs one;
# control-use-cert: no in unbound.conf means no cert files are required, but
# unbound still wants /etc/unbound/unbound_control.pem to exist to be quiet.
# unbound-control-setup is a no-op safety net if certs are ever re-enabled.
if [ ! -f /etc/unbound/unbound_server.key ]; then
    unbound-control-setup >/dev/null 2>&1 || true
fi

echo "[dns-cache] starting unbound with ${CONF}"

# Background stats loop: every 60s, dump cumulative counters (num.queries,
# num.cachehits, num.cachemiss) to stdout via unbound-control over the
# loopback control channel. This is the hit-rate metric surfaced for AC1 —
# `cache_hit_rate = num.cachehits / num.queries`. Runs after a short delay so
# the control socket is up before the first poll.
(
    sleep 10
    while true; do
        if command -v unbound-control >/dev/null 2>&1; then
            STATS=$(unbound-control -c "${CONF}" stats_noreset 2>/dev/null || true)
            QUERIES=$(printf '%s\n' "${STATS}" | grep '^total.num.queries=' | cut -d= -f2)
            HITS=$(printf '%s\n' "${STATS}" | grep '^total.num.cachehits=' | cut -d= -f2)
            MISSES=$(printf '%s\n' "${STATS}" | grep '^total.num.cachemiss=' | cut -d= -f2)
            echo "[dns-cache-stats] queries=${QUERIES:-0} cachehits=${HITS:-0} cachemiss=${MISSES:-0}"
        fi
        sleep 60
    done
) &

exec unbound -d -c "${CONF}"

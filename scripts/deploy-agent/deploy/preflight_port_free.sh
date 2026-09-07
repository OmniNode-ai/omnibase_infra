#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Fail-fast TCP port preflight for the deploy-agent systemd units (OMN-16939).
#
# Usage: preflight_port_free.sh <port>
#
# Exit 0  -- nothing is listening on <port>; the unit may start.
# Exit 1  -- <port> is held. The holder (command + pid) is named on stderr and
#            LEFT RUNNING. Deciding what to stop is an operator decision with a
#            blast radius; a unit file is not the place to make it.
# Exit 2  -- the preflight could not prove either answer (bad argument, or no
#            socket-inspection tool available). Fails CLOSED: an unprovable
#            precondition is not a satisfied one.
#
# WHY THIS EXISTS
#
# The units previously ran a one-line ExecStartPre that force-terminated
# whatever held the port and then discarded the outcome with `|| true`. That
# is a blind SIGKILL of a process the unit knows nothing about -- the peer
# lane's agent, a still-draining rebuild subprocess, or something unrelated
# that merely happened to bind the port -- and because the outcome was
# discarded the step reported success whether or not it had acted, leaving no
# trace either way. A port collision is a fail-fast error.
#
# Portability: `ss` (iproute2) is preferred and is what the .201 host has;
# `lsof` is the fallback. Both are run WITHOUT elevation, so a holder owned by
# another user may be reported without its pid -- that is still a refusal, just
# a less specific one.
#
# A tool that fails to RUN is not a tool that found nothing. `ss` exits 0 for a
# successful query whether or not it matched; any non-zero status means the
# query did not happen (unsupported filter syntax on an older iproute2, a
# netlink error, /proc/net unreadable in a restricted namespace). `lsof` is the
# exception that has to be special-cased: it exits 1 when the query ran and
# matched nothing, so only exit >= 2 is a failure there. Getting this wrong is
# how a fail-closed check silently becomes fail-open.
#
# LIMIT, stated rather than implied: this is a diagnostic refusal, NOT a port
# reservation. It proves the port free at time T; ExecStart binds at some later
# T+d and nothing holds the port across that gap. The authoritative arbiter is
# and remains the bind in ExecStart, which fails the unit if it loses a race.
# What the preflight buys is a legible, early, non-destructive refusal that
# names the holder, instead of a bind error with no attribution -- and, unlike
# the form it replaces, it never disturbs the process it found. The two agent
# units bind different ports (8098 dev, 8099 prod), so they cannot contend with
# each other; the residual race is against an unrelated process.

set -eu

PORT="${1:-}"

case "${PORT}" in
  '' | *[!0-9]*)
    echo "preflight_port_free: usage: $0 <tcp-port>  (got: '${PORT}')" >&2
    exit 2
    ;;
esac

if [ "${PORT}" -lt 1 ] || [ "${PORT}" -gt 65535 ]; then
  echo "preflight_port_free: port out of range: ${PORT}" >&2
  exit 2
fi

HOLDERS=''
PROBE_RC=0

# stderr from the probe is deliberately NOT discarded: when the query fails,
# the reason belongs in the journal right next to the refusal.
if command -v ss >/dev/null 2>&1; then
  # -H no header, -l listening, -t tcp, -n numeric, -p show the owning process.
  HOLDERS="$(ss -Hltnp "sport = :${PORT}")" || PROBE_RC=$?
  if [ "${PROBE_RC}" -ne 0 ]; then
    echo "preflight_port_free: 'ss' exited ${PROBE_RC} querying port ${PORT}, so the" >&2
    echo "  port's state is unproven. Refusing to start (fail-closed)." >&2
    exit 2
  fi
elif command -v lsof >/dev/null 2>&1; then
  # lsof exits 1 when the query ran and matched nothing -- that is the "free"
  # answer, not a failure. Only exit >= 2 means the query itself did not run.
  HOLDERS="$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN)" || PROBE_RC=$?
  if [ "${PROBE_RC}" -gt 1 ]; then
    echo "preflight_port_free: 'lsof' exited ${PROBE_RC} querying port ${PORT}, so the" >&2
    echo "  port's state is unproven. Refusing to start (fail-closed)." >&2
    exit 2
  fi
else
  echo "preflight_port_free: neither 'ss' nor 'lsof' is available, so freedom of" >&2
  echo "  port ${PORT} cannot be proven. Refusing to start (fail-closed)." >&2
  exit 2
fi

if [ -z "${HOLDERS}" ]; then
  exit 0
fi

echo "preflight_port_free: TCP port ${PORT} is already in use. Refusing to start." >&2
echo "  holder(s):" >&2
echo "${HOLDERS}" | sed 's/^/    /' >&2
echo "  This unit deliberately does NOT terminate the holder: it may be a peer" >&2
echo "  lane's agent or a draining rebuild. Identify it above, stop it" >&2
echo "  deliberately (e.g. 'systemctl --user stop <unit>'), then start again." >&2
exit 1

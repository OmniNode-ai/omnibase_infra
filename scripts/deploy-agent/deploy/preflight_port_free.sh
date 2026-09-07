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

if command -v ss >/dev/null 2>&1; then
  # -H no header, -l listening, -t tcp, -n numeric, -p show the owning process.
  HOLDERS="$(ss -Hltnp "sport = :${PORT}" 2>/dev/null || true)"
elif command -v lsof >/dev/null 2>&1; then
  HOLDERS="$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
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

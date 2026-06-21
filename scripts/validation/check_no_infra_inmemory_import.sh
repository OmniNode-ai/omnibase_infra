#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-13419: Enforce a single canonical in-memory event bus transport.
#
# The in-memory transport lives ONCE in omnibase_core:
#   from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
#
# omnibase_infra.event_bus.event_bus_inmemory is now a THIN ADAPTER over that
# core transport (it only re-expresses the infra error taxonomy, the
# dict-shaped health_check, and get_consumer_groups). Node packages and the
# local in-process path MUST depend on the core transport directly and MUST
# NOT import the infra adapter, so that in-memory paths never pull in
# omnibase_infra.
#
# This is a BLOCKING gate (not advisory). It fails on any import of the infra
# in-memory bus module/symbol outside the small allowlist of infra-runtime
# wrapper modules that legitimately own the adapter.
#
# Allowlist (the only files permitted to import the infra adapter):
#   - event_bus/event_bus_inmemory.py        (the adapter itself)
#   - event_bus/__init__.py                  (infra event_bus re-export)
#   - backends/auto_configure.py             (backend factory / inmemory fallback)
#   - runtime/runtime_host_process.py        (owns the bus instance)
#   - runtime/util_wiring.py                 (registers the bus backend)
#   - runtime/service_kernel.py              (kernel bus wiring)
#   - runtime/transition_notification_publisher.py  (ProtocolEventBusLike user)
#   - event_bus/testing/adapter_protocol_event_publisher_inmemory.py (test adapter)
#
# There is no free-text comment escape hatch. To add an allowlist entry, edit
# this script with justification in review.

set -euo pipefail

VIOLATIONS=0

# Files permitted to import the infra in-memory adapter (relative to src/).
ALLOWLIST=(
    "src/omnibase_infra/event_bus/event_bus_inmemory.py"
    "src/omnibase_infra/event_bus/__init__.py"
    "src/omnibase_infra/backends/auto_configure.py"
    "src/omnibase_infra/runtime/runtime_host_process.py"
    "src/omnibase_infra/runtime/util_wiring.py"
    "src/omnibase_infra/runtime/service_kernel.py"
    "src/omnibase_infra/runtime/transition_notification_publisher.py"
    "src/omnibase_infra/event_bus/testing/adapter_protocol_event_publisher_inmemory.py"
)

_is_allowlisted() {
    local f="$1"
    for allowed in "${ALLOWLIST[@]}"; do
        if [[ "$f" == "$allowed" ]]; then
            return 0
        fi
    done
    return 1
}

# Match real import statements (optionally indented for function-local
# imports) of the infra in-memory adapter. Docstring/example lines (e.g. a
# ">>> from ..." doctest) are excluded by anchoring on leading whitespace +
# the import keyword only.
PATTERN='^[[:space:]]*from omnibase_infra\.event_bus\.event_bus_inmemory import|^[[:space:]]*from omnibase_infra\.event_bus import[^#]*EventBusInmemory'

while IFS= read -r line; do
    file="${line%%:*}"
    # Normalize any doubled slash from grep's "src/" prefix (src//foo -> src/foo).
    file="${file//\/\//\/}"
    if _is_allowlisted "$file"; then
        continue
    fi
    echo "  $line"
    VIOLATIONS=$((VIOLATIONS + 1))
done < <(grep -rnE "$PATTERN" src/ --include="*.py" 2>/dev/null || true)

if [[ $VIOLATIONS -gt 0 ]]; then
    echo ""
    echo "Single-canonical in-memory bus (OMN-13419) — $VIOLATIONS violation(s)"
    echo ""
    echo "The in-memory transport lives ONCE in omnibase_core. Import it directly:"
    echo "  from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory"
    echo ""
    echo "Only the infra-runtime wrapper modules in this script's ALLOWLIST may"
    echo "import the thin infra adapter (omnibase_infra.event_bus.event_bus_inmemory)."
    echo ""
    exit 1
fi

echo "OK: no disallowed imports of the infra in-memory bus adapter."
exit 0

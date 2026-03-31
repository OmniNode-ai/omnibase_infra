#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-7077: Prevent new direct imports of EventBusInmemory from omnibase_infra.
#
# EventBusInmemory is migrating to omnibase_core. New code should import from:
#   from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
# Or use the try/except pattern while core Part 1 PRs are in flight.
#
# Allowlisted files:
#   - event_bus_inmemory.py itself (the implementation being migrated)
#   - __init__.py re-exports (use try/except pattern)
#   - auto_configure.py (uses _import_event_bus_inmemory helper)
#   - TYPE_CHECKING blocks (already annotated with OMN-7077 comment)

set -euo pipefail

VIOLATIONS=0

# Find direct imports of EventBusInmemory from the infra path in NEW code.
# Exclude the implementation file itself, __init__.py, and files with the
# OMN-7077 migration comment on the same or preceding line.
while IFS= read -r line; do
    file="${line%%:*}"
    basename="$(basename "$file")"

    # Allowlist: the implementation itself
    if [[ "$basename" == "event_bus_inmemory.py" ]]; then
        continue
    fi

    # Allowlist: __init__.py files (re-export with try/except)
    if [[ "$basename" == "__init__.py" ]]; then
        continue
    fi

    # Allowlist: auto_configure.py (uses helper function)
    if [[ "$basename" == "auto_configure.py" ]]; then
        continue
    fi

    # Allowlist: files that have the OMN-7077 migration comment
    if grep -q "OMN-7077" "$file" 2>/dev/null; then
        continue
    fi

    echo "  $line"
    VIOLATIONS=$((VIOLATIONS + 1))
done < <(grep -rn "from omnibase_infra\.event_bus\.event_bus_inmemory import" src/ --include="*.py" 2>/dev/null || true)

# Also check the shorthand import
while IFS= read -r line; do
    file="${line%%:*}"
    basename="$(basename "$file")"

    if [[ "$basename" == "__init__.py" ]] || [[ "$basename" == "event_bus_inmemory.py" ]] || [[ "$basename" == "auto_configure.py" ]]; then
        continue
    fi

    if grep -q "OMN-7077" "$file" 2>/dev/null; then
        continue
    fi

    echo "  $line"
    VIOLATIONS=$((VIOLATIONS + 1))
done < <(grep -rn "from omnibase_infra\.event_bus import.*EventBusInmemory" src/ --include="*.py" 2>/dev/null || true)

if [[ $VIOLATIONS -gt 0 ]]; then
    echo ""
    echo "EventBusInmemory Import Migration (OMN-7077) — $VIOLATIONS violation(s)"
    echo ""
    echo "EventBusInmemory is migrating to omnibase_core. Use:"
    echo "  try:"
    echo "      from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory"
    echo "  except ImportError:"
    echo '      from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory  # type: ignore[assignment]'
    echo ""
    exit 1
fi

exit 0

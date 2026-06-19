# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read contract-declared publish topics for the coding-agent orchestrator.

The contract's ``event_bus.publish_topics`` is the single source of truth for the
topic names the orchestrator emits onto. Handlers resolve topics by suffix from
this list so no topic string is ever hardcoded in source (CLAUDE.md /
``check-arch-invariants``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

# ``${env.VAR}`` / ``${env.VAR:default}`` expansion in contract string values —
# the same env-overlay convention node_contract_loader_effect uses for paths.
_ENV_REF = re.compile(
    r"\$\{env\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>[^}]*))?\}"
)


def _expand_env_refs(value: str) -> str:
    """Expand ``${env.VAR}`` / ``${env.VAR:default}`` references in ``value``.

    An unset var with no default expands to the empty string so the caller's
    fail-closed check rejects it (rather than leaving a literal placeholder).
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        # ONEX_EXCLUDE: io_audit - contract-config overlay: resolves the lane-specific
        # workspace-root path declared as ${env.…} in the contract (same pattern as
        # node_contract_loader_effect's ONEX_CONTRACTS_DIR); NOT a model/endpoint env.
        return os.environ.get(name, default if default is not None else "")

    return _ENV_REF.sub(_sub, value)


def _load_contract(contract_path: Path) -> dict[str, object]:
    # ONEX_EXCLUDE: io_audit - Module-level contract load keeps handler policy contract-owned
    with contract_path.open(encoding="utf-8") as contract_file:
        raw = yaml.safe_load(contract_file)
    if not isinstance(raw, dict):
        raise ValueError(f"contract {contract_path} must contain a mapping")
    return raw


def contract_publish_topics(contract_path: Path) -> tuple[str, ...]:
    """Return the ``event_bus.publish_topics`` declared in the given contract."""
    raw = _load_contract(contract_path)
    event_bus = raw.get("event_bus")
    if not isinstance(event_bus, dict):
        raise ValueError(f"contract {contract_path} must declare an event_bus mapping")
    topics = event_bus.get("publish_topics")
    if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
        raise ValueError(
            f"contract {contract_path} event_bus.publish_topics must be a list of "
            "strings"
        )
    return tuple(topics)


def contract_allowed_workspace_roots(contract_path: Path) -> tuple[str, ...]:
    """Return ``descriptor.allowed_workspace_roots`` declared in the contract.

    The allowed-root policy is contract-declared (overridable by an operator
    overlay contract) — never hardcoded in source and never read from an env var.
    Returns an empty tuple when the field is absent or empty so the caller can
    fail closed (an unscoped workspace is never silently permitted).
    """
    raw = _load_contract(contract_path)
    descriptor = raw.get("descriptor")
    if not isinstance(descriptor, dict):
        return ()
    roots = descriptor.get("allowed_workspace_roots")
    if not isinstance(roots, list):
        return ()
    expanded = (_expand_env_refs(r) for r in roots if isinstance(r, str))
    return tuple(r for r in expanded if r.strip())


__all__: list[str] = [
    "contract_allowed_workspace_roots",
    "contract_publish_topics",
]

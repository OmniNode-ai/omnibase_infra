# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve the event-forward backend endpoint from its node contract."""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs

_CONTRACT = Path(__file__).resolve().parent / "contract.yaml"


def _load_contract(contract_path: Path) -> dict[str, object]:
    # ONEX_EXCLUDE: io_audit - The descriptor is the contract-owned configuration boundary.
    with contract_path.open(encoding="utf-8") as contract_file:
        raw = yaml.safe_load(contract_file)
    if not isinstance(raw, dict):
        raise ValueError(f"contract {contract_path} must contain a mapping")
    return raw


def contract_event_forward_backend_url(contract_path: Path = _CONTRACT) -> str:
    """Return the fail-closed event-forward backend URL declared by the contract."""
    descriptor = _load_contract(contract_path).get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError(
            f"contract {contract_path} must declare a descriptor mapping with backend_url"
        )
    declared = descriptor.get("backend_url")
    if not isinstance(declared, str):
        raise ValueError(
            f"contract {contract_path} must declare a string descriptor.backend_url"
        )
    resolved = expand_contract_env_refs(declared).strip()
    if not resolved:
        raise ValueError(
            "descriptor.backend_url resolved empty — configure the event-forward "
            "backend through EVENT_FORWARD_BACKEND_URL."
        )
    return resolved


__all__: list[str] = ["contract_event_forward_backend_url"]

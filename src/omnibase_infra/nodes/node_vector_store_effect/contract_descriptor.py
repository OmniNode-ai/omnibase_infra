# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the contract-declared Qdrant endpoint for the vector-store effect.

The vector-store node's ``descriptor.qdrant_url`` is the single source of truth
for the Qdrant HTTP endpoint the upsert/search handlers (and the registry-api
health probe) connect to (OMN-13558 Wave-1 endpoint→overlay migration). It is
declared with the ``${env.VAR}`` overlay convention so an operator overlay / the
per-lane service env supplies the real endpoint per lane — never a hardcoded
``http://localhost:6333`` in source.

Resolution goes through ``expand_contract_env_refs`` — the one sanctioned
env-reading boundary in the overlay package — so the handlers never read
``os.environ`` directly. It is resolved fail-closed: an unset/empty value raises
rather than silently defaulting to localhost (which would mask a missing-config
deploy and connect to the wrong endpoint).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs

_CONTRACT = Path(__file__).resolve().parent / "contract.yaml"


def _load_contract(contract_path: Path) -> dict[str, object]:
    # ONEX_EXCLUDE: io_audit - Module-level contract load keeps handler policy contract-owned
    with contract_path.open(encoding="utf-8") as contract_file:
        raw = yaml.safe_load(contract_file)
    if not isinstance(raw, dict):
        raise ValueError(f"contract {contract_path} must contain a mapping")
    return raw


def contract_qdrant_url(contract_path: Path = _CONTRACT) -> str:
    """Return the resolved ``descriptor.qdrant_url`` for the vector-store node.

    The value is contract-declared (overridable by an operator overlay contract)
    via the ``${env.QDRANT_URL}`` convention — never hardcoded in source. Fails
    closed: raises ``ValueError`` when the field is absent or resolves to an empty
    string, so the vector handlers never silently fall back to
    ``http://localhost:6333`` when ``QDRANT_URL`` is unset.
    """
    raw = _load_contract(contract_path)
    descriptor = raw.get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError(
            f"contract {contract_path} must declare a descriptor mapping with "
            "qdrant_url"
        )
    declared = descriptor.get("qdrant_url")
    if not isinstance(declared, str):
        raise ValueError(
            f"contract {contract_path} must declare a string "
            "descriptor.qdrant_url (the ${env.QDRANT_URL} overlay value the "
            "vector-store handlers use as the Qdrant endpoint)"
        )
    resolved = expand_contract_env_refs(declared).strip()
    if not resolved:
        raise ValueError(
            "descriptor.qdrant_url resolved empty — set QDRANT_URL (the Qdrant "
            "HTTP endpoint the vector-store effect connects to). The vector-store "
            "effect fails closed rather than silently default to "
            "http://localhost:6333."
        )
    return resolved


__all__: list[str] = ["contract_qdrant_url"]

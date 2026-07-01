# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the contract-declared graph (Bolt) endpoint for the graph handler.

The graph handler's ``descriptor.graph_bolt_uri`` is the single source of truth
for the Bolt endpoint the Cypher operations (Memgraph/Neo4j) connect to when no
explicit ``connection_uri`` / ``bolt_uri`` override is supplied in the handler
config dict (OMN-13558 Wave-1 endpoint→overlay migration). It is declared with
the ``${env.VAR}`` overlay convention so an operator overlay / the per-lane
service env supplies the real endpoint per lane — never a hardcoded
``bolt://localhost:7687`` in source.

Resolution goes through ``expand_contract_env_refs`` — the one sanctioned
env-reading boundary in the overlay package — so the handler never reads
``os.environ`` directly. It is resolved fail-closed: an unset/empty value raises
rather than silently defaulting to localhost (which would mask a missing-config
deploy and connect to the wrong endpoint).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs

# The graph handler's authoritative contract lives under the infra contracts tree.
# Resolved relative to this module so it is portable across machines / install
# layouts (no hardcoded absolute path).
_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "handlers"
    / "graph"
    / "handler_contract.yaml"
)


def _load_contract(contract_path: Path) -> dict[str, object]:
    # ONEX_EXCLUDE: io_audit - Module-level contract load keeps handler policy contract-owned
    with contract_path.open(encoding="utf-8") as contract_file:
        raw = yaml.safe_load(contract_file)
    if not isinstance(raw, dict):
        raise ValueError(f"contract {contract_path} must contain a mapping")
    return raw


def contract_graph_bolt_uri(contract_path: Path = _CONTRACT) -> str:
    """Return the resolved ``descriptor.graph_bolt_uri`` for the graph handler.

    The value is contract-declared (overridable by an operator overlay contract)
    via the ``${env.GRAPH_BOLT_URI}`` convention — never hardcoded in source.
    Fails closed: raises ``ValueError`` when the field is absent or resolves to
    an empty string, so the graph handler never silently falls back to
    ``bolt://localhost:7687`` when ``GRAPH_BOLT_URI`` is unset.
    """
    raw = _load_contract(contract_path)
    descriptor = raw.get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError(
            f"contract {contract_path} must declare a descriptor mapping with "
            "graph_bolt_uri"
        )
    declared = descriptor.get("graph_bolt_uri")
    if not isinstance(declared, str):
        raise ValueError(
            f"contract {contract_path} must declare a string "
            "descriptor.graph_bolt_uri (the ${env.GRAPH_BOLT_URI} overlay value "
            "the graph handler uses as the Bolt endpoint)"
        )
    resolved = expand_contract_env_refs(declared).strip()
    if not resolved:
        raise ValueError(
            "descriptor.graph_bolt_uri resolved empty — set GRAPH_BOLT_URI (the "
            "Bolt endpoint the graph handler connects to). The graph handler "
            "fails closed rather than silently default to bolt://localhost:7687."
        )
    return resolved


__all__: list[str] = ["contract_graph_bolt_uri"]

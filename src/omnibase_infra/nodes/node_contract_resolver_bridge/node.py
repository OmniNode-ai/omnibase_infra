# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for the Contract Resolver Bridge.

Exposes NodeContractResolveCompute (OMN-2754) via synchronous HTTP so the
dashboard can resolve overlaid contracts without a Kafka round-trip. Registered
identically to the production node — only the execution path changes when the
real ONEX node runner supports HTTP invocation.

Architecture
------------
Pure declarative shell — ``node.py`` contains NO custom business logic.
All configuration lives in ``contract.yaml`` (see ``config:`` section).
The FastAPI service lives in ``services/contract_resolver/``.

Ticket: OMN-2756
Related:
    - contract.yaml: Node contract with config, I/O, and routing definitions
    - services/contract_resolver/main.py: FastAPI application factory
    - services/contract_resolver/routes.py: Route implementations
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from omnibase_core.nodes.node_effect import NodeEffect

# Type alias for the heterogeneous contract config dict.
# Values are int/str/bool/list — no single concrete type covers all keys.
# ONEX_EXCLUDE: any_type - contract.yaml config values are heterogeneous (int/str/bool/list)
ContractConfig = dict[str, Any]


@lru_cache(maxsize=1)
def load_contract_resolver_bridge_config() -> ContractConfig:
    """Load and return the ``config:`` section from ``contract.yaml``.

    Results are cached for the lifetime of the process (``lru_cache``).
    Call ``load_contract_resolver_bridge_config.cache_clear()`` in tests that
    need a fresh load.

    Returns:
        Dict containing all keys from the ``config:`` block in
        ``contract.yaml``.  Returns an empty dict if the section is
        absent or the file cannot be read.

    Example:
        >>> cfg = load_contract_resolver_bridge_config()
        >>> port = cfg.get("port", 8091)
    """
    contract_path = Path(__file__).parent / "contract.yaml"
    try:
        raw: ContractConfig = yaml.safe_load(contract_path.read_text())
        return dict(raw.get("config", {}))
    except Exception:
        return {}


class NodeContractResolverBridge(NodeEffect):
    """Declarative EFFECT node for the Contract Resolver Bridge.

    Pure declarative shell — all behaviour is defined in ``contract.yaml``.
    No custom business logic or methods.

    The FastAPI service wrapping NodeContractResolveCompute is defined in
    ``services/contract_resolver/``.
    """


__all__ = [
    "ContractConfig",
    "NodeContractResolverBridge",
    "load_contract_resolver_bridge_config",
]

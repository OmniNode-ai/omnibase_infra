# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for the Registry Discovery API.

Wraps the FastAPI Registry API service as a proper ONEX EFFECT node,
replacing hardcoded constants with contract-driven configuration sourced
from ``contract.yaml``.

Architecture
------------
This node follows the ONEX declarative pattern established by
``NodeRegistryEffect`` and ``NodeServiceDiscoveryEffect``:

- Pure declarative shell — ``node.py`` contains NO custom business logic.
- All configuration lives in ``contract.yaml`` (see ``config:`` section).
- Handler routing is 100% contract-driven.
- Dependencies are injected via ``ModelONEXContainer``.

Configuration Loading
---------------------
Contract configuration is loaded via the module-level helper
``load_registry_api_config()`` defined in this module.  The existing
``ServiceRegistryDiscovery`` class imports that helper to resolve
settings at construction time, replacing the former module-level constants:

- ``MAX_NODE_TYPE_FILTER_FETCH``    → ``config.max_node_type_filter_fetch``
- ``DEFAULT_WIDGET_MAPPING_PATH``   → ``config.default_widget_mapping_path``

Ticket: OMN-1441
Related:
    - contract.yaml: Node contract with config, I/O, and routing definitions
    - services/registry_api/service.py: imports load_registry_api_config()
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
def load_registry_api_config() -> ContractConfig:
    """Load and return the ``config:`` section from ``contract.yaml``.

    Results are cached for the lifetime of the process (``lru_cache``).
    Call ``load_registry_api_config.cache_clear()`` in tests that need a
    fresh load.

    Returns:
        Dict containing all keys from the ``config:`` block in
        ``contract.yaml``.  Returns an empty dict if the section is
        absent or the file cannot be read.

    Example:
        >>> cfg = load_registry_api_config()
        >>> max_fetch = cfg.get("max_node_type_filter_fetch", 10000)
    """
    contract_path = Path(__file__).parent / "contract.yaml"
    try:
        raw: ContractConfig = yaml.safe_load(contract_path.read_text())
        return dict(raw.get("config", {}))
    except Exception:
        return {}


class NodeRegistryApiEffect(NodeEffect):
    """Declarative EFFECT node for the Registry Discovery API.

    Pure declarative shell — all behaviour is defined in ``contract.yaml``.
    No custom business logic or methods.

    See module docstring for configuration loading details.
    """


__all__ = ["ContractConfig", "NodeRegistryApiEffect", "load_registry_api_config"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for contract filesystem scanning [OMN-6347].

Delegates to RuntimeContractConfigLoader for the actual scan operation.
This handler bridges the ONEX node interface to the existing imperative
contract scanning infrastructure.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )

logger = logging.getLogger(__name__)

# Default contracts directory env var
_DEFAULT_CONTRACTS_DIR_ENV = "ONEX_CONTRACTS_DIR"
_DEFAULT_CONTRACTS_DIR_FALLBACK = "./contracts"


class HandlerContractScan:
    """Scans directories for ONEX contract YAML files.

    Wraps ``RuntimeContractConfigLoader.load_all_contracts()`` as an ONEX
    handler, enabling contract discovery as part of the declarative runtime boot.
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        *,
        node_graph_config: ModelRuntimeNodeGraphConfig | None = None,
    ) -> None:
        """Initialize with ONEX container for dependency resolution.

        Args:
            container: ONEX container for service resolution.
            node_graph_config: Optional runtime node graph config providing
                scan_exclude_patterns and scan_deny_paths.
        """
        self._container = container
        self._node_graph_config = node_graph_config

    async def handle(self, scan_paths: list[str] | None = None) -> dict[str, object]:
        """Execute contract scan across configured directories.

        Args:
            scan_paths: Optional override paths to scan. If None, uses
                ONEX_CONTRACTS_DIR env var or falls back to ./contracts.

        Returns:
            Summary dict with scan status.
        """
        from omnibase_infra.runtime.runtime_contract_config_loader import (
            RuntimeContractConfigLoader,
        )

        # Resolve scan paths: explicit > env var > fallback
        if scan_paths:
            path_objects = [Path(p) for p in scan_paths]
        else:
            contracts_dir = os.environ.get(
                _DEFAULT_CONTRACTS_DIR_ENV, _DEFAULT_CONTRACTS_DIR_FALLBACK
            )
            path_objects = [Path(contracts_dir)]

        # Thread scan policies from node_graph_config
        exclude_patterns: tuple[str, ...] = ()
        deny_paths: tuple[str, ...] = ()
        if self._node_graph_config is not None:
            exclude_patterns = self._node_graph_config.scan_exclude_patterns
            deny_paths = self._node_graph_config.scan_deny_paths

        loader = RuntimeContractConfigLoader(
            scan_exclude_patterns=exclude_patterns,
            scan_deny_paths=deny_paths,
        )
        config = loader.load_all_contracts(path_objects)
        logger.info(
            "Contract scan complete: loaded %d contracts",
            config.total_contracts_loaded,
        )
        return {
            "status": "complete",
            "total_loaded": config.total_contracts_loaded,
        }

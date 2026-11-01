# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for contract filesystem scanning [OMN-6347].

Delegates to RuntimeContractConfigLoader for the actual scan operation.
This handler bridges the ONEX node interface to the existing imperative
contract scanning infrastructure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )

logger = logging.getLogger(__name__)


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
        default_scan_path: str = "./contracts",
    ) -> None:
        """Initialize with ONEX container for dependency resolution.

        Args:
            container: ONEX container for service resolution.
            node_graph_config: Optional runtime node graph config providing
                scan_exclude_patterns and scan_deny_paths.
            default_scan_path: Fallback scan path when no explicit paths are
                provided. Typically set from ``ONEX_CONTRACTS_DIR`` env var
                by the caller (kernel bootstrap), not read from env here.
        """
        self._container = container
        self._node_graph_config = node_graph_config
        self._default_scan_path = default_scan_path

    async def handle(self, scan_paths: list[str] | None = None) -> dict[str, object]:
        """Execute contract scan across configured directories.

        Args:
            scan_paths: Optional override paths to scan. If None, uses
                the default_scan_path from constructor.

        Returns:
            Summary dict with scan status.
        """
        from omnibase_infra.runtime.runtime_contract_config_loader import (
            RuntimeContractConfigLoader,
        )

        # Resolve scan paths: explicit > default from constructor
        path_objects = (
            [Path(p) for p in scan_paths]
            if scan_paths
            else [Path(self._default_scan_path)]
        )

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

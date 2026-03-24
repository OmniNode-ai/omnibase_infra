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

logger = logging.getLogger(__name__)


class HandlerContractScan:
    """Scans directories for ONEX contract YAML files.

    Wraps ``RuntimeContractConfigLoader.load_all_contracts()`` as an ONEX
    handler, enabling contract discovery as part of the declarative runtime boot.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with ONEX container for dependency resolution."""
        self._container = container

    async def handle(self, scan_paths: list[str] | None = None) -> dict[str, object]:
        """Execute contract scan across configured directories.

        Args:
            scan_paths: Optional override paths to scan. If None, uses
                paths from the runtime contract config.

        Returns:
            Summary dict with scan status.
        """
        from omnibase_infra.runtime.runtime_contract_config_loader import (
            RuntimeContractConfigLoader,
        )

        loader = RuntimeContractConfigLoader()
        path_objects = [Path(p) for p in scan_paths] if scan_paths else []
        config = loader.load_all_contracts(path_objects)
        logger.info(
            "Contract scan complete: loaded %d contracts",
            config.total_contracts_loaded,
        )
        return {
            "status": "complete",
            "total_loaded": config.total_contracts_loaded,
        }

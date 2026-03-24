# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for contract filesystem scanning [OMN-6347].

Delegates to RuntimeContractConfigLoader for the actual scan operation.
This handler bridges the ONEX node interface to the existing imperative
contract scanning infrastructure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)


class HandlerContractScan:
    """Scans directories for ONEX contract YAML files.

    Wraps ``RuntimeContractConfigLoader.scan()`` as an ONEX handler,
    enabling contract discovery as part of the declarative runtime boot.
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
            Summary dict with discovered contract count and paths.
        """
        from omnibase_infra.runtime.runtime_contract_config_loader import (
            RuntimeContractConfigLoader,
        )

        loader = RuntimeContractConfigLoader()
        # Use provided paths or default scan behavior
        results = loader.discover_contracts(scan_paths or [])
        logger.info(
            "Contract scan complete: %d contracts discovered",
            len(results),
        )
        return {
            "contracts_discovered": len(results),
            "scan_paths": scan_paths or [],
        }

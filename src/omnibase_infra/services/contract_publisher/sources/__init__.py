# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Publisher Sources Module.

Provides contract source implementations for discovering contracts from
different backends:

- SourceContractFilesystem: Discovers from directory tree
- SourceContractPackage: Discovers from package resources
- SourceContractComposite: Merges multiple sources

All sources implement ProtocolContractPublisherSource.

Example:
    >>> from omnibase_infra.services.contract_publisher.sources import (
    ...     SourceContractFilesystem,
    ...     SourceContractPackage,
    ...     SourceContractComposite,
    ... )
    >>> filesystem = SourceContractFilesystem(Path("/app/contracts"))
    >>> contracts = await filesystem.discover_contracts()

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from omnibase_infra.services.contract_publisher.sources.model_discovered import (
    ModelDiscoveredContract,
)
from omnibase_infra.services.contract_publisher.sources.protocol import (
    ProtocolContractPublisherSource,
)
from omnibase_infra.services.contract_publisher.sources.source_composite import (
    SourceContractComposite,
)
from omnibase_infra.services.contract_publisher.sources.source_filesystem import (
    SourceContractFilesystem,
)
from omnibase_infra.services.contract_publisher.sources.source_package import (
    SourceContractPackage,
)

__all__ = [
    # Protocol
    "ProtocolContractPublisherSource",
    # Model
    "ModelDiscoveredContract",
    # Implementations
    "SourceContractFilesystem",
    "SourceContractPackage",
    "SourceContractComposite",
]

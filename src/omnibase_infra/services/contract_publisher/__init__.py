# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Publisher Service Module.

Provides infrastructure for bulk contract discovery and publishing to Kafka.
This service discovers contracts from configured sources (filesystem, package)
and publishes them to the contract registration topic for dynamic discovery.

Moved from omniclaude as part of OMN-1752 (ARCH-002 compliance).

Design Principle:
    Infra standardizes the publishing *engine*; apps provide *source configuration*.

Flow:
    Source → Validate → Normalize → Publish → Report
    - Sources provide origin (filesystem, package)
    - Event bus provides distribution (broadcast plane)

Example:
    >>> from omnibase_infra.services.contract_publisher import (
    ...     ServiceContractPublisher,
    ...     ModelContractPublisherConfig,
    ... )
    >>> config = ModelContractPublisherConfig(
    ...     mode="filesystem",
    ...     filesystem_root=Path("/app/contracts/handlers"),
    ... )
    >>> publisher = await ServiceContractPublisher.from_container(container, config)
    >>> result = await publisher.publish_all()
    >>> if result:
    ...     print(f"Published {len(result.published)} contracts")
    ... else:
    ...     print(f"No contracts published, {len(result.contract_errors)} errors")

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

# Config
from omnibase_infra.services.contract_publisher.config import (
    ModelContractPublisherConfig,
)

# Errors
from omnibase_infra.services.contract_publisher.errors import (
    ContractPublisherError,
    ContractPublishingInfraError,
    ContractSourceNotConfiguredError,
    NoContractsFoundError,
)

# Result models
from omnibase_infra.services.contract_publisher.models import (
    ModelContractError,
    ModelInfraError,
    ModelPublishResult,
    ModelPublishStats,
)

# Service
from omnibase_infra.services.contract_publisher.service import (
    ServiceContractPublisher,
)

# Sources
from omnibase_infra.services.contract_publisher.sources import (
    ModelDiscoveredContract,
    ProtocolContractPublisherSource,
    SourceContractComposite,
    SourceContractFilesystem,
    SourceContractPackage,
)

__all__ = [
    # Service
    "ServiceContractPublisher",
    # Config
    "ModelContractPublisherConfig",
    # Result models
    "ModelPublishResult",
    "ModelPublishStats",
    "ModelContractError",
    "ModelInfraError",
    # Errors
    "ContractPublisherError",
    "ContractSourceNotConfiguredError",
    "ContractPublishingInfraError",
    "NoContractsFoundError",
    # Sources
    "ProtocolContractPublisherSource",
    "ModelDiscoveredContract",
    "SourceContractFilesystem",
    "SourceContractPackage",
    "SourceContractComposite",
]

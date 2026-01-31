# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Publisher Service.

Main service for discovering and publishing contracts to Kafka.
Follows the flow: Source → Validate → Normalize → Publish → Report.

This service standardizes the publishing *engine* while apps provide
*source configuration*. It enforces ARCH-002: "Runtime owns all Kafka plumbing."

Related:
    - OMN-1752: Extract ContractPublisher to omnibase_infra
    - ARCH-002: Runtime owns all Kafka plumbing

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING
from uuid import uuid4

import yaml
from pydantic import ValidationError

from omnibase_core.constants import TOPIC_SUFFIX_CONTRACT_REGISTERED
from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract
from omnibase_core.models.events import ModelContractRegisteredEvent
from omnibase_core.models.primitives import ModelSemVer
from omnibase_core.protocols.event_bus import ProtocolEventBusPublisher
from omnibase_infra.services.contract_publisher.config import (
    ModelContractPublisherConfig,
)
from omnibase_infra.services.contract_publisher.errors import (
    ContractPublishingInfraError,
    ContractSourceNotConfiguredError,
    NoContractsFoundError,
)
from omnibase_infra.services.contract_publisher.models import (
    ModelContractError,
    ModelInfraError,
    ModelPublishResult,
    ModelPublishStats,
)
from omnibase_infra.services.contract_publisher.sources import (
    ModelDiscoveredContract,
    ProtocolContractPublisherSource,
    SourceContractComposite,
    SourceContractFilesystem,
    SourceContractPackage,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)


class ServiceContractPublisher:
    """Contract publishing service with injectable sources.

    Discovers contracts from configured sources (filesystem, package, composite)
    and publishes them to Kafka for dynamic discovery by KafkaContractSource.

    Architecture:
        Container → ServiceContractPublisher → ProtocolContractPublisherSource → Kafka
                                            ↓
                                    ModelPublishResult

    Flow:
        1. Discover: Get contracts from source
        2. Sort: Order by (handler_id, origin, ref) for determinism
        3. Validate: Parse YAML, validate schema, extract handler_id
        4. Normalize: Compute SHA-256 hash for change detection
        5. Publish: Create ModelContractRegisteredEvent, publish to Kafka
        6. Report: Return ModelPublishResult with errors and stats

    Thread Safety:
        This service is async-safe. Each call to publish_all() is independent.

    Example:
        >>> config = ModelContractPublisherConfig(
        ...     mode="filesystem",
        ...     filesystem_root=Path("/app/contracts"),
        ... )
        >>> publisher = await ServiceContractPublisher.from_container(container, config)
        >>> result = await publisher.publish_all()
        >>> if result:
        ...     print(f"Published {len(result.published)} contracts")

    .. versionadded:: 0.3.0
    """

    __slots__ = ("_config", "_environment", "_publisher", "_source")

    def __init__(
        self,
        publisher: ProtocolEventBusPublisher,
        source: ProtocolContractPublisherSource,
        config: ModelContractPublisherConfig,
    ) -> None:
        """Initialize service with publisher and source.

        Args:
            publisher: Event bus publisher for Kafka
            source: Contract source for discovery
            config: Publishing configuration
        """
        self._publisher = publisher
        self._source = source
        self._config = config
        self._environment = config.resolve_environment()

    @classmethod
    async def from_container(
        cls,
        container: ModelONEXContainer,
        config: ModelContractPublisherConfig,
    ) -> ServiceContractPublisher:
        """Factory method for DI resolution.

        Resolves ProtocolEventBusPublisher from container and creates
        appropriate source based on config.mode.

        Args:
            container: ONEX container with event bus publisher
            config: Publishing configuration

        Returns:
            Configured ServiceContractPublisher instance

        Raises:
            ContractSourceNotConfiguredError: If required source not configured
            RuntimeError: If publisher not available in container
        """
        # Resolve publisher from container
        # NOTE: Protocol type passed for duck-typed resolution per ONEX patterns
        publisher = await container.get_service_async(
            ProtocolEventBusPublisher  # type: ignore[type-abstract]
        )
        if publisher is None:
            raise ContractSourceNotConfiguredError(
                mode=config.mode,
                missing_field="publisher",
                message="ProtocolEventBusPublisher not available in container",
            )

        # Create source based on mode
        source = cls._create_source(config)

        return cls(publisher, source, config)

    @staticmethod
    def _create_source(
        config: ModelContractPublisherConfig,
    ) -> ProtocolContractPublisherSource:
        """Create source based on config mode.

        Args:
            config: Publishing configuration

        Returns:
            Appropriate source implementation

        Raises:
            ContractSourceNotConfiguredError: If required source not configured
        """
        match config.mode:
            case "filesystem":
                if not config.filesystem_root:
                    raise ContractSourceNotConfiguredError(
                        mode="filesystem",
                        missing_field="filesystem_root",
                    )
                return SourceContractFilesystem(config.filesystem_root)

            case "package":
                if not config.package_module:
                    raise ContractSourceNotConfiguredError(
                        mode="package",
                        missing_field="package_module",
                    )
                return SourceContractPackage(config.package_module)

            case "composite":
                filesystem_source = None
                package_source = None

                if config.filesystem_root:
                    filesystem_source = SourceContractFilesystem(config.filesystem_root)
                if config.package_module:
                    package_source = SourceContractPackage(config.package_module)

                if not filesystem_source and not package_source:
                    raise ContractSourceNotConfiguredError(
                        mode="composite",
                        missing_field="filesystem_root or package_module",
                    )

                return SourceContractComposite(filesystem_source, package_source)

            case _:
                raise ContractSourceNotConfiguredError(
                    mode=config.mode,
                    missing_field="mode",
                    message=f"Unknown mode: {config.mode}",
                )

    def resolve_topic(self, topic_suffix: str) -> str:
        """Resolve topic suffix to full topic name with environment prefix.

        Uses the same pattern as EventBusSubcontractWiring.

        Args:
            topic_suffix: Topic suffix (e.g., "onex.evt.contract-registered.v1")

        Returns:
            Full topic name (e.g., "dev.onex.evt.contract-registered.v1")
        """
        return f"{self._environment}.{topic_suffix}"

    async def publish_all(self) -> ModelPublishResult:
        """Discover and publish all contracts from configured source.

        Executes the full flow: Discover → Sort → Validate → Publish → Report

        Returns:
            ModelPublishResult with published handlers and any errors

        Raises:
            NoContractsFoundError: If no contracts found and allow_zero_contracts=False
            ContractPublishingInfraError: If infrastructure error and fail_fast=True
        """
        start_time = time.perf_counter()
        correlation_id = uuid4()

        # Initialize result tracking
        published: list[str] = []
        contract_errors: list[ModelContractError] = []
        infra_errors: list[ModelInfraError] = []
        dedup_count = 0
        filesystem_count = 0
        package_count = 0

        # Phase 1: Discover
        discover_start = time.perf_counter()
        contracts, source_errors = await self._discover()
        discover_ms = (time.perf_counter() - discover_start) * 1000

        # Add any source errors (from composite conflict detection)
        contract_errors.extend(source_errors)

        # Count per-origin
        for contract in contracts:
            if contract.origin == "filesystem":
                filesystem_count += 1
            elif contract.origin == "package":
                package_count += 1

        discovered_count = len(contracts)

        logger.info(
            "Contract discovery complete: %d contracts from %s",
            discovered_count,
            self._source.source_description,
            extra={"correlation_id": str(correlation_id)},
        )

        # Phase 2: Validate
        validate_start = time.perf_counter()
        valid_contracts: list[tuple[ModelDiscoveredContract, ModelHandlerContract]] = []

        for contract in contracts:
            parsed = self._validate_contract(contract)
            if parsed is None:
                # Validation failed - error already added
                error = self._create_validation_error(contract)
                if error:
                    contract_errors.append(error)
            else:
                valid_contracts.append((contract, parsed))

        validate_ms = (time.perf_counter() - validate_start) * 1000
        valid_count = len(valid_contracts)

        logger.info(
            "Contract validation complete: %d valid, %d errors",
            valid_count,
            len(contract_errors),
            extra={"correlation_id": str(correlation_id)},
        )

        # Check for zero contracts
        if valid_count == 0 and not self._config.allow_zero_contracts:
            raise NoContractsFoundError(
                source_description=self._source.source_description,
                discovered_count=discovered_count,
                valid_count=0,
            )

        # Phase 3: Publish
        publish_start = time.perf_counter()
        topic = self.resolve_topic(TOPIC_SUFFIX_CONTRACT_REGISTERED)

        for contract, parsed in valid_contracts:
            handler_id = parsed.handler_id

            # Create event
            try:
                event = ModelContractRegisteredEvent(
                    node_name=handler_id,
                    node_version=str(parsed.contract_version),
                    contract_hash=contract.content_hash or "",
                    contract_yaml=contract.text,
                    correlation_id=correlation_id,
                )

                # Serialize to bytes
                event_bytes = event.model_dump_json().encode("utf-8")
                key_bytes = handler_id.encode("utf-8")

                # Publish
                await self._publisher.publish(
                    topic=topic,
                    key=key_bytes,
                    value=event_bytes,
                )

                published.append(handler_id)

                logger.debug(
                    "Published contract: %s to %s",
                    handler_id,
                    topic,
                    extra={"correlation_id": str(correlation_id)},
                )

            except Exception as e:
                # Infrastructure error
                infra_error = ModelInfraError(
                    error_type="publish_failed",
                    message=f"Failed to publish {handler_id}: {e}",
                    retriable=True,
                )
                infra_errors.append(infra_error)

                logger.exception(
                    "Failed to publish contract %s",
                    handler_id,
                    extra={"correlation_id": str(correlation_id)},
                )

                if self._config.fail_fast:
                    raise ContractPublishingInfraError(infra_errors)

        publish_ms = (time.perf_counter() - publish_start) * 1000
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Build stats
        stats = ModelPublishStats(
            discovered_count=discovered_count,
            valid_count=valid_count,
            published_count=len(published),
            errored_count=len(contract_errors),
            dedup_count=dedup_count,
            duration_ms=duration_ms,
            discover_ms=discover_ms,
            validate_ms=validate_ms,
            publish_ms=publish_ms,
            environment=self._environment,
            filesystem_count=filesystem_count,
            package_count=package_count,
        )

        # Build result
        result = ModelPublishResult(
            published=published,
            contract_errors=contract_errors,
            infra_errors=infra_errors,
            stats=stats,
        )

        logger.info(
            "Contract publishing complete: %d published, %d contract errors, %d infra errors",
            len(published),
            len(contract_errors),
            len(infra_errors),
            extra={
                "correlation_id": str(correlation_id),
                "duration_ms": duration_ms,
            },
        )

        return result

    async def _discover(
        self,
    ) -> tuple[list[ModelDiscoveredContract], list[ModelContractError]]:
        """Discover contracts from source.

        Returns:
            Tuple of (contracts, errors from composite merge)
        """
        # All sources now return list[ModelDiscoveredContract]
        contracts = await self._source.discover_contracts()

        # Add content hash to all contracts
        contracts = [c.with_content_hash() for c in contracts]

        # Sort for determinism
        contracts.sort(key=lambda c: c.sort_key())

        # Get merge errors from composite source
        errors: list[ModelContractError] = []
        if isinstance(self._source, SourceContractComposite):
            errors = self._source.get_merge_errors()

        return contracts, errors

    def _validate_contract(
        self,
        contract: ModelDiscoveredContract,
    ) -> ModelHandlerContract | None:
        """Validate contract YAML and extract handler_id.

        Args:
            contract: Discovered contract to validate

        Returns:
            Parsed ModelHandlerContract if valid, None if invalid
        """
        try:
            # Parse YAML
            data = yaml.safe_load(contract.text)
            if not isinstance(data, dict):
                return None

            # Validate against ModelHandlerContract
            parsed = ModelHandlerContract.model_validate(data)

            # Update contract with handler_id
            # Note: We can't update frozen model, but we have the parsed data

            return parsed

        except yaml.YAMLError:
            return None
        except ValidationError:
            return None

    def _create_validation_error(
        self,
        contract: ModelDiscoveredContract,
    ) -> ModelContractError | None:
        """Create validation error for a contract.

        Args:
            contract: Contract that failed validation

        Returns:
            ModelContractError describing the failure
        """
        try:
            # Try to determine specific error
            data = yaml.safe_load(contract.text)
            if not isinstance(data, dict):
                return ModelContractError(
                    contract_path=str(contract.ref),
                    handler_id=None,
                    error_type="yaml_parse",
                    message="Contract YAML must be a dictionary",
                )

            # Try Pydantic validation to get specific error
            try:
                ModelHandlerContract.model_validate(data)
                return None  # Should not reach here
            except ValidationError as e:
                # Extract first error
                first_error = e.errors()[0] if e.errors() else None
                if first_error:
                    field = ".".join(str(loc) for loc in first_error.get("loc", []))
                    msg = first_error.get("msg", "Validation failed")
                    return ModelContractError(
                        contract_path=str(contract.ref),
                        handler_id=data.get("handler_id"),
                        error_type="schema_validation",
                        message=f"Validation error in '{field}': {msg}",
                    )
                return ModelContractError(
                    contract_path=str(contract.ref),
                    handler_id=data.get("handler_id"),
                    error_type="schema_validation",
                    message="Contract validation failed",
                )

        except yaml.YAMLError as e:
            return ModelContractError(
                contract_path=str(contract.ref),
                handler_id=None,
                error_type="yaml_parse",
                message=f"Invalid YAML: {e}",
            )


__all__ = ["ServiceContractPublisher"]

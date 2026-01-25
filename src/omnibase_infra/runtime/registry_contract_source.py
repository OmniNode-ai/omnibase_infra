# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry-Based Contract Source using Consul KV.

This module provides RegistryContractSource for discovering handler contracts
from Consul KV storage instead of the filesystem.

Part of OMN-1100: Registry-Based Handler Contract Discovery.

Contract Storage Convention:
    - Prefix: onex/contracts/handlers/
    - Key format: onex/contracts/handlers/{handler_id}
    - Value: Full YAML contract content

Example Consul KV Structure:
    onex/contracts/handlers/effect.consul.handler → (YAML content)
    onex/contracts/handlers/effect.db.handler → (YAML content)
    onex/contracts/handlers/compute.auth.handler → (YAML content)

See Also:
    - HandlerContractSource: Filesystem-based discovery
    - ProtocolContractSource: Protocol definition

.. versionadded:: 0.7.0
    Created as part of OMN-1100 registry-based discovery.
"""

from __future__ import annotations

import logging
import os
from asyncio import to_thread
from typing import cast
from uuid import uuid4

import consul
import yaml
from pydantic import ValidationError

from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import (
    EnumHandlerErrorType,
    EnumHandlerSourceType,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraConnectionError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import (
    LiteralHandlerKind,
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
    ModelHandlerIdentifier,
)
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource

logger = logging.getLogger(__name__)

# Forward Reference Resolution:
# ModelContractDiscoveryResult uses a forward reference to ModelHandlerValidationError.
# Since we import ModelHandlerValidationError above, we can call model_rebuild() here
# to resolve the forward reference. This call is idempotent - multiple calls are harmless.
ModelContractDiscoveryResult.model_rebuild()

# Default Consul KV prefix for contract storage
DEFAULT_CONTRACT_PREFIX = "onex/contracts/handlers/"

# Default Consul connection settings
# NOTE: Standard Consul port is 8500. Production deployments typically
# override via CONSUL_PORT env var (e.g., 28500 per CLAUDE.md infrastructure).
DEFAULT_CONSUL_HOST = "localhost"
DEFAULT_CONSUL_PORT = 8500

# Maximum contract size (same as filesystem source)
MAX_CONTRACT_SIZE = 10 * 1024 * 1024  # 10MB


class RegistryContractSource(ProtocolContractSource):
    """Contract source that discovers handlers from Consul KV.

    This source fetches handler contracts stored in Consul KV under a
    configurable prefix, enabling remote/centralized contract management.

    Attributes:
        host: Consul host address.
        port: Consul port.
        token: Optional ACL token for authentication.
        prefix: KV prefix for contract storage.
        graceful_mode: If True, continue on errors and collect them.

    Example:
        >>> source = RegistryContractSource(
        ...     host="consul.example.com",
        ...     port=8500,
        ...     token="my-acl-token",
        ... )
        >>> result = await source.discover_handlers()
        >>> for desc in result.descriptors:
        ...     print(f"Found: {desc.handler_id}")

    .. versionadded:: 0.7.0
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        token: str | None = None,
        scheme: str | None = None,
        prefix: str = DEFAULT_CONTRACT_PREFIX,
        graceful_mode: bool = False,
    ) -> None:
        """Initialize the registry contract source.

        Args:
            host: Consul host address (default from CONSUL_HOST env var).
            port: Consul port (default from CONSUL_PORT env var).
            token: Optional ACL token (default from CONSUL_TOKEN env var).
            scheme: HTTP scheme (default from CONSUL_SCHEME env var, or "http").
            prefix: KV prefix for contracts.
            graceful_mode: If True, collect errors instead of raising.
        """
        # Configuration from environment variables (per CLAUDE.md)
        self._host = host or os.environ.get("CONSUL_HOST", DEFAULT_CONSUL_HOST)
        self._port = (
            port
            if port is not None
            else int(os.environ.get("CONSUL_PORT", str(DEFAULT_CONSUL_PORT)))
        )
        self._token = token or os.environ.get("CONSUL_TOKEN")
        self._scheme = scheme or os.environ.get("CONSUL_SCHEME", "http")
        self._prefix = prefix
        self._graceful_mode = graceful_mode
        self._correlation_id = uuid4()

        # Initialize Consul client with resolved configuration
        self._client = consul.Consul(
            host=self._host,
            port=self._port,
            token=self._token,
            scheme=self._scheme,
        )

        logger.info(
            "RegistryContractSource initialized",
            extra={
                "host": self._host,
                "port": self._port,
                "prefix": prefix,
                "correlation_id": str(self._correlation_id),
            },
        )

    @property
    def source_type(self) -> str:
        """Return source type identifier."""
        return "REGISTRY"

    async def discover_handlers(self) -> ModelContractDiscoveryResult:
        """Discover handlers from Consul KV.

        Fetches all keys under the configured prefix and parses each
        value as a YAML handler contract.

        Returns:
            ModelContractDiscoveryResult with descriptors and any errors.

        Raises:
            ModelOnexError: In strict mode, if discovery fails.
        """
        descriptors: list[ModelHandlerDescriptor] = []
        validation_errors: list[ModelHandlerValidationError] = []

        try:
            # Fetch all keys under prefix (recurse=True)
            # Wrap synchronous Consul client call to avoid blocking the event loop
            _index, data = await to_thread(
                self._client.kv.get, self._prefix, recurse=True
            )

            if data is None:
                logger.info(
                    "No contracts found in registry",
                    extra={
                        "prefix": self._prefix,
                        "correlation_id": str(self._correlation_id),
                    },
                )
                return ModelContractDiscoveryResult(
                    descriptors=descriptors,
                    validation_errors=validation_errors,
                )

            logger.info(
                "Found contracts in registry",
                extra={
                    "count": len(data),
                    "prefix": self._prefix,
                    "correlation_id": str(self._correlation_id),
                },
            )

            for item in data:
                key = item.get("Key", "")
                value = item.get("Value")

                # Skip the prefix directory key itself
                if key == self._prefix or key == self._prefix.rstrip("/"):
                    continue

                # Extract handler_id from key
                handler_id = key.removeprefix(self._prefix)
                if not handler_id:
                    continue

                try:
                    descriptor = self._parse_contract(
                        key=key,
                        value=value,
                        handler_id=handler_id,
                    )
                    if descriptor:
                        descriptors.append(descriptor)
                        logger.debug(
                            "Discovered handler from registry",
                            extra={
                                "handler_id": descriptor.handler_id,
                                "key": key,
                            },
                        )
                except (
                    yaml.YAMLError,
                    ValidationError,
                    ValueError,
                    ModelOnexError,
                ) as e:
                    error = self._create_parse_error(key, handler_id, e)
                    if self._graceful_mode:
                        validation_errors.append(error)
                        logger.warning(
                            "Failed to parse contract (graceful mode)",
                            extra={"key": key, "error": str(e)},
                        )
                    else:
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=self._correlation_id,
                            transport_type=EnumInfraTransportType.CONSUL,
                            operation="parse_contract",
                        )
                        raise ProtocolConfigurationError(
                            f"Failed to parse contract from registry key '{key}': {e}",
                            context=context,
                        ) from e

        except consul.ConsulException as e:
            error_msg = f"Consul connection failed: {e}"
            logger.exception(
                error_msg,
                extra={
                    "host": self._host,
                    "port": self._port,
                    "correlation_id": str(self._correlation_id),
                },
            )
            if not self._graceful_mode:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=self._correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="discover_handlers",
                )
                raise InfraConnectionError(
                    f"Consul connection failed: {e}",
                    context=context,
                ) from e
            validation_errors.append(self._create_connection_error(e))

        return ModelContractDiscoveryResult(
            descriptors=descriptors,
            validation_errors=validation_errors,
        )

    def _parse_contract(
        self,
        key: str,
        value: bytes | None,
        handler_id: str,
    ) -> ModelHandlerDescriptor | None:
        """Parse a contract from Consul KV value.

        Args:
            key: The Consul KV key.
            value: The raw bytes value from Consul.
            handler_id: Extracted handler ID from key.

        Returns:
            ModelHandlerDescriptor if valid, None if empty.

        Raises:
            ValueError: If contract is invalid.
        """
        if value is None:
            logger.debug(
                "Skipping contract with None value",
                extra={"key": key, "handler_id": handler_id},
            )
            return None

        # Check size limit
        if len(value) > MAX_CONTRACT_SIZE:
            raise ValueError(
                f"Contract exceeds size limit: {len(value)} > {MAX_CONTRACT_SIZE}"
            )

        # Decode and parse YAML
        content = value.decode("utf-8")
        contract_data = yaml.safe_load(content)

        if not contract_data:
            logger.debug(
                "Skipping contract with empty YAML content",
                extra={"key": key, "handler_id": handler_id},
            )
            return None

        # Validate against ModelHandlerContract
        contract = ModelHandlerContract.model_validate(contract_data)

        # Validate handler_id consistency between key and contract content.
        # NOTE: handler_id mismatch is NEVER silently ignored - it's always treated
        # as an error in both modes. The difference is how the error is surfaced:
        # - Strict mode: Raises ProtocolConfigurationError immediately
        # - Graceful mode: Collects in validation_errors (ValueError caught upstream)
        # This ensures data integrity issues are always reported, not swallowed.
        if contract.handler_id != handler_id:
            raise ValueError(
                f"handler_id mismatch: key='{handler_id}' vs "
                f"contract='{contract.handler_id}'"
            )

        # Extract handler_class from metadata (canonical location per contract schema)
        metadata = contract_data.get("metadata", {})
        handler_class = metadata.get("handler_class")

        if handler_class is None:
            logger.debug(
                "handler_class missing from metadata, handler may not be loadable",
                extra={"handler_id": handler_id, "key": key},
            )

        # Access node_archetype from the validated Pydantic model and cast to LiteralHandlerKind.
        # ModelHandlerBehavior.node_archetype is an EnumNodeArchetype enum.
        # Its .value property returns the handler_kind string ("compute", "effect", etc.).
        # The cast is type-safe: EnumNodeArchetype values exactly match LiteralHandlerKind.
        handler_kind = cast(
            "LiteralHandlerKind", contract.descriptor.node_archetype.value
        )

        return ModelHandlerDescriptor(
            handler_id=contract.handler_id,
            name=contract.name,
            # Use contract_version directly - it's already a ModelSemVer from Pydantic validation
            version=contract.contract_version,
            handler_kind=handler_kind,
            input_model=contract.input_model,
            output_model=contract.output_model,
            description=contract.description,
            handler_class=handler_class,
            contract_path=f"consul://{self._host}:{self._port}/{key}",
            contract_config=contract_data,
        )

    def _create_parse_error(
        self,
        key: str,
        handler_id: str,
        error: Exception,
    ) -> ModelHandlerValidationError:
        """Create a validation error for contract parse failures.

        Args:
            key: The Consul KV key.
            handler_id: The handler identifier.
            error: The parsing error.

        Returns:
            ModelHandlerValidationError with parse error details.
        """
        handler_identity = ModelHandlerIdentifier.from_handler_id(
            f"registry://{handler_id}"
        )

        return ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="REGISTRY-001",
            handler_identity=handler_identity,
            source_type=EnumHandlerSourceType.CONTRACT,
            message=f"Failed to parse contract from registry key '{key}': {error}",
            remediation_hint="Check YAML syntax and required contract fields",
            file_path=f"consul://{self._host}:{self._port}/{key}",
            correlation_id=self._correlation_id,
        )

    def _create_connection_error(
        self,
        error: consul.ConsulException,
    ) -> ModelHandlerValidationError:
        """Create a validation error for Consul connection failures.

        Args:
            error: The Consul exception.

        Returns:
            ModelHandlerValidationError with connection error details.
        """
        handler_identity = ModelHandlerIdentifier.from_handler_id("registry-source")

        return ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.DISCOVERY_ERROR,
            rule_id="REGISTRY-002",
            handler_identity=handler_identity,
            source_type=EnumHandlerSourceType.CONTRACT,
            message=f"Failed to connect to Consul registry at {self._host}:{self._port}: {error}",
            remediation_hint="Verify Consul is running and accessible",
            correlation_id=self._correlation_id,
        )


def _create_consul_client_from_env() -> consul.Consul:
    """Create Consul client using environment variables.

    Configuration is sourced from environment (per CLAUDE.md):
    - CONSUL_HOST: Host address (default: localhost)
    - CONSUL_PORT: Port number (default: 8500)
    - CONSUL_TOKEN: ACL token (default: None)
    - CONSUL_SCHEME: HTTP scheme (default: http)

    Returns:
        Configured Consul client instance.
    """
    return consul.Consul(
        host=os.environ.get("CONSUL_HOST", DEFAULT_CONSUL_HOST),
        port=int(os.environ.get("CONSUL_PORT", str(DEFAULT_CONSUL_PORT))),
        token=os.environ.get("CONSUL_TOKEN"),
        scheme=os.environ.get("CONSUL_SCHEME", "http"),
    )


def store_contract_in_consul(
    contract_yaml: str,
    handler_id: str,
    prefix: str = DEFAULT_CONTRACT_PREFIX,
    client: consul.Consul | None = None,
) -> bool:
    """Store a handler contract in Consul KV.

    This is a utility function for populating contracts in Consul,
    useful for demos and initial setup. Consul connection is configured
    via environment variables (CONSUL_HOST, CONSUL_PORT, etc.).

    Args:
        contract_yaml: The full YAML contract content.
        handler_id: Handler ID (used as key suffix).
        prefix: KV prefix (default: onex/contracts/handlers/).
        client: Optional Consul client. If None, creates a new client from
            environment variables.

    Returns:
        True if successful.

    Example:
        >>> contract = '''
        ... handler_id: effect.filesystem.handler
        ... name: Filesystem Handler
        ... version: "1.0.0"
        ... input_model: omnibase_infra.models.types.JsonDict
        ... output_model: omnibase_core.models.dispatch.ModelHandlerOutput
        ... descriptor:
        ...   handler_kind: effect
        ... '''
        >>> store_contract_in_consul(contract, "effect.filesystem.handler")
        True

    Note:
        This is a synchronous function that makes blocking network calls.
        Do not call from async code without wrapping in ``asyncio.to_thread()``.
        For async usage, use ``RegistryContractSource.discover_handlers()`` instead.

        For batch operations, pass an existing ``client`` to avoid creating
        a new connection for each call. This improves performance when
        storing multiple contracts.
    """
    client = client or _create_consul_client_from_env()
    key = f"{prefix}{handler_id}"

    success: bool = client.kv.put(key, contract_yaml)

    if success:
        logger.info(
            "Stored contract in Consul",
            extra={"key": key, "handler_id": handler_id},
        )
    else:
        logger.error(
            "Failed to store contract in Consul",
            extra={"key": key, "handler_id": handler_id},
        )

    return success


def list_contracts_in_consul(
    prefix: str = DEFAULT_CONTRACT_PREFIX,
    client: consul.Consul | None = None,
) -> list[str]:
    """List all contract keys in Consul KV.

    Consul connection is configured via environment variables
    (CONSUL_HOST, CONSUL_PORT, CONSUL_TOKEN, CONSUL_SCHEME).

    Args:
        prefix: KV prefix (default: onex/contracts/handlers/).
        client: Optional Consul client. If None, creates a new client from
            environment variables.

    Returns:
        List of handler IDs found.

    Note:
        This is a synchronous function that makes blocking network calls.
        Do not call from async code without wrapping in ``asyncio.to_thread()``.

        For batch operations, pass an existing ``client`` to avoid creating
        a new connection for each call. This improves performance when
        listing and then fetching multiple contracts.
    """
    client = client or _create_consul_client_from_env()
    _index, keys = client.kv.get(prefix, keys=True)

    if keys is None:
        return []

    handler_ids = [
        k.removeprefix(prefix) for k in keys if k != prefix and k != prefix.rstrip("/")
    ]

    return handler_ids


def delete_contract_from_consul(
    handler_id: str,
    prefix: str = DEFAULT_CONTRACT_PREFIX,
    client: consul.Consul | None = None,
) -> bool:
    """Delete a handler contract from Consul KV.

    This is a utility function for removing contracts from Consul,
    useful for cleanup and testing. Consul connection is configured
    via environment variables (CONSUL_HOST, CONSUL_PORT, etc.).

    Args:
        handler_id: Handler ID (used as key suffix).
        prefix: KV prefix (default: onex/contracts/handlers/).
        client: Optional Consul client. If None, creates a new client from
            environment variables.

    Returns:
        True if successful.

    Example:
        >>> delete_contract_from_consul("effect.filesystem.handler")
        True

    Note:
        This is a synchronous function that makes blocking network calls.
        Do not call from async code without wrapping in ``asyncio.to_thread()``.

        For batch operations, pass an existing ``client`` to avoid creating
        a new connection for each call. This improves performance when
        deleting multiple contracts.
    """
    client = client or _create_consul_client_from_env()
    key = f"{prefix}{handler_id}"

    success: bool = client.kv.delete(key)

    if success:
        logger.info(
            "Deleted contract from Consul",
            extra={"key": key, "handler_id": handler_id},
        )
    else:
        logger.error(
            "Failed to delete contract from Consul",
            extra={"key": key, "handler_id": handler_id},
        )

    return success


async def astore_contract_in_consul(
    contract_yaml: str,
    handler_id: str,
    prefix: str = DEFAULT_CONTRACT_PREFIX,
    client: consul.Consul | None = None,
) -> bool:
    """Async variant of store_contract_in_consul.

    Wraps the synchronous function in asyncio.to_thread() for non-blocking
    execution in async contexts.

    Args:
        contract_yaml: The full YAML contract content.
        handler_id: Handler ID (used as key suffix).
        prefix: KV prefix (default: onex/contracts/handlers/).
        client: Optional Consul client.

    Returns:
        True if successful.
    """
    return await to_thread(
        store_contract_in_consul,
        contract_yaml,
        handler_id,
        prefix,
        client,
    )


async def alist_contracts_in_consul(
    prefix: str = DEFAULT_CONTRACT_PREFIX,
    client: consul.Consul | None = None,
) -> list[str]:
    """Async variant of list_contracts_in_consul.

    Wraps the synchronous function in asyncio.to_thread() for non-blocking
    execution in async contexts.

    Args:
        prefix: KV prefix (default: onex/contracts/handlers/).
        client: Optional Consul client.

    Returns:
        List of handler IDs found.
    """
    return await to_thread(
        list_contracts_in_consul,
        prefix,
        client,
    )


async def adelete_contract_from_consul(
    handler_id: str,
    prefix: str = DEFAULT_CONTRACT_PREFIX,
    client: consul.Consul | None = None,
) -> bool:
    """Async variant of delete_contract_from_consul.

    Wraps the synchronous function in asyncio.to_thread() for non-blocking
    execution in async contexts.

    Args:
        handler_id: Handler ID (used as key suffix).
        prefix: KV prefix (default: onex/contracts/handlers/).
        client: Optional Consul client.

    Returns:
        True if successful.
    """
    return await to_thread(
        delete_contract_from_consul,
        handler_id,
        prefix,
        client,
    )


__all__ = [
    "DEFAULT_CONSUL_HOST",
    "DEFAULT_CONSUL_PORT",
    "DEFAULT_CONTRACT_PREFIX",
    "RegistryContractSource",
    "adelete_contract_from_consul",
    "alist_contracts_in_consul",
    "astore_contract_in_consul",
    "delete_contract_from_consul",
    "list_contracts_in_consul",
    "store_contract_in_consul",
]

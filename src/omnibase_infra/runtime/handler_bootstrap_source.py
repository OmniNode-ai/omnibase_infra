# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Bootstrap Source for Hardcoded Handler Registration.

This module provides HandlerBootstrapSource, which centralizes all hardcoded handler
wiring that was previously scattered in util_wiring.py. This source implements
ProtocolContractSource and provides handler descriptors for the core infrastructure
handlers (Consul, Database, HTTP, Vault) without requiring contract.yaml files.

Part of OMN-1087: HandlerBootstrapSource for hardcoded handler registration.

The bootstrap source provides handler descriptors for effect handlers that interact
with external infrastructure services. These handlers use envelope-based routing
and are registered as the foundation of the ONEX runtime handler ecosystem.

Registered Handlers:
    - consul: HandlerConsul for HashiCorp Consul service discovery
    - db: HandlerDb for PostgreSQL database operations
    - http: HandlerHttpRest for HTTP/REST protocol operations
    - vault: HandlerVault for HashiCorp Vault secret management

All handlers are registered with handler_kind="effect" as they perform external I/O
operations with infrastructure services.

See Also:
    - ProtocolContractSource: Protocol definition for handler sources
    - HandlerContractSource: Filesystem-based contract discovery source
    - util_wiring: Module that previously contained hardcoded handler wiring
    - ModelHandlerDescriptor: Descriptor model for discovered handlers

.. versionadded:: 0.6.4
    Created as part of OMN-1087 bootstrap handler registration.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TypedDict

from omnibase_infra.models.handlers import (
    LiteralHandlerKind,
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
)
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource


class BootstrapEffectDefinition(TypedDict):
    """Type definition for bootstrap effect node configuration entries.

    This TypedDict provides compile-time type safety for the hardcoded effect
    definitions, ensuring kind values are correctly typed as LiteralHandlerKind
    rather than generic str. This eliminates the need for type: ignore comments
    when constructing ModelHandlerDescriptor instances.
    """

    handler_id: str
    name: str
    description: str
    handler_kind: LiteralHandlerKind
    handler_class: str
    input_model: str
    output_model: str


# Thread-safe initialization for model rebuild
# Lock ensures only one thread performs the rebuild
_model_rebuild_lock = threading.Lock()

# Mutable container to track if model_rebuild() has been called
# Using a list avoids the need for global statement (PLW0603)
_model_rebuild_state: list[bool] = [False]


def _ensure_model_rebuilt() -> None:
    """Ensure ModelContractDiscoveryResult has resolved forward references.

    This must be called before creating ModelContractDiscoveryResult instances.
    It's deferred from module load time to avoid circular import issues when
    this module is imported through the runtime.__init__.py chain.

    The rebuild resolves the forward reference to ModelHandlerValidationError
    in the validation_errors field of ModelContractDiscoveryResult.

    Thread Safety:
        Uses double-checked locking pattern to ensure thread-safe initialization
        while minimizing lock contention after the first successful rebuild.
    """
    # Fast path - already rebuilt (no lock needed)
    if _model_rebuild_state[0]:
        return

    # Thread-safe initialization with double-checked locking
    with _model_rebuild_lock:
        # Re-check after acquiring lock (another thread may have completed rebuild)
        if _model_rebuild_state[0]:
            return

        # Import ModelHandlerValidationError here to avoid circular import at module load
        from omnibase_infra.models.errors import ModelHandlerValidationError

        # Rebuild the model to resolve forward references
        # This needs ModelHandlerValidationError in scope for Pydantic to resolve the type
        ModelContractDiscoveryResult.model_rebuild()

        # Suppress unused variable warning - the import is needed for model_rebuild()
        _ = ModelHandlerValidationError
        _model_rebuild_state[0] = True


logger = logging.getLogger(__name__)

# Source type identifier for bootstrap handlers
SOURCE_TYPE_BOOTSTRAP = "BOOTSTRAP"

# Handler type constants (matching handler_registry.py)
_HANDLER_TYPE_CONSUL = "consul"
_HANDLER_TYPE_DATABASE = "db"
_HANDLER_TYPE_HTTP = "http"
_HANDLER_TYPE_VAULT = "vault"

# Bootstrap handler definitions.
#
# Each entry contains the metadata needed to create a ModelHandlerDescriptor:
#   handler_id: Unique identifier with "bootstrap." prefix
#   name: Human-readable display name
#   description: Handler purpose description
#   handler_kind: ONEX handler archetype (all are "effect" for I/O handlers)
#   handler_class: Fully qualified Python class path for dynamic import
#   input_model: Fully qualified path to input type (envelope-based handlers use JsonDict)
#   output_model: Fully qualified path to output type (all handlers return ModelHandlerOutput)
#
# Design Note (handler_class vs handler_module):
#   ModelHandlerDescriptor uses a single handler_class field with the fully qualified
#   path (e.g., "module.path.ClassName") rather than separate handler_module and
#   handler_class fields. This follows the standard Python import convention and
#   avoids redundancy. The runtime extracts module/class via rsplit(".", 1):
#       module_path, class_name = handler_class.rsplit(".", 1)
#   See: handler_plugin_loader.py::_import_handler_class() for implementation.
#
# These handlers are the core infrastructure handlers that support envelope-based
# routing patterns for external service integration.
#
# The BootstrapEffectDefinition TypedDict ensures handler_kind is typed as LiteralHandlerKind,
# providing compile-time type safety for the hardcoded values.
_BOOTSTRAP_HANDLER_DEFINITIONS: list[BootstrapEffectDefinition] = [
    {
        "handler_id": f"bootstrap.{_HANDLER_TYPE_CONSUL}",
        "name": "Consul Handler",
        "description": "HashiCorp Consul service discovery handler",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_consul.HandlerConsul",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
    },
    {
        "handler_id": f"bootstrap.{_HANDLER_TYPE_DATABASE}",
        "name": "Database Handler",
        "description": "PostgreSQL database handler",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_db.HandlerDb",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
    },
    {
        "handler_id": f"bootstrap.{_HANDLER_TYPE_HTTP}",
        "name": "HTTP Handler",
        "description": "HTTP REST protocol handler",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_http.HandlerHttpRest",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
    },
    {
        "handler_id": f"bootstrap.{_HANDLER_TYPE_VAULT}",
        "name": "Vault Handler",
        "description": "HashiCorp Vault secret management handler",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_vault.HandlerVault",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
    },
]

# Version for all bootstrap handlers (hardcoded handlers use stable version)
_BOOTSTRAP_HANDLER_VERSION = "1.0.0"


class HandlerBootstrapSource(
    ProtocolContractSource
):  # naming-ok: Handler prefix required by ProtocolHandlerSource convention
    """Handler source that provides hardcoded bootstrap handler descriptors.

    This class implements ProtocolContractSource by returning predefined handler
    descriptors for core infrastructure handlers. Unlike HandlerContractSource
    which discovers handlers from filesystem contracts, this source provides
    handlers that are essential for the ONEX runtime bootstrap process.

    Protocol Compliance:
        This class explicitly inherits from ProtocolContractSource and implements
        all required protocol methods: discover_handlers() async method and
        source_type property. Protocol compliance is verified at runtime through
        Python's structural subtyping and enforced by type checkers.

    API Consistency Note:
        The ``graceful_mode`` parameter is accepted but **currently unused**.
        Bootstrap handlers are hardcoded definitions that cannot fail validation,
        so error collection is never needed. The parameter exists solely to
        maintain interface compatibility with :class:`HandlerContractSource`
        and :class:`ProtocolContractSource`, enabling callers to use these
        sources interchangeably without conditional parameter handling.

    Attributes:
        source_type: Returns "BOOTSTRAP" as the source type identifier.

    Example:
        >>> source = HandlerBootstrapSource()
        >>> result = await source.discover_handlers()
        >>> print(f"Found {len(result.descriptors)} bootstrap handlers")
        Found 4 bootstrap handlers
        >>> for desc in result.descriptors:
        ...     print(f"  - {desc.handler_id}: {desc.description}")
        - bootstrap.consul: HashiCorp Consul service discovery handler
        - bootstrap.db: PostgreSQL database handler
        - bootstrap.http: HTTP REST protocol handler
        - bootstrap.vault: HashiCorp Vault secret management handler

    Performance Characteristics:
        - No filesystem or network I/O required
        - Constant time O(1) discovery (hardcoded definitions)
        - Typical performance: <1ms for all handlers
        - Memory: ~500 bytes per handler descriptor

    .. versionadded:: 0.6.4
        Created as part of OMN-1087 bootstrap handler registration.
    """

    def __init__(
        self,
        graceful_mode: bool = False,
    ) -> None:
        """Initialize the bootstrap handler source.

        Args:
            graceful_mode: Error handling mode for API consistency.

                **Note**: This parameter is currently unused. Bootstrap handlers
                are hardcoded definitions that cannot fail validation, so error
                collection logic is never invoked. The parameter is retained
                solely to maintain a consistent interface with
                :class:`ProtocolContractSource` implementations, enabling
                interchangeable use with :class:`HandlerContractSource`.

                When True, would collect errors and continue discovery.
                When False (default), would raise on first error.
        """
        # Stored for API consistency only - bootstrap handlers cannot fail validation
        self._graceful_mode = graceful_mode

    @property
    def source_type(self) -> str:
        """Return the source type identifier.

        Returns:
            "BOOTSTRAP" as the source type.
        """
        return SOURCE_TYPE_BOOTSTRAP

    async def discover_handlers(
        self,
    ) -> ModelContractDiscoveryResult:
        """Discover bootstrap handler descriptors.

        Returns predefined handler descriptors for core infrastructure handlers.
        Unlike filesystem-based discovery, this method returns hardcoded
        definitions that are essential for ONEX runtime bootstrap.

        Returns:
            ModelContractDiscoveryResult containing bootstrap handler descriptors.
            The validation_errors list will always be empty since bootstrap
            handlers are hardcoded and validated at development time.

        Note:
            This method is idempotent and can be called multiple times safely.
            Each call returns the same set of handler descriptors.
        """
        # Ensure forward references are resolved before creating result
        _ensure_model_rebuilt()

        start_time = time.perf_counter()
        descriptors: list[ModelHandlerDescriptor] = []

        logger.debug(
            "Starting bootstrap handler discovery",
            extra={
                "source_type": SOURCE_TYPE_BOOTSTRAP,
                "graceful_mode": self._graceful_mode,
                "expected_handler_count": len(_BOOTSTRAP_HANDLER_DEFINITIONS),
            },
        )

        # Create descriptors from hardcoded definitions
        for handler_def in _BOOTSTRAP_HANDLER_DEFINITIONS:
            descriptor = ModelHandlerDescriptor(
                handler_id=handler_def["handler_id"],
                name=handler_def["name"],
                version=_BOOTSTRAP_HANDLER_VERSION,
                handler_kind=handler_def["handler_kind"],
                input_model=handler_def["input_model"],
                output_model=handler_def["output_model"],
                description=handler_def["description"],
                handler_class=handler_def["handler_class"],
                contract_path=None,  # No contract file for bootstrap handlers
            )
            descriptors.append(descriptor)

            logger.debug(
                "Created bootstrap handler descriptor",
                extra={
                    "handler_id": descriptor.handler_id,
                    "handler_name": descriptor.name,
                    "handler_kind": descriptor.handler_kind,
                    "source_type": SOURCE_TYPE_BOOTSTRAP,
                },
            )

        # Calculate duration and log results
        duration_seconds = time.perf_counter() - start_time
        self._log_discovery_results(len(descriptors), duration_seconds)

        return ModelContractDiscoveryResult(
            descriptors=descriptors,
            validation_errors=[],  # Bootstrap handlers have no validation errors
        )

    def _log_discovery_results(
        self,
        discovered_count: int,
        duration_seconds: float,
    ) -> None:
        """Log the discovery results with structured counts and timing.

        Args:
            discovered_count: Number of successfully discovered handlers.
            duration_seconds: Total discovery duration in seconds.
        """
        handlers_per_sec = (
            discovered_count / duration_seconds
            if duration_seconds > 0
            else float("inf")
            if discovered_count > 0
            else 0.0
        )

        logger.info(
            "Bootstrap handler discovery completed: "
            "discovered_handler_count=%d, "
            "duration_seconds=%.6f, handlers_per_second=%.1f",
            discovered_count,
            duration_seconds,
            handlers_per_sec,
            extra={
                "discovered_handler_count": discovered_count,
                "validation_failure_count": 0,
                "source_type": SOURCE_TYPE_BOOTSTRAP,
                "graceful_mode": self._graceful_mode,
                "duration_seconds": duration_seconds,
                "handlers_per_second": handlers_per_sec,
            },
        )


__all__ = [
    "HandlerBootstrapSource",
    "SOURCE_TYPE_BOOTSTRAP",
]

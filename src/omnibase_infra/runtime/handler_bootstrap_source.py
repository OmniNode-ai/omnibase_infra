# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Bootstrap Source for Hardcoded Handler Registration.

This module provides HandlerBootstrapSource, which centralizes all hardcoded handler
wiring that was previously scattered in util_wiring.py. This source implements
ProtocolContractSource and provides handler descriptors for the core infrastructure
handlers (Consul, Database, HTTP, Vault, MCP).

Part of OMN-1087: HandlerBootstrapSource for hardcoded handler registration.

The bootstrap source provides handler descriptors for effect handlers that interact
with external infrastructure services. These handlers use envelope-based routing
and are registered as the foundation of the ONEX runtime handler ecosystem.

Contract Loading:
    Bootstrap handlers now load their contract YAML files during discovery.
    This enables:
    - Security metadata (trusted_namespace, audit_logging, requires_authentication)
    - Tags for handler discovery and filtering
    - Validation that contract files exist and are well-formed

    Contract file locations (relative to repo root):
    - contracts/handlers/consul/handler_contract.yaml (basic contract)
    - contracts/handlers/db/handler_contract.yaml (basic contract)
    - contracts/handlers/http/handler_contract.yaml (basic contract)
    - contracts/handlers/vault/handler_contract.yaml (basic contract)
    - src/omnibase_infra/contracts/handlers/mcp/handler_contract.yaml (rich contract with transport config)

    Basic contracts provide: name, handler_class, handler_type, tags, security
    Rich contracts (MCP only) additionally provide: descriptor, metadata.transport, metadata.security

    If a contract file is missing or invalid, discovery fails fast with
    ProtocolConfigurationError. This ensures bootstrap handlers always have
    valid configuration.

Registered Handlers:
    - consul: HandlerConsul for HashiCorp Consul service discovery
    - db: HandlerDb for PostgreSQL database operations
    - http: HandlerHttpRest for HTTP/REST protocol operations
    - vault: HandlerVault for HashiCorp Vault secret management
    - mcp: HandlerMCP for Model Context Protocol AI agent integration

All handlers are registered with handler_kind="effect" as they perform external I/O
operations with infrastructure services.

See Also:
    - ProtocolContractSource: Protocol definition for handler sources
    - HandlerContractSource: Filesystem-based contract discovery source
    - handler_contract_config_loader: Loads and parses handler contract YAML files
    - util_wiring: Module that previously contained hardcoded handler wiring
    - ModelHandlerDescriptor: Descriptor model for discovered handlers

.. versionadded:: 0.6.4
    Created as part of OMN-1087 bootstrap handler registration.

.. versionchanged:: 0.6.5
    Added contract loading support. Bootstrap handlers now load and validate
    their contract YAML files during discovery, populating contract_config
    in the descriptor.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TypedDict, final

from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.handlers import (
    LiteralHandlerKind,
    ModelBootstrapHandlerDescriptor,
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
)
from omnibase_infra.runtime.handler_contract_config_loader import (
    extract_handler_config,
    load_handler_contract_config,
)
from omnibase_infra.runtime.handler_identity import handler_identity
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource


class BootstrapEffectDefinition(TypedDict):
    """Type definition for bootstrap effect node configuration entries.

    This TypedDict provides compile-time type safety for the hardcoded effect
    definitions, ensuring kind values are correctly typed as LiteralHandlerKind
    rather than generic str. This eliminates the need for type: ignore comments
    when constructing ModelBootstrapHandlerDescriptor instances.

    Note that handler_class is a required field here, matching the
    ModelBootstrapHandlerDescriptor requirement that bootstrap handlers
    must always specify their implementation class.

    Attributes:
        handler_id: Unique identifier with "proto." prefix (protocol identity namespace).
        name: Human-readable display name.
        description: Handler purpose description.
        handler_kind: ONEX handler archetype (all are "effect" for I/O handlers).
        handler_class: Fully qualified Python class path for dynamic import.
        input_model: Fully qualified path to input type.
        output_model: Fully qualified path to output type.
        contract_path: Relative path to handler_contract.yaml from repo root.
    """

    handler_id: str
    name: str
    description: str
    handler_kind: LiteralHandlerKind
    handler_class: str
    input_model: str
    output_model: str
    contract_path: str


# =============================================================================
# Thread-Safe Model Rebuild Pattern (Safety Net)
# =============================================================================
#
# ModelContractDiscoveryResult.model_rebuild() is now called CENTRALLY in
# omnibase_infra.models.handlers.__init__ to resolve forward references.
# This ensures the forward reference to ModelHandlerValidationError is resolved
# as soon as the handlers package is imported.
#
# This module retains a DEFERRED, thread-safe model_rebuild() call as a SAFETY NET:
#   - model_rebuild() is idempotent - multiple calls are harmless
#   - The flag-guarded pattern ensures at most one rebuild per process
#   - This provides fallback protection if import order changes in the future
#
# WHY THREAD-SAFE (historical context):
#   - discover_handlers() may be called concurrently from multiple threads
#   - Unlike module-level code (which Python imports once, thread-safely),
#     runtime-invoked code needs explicit synchronization
#   - The double-checked locking pattern minimizes lock contention
#
# See Also:
#   - omnibase_infra.models.handlers.__init__: Central model_rebuild() location
#   - OMN-1087 for the ticket tracking this design decision
# =============================================================================

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

    Why Deferred (Not Module-Level):
        Unlike HandlerContractSource which uses module-level model_rebuild(),
        this module is imported early in the runtime bootstrap chain before all
        model dependencies are resolved. Deferring the rebuild to first use
        avoids circular import failures.

    Thread Safety:
        Uses double-checked locking pattern to ensure thread-safe initialization
        while minimizing lock contention after the first successful rebuild.
        This is necessary because discover_handlers() may be called from multiple
        threads, unlike module-level code which Python imports once.

    See Also:
        handler_contract_source.py for the simpler immediate pattern used when
        import order constraints don't apply.
    """
    # Fast path - already rebuilt (no lock needed)
    if _model_rebuild_state[0]:
        return

    # Thread-safe initialization with double-checked locking
    with _model_rebuild_lock:
        # Re-check after acquiring lock (another thread may have completed rebuild)
        if _model_rebuild_state[0]:
            return

        # Import ModelHandlerValidationError here to avoid circular import at module load.
        # This import MUST be in scope when model_rebuild() is called, as Pydantic uses
        # the local namespace to resolve forward references in the validation_errors field.
        from omnibase_infra.models.errors import ModelHandlerValidationError

        # Rebuild the model to resolve forward references.
        # If this fails, provide a clear error message rather than obscure Pydantic errors.
        try:
            ModelContractDiscoveryResult.model_rebuild()
        except Exception as e:
            raise RuntimeError(
                f"Failed to rebuild ModelContractDiscoveryResult during bootstrap "
                f"initialization. This typically indicates a circular import or missing "
                f"type definition: {e}"
            ) from e

        # Keep import reference in scope - required for Pydantic forward reference resolution
        _ = ModelHandlerValidationError
        _model_rebuild_state[0] = True


logger = logging.getLogger(__name__)

# Source type identifier for bootstrap handlers
SOURCE_TYPE_BOOTSTRAP = "BOOTSTRAP"

# Handler type constants (matching handler_registry.py)
_HANDLER_TYPE_CONSUL = "consul"
_HANDLER_TYPE_DATABASE = "db"
_HANDLER_TYPE_HTTP = "http"
_HANDLER_TYPE_MCP = "mcp"
_HANDLER_TYPE_VAULT = "vault"

# Bootstrap handler definitions.
#
# Each entry contains the metadata needed to create a ModelHandlerDescriptor:
#   handler_id: Unique identifier with "proto." prefix (protocol identity namespace)
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
        "handler_id": handler_identity(_HANDLER_TYPE_CONSUL),
        "name": "Consul Handler",
        "description": "HashiCorp Consul service discovery handler",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_consul.HandlerConsul",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
        "contract_path": "contracts/handlers/consul/handler_contract.yaml",
    },
    {
        "handler_id": handler_identity(_HANDLER_TYPE_DATABASE),
        "name": "Database Handler",
        "description": "PostgreSQL database handler",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_db.HandlerDb",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
        "contract_path": "contracts/handlers/db/handler_contract.yaml",
    },
    {
        "handler_id": handler_identity(_HANDLER_TYPE_HTTP),
        "name": "HTTP Handler",
        "description": "HTTP REST protocol handler",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_http.HandlerHttpRest",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
        "contract_path": "contracts/handlers/http/handler_contract.yaml",
    },
    {
        "handler_id": handler_identity(_HANDLER_TYPE_VAULT),
        "name": "Vault Handler",
        "description": "HashiCorp Vault secret management handler",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_vault.HandlerVault",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
        "contract_path": "contracts/handlers/vault/handler_contract.yaml",
    },
    {
        "handler_id": handler_identity(_HANDLER_TYPE_MCP),
        "name": "MCP Handler",
        "description": "Model Context Protocol handler for AI agent integration",
        "handler_kind": "effect",
        "handler_class": "omnibase_infra.handlers.handler_mcp.HandlerMCP",
        "input_model": "omnibase_infra.models.types.JsonDict",
        "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
        "contract_path": "src/omnibase_infra/contracts/handlers/mcp/handler_contract.yaml",
    },
]

# Version for all bootstrap handlers (hardcoded handlers use stable version)
_BOOTSTRAP_HANDLER_VERSION = ModelSemVer(major=1, minor=0, patch=0)


@final
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

    Attributes:
        source_type: Returns "BOOTSTRAP" as the source type identifier.

    Example:
        >>> source = HandlerBootstrapSource()
        >>> result = await source.discover_handlers()
        >>> print(f"Found {len(result.descriptors)} bootstrap handlers")
        Found 5 bootstrap handlers
        >>> for desc in result.descriptors:
        ...     print(f"  - {desc.handler_id}: {desc.description}")
        - proto.consul: HashiCorp Consul service discovery handler
        - proto.db: PostgreSQL database handler
        - proto.http: HTTP REST protocol handler
        - proto.mcp: Model Context Protocol handler for AI agent integration
        - proto.vault: HashiCorp Vault secret management handler

    Performance Characteristics:
        - No filesystem or network I/O required
        - Constant time O(1) discovery (hardcoded definitions)
        - Typical performance: <1ms for all handlers (local dev)
        - Test threshold: 100ms (generous for CI runner variance)
        - Memory: ~500 bytes per handler descriptor

    .. versionadded:: 0.6.4
        Created as part of OMN-1087 bootstrap handler registration.
    """

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

        Implementation Note:
            Uses ModelBootstrapHandlerDescriptor (which requires handler_class)
            for construction validation, ensuring all bootstrap handlers have
            the required handler_class field. The descriptors are instances
            of ModelHandlerDescriptor due to inheritance.
        """
        # Ensure forward references are resolved before creating result
        _ensure_model_rebuilt()

        start_time = time.perf_counter()
        descriptors: list[ModelHandlerDescriptor] = []

        logger.debug(
            "Starting bootstrap handler discovery",
            extra={
                "source_type": SOURCE_TYPE_BOOTSTRAP,
                "expected_handler_count": len(_BOOTSTRAP_HANDLER_DEFINITIONS),
            },
        )

        # Create descriptors from hardcoded definitions
        # Uses ModelBootstrapHandlerDescriptor to enforce handler_class requirement
        for handler_def in _BOOTSTRAP_HANDLER_DEFINITIONS:
            contract_path = handler_def["contract_path"]
            contract_config = None

            # Load contract configuration if path is specified
            try:
                contract = load_handler_contract_config(
                    contract_path,
                    handler_def["handler_id"],
                )
                handler_type = handler_def["handler_id"].split(".")[-1]
                # Bootstrap handlers get handler_class from _BOOTSTRAP_HANDLER_DEFINITIONS,
                # not from the contract file. Rich contracts (like MCP) and basic contracts
                # don't include handler_class since it would be redundant with the definition.
                contract_config = extract_handler_config(
                    contract, handler_type, require_basic_fields=False
                )
                logger.debug(
                    "Loaded contract config for bootstrap handler",
                    extra={
                        "handler_id": handler_def["handler_id"],
                        "contract_path": contract_path,
                        "config_keys": list(contract_config.keys()),
                    },
                )
            except ProtocolConfigurationError:
                # Fail fast for bootstrap handlers - contracts must exist
                logger.exception(
                    "Failed to load contract config for bootstrap handler",
                    extra={
                        "handler_id": handler_def["handler_id"],
                        "contract_path": contract_path,
                    },
                )
                raise

            descriptor = ModelBootstrapHandlerDescriptor(
                handler_id=handler_def["handler_id"],
                name=handler_def["name"],
                version=_BOOTSTRAP_HANDLER_VERSION,
                handler_kind=handler_def["handler_kind"],
                input_model=handler_def["input_model"],
                output_model=handler_def["output_model"],
                description=handler_def["description"],
                handler_class=handler_def["handler_class"],
                contract_path=contract_path,
                contract_config=contract_config,
            )
            descriptors.append(descriptor)

            logger.debug(
                "Created bootstrap handler descriptor",
                extra={
                    "handler_id": descriptor.handler_id,
                    "handler_name": descriptor.name,
                    "handler_kind": descriptor.handler_kind,
                    "contract_path": descriptor.contract_path,
                    "has_contract_config": descriptor.contract_config is not None,
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
        # Cap handlers_per_sec at 1M to avoid float("inf") which can cause issues
        # in downstream logging/monitoring systems expecting finite numbers.
        # A value of 1M represents "effectively instant" discovery.
        if duration_seconds > 0:
            handlers_per_sec = discovered_count / duration_seconds
        elif discovered_count > 0:
            handlers_per_sec = 1_000_000.0  # Cap for instant discovery
        else:
            handlers_per_sec = 0.0

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
                "duration_seconds": duration_seconds,
                "handlers_per_second": handlers_per_sec,
            },
        )


__all__ = [
    "HandlerBootstrapSource",
    "SOURCE_TYPE_BOOTSTRAP",
]

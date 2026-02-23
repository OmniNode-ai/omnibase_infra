# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Transport type to Infisical path mapping.

Maps each ``EnumInfraTransportType`` to its expected Infisical folder path
and configuration keys. Supports both shared config (``/shared/<transport>/``)
and per-service config (``/services/<service>/<transport>/``).

Path Convention:
    Shared:      ``/shared/<transport>/KEY``
    Per-service: ``/services/<service>/<transport>/KEY``

The ``<transport>`` segment is the enum value of ``EnumInfraTransportType``,
not its name.  For example, ``DATABASE`` has value ``"db"``, so its shared
path is ``/shared/db/KEY``.

Multiple instances of the same transport are handled via service namespacing.
For example, two PostgreSQL connections (one for the main runtime, one for
intelligence) would live at:
    ``/services/omnibase-runtime/db/POSTGRES_DATABASE``
    ``/services/omniintelligence/db/POSTGRES_DATABASE``

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.models.model_transport_config_spec import (
    ModelTransportConfigSpec,
)

logger = logging.getLogger(__name__)

# Canonical configuration keys per transport type.
# These represent the standard keys each transport expects in Infisical.
_TRANSPORT_KEYS: dict[EnumInfraTransportType, tuple[str, ...]] = {
    EnumInfraTransportType.DATABASE: (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_POOL_MIN_SIZE",
        "POSTGRES_POOL_MAX_SIZE",
        "POSTGRES_TIMEOUT_MS",
        "QUERY_TIMEOUT_SECONDS",
    ),
    EnumInfraTransportType.KAFKA: (
        "KAFKA_GROUP_ID",
        "KAFKA_ACKS",
        "KAFKA_REQUEST_TIMEOUT_MS",
    ),
    EnumInfraTransportType.CONSUL: (
        "CONSUL_HOST",
        "CONSUL_PORT",
        "CONSUL_SCHEME",
        "CONSUL_ACL_TOKEN",
        "CONSUL_ENABLED",
    ),
    # INFISICAL is listed here for completeness (so introspection tools can
    # enumerate all known keys), but these keys MUST NEVER be fetched from
    # Infisical — they form the bootstrap credentials that Infisical itself
    # needs to start.  Fetching them from Infisical would create a circular
    # dependency.
    #
    # EnumInfraTransportType.INFISICAL is in _BOOTSTRAP_TRANSPORTS, so
    # specs_for_transports() and all_shared_specs() already skip it.
    # If you call keys_for_transport(INFISICAL) directly, check
    # is_bootstrap_transport(INFISICAL) first and resolve from the
    # environment (e.g. .env file), NOT from Infisical.
    EnumInfraTransportType.INFISICAL: (
        "INFISICAL_ADDR",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
    ),
    EnumInfraTransportType.VALKEY: (
        "VALKEY_HOST",
        "VALKEY_PORT",
        "VALKEY_PASSWORD",
        "VALKEY_DB",
    ),
    EnumInfraTransportType.HTTP: (
        "HTTP_BASE_URL",
        "HTTP_TIMEOUT_MS",
        "HTTP_MAX_RETRIES",
    ),
    EnumInfraTransportType.LLM: (
        "REMOTE_SERVER_IP",
        "LLM_CODER_URL",
        "LLM_CODER_FAST_URL",
        "LLM_EMBEDDING_URL",
        "LLM_DEEPSEEK_R1_URL",
        "EMBEDDING_MODEL_URL",
        "VLLM_SERVICE_URL",
        "VLLM_DEEPSEEK_URL",
        "VLLM_LLAMA_URL",
        "ONEX_TREE_SERVICE_URL",
        "METADATA_STAMPING_SERVICE_URL",
    ),
    EnumInfraTransportType.GRPC: (
        "GRPC_HOST",
        "GRPC_PORT",
        "GRPC_TLS_ENABLED",
    ),
    EnumInfraTransportType.MCP: (
        "MCP_SERVER_HOST",
        "MCP_SERVER_PORT",
    ),
    EnumInfraTransportType.FILESYSTEM: ("FS_BASE_PATH",),
    EnumInfraTransportType.QDRANT: (
        "QDRANT_HOST",
        "QDRANT_PORT",
        "QDRANT_API_KEY",
        "QDRANT_URL",
    ),
    EnumInfraTransportType.GRAPH: (
        "GRAPH_HOST",
        "GRAPH_PORT",
        "GRAPH_PROTOCOL",
    ),
    EnumInfraTransportType.INMEMORY: (),
    EnumInfraTransportType.RUNTIME: (),
}


class TransportConfigMap:
    """Maps transport types to Infisical folder paths and expected keys.

    This class provides the canonical mapping between ONEX transport types
    and their Infisical storage layout. It is stateless and deterministic.

    Note:
        Naming convention: CLAUDE.md defines ``Service<Name>`` for service
        classes.  This class predates that convention and is named
        ``TransportConfigMap`` rather than ``ServiceTransportConfigMap``.
        Renaming it would break imports across tests, scripts, and runtime
        modules; the name is intentionally left unchanged here.

    Usage::

        tcm = TransportConfigMap()

        # Shared config for database (slug "db" from EnumInfraTransportType.DATABASE.value)
        spec = tcm.shared_spec(EnumInfraTransportType.DATABASE)
        # -> folder=/shared/db/, keys=(POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, ...)

        # Per-service config
        spec = tcm.service_spec(
            EnumInfraTransportType.DATABASE,
            service_slug="omnibase-runtime",
        )
        # -> folder=/services/omnibase-runtime/db/, keys=(POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, ...)

        # All specs for a list of transport types
        specs = tcm.specs_for_transports(
            [EnumInfraTransportType.DATABASE, EnumInfraTransportType.KAFKA],
            service_slug="omnibase-runtime",
        )
    """

    # Transport types whose credentials must come from the environment (e.g.
    # .env file), NOT from Infisical.  Fetching these from Infisical would
    # create a circular dependency because Infisical itself needs them to start.
    _BOOTSTRAP_TRANSPORTS: frozenset[EnumInfraTransportType] = frozenset(
        {EnumInfraTransportType.INFISICAL}
    )

    def is_bootstrap_transport(self, transport: EnumInfraTransportType) -> bool:
        """Return True if this transport's credentials come from the environment.

        Bootstrap transports (e.g. ``INFISICAL``) must be resolved from the
        environment (e.g. ``.env`` file), never from Infisical itself.
        Fetching their credentials from Infisical would create a circular
        dependency because Infisical needs those credentials to start.

        Args:
            transport: The transport type to check.

        Returns:
            ``True`` if the transport is bootstrap-only, ``False`` otherwise.
        """
        return transport in self._BOOTSTRAP_TRANSPORTS

    @staticmethod
    def _transport_slug(transport: EnumInfraTransportType) -> str:
        """Convert transport type to folder slug.

        Uses the enum value directly (e.g., ``"db"`` for DATABASE,
        ``"kafka"`` for KAFKA).

        Args:
            transport: The transport type enum member.

        Returns:
            The string slug used as the Infisical folder name.
        """
        return transport.value

    @staticmethod
    def keys_for_transport(transport: EnumInfraTransportType) -> tuple[str, ...]:
        """Return the canonical config keys for a transport type.

        Args:
            transport: The transport type to look up.

        Returns:
            Tuple of expected key names. Empty tuple for transports
            that have no external configuration (INMEMORY, RUNTIME).

        Warning:
            ``INFISICAL`` is included in the key map for completeness, but its
            keys (``INFISICAL_ADDR``, ``INFISICAL_CLIENT_ID``, etc.) must
            **never** be resolved from Infisical.  They are bootstrap
            credentials that Infisical needs to start; fetching them from
            Infisical creates a circular dependency.

            Before calling this method for a transport that originates from
            contract scanning or dynamic dispatch, check
            ``is_bootstrap_transport(transport)`` and resolve bootstrap
            transports from the environment (e.g. ``.env`` file) instead.
            The higher-level helpers ``specs_for_transports()`` and
            ``all_shared_specs()`` already skip bootstrap transports
            automatically.
        """
        return _TRANSPORT_KEYS.get(transport, ())

    def shared_spec(
        self,
        transport: EnumInfraTransportType,
        *,
        required: bool = False,
    ) -> ModelTransportConfigSpec:
        """Build a spec for shared (non-service-specific) transport config.

        Args:
            transport: The transport type.
            required: Whether this config is required for startup.

        Returns:
            A ``ModelTransportConfigSpec`` with the shared folder path.
        """
        slug = self._transport_slug(transport)
        return ModelTransportConfigSpec(
            transport_type=transport,
            infisical_folder=f"/shared/{slug}/",
            keys=self.keys_for_transport(transport),
            required=required,
        )

    def service_spec(
        self,
        transport: EnumInfraTransportType,
        *,
        service_slug: str,
        required: bool = False,
    ) -> ModelTransportConfigSpec:
        """Build a spec for per-service transport config.

        Args:
            transport: The transport type.
            service_slug: The service name for namespacing.
            required: Whether this config is required for startup.

        Returns:
            A ``ModelTransportConfigSpec`` with the per-service folder path.

        Raises:
            ValueError: If service_slug is empty.
        """
        if not service_slug:
            msg = "service_slug must not be empty for per-service config"
            raise ValueError(msg)

        slug = self._transport_slug(transport)
        return ModelTransportConfigSpec(
            transport_type=transport,
            infisical_folder=f"/services/{service_slug}/{slug}/",
            keys=self.keys_for_transport(transport),
            required=required,
            service_slug=service_slug,
        )

    def specs_for_transports(
        self,
        transports: list[EnumInfraTransportType],
        *,
        service_slug: str = "",
        required: bool = False,
    ) -> list[ModelTransportConfigSpec]:
        """Build specs for multiple transport types.

        If ``service_slug`` is provided, returns per-service specs.
        Otherwise, returns shared specs.

        Transports with no config keys (INMEMORY, RUNTIME) are skipped.

        Args:
            transports: List of transport types.
            service_slug: Optional service name for per-service paths.
            required: Whether these configs are required.

        Returns:
            List of ``ModelTransportConfigSpec`` instances.
        """
        specs: list[ModelTransportConfigSpec] = []
        for transport in transports:
            if transport in self._BOOTSTRAP_TRANSPORTS:
                logger.debug(
                    "Skipping bootstrap transport %s (credentials come from env, "
                    "not Infisical)",
                    transport.value,
                )
                continue

            keys = self.keys_for_transport(transport)
            if not keys:
                logger.debug("Skipping transport %s (no config keys)", transport.value)
                continue

            if service_slug:
                specs.append(
                    self.service_spec(
                        transport,
                        service_slug=service_slug,
                        required=required,
                    )
                )
            else:
                specs.append(self.shared_spec(transport, required=required))

        return specs

    def all_shared_specs(
        self, *, required: bool = False
    ) -> list[ModelTransportConfigSpec]:
        """Build shared specs for ALL transport types that have config keys.

        Convenience method that calls ``specs_for_transports`` with every
        member of ``EnumInfraTransportType``.  Bootstrap-only transports
        (e.g. ``INFISICAL``) and transports with no keys (e.g. ``INMEMORY``)
        are automatically excluded.

        Args:
            required: Whether these configs are required for startup.

        Returns:
            List of shared ``ModelTransportConfigSpec`` for every transport
            with defined keys.
        """
        return self.specs_for_transports(
            list(EnumInfraTransportType),
            required=required,
        )

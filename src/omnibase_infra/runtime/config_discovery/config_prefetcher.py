# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration prefetcher service for ONEX Infrastructure.

Orchestrates the prefetching of configuration values from Infisical during
runtime bootstrap. This module:

    1. Takes config requirements (from ``ContractConfigExtractor``)
    2. Resolves Infisical paths (via ``TransportConfigMap``)
    3. Fetches values through ``HandlerInfisical`` (not the adapter directly)
    4. Returns a dict of resolved key-value pairs

Design Decisions:
    - **No caching**: The prefetcher does NOT cache. Caching is owned by
      ``HandlerInfisical``.
    - **Handler, not adapter**: Calls go through the handler layer so that
      circuit breaking, caching, and audit logging are applied.
    - **INFISICAL_REQUIRED policy**: When the policy is enforced and Infisical
      is unavailable, prefetch fails loudly. When not enforced, missing values
      are logged as warnings and skipped.

.. versionadded:: 0.10.0
    Created as part of OMN-2287.

.. versionchanged:: 0.10.1
    ``ModelPrefetchResult`` renamed from ``PrefetchResult``.
    Class ``ConfigPrefetcher`` renamed to ``ConfigPrefetcher``.
    Dataclass ``PrefetchResult`` renamed to ``ModelPrefetchResult``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from pydantic import SecretStr

from omnibase_infra.runtime.config_discovery.models.model_config_requirements import (
    ModelConfigRequirements,
)
from omnibase_infra.runtime.config_discovery.models.model_transport_config_spec import (
    ModelTransportConfigSpec,
)

# ProtocolSecretResolver lives in the local models subpackage (not the handler
# layer), so importing it here does NOT introduce circular imports.  The
# protocol was created specifically to break the previous circular-import
# risk that motivated the original ``handler: object`` typing.
from omnibase_infra.runtime.config_discovery.models.protocol_secret_resolver import (
    ProtocolSecretResolver,
)
from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)


@dataclass
class ModelPrefetchResult:
    """Result of a config prefetch operation.

    Attributes:
        resolved: Successfully resolved key-value pairs. Values are
            ``SecretStr`` to prevent accidental logging of secret material.
        missing: Keys that could not be resolved (handler returned ``None``).
        errors: Per-key error messages for failed resolutions. Keys are the
            config key names; values are human-readable error descriptions.
        specs_attempted: Number of transport specs that were attempted during
            the prefetch operation.
    """

    resolved: dict[str, SecretStr] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    specs_attempted: int = 0

    @property
    def success_count(self) -> int:
        """Return the number of successfully resolved keys."""
        return len(self.resolved)

    @property
    def failure_count(self) -> int:
        """Return the number of keys that failed to resolve."""
        return len(self.missing) + len(self.errors)


class ConfigPrefetcher:
    """Prefetches configuration values from Infisical for runtime bootstrap.

    This service coordinates the extraction of configuration requirements from
    ONEX contracts, maps them to Infisical folder paths via
    ``TransportConfigMap``, and fetches values through a
    ``ProtocolSecretResolver`` (typically ``HandlerInfisical``).

    Usage::

        from omnibase_infra.handlers.handler_infisical import HandlerInfisical

        handler = HandlerInfisical(container)
        await handler.initialize(config)

        prefetcher = ConfigPrefetcher(handler=handler)
        result = prefetcher.prefetch(requirements)

        for key, value in result.resolved.items():
            os.environ[key] = value.get_secret_value()
    """

    def __init__(
        self,
        *,
        handler: ProtocolSecretResolver,
        service_slug: str = "",
        infisical_required: bool = False,
    ) -> None:
        """Initialize the config prefetcher.

        Args:
            handler: Any object satisfying ``ProtocolSecretResolver`` (e.g.
                ``HandlerInfisical``).  The protocol is defined in the
                local models subpackage to avoid circular imports with the
                handler layer.
            service_slug: Optional service name for per-service paths.
                If empty, shared paths are used.
            infisical_required: If True, missing keys cause errors.
                If False (default), missing keys are logged as warnings.
        """
        self._handler = handler
        self._service_slug = service_slug
        self._infisical_required = infisical_required
        self._transport_map = TransportConfigMap()

    def _fetch_key(
        self,
        key: str,
        spec: ModelTransportConfigSpec,
    ) -> SecretStr | None:
        """Fetch a single key from Infisical via the handler.

        Args:
            key: The secret key name.
            spec: The transport config spec (provides the folder path).

        Returns:
            The secret value, or ``None`` if not found.
        """
        try:
            result: SecretStr | None = self._handler.get_secret_sync(
                secret_name=key,
                secret_path=spec.infisical_folder,
            )
            return result
        except Exception as exc:
            logger.warning(
                "Failed to prefetch key %s from %s: %s",
                key,
                spec.infisical_folder,
                sanitize_error_message(exc),
            )
            return None

    def prefetch(
        self,
        requirements: ModelConfigRequirements,
    ) -> ModelPrefetchResult:
        """Prefetch all configuration values for the given requirements.

        Builds transport specs from the requirements' transport types,
        then fetches each key via the handler. Keys already present in
        the process environment are skipped (env always takes precedence).

        Args:
            requirements: Config requirements extracted from contracts.

        Returns:
            ``ModelPrefetchResult`` with resolved values and any errors.

        Raises:
            No exceptions are raised; all errors are captured in the result
            object or logged as warnings.
        """
        result = ModelPrefetchResult()

        # Build specs from discovered transport types
        specs = self._transport_map.specs_for_transports(
            list(requirements.transport_types),
            service_slug=self._service_slug,
        )
        result.specs_attempted = len(specs)

        # Also include any explicit environment dependencies as
        # individual key fetches from the runtime folder
        env_keys: list[str] = []
        for req in requirements.requirements:
            if req.source_field.startswith("dependencies["):
                env_keys.append(req.key)

        logger.info(
            "Prefetching config: %d transport specs, %d env keys",
            len(specs),
            len(env_keys),
        )

        # Fetch transport-based keys
        for spec in specs:
            for key in spec.keys:
                # Skip if already in environment (env overrides Infisical).
                # Use ``key in os.environ`` (not ``os.environ.get(key)``) so
                # that intentionally empty values are respected and not
                # overwritten by Infisical.
                if key in os.environ:
                    logger.debug(
                        "Key %s already in environment, skipping prefetch",
                        key,
                    )
                    result.resolved[key] = SecretStr(os.environ[key])
                    continue

                value = self._fetch_key(key, spec)
                if value is not None:
                    result.resolved[key] = value
                elif self._infisical_required and spec.required:
                    result.errors[key] = (
                        f"Required key {key} not found at {spec.infisical_folder}"
                    )
                else:
                    result.missing.append(key)

        # Fetch explicit environment dependencies
        if env_keys:
            # Use a generic /shared/env/ path for env dependencies
            from omnibase_infra.enums import EnumInfraTransportType

            env_spec = ModelTransportConfigSpec(
                transport_type=EnumInfraTransportType.RUNTIME,
                infisical_folder="/shared/env/",
                keys=tuple(env_keys),
            )
            for key in env_keys:
                if key in os.environ:
                    result.resolved[key] = SecretStr(os.environ[key])
                    continue

                value = self._fetch_key(key, env_spec)
                if value is not None:
                    result.resolved[key] = value
                elif self._infisical_required:
                    result.errors[key] = (
                        f"Required env dependency key {key} not found at"
                        f" {env_spec.infisical_folder}"
                    )
                else:
                    result.missing.append(key)

        logger.info(
            "Prefetch complete: %d resolved, %d missing, %d errors",
            result.success_count,
            len(result.missing),
            len(result.errors),
        )

        return result

    def apply_to_environment(self, result: ModelPrefetchResult) -> int:
        """Apply prefetched values to the process environment.

        Only sets keys that are NOT already in the environment (environment
        variables always take precedence over Infisical values).

        Args:
            result: The prefetch result to apply.

        Returns:
            Number of keys actually set in the environment.
        """
        applied = 0
        for key, value in result.resolved.items():
            if key not in os.environ:
                os.environ[key] = value.get_secret_value()
                applied += 1
                logger.debug("Applied prefetched key %s to environment", key)

        logger.info(
            "Applied %d/%d prefetched keys to environment",
            applied,
            len(result.resolved),
        )
        return applied

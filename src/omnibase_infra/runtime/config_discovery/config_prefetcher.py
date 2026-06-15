# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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

Precedence Rules (OMN-13070):
    On **controlled lanes** (``infisical_required=True``):
        1. Fetched/overlay config (Infisical) **always wins** over ambient env.
        2. When Infisical returns ``None`` for a key, ambient env is used as a
           **declared bootstrap fallback** — but only if the key is present, and
           a ``INFO``-level provenance log line is emitted identifying that the
           value came from ambient env rather than the controlled store.
        3. A key missing from both Infisical and ambient env is reported as an
           error (not silently dropped).

    On **uncontrolled lanes** (``infisical_required=False``, the default):
        - Keys already present in the process environment are skipped; the
          ambient value is used directly (legacy behaviour — local dev
          without Infisical continues to work).

.. versionadded:: 0.10.0
    Created as part of OMN-2287.

.. versionchanged:: 0.10.1
    ``ModelPrefetchResult`` renamed from ``PrefetchResult``.

.. versionchanged:: 0.10.2
    ``ModelPrefetchResult`` converted from ``@dataclass`` to a frozen Pydantic
    ``BaseModel`` (``ConfigDict(frozen=True, extra="forbid")``).  The
    ``missing`` field type changed from ``list[str]`` to ``tuple[str, ...]``
    to satisfy immutability requirements.

.. versionchanged:: 0.10.3
    OMN-13070: On controlled lanes (``infisical_required=True``) fetched config
    wins over ambient env.  Ambient env is now only a declared bootstrap
    fallback on controlled lanes, with a provenance log line.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr

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


class ModelPrefetchResult(BaseModel):
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

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolved: dict[str, SecretStr] = Field(default_factory=dict)
    missing: tuple[str, ...] = Field(default_factory=tuple)
    errors: dict[str, str] = Field(default_factory=dict)
    specs_attempted: int = Field(default=0)

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
            infisical_required: If True, missing keys for both transport-based
                config and explicit env dependencies cause errors.
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
        except Exception as exc:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "Failed to prefetch key %s from %s: %s",
                key,
                spec.infisical_folder,
                sanitize_error_message(exc),
            )
            return None

    def _resolve_key(
        self,
        key: str,
        spec: ModelTransportConfigSpec,
        *,
        is_env_dep: bool = False,
    ) -> tuple[str, SecretStr | None, str | None]:
        """Resolve a single config key and return the outcome to the caller.

        Returns a 3-tuple ``(outcome, value, error)`` where:
        - ``outcome`` is one of ``"resolved"``, ``"missing"``, or ``"error"``
        - ``value`` is the ``SecretStr`` when ``outcome == "resolved"``
        - ``error`` is the error message string when ``outcome == "error"``

        On controlled lanes (``infisical_required=True``), Infisical is
        always consulted first.  Ambient env is used only as a declared
        bootstrap fallback when Infisical returns ``None``, and a provenance
        ``INFO`` log line is emitted in that case.

        On uncontrolled lanes (``infisical_required=False``), a key already
        present in the process environment is used directly (legacy behaviour
        — local dev without Infisical continues to work unchanged).

        Args:
            key: The config key name.
            spec: The transport config spec (provides the Infisical folder).
            is_env_dep: ``True`` when the key comes from an explicit
                ``dependencies[]`` declaration rather than a transport spec.

        Returns:
            ``(outcome, value, error)`` 3-tuple; callers record into accumulators.
        """
        if self._infisical_required:
            # Controlled lane: Infisical/overlay WINS over ambient env.
            # Always attempt the fetch; fall back to ambient env only when
            # Infisical returns None, with an explicit provenance log line.
            value = self._fetch_key(key, spec)
            if value is not None:
                return ("resolved", value, None)
            if key in os.environ:
                # Declared bootstrap fallback — log provenance explicitly.
                logger.info(
                    "Key %s not found in Infisical; using ambient env as "
                    "bootstrap fallback (controlled lane) — "
                    "provenance: process environment",
                    key,
                )
                return ("resolved", SecretStr(os.environ[key]), None)
            # Missing from both Infisical and ambient env on a controlled
            # lane: record as error, not silently dropped.
            if is_env_dep or spec.required:
                return (
                    "error",
                    None,
                    f"Required key {key} not found at"
                    f" {spec.infisical_folder} and absent from ambient env",
                )
            return ("missing", None, None)

        # Uncontrolled lane: ambient env takes precedence (legacy).
        if key in os.environ:
            logger.debug(
                "Key %s already in environment, skipping prefetch",
                key,
            )
            return ("resolved", SecretStr(os.environ[key]), None)

        value = self._fetch_key(key, spec)
        if value is not None:
            return ("resolved", value, None)
        return ("missing", None, None)

    def prefetch(
        self,
        requirements: ModelConfigRequirements,
    ) -> ModelPrefetchResult:
        """Prefetch all configuration values for the given requirements.

        Builds transport specs from the requirements' transport types, then
        fetches each key via the handler.

        Precedence on controlled lanes (``infisical_required=True``):
            Infisical/overlay config **wins** over ambient env.  Ambient env is
            used only as a declared bootstrap fallback when Infisical returns
            ``None``, and a provenance ``INFO`` log line is emitted.

        Precedence on uncontrolled lanes (``infisical_required=False``):
            Keys already present in the process environment are skipped; the
            ambient value is used directly (legacy behaviour for local dev).

        Args:
            requirements: Config requirements extracted from contracts.

        Returns:
            ``ModelPrefetchResult`` with resolved values and any errors.

        Raises:
            No exceptions are raised; all errors are captured in the result
            object or logged as warnings.
        """
        # Local mutable accumulators — ModelPrefetchResult is frozen, so all
        # accumulation happens here and the immutable model is built at the end.
        resolved: dict[str, SecretStr] = {}
        missing: list[str] = []
        errors: dict[str, str] = {}

        # Build specs from discovered transport types.
        # When infisical_required=True, mark specs as required so that missing
        # transport-based keys are routed to errors rather than missing.
        # Without this, the spec.required flag would always be False (the
        # default), and the ``elif self._infisical_required and spec.required``
        # guard below would never fire for transport keys.
        specs = self._transport_map.specs_for_transports(
            list(requirements.transport_types),
            service_slug=self._service_slug,
            required=self._infisical_required,
        )
        specs_attempted = len(specs)

        # Also include any explicit environment dependencies as
        # individual key fetches from the runtime folder
        env_keys: list[str] = []
        for req in requirements.requirements:
            if req.source_field.startswith("dependencies["):
                env_keys.append(req.key)

        logger.info(
            "Prefetching config: %d transport specs, %d env keys (controlled_lane=%s)",
            len(specs),
            len(env_keys),
            self._infisical_required,
        )

        # Fetch transport-based keys
        for spec in specs:
            for key in spec.keys:
                outcome, value, error = self._resolve_key(key, spec)
                if outcome == "resolved" and value is not None:
                    resolved[key] = value
                elif outcome == "error" and error is not None:
                    errors[key] = error
                else:
                    missing.append(key)

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
                outcome, value, error = self._resolve_key(
                    key, env_spec, is_env_dep=True
                )
                if outcome == "resolved" and value is not None:
                    resolved[key] = value
                elif outcome == "error" and error is not None:
                    errors[key] = error
                else:
                    missing.append(key)

        result = ModelPrefetchResult(
            resolved=resolved,
            missing=tuple(missing),
            errors=errors,
            specs_attempted=specs_attempted,
        )

        logger.info(
            "Prefetch complete: %d resolved, %d missing, %d errors",
            result.success_count,
            len(result.missing),
            len(result.errors),
        )

        return result

    def prefetch_for_contracts(
        self,
        contracts_dir: Path,
    ) -> ModelPrefetchResult:
        """Extract requirements from a contracts directory and prefetch them.

        Convenience wrapper: calls ``ContractConfigExtractor.extract_from_paths``
        on ``contracts_dir``, then delegates to ``prefetch``.

        Args:
            contracts_dir: Path to a directory (or file) containing ONEX
                contract YAML files.  Passed directly to
                ``ContractConfigExtractor.extract_from_paths``.

        Returns:
            ``ModelPrefetchResult`` with resolved values and any errors.
        """
        from omnibase_infra.runtime.config_discovery.contract_config_extractor import (
            ContractConfigExtractor,
        )

        extractor = ContractConfigExtractor()
        requirements = extractor.extract_from_paths([contracts_dir])
        return self.prefetch(requirements)

    def apply_to_environment(self, result: ModelPrefetchResult) -> int:
        """Apply prefetched values to the process environment.

        On controlled lanes (``infisical_required=True``), fetched config
        **always wins**: existing ambient env values are overwritten so that
        stale shell or compose state cannot survive the prefetch boundary.

        On uncontrolled lanes (``infisical_required=False``), only keys that
        are NOT already present in the environment are set (legacy behaviour
        — local dev without Infisical continues to work unchanged).

        Args:
            result: The prefetch result to apply.

        Returns:
            Number of keys actually set (or overwritten) in the environment.
        """
        applied = 0
        for key, value in result.resolved.items():
            if self._infisical_required:
                # Controlled lane: fetched config wins — overwrite ambient env.
                existing = os.environ.get(key)
                os.environ[key] = value.get_secret_value()
                applied += 1
                if existing is not None:
                    logger.info(
                        "Applied prefetched key %s to environment "
                        "(controlled lane — overwrote ambient env value)",
                        key,
                    )
                else:
                    logger.debug("Applied prefetched key %s to environment", key)
            # Uncontrolled lane: ambient env takes precedence (legacy).
            elif key not in os.environ:
                os.environ[key] = value.get_secret_value()
                applied += 1
                logger.debug("Applied prefetched key %s to environment", key)

        logger.info(
            "Applied %d/%d prefetched keys to environment (controlled_lane=%s)",
            applied,
            len(result.resolved),
            self._infisical_required,
        )
        return applied

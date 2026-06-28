# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration model for the LLM endpoint health checker.

Defines probe intervals, HTTP timeouts, endpoint URLs, and per-endpoint
circuit breaker thresholds consumed by ``ServiceLlmEndpointHealth``.

.. versionadded:: 0.9.0
    Part of OMN-2255 LLM endpoint health checker.

.. versionchanged:: OMN-13699
    Added ``from_model_registry`` factory that sources model aliases from a
    routing-contract YAML and resolves URLs through an injected
    ``env_resolver`` callable rather than reading ``os.getenv`` directly.
    The ``_validate_endpoint_urls`` validator now also rejects explicitly-set
    empty strings with a diagnostic message that names the violating endpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils.util_error_sanitization import sanitize_url


class ModelLlmEndpointHealthConfig(BaseModel):
    """Configuration for the LLM endpoint health checker.

    Attributes:
        endpoints: Mapping of logical endpoint name to base URL.  Keys must
            be ``model_key`` values from the routing contract YAML (e.g.
            ``"qwen3-coder-30b"``), not hardcoded legacy aliases.  Use
            ``from_model_registry`` to build this map from a contract file.
        probe_interval_seconds: Seconds between probe cycles. Default: 30.
        probe_timeout_seconds: HTTP timeout for individual probe requests.
            Default: 5.0.
        circuit_breaker_threshold: Consecutive failures before opening
            the circuit for an endpoint. Default: 3.
        circuit_breaker_reset_timeout: Seconds before a tripped circuit
            transitions to half-open. Default: 60.0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of logical name to base URL",
    )

    @field_validator("endpoints")
    @classmethod
    def _validate_endpoint_urls(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that every endpoint URL uses an HTTP(S) scheme and has a hostname.

        Rejects non-HTTP schemes to prevent accidental use of ``file://``,
        ``ftp://``, or bare hostnames.  Also rejects URLs with empty netloc
        (e.g. ``http://``) which would produce invalid probe requests.
        Explicitly rejects empty strings with a diagnostic message rather
        than permitting them to silently cause probe failures.
        Error messages are sanitized via ``sanitize_url`` to avoid leaking
        credentials embedded in URLs.

        Raises:
            ValueError: If any URL is empty, does not start with ``http://``
                or ``https://``, or has an empty netloc (no hostname).
        """
        for name, url in v.items():
            if not url:
                msg = (
                    f"Endpoint '{name}' has an empty URL. "
                    "URLs must be sourced from the routing contract via "
                    "ModelLlmEndpointHealthConfig.from_model_registry(); "
                    "do not use os.getenv with an empty-string default."
                )
                raise ValueError(msg)
            if not url.startswith(("http://", "https://")):
                safe_url = sanitize_url(url)
                msg = (
                    f"Endpoint '{name}' has invalid URL '{safe_url}': "
                    "must start with 'http://' or 'https://'"
                )
                raise ValueError(msg)
            parsed = urlparse(url)
            if not parsed.netloc:
                safe_url = sanitize_url(url)
                msg = (
                    f"Endpoint '{name}' has invalid URL '{safe_url}': "
                    "URL must have a hostname"
                )
                raise ValueError(msg)
        return v

    probe_interval_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Seconds between probe cycles",
    )
    probe_timeout_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="HTTP timeout per probe request",
    )
    circuit_breaker_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive failures before opening circuit per endpoint",
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        ge=1.0,
        description=(
            "Minimum open-state cooling period in seconds before the circuit "
            "transitions from OPEN to HALF_OPEN"
        ),
    )

    @classmethod
    def from_model_registry(
        cls,
        registry_path: Path,
        env_resolver: Callable[[str], str | None],
    ) -> ModelLlmEndpointHealthConfig:
        """Build config by reading model aliases and URL env-var names from a routing
        contract YAML (e.g. ``docker/catalog/model_registry.yaml``).

        This is the correct construction path — it sources model aliases from the
        contract, not from hardcoded strings, and resolves URLs through an injected
        ``env_resolver`` rather than reading ``os.getenv`` directly.  This makes
        the factory fully testable without environment mutation.

        Only models with ``transport: http`` and a ``base_url_env`` field are
        included.  Non-HTTP transports (``oauth``, ``sdk``, etc.) are skipped
        because they are not probeable via HTTP health endpoints.

        The ``env_resolver`` is called with each model's ``base_url_env`` value:

        - Returns ``None`` → endpoint not configured in this environment; skipped.
        - Returns ``""`` → endpoint env var is set but empty; raises ``ValueError``
          with a diagnostic message naming the var — **never silently ignored**.
        - Returns a non-empty string → validated and included in the config.

        Probe settings (``probe_interval_seconds``, ``probe_timeout_seconds``,
        ``circuit_breaker_threshold``, ``circuit_breaker_reset_timeout``) use
        field defaults.  To override them, use the resolved ``endpoints`` map
        from this factory and construct a new config directly::

            base = ModelLlmEndpointHealthConfig.from_model_registry(
                registry_path=registry, env_resolver=os.getenv
            )
            config = ModelLlmEndpointHealthConfig(
                endpoints=base.endpoints,
                probe_interval_seconds=60.0,
            )

        Args:
            registry_path: Path to the model registry YAML contract.  Must exist
                and contain a ``models`` list.
            env_resolver: Callable mapping an env-var name to its value, or
                ``None`` if unset.  Pass ``os.getenv`` in production; pass a
                mock dict's ``.get`` method in tests.

        Returns:
            A ``ModelLlmEndpointHealthConfig`` whose ``endpoints`` map contains
            only the models whose env vars are set and non-empty, with all
            probe settings at their field defaults.

        Raises:
            ValueError: If ``registry_path`` does not exist.
            ValueError: If the YAML root is missing the ``models`` key.
            ValueError: If ``env_resolver`` returns an empty string for any
                model's ``base_url_env`` (empty = misconfigured, not absent).

        Example::

            import os
            from pathlib import Path
            from omnibase_infra.models.health.model_llm_endpoint_health_config import (
                ModelLlmEndpointHealthConfig,
            )

            registry = Path("docker/catalog/model_registry.yaml")
            config = ModelLlmEndpointHealthConfig.from_model_registry(
                registry_path=registry,
                env_resolver=os.getenv,
            )
            svc = ServiceLlmEndpointHealth(config=config, event_bus=bus)
        """
        if not registry_path.exists():
            msg = f"Model registry not found: {registry_path}"
            raise ValueError(msg)

        import yaml  # guarded: pyyaml is a declared dep; import here avoids module-level cost

        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "models" not in raw:
            msg = (
                f"Invalid model registry at {registry_path}: "
                "expected a mapping with a 'models' key"
            )
            raise ValueError(msg)

        models = raw["models"]
        if not isinstance(models, list):
            msg = (
                f"Invalid model registry at {registry_path}: "
                "'models' must be a list of model entries"
            )
            raise ValueError(msg)

        endpoints: dict[str, str] = {}
        for entry in models:
            if not isinstance(entry, dict):
                continue
            # Only probe HTTP-transport endpoints
            if entry.get("transport") != "http":
                continue
            base_url_env = entry.get("base_url_env", "")
            if not base_url_env:
                continue
            model_key = entry.get("model_key", "")
            if not model_key:
                continue

            url = env_resolver(str(base_url_env))
            if url is None:
                # Env var absent — endpoint not deployed in this environment; skip.
                continue
            if not url:
                msg = (
                    f"Model '{model_key}' env var '{base_url_env}' is set but empty. "
                    "Provide a valid http/https URL sourced from the routing contract. "
                    "Do not use os.getenv with an empty-string default."
                )
                raise ValueError(msg)
            endpoints[str(model_key)] = url

        return cls(endpoints=endpoints)


__all__: list[str] = ["ModelLlmEndpointHealthConfig"]

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

.. versionchanged:: OMN-16900
    ``from_model_registry`` now also resolves each entry's ``api_key_env``.  An
    endpoint that declares an auth secret which is absent or unresolvable is
    routed to ``unauthenticated_endpoints`` instead of ``endpoints`` — it is
    classified once and never probed, rather than 401-ing forever at the probe
    interval.  Adds ``auth_failure_threshold`` and
    ``auth_failure_backoff_max_seconds`` for the terminal-auth-failure backoff.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.utils.util_error_sanitization import sanitize_url


def _validate_url_map(v: dict[str, str], field_label: str) -> dict[str, str]:
    """Validate that every URL in a name -> URL map is a usable HTTP(S) URL.

    Args:
        v: Mapping of logical endpoint name to base URL.
        field_label: Field name used in diagnostic messages.

    Returns:
        The same mapping, unchanged, once every entry validates.

    Raises:
        ValueError: If any URL is empty, does not start with ``http://`` or
            ``https://``, or has no hostname.
    """
    for name, url in v.items():
        if not url:
            msg = (
                f"Endpoint '{name}' in '{field_label}' has an empty URL. "
                "URLs must be sourced from the routing contract via "
                "ModelLlmEndpointHealthConfig.from_model_registry(); "
                "an empty-string default is never a valid endpoint URL."
            )
            raise ValueError(msg)
        if not url.startswith(("http://", "https://")):
            safe_url = sanitize_url(url)
            msg = (
                f"Endpoint '{name}' in '{field_label}' has invalid URL "
                f"'{safe_url}': must start with 'http://' or 'https://'"
            )
            raise ValueError(msg)
        parsed = urlparse(url)
        if not parsed.hostname:
            safe_url = sanitize_url(url)
            msg = (
                f"Endpoint '{name}' in '{field_label}' has invalid URL "
                f"'{safe_url}': URL must have a hostname"
            )
            raise ValueError(msg)
    return v


class ModelLlmEndpointHealthConfig(BaseModel):
    """Configuration for the LLM endpoint health checker.

    Attributes:
        endpoints: Mapping of logical endpoint name to base URL.  Keys must
            be ``model_key`` values from the routing contract YAML (e.g.
            ``"qwen3-coder-30b"``), not hardcoded legacy aliases.  Use
            ``from_model_registry`` to build this map from a contract file.
        unauthenticated_endpoints: Endpoints whose declared auth secret is
            absent or unresolvable.  Same key/URL shape as ``endpoints``, but
            these are classified ``SKIPPED_NO_AUTH`` once and never probed.
            Must be disjoint from ``endpoints``.
        probe_interval_seconds: Seconds between probe cycles. Default: 30.
        probe_timeout_seconds: HTTP timeout for individual probe requests.
            Default: 5.0.
        circuit_breaker_threshold: Consecutive failures before opening
            the circuit for an endpoint. Default: 3.
        circuit_breaker_reset_timeout: Seconds before a tripped circuit
            transitions to half-open. Default: 60.0.
        auth_failure_threshold: Consecutive 401/403 probe results before an
            endpoint is treated as terminally auth-failed. Default: 2.
        auth_failure_backoff_max_seconds: Ceiling on the exponential backoff
            applied to a terminally auth-failed endpoint. Default: 3600.0.
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
        ``ftp://``, or bare hostnames.  Also rejects URLs with no hostname
        (e.g. ``http://`` or ``http://user:pass@``) which would produce
        invalid probe requests.  ``parsed.hostname`` is checked rather than
        ``parsed.netloc`` because ``netloc`` is non-empty for userinfo-only
        authorities like ``http://user:pass@`` that have no actual host.
        Explicitly rejects empty strings with a diagnostic message rather
        than permitting them to silently cause probe failures.
        Error messages are sanitized via ``sanitize_url`` to avoid leaking
        credentials embedded in URLs.

        Raises:
            ValueError: If any URL is empty, does not start with ``http://``
                or ``https://``, or has no hostname.
        """
        return _validate_url_map(v, "endpoints")

    unauthenticated_endpoints: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Endpoints whose declared auth secret is absent or unresolvable. "
            "Classified SKIPPED_NO_AUTH once and never probed (OMN-16900)."
        ),
    )

    @field_validator("unauthenticated_endpoints")
    @classmethod
    def _validate_unauthenticated_endpoint_urls(
        cls, v: dict[str, str]
    ) -> dict[str, str]:
        """Apply the same URL rules to the auth-dead endpoint map.

        These endpoints are never probed, but their URLs still surface in
        status snapshots and health events, so they must be well-formed.

        Raises:
            ValueError: On the same conditions as ``_validate_endpoint_urls``.
        """
        return _validate_url_map(v, "unauthenticated_endpoints")

    @model_validator(mode="after")
    def _validate_endpoint_sets_disjoint(self) -> ModelLlmEndpointHealthConfig:
        """Reject a name that is both probeable and classified auth-dead.

        An endpoint in both maps would be probed *and* reported as skipped,
        which is a wiring bug rather than a representable state.

        Raises:
            ValueError: If the two maps share any key.
        """
        overlap = sorted(set(self.endpoints) & set(self.unauthenticated_endpoints))
        if overlap:
            msg = (
                f"Endpoint(s) {overlap} appear in both 'endpoints' and "
                "'unauthenticated_endpoints'; an endpoint is either probeable "
                "or classified SKIPPED_NO_AUTH, never both."
            )
            raise ValueError(msg)
        return self

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
    auth_failure_threshold: int = Field(
        default=2,
        ge=1,
        description=(
            "Consecutive 401/403 probe results before the endpoint is treated "
            "as terminally auth-failed and moved to backoff (OMN-16900)"
        ),
    )
    auth_failure_backoff_max_seconds: float = Field(
        default=3600.0,
        ge=1.0,
        description=(
            "Ceiling for the exponential backoff applied to a terminally "
            "auth-failed endpoint (OMN-16900)"
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

        **Auth partitioning (OMN-16900).**  When an entry also declares
        ``api_key_env``, that variable is resolved through the same
        ``env_resolver``.  If it is absent or empty the endpoint is placed in
        ``unauthenticated_endpoints`` rather than ``endpoints``: the service
        classifies it ``SKIPPED_NO_AUTH`` once and never probes it.  A missing
        credential is a permanent condition, not a transient outage, so
        retrying it at the probe interval is pure waste — on .201 exactly this
        gap produced 5+ days of 401s against the GLM endpoints at up to 4525
        probes per container.

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
            only the models whose URL env vars are set and non-empty **and**
            whose declared auth secret resolves, with the auth-dead remainder
            in ``unauthenticated_endpoints`` and all probe settings at their
            field defaults.

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
        unauthenticated: dict[str, str] = {}
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

            # OMN-16900: an entry that declares an auth secret is only
            # probeable while that secret resolves.  Absent or empty means the
            # endpoint can never answer — classify it once, never probe it.
            api_key_env = entry.get("api_key_env", "")
            if api_key_env and not env_resolver(str(api_key_env)):
                unauthenticated[str(model_key)] = url
                continue

            endpoints[str(model_key)] = url

        return cls(endpoints=endpoints, unauthenticated_endpoints=unauthenticated)


__all__: list[str] = ["ModelLlmEndpointHealthConfig"]

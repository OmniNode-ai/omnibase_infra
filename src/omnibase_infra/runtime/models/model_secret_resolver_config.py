# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration model for SecretResolver.

.. versionadded:: 0.8.0
    Initial implementation for OMN-764.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping


class ModelSecretResolverConfig(BaseModel):
    """Configuration for SecretResolver.

    Configures the centralized secret resolution system that supports
    multiple secret sources with priority-based resolution.

    Source Priority Order:
        1. Infisical (if configured) - for production secrets (OMN-2286)
        2. Environment variables - for local development
        3. File-based secrets - for Kubernetes deployments

    Attributes:
        mappings: Explicit mappings from logical names to source specs.
        default_ttl_env_seconds: Default TTL for environment variable secrets.
        default_ttl_file_seconds: Default TTL for file-based secrets.
        enable_convention_fallback: Enable automatic source discovery by convention.
        convention_env_prefix: Prefix for environment variable convention lookup.
        bootstrap_secrets: Secrets resolved directly from env (never through chain).
        secrets_dir: Directory for file-based secrets (K8s secrets volume).
        required_secrets: The authoritative declared-required key set (OMN-14951).
            Logical names only -- never values. Every name here MUST resolve
            through ``SecretResolver.validate_required_secrets()`` (Infisical or
            an explicit non-Infisical mapping when
            ``require_infisical_for_required_secrets=False``) or boot/deploy
            fails, naming every missing key at once.
        require_infisical_for_required_secrets: When True (default), every
            logical name in ``required_secrets`` MUST have an explicit
            ``mappings`` entry with ``source_type="infisical"``. Convention
            fallback (``enable_convention_fallback``) is structurally forbidden
            for declared-required keys: a required key silently resolving from
            ambient env is exactly the failure class this gate closes
            (OMN-14951). Config construction itself fails, naming every
            offending key, when this invariant is violated.

    Example:
        >>> config = ModelSecretResolverConfig(
        ...     enable_convention_fallback=True,
        ...     convention_env_prefix="ONEX_",
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,  # Allow mutation for runtime configuration
        extra="forbid",
        from_attributes=True,
    )

    # Explicit mappings: logical_name -> source_spec
    mappings: list[ModelSecretMapping] = Field(
        default_factory=list,
        description="Explicit mappings from logical secret names to source specifications. "
        "Takes precedence over convention-based resolution.",
    )

    # Default TTLs by source type (in seconds)
    default_ttl_env_seconds: int = Field(
        default=86400,
        ge=0,
        description="Default TTL for environment variable secrets (24 hours).",
    )
    default_ttl_file_seconds: int = Field(
        default=86400,
        ge=0,
        description="Default TTL for file-based secrets (24 hours).",
    )

    # Convention fallback when no explicit mapping exists
    enable_convention_fallback: bool = Field(
        default=True,
        description="Enable automatic source discovery using naming conventions. "
        "When True, 'database.postgres.password' becomes 'DATABASE_POSTGRES_PASSWORD'.",
    )
    convention_env_prefix: str = Field(
        default="",
        description="Prefix added to environment variable names during convention lookup. "
        "E.g., 'ONEX_' makes 'database.password' look for 'ONEX_DATABASE_PASSWORD'.",
    )

    # Bootstrap secrets (always resolved from env, never through resolver chain)
    bootstrap_secrets: list[str] = Field(
        default_factory=list,
        description="Secrets that are always resolved directly from environment variables, "
        "never through the resolver chain.",
    )

    # File-based secrets directory (K8s secrets volume mount)
    secrets_dir: Path = Field(
        default=Path("/run/secrets"),
        description="Directory containing file-based secrets (K8s secrets volume mount).",
    )

    # Declared-required key set (OMN-14951): the authoritative env/secret
    # surface for a lane. Names only -- never values. See class docstring.
    required_secrets: list[str] = Field(
        default_factory=list,
        description="Authoritative declared-required logical secret names for this "
        "lane (OMN-14951). Every name must resolve via "
        "SecretResolver.validate_required_secrets() or boot/deploy fails, naming "
        "every missing key at once. Names only, never values.",
    )
    require_infisical_for_required_secrets: bool = Field(
        default=True,
        description="When True, every name in required_secrets must have an explicit "
        "mappings entry with source_type='infisical' -- convention/env fallback is "
        "structurally forbidden for declared-required keys (OMN-14951). Config "
        "construction fails, naming every offending key, when violated.",
    )

    # NOTE: Vault configuration will be added when ModelVaultHandlerConfig is available
    # vault_config: ModelVaultHandlerConfig | None = None

    @model_validator(mode="after")
    def _validate_required_secrets_route_through_infisical(
        self,
    ) -> ModelSecretResolverConfig:
        """Fail loud at construction time, naming every offending key at once.

        Closes the core silent-fallback defect (OMN-14951 finding #1):
        ``_get_source_spec`` silently builds an env-var convention spec for any
        unmapped logical name, so whether a key is Infisical-backed is decided
        per-key by static config -- never by runtime fallback-on-failure. This
        validator makes that routing decision structurally impossible to get
        wrong for declared-required keys: either every required key has an
        explicit ``source_type="infisical"`` mapping, or config construction
        itself raises, naming every missing-mapping and wrong-source-type key
        together (never one at a time).
        """
        if not self.required_secrets:
            return self

        bootstrap_overlap = sorted(
            set(self.required_secrets) & set(self.bootstrap_secrets)
        )
        if bootstrap_overlap:
            raise ValueError(
                "required_secrets must not overlap bootstrap_secrets -- bootstrap "
                "keys bypass the resolver chain entirely (a keyring cannot unlock "
                "itself) and can never be Infisical-routed required secrets: "
                f"{bootstrap_overlap}"
            )

        if not self.require_infisical_for_required_secrets:
            return self

        mapped_source_types = {
            mapping.logical_name: mapping.source.source_type
            for mapping in self.mappings
        }
        missing_mapping: list[str] = []
        wrong_source_type: list[str] = []
        for name in self.required_secrets:
            source_type = mapped_source_types.get(name)
            if source_type is None:
                missing_mapping.append(name)
            elif source_type != "infisical":
                wrong_source_type.append(name)

        if missing_mapping or wrong_source_type:
            details: list[str] = []
            if missing_mapping:
                details.append(f"no explicit mapping: {sorted(missing_mapping)}")
            if wrong_source_type:
                details.append(
                    f"mapped to non-infisical source_type: {sorted(wrong_source_type)}"
                )
            raise ValueError(
                "required_secrets must each have an explicit mappings entry with "
                "source_type='infisical' (require_infisical_for_required_secrets=True); "
                "convention/env fallback is forbidden for declared-required keys "
                "(OMN-14951). Offending keys -- " + "; ".join(details)
            )

        return self


__all__: list[str] = ["ModelSecretResolverConfig"]

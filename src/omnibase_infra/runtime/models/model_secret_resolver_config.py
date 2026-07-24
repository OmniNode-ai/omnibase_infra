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
            Logical names only -- never values. Every name here MUST have an
            explicit ``mappings`` entry (convention/env fallback is
            structurally forbidden for declared-required keys, unconditionally
            -- this holds regardless of ``require_infisical_for_required_secrets``
            or ``enable_convention_fallback``) and MUST resolve through
            ``SecretResolver.validate_required_secrets()`` or boot/deploy
            fails, naming every missing key at once.
        require_infisical_for_required_secrets: When True (default), every
            explicit mapping for a name in ``required_secrets`` must have
            ``source_type="infisical"``. When False, a required key may
            instead have an explicit **non**-Infisical mapping (e.g.
            ``source_type="file"`` for a K8s secrets-volume lane) -- but an
            explicit mapping of SOME source type is still mandatory either
            way. This flag only ever narrows *which* source type an explicit
            mapping may use; it can never be used to skip declaring a mapping
            at all, so it can never reopen the convention-fallback path for a
            declared-required key (OMN-14951 gap 1). Config construction
            itself fails, naming every offending key, when this invariant is
            violated.

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
        "When True, 'database.postgres.password' becomes 'DATABASE_POSTGRES_PASSWORD'. "
        "Never applies to a name in required_secrets (OMN-14951) -- that "
        "structural exclusion is enforced by SecretResolver._get_source_spec "
        "at the resolution choke point, independent of this flag's value.",
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
        description="When True, every explicit mapping for a name in required_secrets "
        "must have source_type='infisical'. When False, a required key's explicit "
        "mapping may use a different source_type instead -- an explicit mapping is "
        "mandatory either way, so convention/env fallback stays structurally forbidden "
        "for declared-required keys regardless of this flag (OMN-14951 gap 1). Config "
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
        wrong for declared-required keys: every required key MUST have an
        explicit mapping, unconditionally -- config construction itself raises,
        naming every missing-mapping key, otherwise.

        ``require_infisical_for_required_secrets`` narrows this further when
        True (the default): it additionally requires the explicit mapping's
        ``source_type`` to be ``"infisical"``. When False, an explicit mapping
        of any other source type (e.g. ``"file"`` for a K8s secrets-volume
        lane) is accepted -- but the mapping itself is never optional. This is
        gap 1 from the 2026-07-23 hardening pass: the flag previously
        short-circuited the *entire* check (including the missing-mapping
        check) when False, so a required key with zero mappings passed
        construction and silently resolved via
        ``enable_convention_fallback`` at runtime -- exactly the silent-miss
        this gate exists to kill. That early return is gone; only the
        source-type restriction is now conditional on the flag.

        This is the construction-time half of the fix. The resolution-time
        half lives in ``SecretResolver._get_source_spec``, which refuses
        convention fallback for any name in ``required_secrets`` unconditionally
        -- a second, independent enforcement point so the invariant holds even
        if this config were mutated after construction (the model is
        ``frozen=False``) or constructed via ``model_construct()``, bypassing
        this validator.
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

        mapped_source_types = {
            mapping.logical_name: mapping.source.source_type
            for mapping in self.mappings
        }
        missing_mapping: list[str] = []
        wrong_source_type: list[str] = []
        for name in self.required_secrets:
            source_type = mapped_source_types.get(name)
            if source_type is None:
                # OMN-14951 gap 1: unconditional. An explicit mapping is
                # mandatory for every declared-required key regardless of
                # require_infisical_for_required_secrets -- that flag only
                # ever narrows *which* source_type the explicit mapping may
                # use (see below), never whether a mapping must exist at all.
                missing_mapping.append(name)
            elif (
                self.require_infisical_for_required_secrets
                and source_type != "infisical"
            ):
                wrong_source_type.append(name)

        if missing_mapping or wrong_source_type:
            details: list[str] = []
            if missing_mapping:
                details.append(f"no explicit mapping: {sorted(missing_mapping)}")
            if wrong_source_type:
                details.append(
                    f"mapped to non-infisical source_type "
                    f"(require_infisical_for_required_secrets=True): "
                    f"{sorted(wrong_source_type)}"
                )
            raise ValueError(
                "required_secrets must each have an explicit mappings entry "
                "(convention/env fallback is forbidden for declared-required keys, "
                "OMN-14951); require_infisical_for_required_secrets=True additionally "
                "requires source_type='infisical'. Offending keys -- "
                + "; ".join(details)
            )

        return self


__all__: list[str] = ["ModelSecretResolverConfig"]

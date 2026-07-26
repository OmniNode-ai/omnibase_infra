# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for OMN-14951: ModelSecretResolverConfig.required_secrets.

Construction-time counterpart of the boot-time gate in
``test_runtime_host_process_required_secrets.py``. Proves that a declared-
required secret without an explicit ``source_type="infisical"`` mapping makes
config construction itself fail -- convention/env fallback is structurally
forbidden for declared-required keys, and every offending key is named
together, not one at a time.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import (
    ModelSecretSourceSpec,
)


def _infisical_mapping(logical_name: str, source_path: str) -> ModelSecretMapping:
    return ModelSecretMapping(
        logical_name=logical_name,
        source=ModelSecretSourceSpec(source_type="infisical", source_path=source_path),
    )


def _env_mapping(logical_name: str, source_path: str) -> ModelSecretMapping:
    return ModelSecretMapping(
        logical_name=logical_name,
        source=ModelSecretSourceSpec(source_type="env", source_path=source_path),
    )


class TestRequiredSecretsHappyPath:
    def test_required_secret_with_infisical_mapping_constructs(self) -> None:
        config = ModelSecretResolverConfig(
            required_secrets=["db.postgres.password"],
            mappings=[_infisical_mapping("db.postgres.password", "POSTGRES_PASSWORD")],
        )
        assert config.required_secrets == ["db.postgres.password"]

    def test_empty_required_secrets_is_default_and_unconstrained(self) -> None:
        # Backward compatible: existing configs with no required_secrets are
        # untouched by this validator (regression guard for OMN-764 baseline).
        config = ModelSecretResolverConfig(enable_convention_fallback=True)
        assert config.required_secrets == []


class TestRequiredSecretsRedProofMissingMapping:
    """RED proof: a required key with NO mapping fails construction, named."""

    def test_required_secret_without_any_mapping_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ModelSecretResolverConfig(required_secrets=["db.postgres.password"])
        assert "db.postgres.password" in str(exc_info.value)

    def test_multiple_missing_mappings_named_together(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ModelSecretResolverConfig(
                required_secrets=["a.secret", "b.secret", "c.secret"],
                mappings=[_infisical_mapping("c.secret", "C_SECRET")],
            )
        message = str(exc_info.value)
        assert "a.secret" in message
        assert "b.secret" in message
        # c.secret has a valid infisical mapping -- must not be reported.
        assert "no explicit mapping: ['a.secret', 'b.secret']" in message


class TestRequiredSecretsRedProofWrongSourceType:
    """RED proof: convention/env fallback is structurally forbidden for
    declared-required keys -- an explicit env mapping is not good enough."""

    def test_required_secret_mapped_to_env_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ModelSecretResolverConfig(
                required_secrets=["db.postgres.password"],
                mappings=[_env_mapping("db.postgres.password", "POSTGRES_PASSWORD")],
            )
        message = str(exc_info.value)
        assert "db.postgres.password" in message
        assert "non-infisical" in message

    def test_convention_fallback_enabled_does_not_bypass_the_gate(self) -> None:
        """Even with enable_convention_fallback=True (the SecretResolver
        default), an unmapped required key must NOT silently pass -- that
        silent env-fallback routing is the core defect OMN-14951 closes."""
        with pytest.raises(ValidationError):
            ModelSecretResolverConfig(
                enable_convention_fallback=True,
                required_secrets=["db.postgres.password"],
            )


class TestRequiredSecretsOptOutEscapeHatch:
    """require_infisical_for_required_secrets=False is an explicit, named
    escape hatch for a required key's explicit mapping to use a non-Infisical
    source_type -- NOT an escape hatch from declaring a mapping at all. Gap 1
    (2026-07-23 hardening): prior to this fix, the flag short-circuited the
    ENTIRE validator (including the missing-mapping check) when False, so a
    required key with zero mappings passed construction and silently resolved
    via enable_convention_fallback at runtime -- exactly the operator-flagged
    live reproduction. The missing-mapping check is now unconditional."""

    def test_opt_out_allows_env_mapping(self) -> None:
        config = ModelSecretResolverConfig(
            required_secrets=["db.postgres.password"],
            require_infisical_for_required_secrets=False,
            mappings=[_env_mapping("db.postgres.password", "POSTGRES_PASSWORD")],
        )
        assert config.required_secrets == ["db.postgres.password"]

    def test_opt_out_does_not_waive_the_explicit_mapping_requirement(
        self,
    ) -> None:
        # require_infisical_for_required_secrets=False only removes the
        # infisical-only source_type constraint; it does NOT create a new
        # fallback path or waive the explicit-mapping requirement. With no
        # mapping at all, construction must still raise -- naming the key --
        # regardless of the flag value.
        with pytest.raises(ValidationError) as exc_info:
            ModelSecretResolverConfig(
                required_secrets=["db.postgres.password"],
                require_infisical_for_required_secrets=False,
                enable_convention_fallback=False,
            )
        assert "db.postgres.password" in str(exc_info.value)


class TestRequiredSecretsGap1LeakyFlagComboRedProof:
    """RED proof (OMN-14951 gap 1, 2026-07-23 hardening): the exact leaky
    flag combination the operator/verifier drove live --
    require_infisical_for_required_secrets=False + enable_convention_fallback
    at its True default + a required key with no Infisical mapping -- must
    now raise, naming the key, rather than silently reintroducing the
    convention-fallback silent-miss this gate exists to kill.

    This is a stricter outcome than the live reproduction described: the
    fixed validator now refuses to even CONSTRUCT such a config (closing the
    leak one layer earlier, at declaration time, rather than only at
    resolution time) -- see test_secret_resolver_required_secrets.py's
    TestValidateRequiredSecretsGap1ResolverChokePoint for the independent
    resolution-time choke-point proof (SecretResolver._get_source_spec),
    which holds even if this construction-time validator is bypassed via
    model_construct().
    """

    def test_leaky_flag_combo_raises_naming_the_key(self) -> None:
        # Pre-fix (verify by checking out the unpatched
        # model_secret_resolver_config.py and re-running): this constructs
        # successfully with zero mappings, and the resulting config silently
        # resolves "payments.stripe.secret_key" via convention/env fallback
        # at runtime -- the exact silent-miss the operator flagged.
        with pytest.raises(ValidationError) as exc_info:
            ModelSecretResolverConfig(
                required_secrets=["payments.stripe.secret_key"],
                require_infisical_for_required_secrets=False,
                enable_convention_fallback=True,
            )
        assert "payments.stripe.secret_key" in str(exc_info.value)

    def test_leaky_flag_combo_with_bootstrap_default_also_raises(self) -> None:
        # bootstrap_secrets/require_infisical_for_required_secrets both left
        # at their non-default-opt-out-adjacent values to prove no OTHER flag
        # combination reopens the leak either.
        with pytest.raises(ValidationError):
            ModelSecretResolverConfig(
                required_secrets=["payments.stripe.secret_key"],
                require_infisical_for_required_secrets=False,
                enable_convention_fallback=True,
                bootstrap_secrets=[],
                convention_env_prefix="ONEX_",
            )


class TestBootstrapSecretsCannotOverlapRequiredSecrets:
    """RED proof: the bootstrap allowlist and required_secrets are disjoint
    by construction -- a bootstrap key (irreducible, resolves directly from
    env, never via Infisical) can never also be declared Infisical-required."""

    def test_overlap_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ModelSecretResolverConfig(
                required_secrets=["INFISICAL_ADDR"],
                bootstrap_secrets=["INFISICAL_ADDR"],
            )
        assert "INFISICAL_ADDR" in str(exc_info.value)

    def test_no_overlap_is_fine(self) -> None:
        config = ModelSecretResolverConfig(
            required_secrets=["db.postgres.password"],
            bootstrap_secrets=["INFISICAL_ADDR", "INFISICAL_CLIENT_ID"],
            mappings=[_infisical_mapping("db.postgres.password", "POSTGRES_PASSWORD")],
        )
        assert config.bootstrap_secrets == ["INFISICAL_ADDR", "INFISICAL_CLIENT_ID"]

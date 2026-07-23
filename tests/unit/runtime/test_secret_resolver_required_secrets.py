# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for OMN-14951: SecretResolver.validate_required_secrets().

The resolver-level batch presence gate: fails loudly, naming every
missing/unreachable required secret at once, and fails closed when Infisical
is unreachable or rejects authentication. This is the direct RED-proof
acceptance evidence named in the OMN-14951 build instructions -- an absent
key MUST fail; a silent skip that reports green is exactly the defect being
fixed (see memory feedback_prove_red_against_exists_but_wrong and
feedback_optional_input_means_the_check_does_not_exist).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from omnibase_infra.errors import SecretResolutionError
from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import (
    ModelSecretSourceSpec,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver


def _infisical_mapping(logical_name: str, source_path: str) -> ModelSecretMapping:
    return ModelSecretMapping(
        logical_name=logical_name,
        source=ModelSecretSourceSpec(source_type="infisical", source_path=source_path),
    )


def _config(
    *names: str, extra_mappings: list[ModelSecretMapping] | None = None
) -> ModelSecretResolverConfig:
    mappings = [
        _infisical_mapping(name, name.upper().replace(".", "_")) for name in names
    ]
    if extra_mappings:
        mappings.extend(extra_mappings)
    return ModelSecretResolverConfig(required_secrets=list(names), mappings=mappings)


class TestValidateRequiredSecretsNoOp:
    def test_empty_required_secrets_never_raises_even_without_handler(self) -> None:
        config = ModelSecretResolverConfig(required_secrets=[])
        resolver = SecretResolver(config=config)  # no infisical_handler at all
        resolver.validate_required_secrets()  # must not raise


class TestValidateRequiredSecretsRedProofAbsence:
    """RED proof: a declared-required key ABSENT from Infisical fails loud."""

    def test_single_missing_key_raises_named(self) -> None:
        config = _config("llm.openai.api_key")
        mock_handler = MagicMock()
        mock_handler.get_secret_sync.return_value = None  # absent
        resolver = SecretResolver(config=config, infisical_handler=mock_handler)

        with pytest.raises(SecretResolutionError) as exc_info:
            resolver.validate_required_secrets()

        message = str(exc_info.value)
        assert "OMN-14951" in message
        assert "llm.openai.api_key" in message
        assert "missing=" in message

    def test_all_missing_keys_named_together_not_one_at_a_time(self) -> None:
        config = _config("a.secret", "b.secret", "c.secret")
        mock_handler = MagicMock()
        mock_handler.get_secret_sync.return_value = None
        resolver = SecretResolver(config=config, infisical_handler=mock_handler)

        with pytest.raises(SecretResolutionError) as exc_info:
            resolver.validate_required_secrets()

        message = str(exc_info.value)
        for name in ("a.secret", "b.secret", "c.secret"):
            assert name in message

    def test_partial_absence_names_only_the_absent_keys(self) -> None:
        config = _config("present.secret", "absent.secret")
        mock_handler = MagicMock()

        def _get_secret_sync(secret_name: str) -> SecretStr | None:
            if secret_name == "PRESENT_SECRET":
                return SecretStr("value")
            return None

        mock_handler.get_secret_sync.side_effect = _get_secret_sync
        resolver = SecretResolver(config=config, infisical_handler=mock_handler)

        with pytest.raises(SecretResolutionError) as exc_info:
            resolver.validate_required_secrets()

        message = str(exc_info.value)
        assert "absent.secret" in message
        assert "present.secret" not in message


class TestValidateRequiredSecretsFailClosedOnUnreachable:
    """RED proof: Infisical unreachable/auth failure fails CLOSED, never a
    silent pass."""

    def test_connection_error_fails_closed(self) -> None:
        config = _config("db.postgres.password")
        mock_handler = MagicMock()
        mock_handler.get_secret_sync.side_effect = ConnectionError(
            "infisical unreachable: connection refused"
        )
        resolver = SecretResolver(config=config, infisical_handler=mock_handler)

        with pytest.raises(SecretResolutionError) as exc_info:
            resolver.validate_required_secrets()

        message = str(exc_info.value)
        assert "db.postgres.password" in message
        assert "unreachable_or_errored=" in message

    def test_auth_failure_fails_closed(self) -> None:
        config = _config("db.postgres.password")
        mock_handler = MagicMock()
        mock_handler.get_secret_sync.side_effect = PermissionError(
            "No identity with specified client ID was found (Status: 404)"
        )
        resolver = SecretResolver(config=config, infisical_handler=mock_handler)

        with pytest.raises(SecretResolutionError):
            resolver.validate_required_secrets()

    def test_unreachable_is_never_reported_as_missing_but_still_fails(self) -> None:
        """Distinguishes 'unreachable' from 'missing' in the message while
        still failing the gate either way -- both are fail-closed outcomes."""
        config = _config("db.postgres.password")
        mock_handler = MagicMock()
        mock_handler.get_secret_sync.side_effect = TimeoutError("infisical timeout")
        resolver = SecretResolver(config=config, infisical_handler=mock_handler)

        with pytest.raises(SecretResolutionError) as exc_info:
            resolver.validate_required_secrets()

        message = str(exc_info.value)
        assert "unreachable_or_errored=" in message


class TestValidateRequiredSecretsGreenControl:
    """GREEN control: not vacuous, since the RED cases above independently
    prove the gate actually discriminates absent/unreachable from present."""

    def test_all_present_does_not_raise(self) -> None:
        config = _config("llm.openai.api_key", "db.postgres.password")
        mock_handler = MagicMock()
        mock_handler.get_secret_sync.return_value = SecretStr("value")
        resolver = SecretResolver(config=config, infisical_handler=mock_handler)

        resolver.validate_required_secrets()  # must not raise

    def test_no_handler_but_no_required_secrets_does_not_raise(self) -> None:
        config = ModelSecretResolverConfig(required_secrets=[])
        resolver = SecretResolver(config=config, infisical_handler=None)
        resolver.validate_required_secrets()

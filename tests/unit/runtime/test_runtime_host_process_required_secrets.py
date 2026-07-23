# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for OMN-14951: RuntimeHostProcess required-secrets presence gate.

Exercises ``RuntimeHostProcess._validate_required_secrets_via_resolver``, the
boot-time wiring that loads the rendered ``ModelSecretResolverConfig`` from
``ONEX_SECRET_RESOLVER_CONFIG_PATH`` and, when it declares
``required_secrets``, fails loudly naming every missing/unreachable key at
once via ``SecretResolver.validate_required_secrets()``.

This is the RED-proof acceptance evidence for OMN-14951: the gate MUST fail
when a declared-required key is absent from (or unreachable via) Infisical --
a silent pass on absence is exactly the defect class being closed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import SecretStr

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config

_RENDERED_CONFIG_ENV_VAR = "ONEX_SECRET_RESOLVER_CONFIG_PATH"


def _make_process(**kwargs: object) -> RuntimeHostProcess:
    config = make_runtime_config()
    kwargs.setdefault("prefetch_policy", "required")
    return RuntimeHostProcess(config=config, **kwargs)  # type: ignore[arg-type]


def _write_rendered_config(
    tmp_path: Path,
    *,
    required_secrets: list[str],
    mappings: list[dict[str, object]] | None = None,
) -> Path:
    """Write a rendered secret-resolver config YAML, mirroring what
    ``render_secret_resolver_config.py`` produces at deploy time."""
    data: dict[str, object] = {
        "required_secrets": required_secrets,
        "mappings": mappings or [],
    }
    path = tmp_path / "secret_resolver.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _infisical_mapping(logical_name: str, source_path: str) -> dict[str, object]:
    return {
        "logical_name": logical_name,
        "source": {"source_type": "infisical", "source_path": source_path},
    }


class TestNoOpWhenGateNotOptedIn:
    """The gate must be a strict no-op until a lane opts in."""

    def test_noop_when_env_var_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        process = _make_process()
        monkeypatch.delenv(_RENDERED_CONFIG_ENV_VAR, raising=False)
        # Must not raise, must not require a real handler.
        process._validate_required_secrets_via_resolver(MagicMock())

    def test_noop_when_rendered_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _make_process()
        monkeypatch.setenv(
            _RENDERED_CONFIG_ENV_VAR, str(tmp_path / "does-not-exist.yaml")
        )
        process._validate_required_secrets_via_resolver(MagicMock())

    def test_noop_when_required_secrets_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _make_process()
        path = _write_rendered_config(tmp_path, required_secrets=[])
        monkeypatch.setenv(_RENDERED_CONFIG_ENV_VAR, str(path))
        # No handler call should even be needed -- required_secrets is empty.
        process._validate_required_secrets_via_resolver(MagicMock())


class TestRequiredSecretsFailLoudOnAbsence:
    """RED proof (OMN-14951 acceptance bar): absent key MUST fail loudly."""

    def test_missing_key_raises_naming_the_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A declared-required key absent from Infisical fails the gate by name."""
        process = _make_process(prefetch_policy="required")
        path = _write_rendered_config(
            tmp_path,
            required_secrets=["llm.openai.api_key"],
            mappings=[_infisical_mapping("llm.openai.api_key", "OPENAI_API_KEY")],
        )
        monkeypatch.setenv(_RENDERED_CONFIG_ENV_VAR, str(path))

        mock_handler = MagicMock()
        mock_handler.get_secret_sync.return_value = None  # absent from Infisical

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            process._validate_required_secrets_via_resolver(mock_handler)

        message = str(exc_info.value)
        assert "OMN-14951" in message
        assert "llm.openai.api_key" in message

    def test_multiple_missing_keys_all_named_at_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Every missing key is named together -- not one failure at a time."""
        process = _make_process(prefetch_policy="required")
        path = _write_rendered_config(
            tmp_path,
            required_secrets=["a.secret", "b.secret", "c.secret"],
            mappings=[
                _infisical_mapping("a.secret", "A_SECRET"),
                _infisical_mapping("b.secret", "B_SECRET"),
                _infisical_mapping("c.secret", "C_SECRET"),
            ],
        )
        monkeypatch.setenv(_RENDERED_CONFIG_ENV_VAR, str(path))

        mock_handler = MagicMock()

        def _get_secret_sync(secret_name: str) -> SecretStr | None:
            # Only c.secret (mapped to C_SECRET) is present.
            if secret_name == "C_SECRET":
                return SecretStr("present-value")
            return None

        mock_handler.get_secret_sync.side_effect = _get_secret_sync

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            process._validate_required_secrets_via_resolver(mock_handler)

        message = str(exc_info.value)
        assert "a.secret" in message
        assert "b.secret" in message
        # The present key must NOT appear in the missing/errored report.
        assert "c.secret" not in message

    def test_unreachable_infisical_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """RED proof: Infisical unreachable/auth-failure MUST fail closed, not skip."""
        process = _make_process(prefetch_policy="required")
        path = _write_rendered_config(
            tmp_path,
            required_secrets=["db.postgres.password"],
            mappings=[_infisical_mapping("db.postgres.password", "POSTGRES_PASSWORD")],
        )
        monkeypatch.setenv(_RENDERED_CONFIG_ENV_VAR, str(path))

        mock_handler = MagicMock()
        mock_handler.get_secret_sync.side_effect = ConnectionError(
            "infisical unreachable: connection refused"
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            process._validate_required_secrets_via_resolver(mock_handler)

        message = str(exc_info.value)
        assert "OMN-14951" in message
        assert "db.postgres.password" in message

    def test_best_effort_policy_does_not_raise_but_does_not_hide_the_check(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """best_effort tolerates the failure (logs) rather than crashing boot."""
        import logging

        process = _make_process(prefetch_policy="best_effort")
        path = _write_rendered_config(
            tmp_path,
            required_secrets=["llm.openai.api_key"],
            mappings=[_infisical_mapping("llm.openai.api_key", "OPENAI_API_KEY")],
        )
        monkeypatch.setenv(_RENDERED_CONFIG_ENV_VAR, str(path))

        mock_handler = MagicMock()
        mock_handler.get_secret_sync.return_value = None

        with caplog.at_level(logging.WARNING):
            # Must NOT raise under best_effort.
            process._validate_required_secrets_via_resolver(mock_handler)

        assert any(
            "OMN-14951" in record.message or "Required-secrets" in record.message
            for record in caplog.records
        )


class TestRequiredSecretsGreenControl:
    """GREEN control case -- not vacuous, since the RED cases above exist."""

    def test_all_present_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _make_process(prefetch_policy="required")
        path = _write_rendered_config(
            tmp_path,
            required_secrets=["llm.openai.api_key", "db.postgres.password"],
            mappings=[
                _infisical_mapping("llm.openai.api_key", "OPENAI_API_KEY"),
                _infisical_mapping("db.postgres.password", "POSTGRES_PASSWORD"),
            ],
        )
        monkeypatch.setenv(_RENDERED_CONFIG_ENV_VAR, str(path))

        mock_handler = MagicMock()
        mock_handler.get_secret_sync.return_value = SecretStr("present-value")

        # Must not raise.
        process._validate_required_secrets_via_resolver(mock_handler)


class TestRenderedConfigStructuralInvariant:
    """A rendered config with a required key lacking an infisical mapping
    can never pass ModelSecretResolverConfig validation in the first place --
    this proves the construction-time gate (model_secret_resolver_config.py)
    and the boot-time gate (this module) are consistent, not two independently
    drifting checks."""

    def test_malformed_rendered_config_is_non_fatal_but_does_not_silently_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        process = _make_process(prefetch_policy="required")
        # required_secrets names a key with NO mapping at all -- this could
        # never have been produced by render_secret_resolver_config.py (which
        # validates via ModelSecretResolverConfig before writing), so this
        # simulates deploy-time file corruption/tampering.
        path = tmp_path / "secret_resolver.yaml"
        path.write_text(
            yaml.safe_dump({"required_secrets": ["orphan.secret"], "mappings": []}),
            encoding="utf-8",
        )
        monkeypatch.setenv(_RENDERED_CONFIG_ENV_VAR, str(path))

        with caplog.at_level(logging.WARNING):
            # Must not raise (config load/parse failure is logged, non-fatal --
            # this is a corruption diagnostic, not a routing defect this gate
            # is responsible for; the render-time gate already prevents this
            # shape from ever being written by the canonical path).
            process._validate_required_secrets_via_resolver(MagicMock())

        assert any(
            "Failed to load rendered secret resolver config" in record.message
            for record in caplog.records
        )

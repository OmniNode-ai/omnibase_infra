# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ConfigPrefetcher (OMN-2287)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.config_prefetcher import (
    ConfigPrefetcher,
    ModelPrefetchResult,
)
from omnibase_infra.runtime.config_discovery.models.model_config_requirement import (
    ModelConfigRequirement,
)
from omnibase_infra.runtime.config_discovery.models.model_config_requirements import (
    ModelConfigRequirements,
)


class TestModelPrefetchResult:
    """Tests for ModelPrefetchResult dataclass."""

    def test_empty_result(self) -> None:
        result = ModelPrefetchResult()
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.specs_attempted == 0

    def test_success_count(self) -> None:
        result = ModelPrefetchResult(
            resolved={"KEY1": SecretStr("val1"), "KEY2": SecretStr("val2")}
        )
        assert result.success_count == 2

    def test_failure_count(self) -> None:
        result = ModelPrefetchResult(
            missing=["KEY1"],
            errors={"KEY2": "not found"},
        )
        assert result.failure_count == 2


class TestConfigPrefetcher:
    """Tests for ConfigPrefetcher."""

    def _make_handler(self, secrets: dict[str, str] | None = None) -> MagicMock:
        """Create a mock handler with get_secret_sync."""
        handler = MagicMock()
        _secrets = secrets or {}

        def _get_secret_sync(
            secret_name: str,
            secret_path: str | None = None,
            **kwargs: object,
        ) -> SecretStr | None:
            return SecretStr(_secrets[secret_name]) if secret_name in _secrets else None

        handler.get_secret_sync = MagicMock(side_effect=_get_secret_sync)
        return handler

    def _make_requirements(
        self,
        transport_types: list[EnumInfraTransportType] | None = None,
        env_deps: list[tuple[str, str]] | None = None,
    ) -> ModelConfigRequirements:
        """Create test requirements."""
        reqs: list[ModelConfigRequirement] = []
        if env_deps:
            for idx, (key, source) in enumerate(env_deps):
                reqs.append(
                    ModelConfigRequirement(
                        key=key,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        source_contract=Path("/test/contract.yaml"),
                        source_field=f"dependencies[{idx}]",
                    )
                )
        return ModelConfigRequirements(
            requirements=tuple(reqs),
            transport_types=tuple(transport_types or []),
            contract_paths=(Path("/test/contract.yaml"),),
        )

    def test_prefetch_database_keys(self) -> None:
        """Should prefetch database transport keys."""
        handler = self._make_handler(secrets={"POSTGRES_HOST": "db.example.com"})
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        assert result.specs_attempted == 1
        assert "POSTGRES_HOST" in result.resolved
        assert result.resolved["POSTGRES_HOST"].get_secret_value() == "db.example.com"

    def test_prefetch_missing_keys(self) -> None:
        """Should report missing keys when handler returns None."""
        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        assert len(result.missing) > 0
        assert "POSTGRES_HOST" in result.missing

    def test_prefetch_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keys already in environment should skip Infisical fetch."""
        monkeypatch.setenv("POSTGRES_HOST", "from-env")
        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        assert "POSTGRES_HOST" in result.resolved
        assert result.resolved["POSTGRES_HOST"].get_secret_value() == "from-env"
        # Handler should NOT have been called for POSTGRES_HOST
        postgres_host_calls = [
            call
            for call in handler.get_secret_sync.call_args_list
            if call.kwargs.get("secret_name") == "POSTGRES_HOST"
        ]
        assert len(postgres_host_calls) == 0, (
            "POSTGRES_HOST should not be fetched from Infisical when present in env"
        )

    def test_prefetch_env_dependencies(self) -> None:
        """Should prefetch explicit env dependencies."""
        handler = self._make_handler(
            secrets={"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}
        )
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            env_deps=[("SLACK_WEBHOOK_URL", "/test/contract.yaml")]
        )

        result = prefetcher.prefetch(reqs)

        assert "SLACK_WEBHOOK_URL" in result.resolved

    def test_prefetch_with_service_slug(self) -> None:
        """Should use per-service paths when service_slug is set."""
        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler, service_slug="my-service")
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        # Verify the handler was called with per-service path
        calls = handler.get_secret_sync.call_args_list
        assert len(calls) > 0, "Expected get_secret_sync to be called"
        paths = [c.kwargs.get("secret_path", "") for c in calls]
        assert any("/services/my-service/" in p for p in paths)

    def test_prefetch_handler_error(self) -> None:
        """Should handle handler errors gracefully."""
        handler = MagicMock()
        handler.get_secret_sync = MagicMock(
            side_effect=RuntimeError("connection refused")
        )
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        # Should not raise
        result = prefetcher.prefetch(reqs)
        assert result.failure_count > 0

    def test_prefetch_infisical_required(self) -> None:
        """Should report errors for required keys when infisical_required=True."""
        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler, infisical_required=True)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        # When infisical_required=True, ConfigPrefetcher passes required=True
        # to specs_for_transports(), which sets spec.required=True on all
        # returned specs. The condition ``self._infisical_required and
        # spec.required`` therefore fires for every missing transport key,
        # routing them to result.errors rather than result.missing.
        assert result.failure_count > 0
        assert len(result.errors) > 0
        assert len(result.missing) == 0

    def test_handler_without_get_secret_sync(self) -> None:
        """Should handle handler without get_secret_sync method."""
        handler = MagicMock(spec=[])  # Empty spec = no methods
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)
        assert result.failure_count > 0

    def test_apply_to_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should apply resolved values to os.environ."""
        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler)

        # Clean environment
        monkeypatch.delenv("TEST_PREFETCH_KEY", raising=False)

        result = ModelPrefetchResult(
            resolved={"TEST_PREFETCH_KEY": SecretStr("test-value")}
        )

        applied = prefetcher.apply_to_environment(result)
        assert applied == 1
        assert os.environ.get("TEST_PREFETCH_KEY") == "test-value"

        # Cleanup
        monkeypatch.delenv("TEST_PREFETCH_KEY", raising=False)

    def test_apply_does_not_overwrite_existing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should NOT overwrite keys already in environment."""
        monkeypatch.setenv("EXISTING_KEY", "original")

        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler)

        result = ModelPrefetchResult(
            resolved={"EXISTING_KEY": SecretStr("from-infisical")}
        )

        applied = prefetcher.apply_to_environment(result)
        assert applied == 0
        assert os.environ.get("EXISTING_KEY") == "original"

    def test_prefetch_missing_required_env_dep_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing env-dep key with infisical_required=True must go to errors."""
        # Ensure the key is absent from the process environment
        monkeypatch.delenv("MISSING_ENV_DEP_KEY", raising=False)

        # Handler returns None — key is not in Infisical either
        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler, infisical_required=True)
        reqs = self._make_requirements(
            env_deps=[("MISSING_ENV_DEP_KEY", "/test/contract.yaml")]
        )

        result = prefetcher.prefetch(reqs)

        assert "MISSING_ENV_DEP_KEY" in result.errors, (
            "Key missing from both env and Infisical must appear in errors when "
            "infisical_required=True"
        )
        assert "MISSING_ENV_DEP_KEY" not in result.missing, (
            "Key must not be in missing when infisical_required=True"
        )

    def test_empty_requirements(self) -> None:
        """Should handle empty requirements gracefully."""
        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = ModelConfigRequirements()

        result = prefetcher.prefetch(reqs)
        assert result.success_count == 0
        assert result.failure_count == 0

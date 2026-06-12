# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ConfigPrefetcher (OMN-2287, OMN-13070)."""

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

    def test_prefetch_database_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should prefetch database transport keys."""
        # Remove all DATABASE transport keys from the environment so the
        # prefetcher is forced to call the handler rather than short-circuiting
        # via the "already in os.environ" fast-path (which would resolve to
        # whatever the host machine has set, not the mock value).
        from omnibase_infra.runtime.config_discovery.transport_config_map import (
            TransportConfigMap,
        )

        for key in TransportConfigMap.keys_for_transport(
            EnumInfraTransportType.DATABASE
        ):
            monkeypatch.delenv(key, raising=False)

        handler = self._make_handler(secrets={"POSTGRES_HOST": "db.example.com"})
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        assert result.specs_attempted == 1
        assert "POSTGRES_HOST" in result.resolved
        assert result.resolved["POSTGRES_HOST"].get_secret_value() == "db.example.com"

    def test_prefetch_missing_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should report missing keys when handler returns None."""
        # Remove all DATABASE transport keys from the environment so none of
        # them get resolved via the env fast-path — they should all go through
        # the handler (which returns None) and land in result.missing.
        from omnibase_infra.runtime.config_discovery.transport_config_map import (
            TransportConfigMap,
        )

        for key in TransportConfigMap.keys_for_transport(
            EnumInfraTransportType.DATABASE
        ):
            monkeypatch.delenv(key, raising=False)

        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        assert len(result.missing) > 0
        assert "POSTGRES_HOST" in result.missing

    def test_prefetch_env_override_uncontrolled_lane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On uncontrolled lane, keys already in environment skip Infisical fetch.

        Legacy behaviour: local dev without Infisical continues to work.
        infisical_required=False (default) means ambient env wins.
        """
        monkeypatch.setenv("POSTGRES_HOST", "from-env")
        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler)  # infisical_required=False
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        assert "POSTGRES_HOST" in result.resolved
        assert result.resolved["POSTGRES_HOST"].get_secret_value() == "from-env"
        # Handler should NOT have been called for POSTGRES_HOST on uncontrolled lane
        postgres_host_calls = [
            call
            for call in handler.get_secret_sync.call_args_list
            if call.kwargs.get("secret_name") == "POSTGRES_HOST"
        ]
        assert len(postgres_host_calls) == 0, (
            "POSTGRES_HOST should not be fetched from Infisical on uncontrolled lane "
            "when present in env"
        )

    def test_prefetch_env_dependencies(self) -> None:
        """Should prefetch explicit env dependencies."""
        handler = self._make_handler(secrets={"SLACK_BOT_TOKEN": "xoxb-test-token"})
        prefetcher = ConfigPrefetcher(handler=handler)
        reqs = self._make_requirements(
            env_deps=[("SLACK_BOT_TOKEN", "/test/contract.yaml")]
        )

        result = prefetcher.prefetch(reqs)

        assert "SLACK_BOT_TOKEN" in result.resolved

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

    def test_prefetch_infisical_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On controlled lane, missing keys (not in Infisical or env) go to errors."""
        # Remove all DATABASE transport keys so neither Infisical nor env can
        # supply them — they must land in result.errors on a controlled lane.
        from omnibase_infra.runtime.config_discovery.transport_config_map import (
            TransportConfigMap,
        )

        for key in TransportConfigMap.keys_for_transport(
            EnumInfraTransportType.DATABASE
        ):
            monkeypatch.delenv(key, raising=False)

        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler, infisical_required=True)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        # When infisical_required=True, ConfigPrefetcher passes required=True
        # to specs_for_transports(), which sets spec.required=True on all
        # returned specs. Keys missing from both Infisical and ambient env go to
        # result.errors rather than result.missing.
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

    def test_apply_does_not_overwrite_existing_uncontrolled_lane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On uncontrolled lane, apply_to_environment must NOT overwrite existing env.

        Legacy behaviour: infisical_required=False means ambient env wins.
        """
        monkeypatch.setenv("EXISTING_KEY", "original")

        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler)  # infisical_required=False

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

    # --- OMN-13070 controlled-lane precedence regression tests ---

    def test_controlled_lane_infisical_wins_over_stale_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On controlled lane, Infisical value beats a stale ambient env value.

        Regression for OMN-13070: config_prefetcher previously skipped keys
        already present in os.environ regardless of lane, allowing stale
        shell/compose state to override fetched configuration.
        """
        # Stale value already in env (e.g. from compose or developer's shell)
        monkeypatch.setenv("POSTGRES_HOST", "stale-from-compose")
        # Infisical has the authoritative value
        handler = self._make_handler(
            secrets={"POSTGRES_HOST": "authoritative-from-infisical"}
        )
        prefetcher = ConfigPrefetcher(handler=handler, infisical_required=True)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        assert "POSTGRES_HOST" in result.resolved
        assert (
            result.resolved["POSTGRES_HOST"].get_secret_value()
            == "authoritative-from-infisical"
        ), "Controlled lane: Infisical value must win over stale ambient env"
        # Handler MUST have been called — we must not short-circuit on env presence
        postgres_host_calls = [
            call
            for call in handler.get_secret_sync.call_args_list
            if call.kwargs.get("secret_name") == "POSTGRES_HOST"
        ]
        assert len(postgres_host_calls) == 1, (
            "Controlled lane: handler must be called for POSTGRES_HOST even when "
            "it is present in ambient env"
        )

    def test_controlled_lane_env_as_bootstrap_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On controlled lane, ambient env is used as bootstrap fallback only when
        Infisical returns None for a key, and the result is resolved (not missing).

        This covers the bootstrap case where a secret is seeded in env before
        Infisical is available (e.g. POSTGRES_PASSWORD during first-time startup).
        """
        # Infisical has no value for this key (e.g. not seeded yet)
        handler = self._make_handler(secrets={})
        # But the key is present in the bootstrap env
        monkeypatch.setenv("POSTGRES_HOST", "bootstrap-env-value")
        prefetcher = ConfigPrefetcher(handler=handler, infisical_required=True)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        # Key must be resolved from env fallback, not reported as error/missing
        assert "POSTGRES_HOST" in result.resolved, (
            "Controlled lane: ambient env must be used as bootstrap fallback "
            "when Infisical returns None"
        )
        assert (
            result.resolved["POSTGRES_HOST"].get_secret_value() == "bootstrap-env-value"
        )
        # Handler MUST have been called (we always try Infisical first)
        postgres_host_calls = [
            call
            for call in handler.get_secret_sync.call_args_list
            if call.kwargs.get("secret_name") == "POSTGRES_HOST"
        ]
        assert len(postgres_host_calls) == 1, (
            "Controlled lane: handler must be called before falling back to env"
        )

    def test_controlled_lane_apply_overwrites_stale_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On controlled lane, apply_to_environment must overwrite stale ambient env.

        Regression for OMN-13070: apply_to_environment previously skipped keys
        already present in os.environ on all lanes, allowing stale shell/compose
        state to survive the prefetch boundary.
        """
        monkeypatch.setenv("DB_HOST", "stale-compose-value")

        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler, infisical_required=True)

        result = ModelPrefetchResult(
            resolved={"DB_HOST": SecretStr("authoritative-from-infisical")}
        )

        applied = prefetcher.apply_to_environment(result)

        assert applied == 1, "Controlled lane: stale env key must be overwritten"
        assert os.environ.get("DB_HOST") == "authoritative-from-infisical", (
            "Controlled lane: apply_to_environment must replace stale ambient env "
            "value with the fetched authoritative value"
        )

    def test_controlled_lane_missing_from_both_infisical_and_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On controlled lane, a key absent from both Infisical and env is an error."""
        from omnibase_infra.runtime.config_discovery.transport_config_map import (
            TransportConfigMap,
        )

        for key in TransportConfigMap.keys_for_transport(
            EnumInfraTransportType.DATABASE
        ):
            monkeypatch.delenv(key, raising=False)

        handler = self._make_handler(secrets={})
        prefetcher = ConfigPrefetcher(handler=handler, infisical_required=True)
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        # Must appear in errors, not missing — missing is for uncontrolled-lane
        # soft failures; controlled-lane absence is a hard error.
        assert len(result.errors) > 0, (
            "Controlled lane: keys absent from both Infisical and env must be errors"
        )
        assert len(result.missing) == 0, (
            "Controlled lane: result.missing must be empty; use result.errors"
        )

    def test_precedence_order_uncontrolled_env_wins_infisical_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On uncontrolled lane, Infisical is NOT called when the key is in env.

        Verifies the uncontrolled-lane legacy shortcut is still active and
        does not regress to always-calling-Infisical after OMN-13070.
        """
        monkeypatch.setenv("POSTGRES_HOST", "local-dev-value")
        handler = self._make_handler(secrets={"POSTGRES_HOST": "infisical-value"})
        prefetcher = ConfigPrefetcher(handler=handler)  # infisical_required=False
        reqs = self._make_requirements(
            transport_types=[EnumInfraTransportType.DATABASE]
        )

        result = prefetcher.prefetch(reqs)

        assert result.resolved["POSTGRES_HOST"].get_secret_value() == "local-dev-value"
        # Handler must NOT have been called on uncontrolled lane when env is set
        postgres_host_calls = [
            call
            for call in handler.get_secret_sync.call_args_list
            if call.kwargs.get("secret_name") == "POSTGRES_HOST"
        ]
        assert len(postgres_host_calls) == 0, (
            "Uncontrolled lane: Infisical must not be called when key is in env"
        )

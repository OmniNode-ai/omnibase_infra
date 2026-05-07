# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OMN-10587: ConfigPrefetcher wiring via runtime profile.

Verifies that RuntimeHostProcess honours the ``prefetch_policy`` parameter:
    * ``"disabled"``    -- prefetcher is never invoked
    * ``"best_effort"`` -- prefetcher runs; boot continues despite missing keys
    * ``"required"``    -- prefetcher runs; missing keys raise ProtocolConfigurationError

All tests use a FakeSecretStore (via monkeypatching) so no live Infisical instance
is needed.  The prefetcher call chain is:

    RuntimeHostProcess._prefetch_config_from_infisical()
        -> ContractConfigExtractor.extract_from_paths()
        -> ConfigPrefetcher.prefetch()
        -> ProtocolSecretResolver.get_secret_sync()  (FakeSecretStore)

The INFISICAL_ADDR env var is set to a non-empty value so the early-exit guard
inside _prefetch_config_from_infisical() does not trigger the "skipped" path.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.config_discovery.config_prefetcher import (
    ModelPrefetchResult,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config

_EXTRACTOR_PATH = (
    "omnibase_infra.runtime.config_discovery.contract_config_extractor"
    ".ContractConfigExtractor"
)
_HANDLER_PATH = "omnibase_infra.handlers.handler_infisical.HandlerInfisical"
_PREFETCHER_PATH = (
    "omnibase_infra.runtime.config_discovery.config_prefetcher.ConfigPrefetcher"
)


def _make_process(prefetch_policy: str = "disabled") -> RuntimeHostProcess:
    config = make_runtime_config()
    return RuntimeHostProcess(config=config, prefetch_policy=prefetch_policy)


def _make_mock_extractor_with_requirements() -> MagicMock:
    requirements = MagicMock()
    requirements.requirements = [MagicMock()]
    requirements.errors = ()
    extractor = MagicMock()
    extractor.extract_from_paths.return_value = requirements
    return extractor


class TestDisabledPolicy:
    """profile=disabled: prefetcher must not be invoked."""

    @pytest.mark.asyncio
    async def test_prefetcher_not_invoked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With policy=disabled the prefetch method returns early before touching
        the extractor or handler — even when INFISICAL_ADDR is set."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")

        process = _make_process(prefetch_policy="disabled")

        with patch(_EXTRACTOR_PATH) as mock_extractor_cls:
            await process._prefetch_config_from_infisical()

        mock_extractor_cls.assert_not_called()
        assert process._config_prefetch_status == "skipped"

    @pytest.mark.asyncio
    async def test_prefetcher_not_invoked_without_infisical_addr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """policy=disabled skips before INFISICAL_ADDR is even checked."""
        monkeypatch.delenv("INFISICAL_ADDR", raising=False)
        process = _make_process(prefetch_policy="disabled")

        with patch(_EXTRACTOR_PATH) as mock_extractor_cls:
            await process._prefetch_config_from_infisical()

        mock_extractor_cls.assert_not_called()
        assert process._config_prefetch_status == "skipped"


class TestBestEffortPolicy:
    """profile=best_effort: prefetcher runs; missing keys are warnings, not errors."""

    @pytest.mark.asyncio
    async def test_boots_with_one_missing_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With policy=best_effort and one missing key, boot continues and a
        structured warning is logged."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csecret")
        monkeypatch.setenv("INFISICAL_PROJECT_ID", "pid")

        process = _make_process(prefetch_policy="best_effort")

        mock_extractor = _make_mock_extractor_with_requirements()

        missing_key = "POSTGRES_PASSWORD"
        mock_result = MagicMock(spec=ModelPrefetchResult)
        mock_result.success_count = 0
        mock_result.missing = (missing_key,)
        mock_result.errors = {}

        mock_prefetcher = MagicMock()
        mock_prefetcher.prefetch.return_value = mock_result
        mock_prefetcher.apply_to_environment.return_value = 0

        mock_handler = MagicMock()
        mock_handler.initialize = AsyncMock(return_value=None)
        mock_handler.shutdown = AsyncMock(return_value=None)

        with (
            patch(_EXTRACTOR_PATH, return_value=mock_extractor),
            patch(_PREFETCHER_PATH, return_value=mock_prefetcher),
            patch(_HANDLER_PATH, return_value=mock_handler),
            caplog.at_level(logging.INFO),
        ):
            # Must not raise
            await process._prefetch_config_from_infisical()

        assert process._config_prefetch_status == "ok"
        assert mock_prefetcher.prefetch.called

    @pytest.mark.asyncio
    async def test_prefetch_policy_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Structured log for prefetch_complete includes prefetch_policy field."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csecret")
        monkeypatch.setenv("INFISICAL_PROJECT_ID", "pid")

        process = _make_process(prefetch_policy="best_effort")

        mock_extractor = _make_mock_extractor_with_requirements()

        mock_result = MagicMock(spec=ModelPrefetchResult)
        mock_result.success_count = 2
        mock_result.missing = ()
        mock_result.errors = {}

        mock_prefetcher = MagicMock()
        mock_prefetcher.prefetch.return_value = mock_result
        mock_prefetcher.apply_to_environment.return_value = 2

        mock_handler = MagicMock()
        mock_handler.initialize = AsyncMock(return_value=None)
        mock_handler.shutdown = AsyncMock(return_value=None)

        with (
            patch(_EXTRACTOR_PATH, return_value=mock_extractor),
            patch(_PREFETCHER_PATH, return_value=mock_prefetcher),
            patch(_HANDLER_PATH, return_value=mock_handler),
            caplog.at_level(logging.INFO),
        ):
            await process._prefetch_config_from_infisical()

        assert process._config_prefetch_status == "ok"
        # Verify the prefetch was run (not skipped)
        assert mock_prefetcher.prefetch.called


class TestRequiredPolicy:
    """profile=required: prefetcher runs; missing keys raise ProtocolConfigurationError."""

    @pytest.mark.asyncio
    async def test_raises_on_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With policy=required and a missing key, ProtocolConfigurationError is raised
        with the missing key name in the error message."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csecret")
        monkeypatch.setenv("INFISICAL_PROJECT_ID", "pid")

        process = _make_process(prefetch_policy="required")

        mock_extractor = _make_mock_extractor_with_requirements()

        missing_key = "POSTGRES_PASSWORD"
        mock_result = MagicMock(spec=ModelPrefetchResult)
        mock_result.success_count = 0
        mock_result.missing = (missing_key,)
        mock_result.errors = {}

        mock_prefetcher = MagicMock()
        mock_prefetcher.prefetch.return_value = mock_result
        mock_prefetcher.apply_to_environment.return_value = 0

        mock_handler = MagicMock()
        mock_handler.initialize = AsyncMock(return_value=None)
        mock_handler.shutdown = AsyncMock(return_value=None)

        with (
            patch(_EXTRACTOR_PATH, return_value=mock_extractor),
            patch(_PREFETCHER_PATH, return_value=mock_prefetcher),
            patch(_HANDLER_PATH, return_value=mock_handler),
            pytest.raises(ProtocolConfigurationError) as exc_info,
        ):
            await process._prefetch_config_from_infisical()

        assert missing_key in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_on_error_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With policy=required and a key that errored, raise includes the key name."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csecret")
        monkeypatch.setenv("INFISICAL_PROJECT_ID", "pid")

        process = _make_process(prefetch_policy="required")

        mock_extractor = _make_mock_extractor_with_requirements()

        errored_key = "KAFKA_BOOTSTRAP_SERVERS"
        mock_result = MagicMock(spec=ModelPrefetchResult)
        mock_result.success_count = 0
        mock_result.missing = ()
        mock_result.errors = {errored_key: "connection refused"}

        mock_prefetcher = MagicMock()
        mock_prefetcher.prefetch.return_value = mock_result
        mock_prefetcher.apply_to_environment.return_value = 0

        mock_handler = MagicMock()
        mock_handler.initialize = AsyncMock(return_value=None)
        mock_handler.shutdown = AsyncMock(return_value=None)

        with (
            patch(_EXTRACTOR_PATH, return_value=mock_extractor),
            patch(_PREFETCHER_PATH, return_value=mock_prefetcher),
            patch(_HANDLER_PATH, return_value=mock_handler),
            pytest.raises(ProtocolConfigurationError) as exc_info,
        ):
            await process._prefetch_config_from_infisical()

        assert errored_key in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_raise_when_all_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With policy=required and all keys resolved, boot continues without error."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csecret")
        monkeypatch.setenv("INFISICAL_PROJECT_ID", "pid")

        process = _make_process(prefetch_policy="required")

        mock_extractor = _make_mock_extractor_with_requirements()

        mock_result = MagicMock(spec=ModelPrefetchResult)
        mock_result.success_count = 3
        mock_result.missing = ()
        mock_result.errors = {}

        mock_prefetcher = MagicMock()
        mock_prefetcher.prefetch.return_value = mock_result
        mock_prefetcher.apply_to_environment.return_value = 3

        mock_handler = MagicMock()
        mock_handler.initialize = AsyncMock(return_value=None)
        mock_handler.shutdown = AsyncMock(return_value=None)

        with (
            patch(_EXTRACTOR_PATH, return_value=mock_extractor),
            patch(_PREFETCHER_PATH, return_value=mock_prefetcher),
            patch(_HANDLER_PATH, return_value=mock_handler),
        ):
            # Must not raise
            await process._prefetch_config_from_infisical()

        assert process._config_prefetch_status == "ok"

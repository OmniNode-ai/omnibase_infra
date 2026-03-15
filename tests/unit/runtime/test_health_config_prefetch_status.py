# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for OMN-3902: Health endpoint exposes config prefetch status.

Validates that:
1. health_check() includes `config_prefetch_status` in the response
2. The status reflects the actual prefetch outcome (not hardcoded)
3. Each of the 5 status vocabulary values is observable
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config

# Lazy-import patch targets (same as test_prefetch_contract_isolation.py)
_EXTRACTOR_PATH = (
    "omnibase_infra.runtime.config_discovery.contract_config_extractor"
    ".ContractConfigExtractor"
)
_PREFETCHER_PATH = (
    "omnibase_infra.runtime.config_discovery.config_prefetcher.ConfigPrefetcher"
)
_HANDLER_PATH = "omnibase_infra.handlers.handler_infisical.HandlerInfisical"


def _make_process(**kwargs: object) -> RuntimeHostProcess:
    """Create a RuntimeHostProcess with minimal config for prefetch testing."""
    config = make_runtime_config()
    return RuntimeHostProcess(config=config, **kwargs)  # type: ignore[arg-type]


class TestHealthCheckIncludesConfigPrefetchStatus:
    """Verify config_prefetch_status appears in health_check response."""

    @pytest.mark.asyncio
    async def test_health_check_contains_config_prefetch_status_key(self) -> None:
        """health_check() must include the config_prefetch_status field."""
        process = _make_process()
        health = await process.health_check()

        assert "config_prefetch_status" in health
        assert isinstance(health["config_prefetch_status"], str)

    @pytest.mark.asyncio
    async def test_default_status_is_pending(self) -> None:
        """Before prefetch runs, status should be 'pending'."""
        process = _make_process()

        assert process._config_prefetch_status == "pending"

        health = await process.health_check()
        assert health["config_prefetch_status"] == "pending"

    @pytest.mark.asyncio
    async def test_status_skipped_when_no_infisical_addr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When INFISICAL_ADDR is not set, status should be 'skipped'."""
        monkeypatch.delenv("INFISICAL_ADDR", raising=False)

        process = _make_process()
        await process._prefetch_config_from_infisical()

        health = await process.health_check()
        assert health["config_prefetch_status"] == "skipped"

    @pytest.mark.asyncio
    async def test_status_skipped_when_credentials_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When INFISICAL_ADDR is set but credentials are missing, status is 'skipped'."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://localhost:8880")
        monkeypatch.delenv("INFISICAL_CLIENT_ID", raising=False)
        monkeypatch.delenv("INFISICAL_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("INFISICAL_PROJECT_ID", raising=False)

        # Mock extractor to return requirements so we get past the first gate
        mock_extractor = MagicMock()
        mock_requirements = MagicMock()
        mock_requirements.requirements = [MagicMock()]  # non-empty
        mock_extractor.return_value.extract_from_paths.return_value = mock_requirements

        with patch(_EXTRACTOR_PATH, mock_extractor):
            process = _make_process()
            await process._prefetch_config_from_infisical()

        health = await process.health_check()
        assert health["config_prefetch_status"] == "skipped"

    @pytest.mark.asyncio
    async def test_status_degraded_no_requirements(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When INFISICAL_ADDR is set but no requirements found, status is 'degraded_no_requirements'."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://localhost:8880")

        mock_extractor = MagicMock()
        mock_requirements = MagicMock()
        mock_requirements.requirements = []  # empty
        mock_requirements.errors = ()  # no extraction errors → degraded_no_requirements
        mock_extractor.return_value.extract_from_paths.return_value = mock_requirements

        with patch(_EXTRACTOR_PATH, mock_extractor):
            process = _make_process()
            await process._prefetch_config_from_infisical()

        health = await process.health_check()
        assert health["config_prefetch_status"] == "degraded_no_requirements"

    @pytest.mark.asyncio
    async def test_status_ok_after_successful_prefetch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a successful prefetch, status should be 'ok'."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://localhost:8880")
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "test-id")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "test-secret")
        monkeypatch.setenv("INFISICAL_PROJECT_ID", "test-project")

        mock_extractor = MagicMock()
        mock_requirements = MagicMock()
        mock_requirements.requirements = [MagicMock()]
        mock_requirements.errors = ()  # no extraction errors
        mock_extractor.return_value.extract_from_paths.return_value = mock_requirements

        mock_result = MagicMock()
        mock_result.success_count = 1
        mock_result.missing = []
        mock_result.errors = {}

        mock_prefetcher = MagicMock()
        mock_prefetcher.return_value.prefetch.return_value = mock_result
        mock_prefetcher.return_value.apply_to_environment.return_value = 1

        mock_handler_cls = MagicMock()
        mock_handler_instance = AsyncMock()
        mock_handler_cls.return_value = mock_handler_instance

        with (
            patch(_EXTRACTOR_PATH, mock_extractor),
            patch(_PREFETCHER_PATH, mock_prefetcher),
            patch(_HANDLER_PATH, mock_handler_cls),
        ):
            process = _make_process()
            await process._prefetch_config_from_infisical()

        health = await process.health_check()
        assert health["config_prefetch_status"] == "ok"

    @pytest.mark.asyncio
    async def test_status_degraded_error_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When prefetch raises an exception, status should be 'degraded_error'."""
        monkeypatch.setenv("INFISICAL_ADDR", "http://localhost:8880")

        # Make the extractor raise to trigger the outer except
        with patch(_EXTRACTOR_PATH, side_effect=RuntimeError("connection refused")):
            process = _make_process()
            await process._prefetch_config_from_infisical()

        health = await process.health_check()
        assert health["config_prefetch_status"] == "degraded_error"

    @pytest.mark.asyncio
    async def test_status_degraded_error_on_soft_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When prefetch returns result.errors (soft failure), status must be 'degraded_error'.

        Regression test for the bug where config_prefetch_status was set to 'ok'
        unconditionally even when ConfigPrefetcher.prefetch() returned soft errors
        via result.errors without raising an exception.
        """
        monkeypatch.setenv("INFISICAL_ADDR", "http://localhost:8880")
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "test-id")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "test-secret")
        monkeypatch.setenv("INFISICAL_PROJECT_ID", "test-project")

        mock_extractor = MagicMock()
        mock_requirements = MagicMock()
        mock_requirements.requirements = [MagicMock()]
        mock_extractor.return_value.extract_from_paths.return_value = mock_requirements

        # Simulate a soft failure: prefetch() returns without raising,
        # but result.errors is non-empty.
        mock_result = MagicMock()
        mock_result.success_count = 0
        mock_result.missing = []
        mock_result.errors = {"SOME_KEY": "fetch failed"}

        mock_prefetcher = MagicMock()
        mock_prefetcher.return_value.prefetch.return_value = mock_result
        mock_prefetcher.return_value.apply_to_environment.return_value = 0

        mock_handler_cls = MagicMock()
        mock_handler_instance = AsyncMock()
        mock_handler_cls.return_value = mock_handler_instance

        with (
            patch(_EXTRACTOR_PATH, mock_extractor),
            patch(_PREFETCHER_PATH, mock_prefetcher),
            patch(_HANDLER_PATH, mock_handler_cls),
        ):
            process = _make_process()
            await process._prefetch_config_from_infisical()

        health = await process.health_check()
        assert health["config_prefetch_status"] == "degraded_error"

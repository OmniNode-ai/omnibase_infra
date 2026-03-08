# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for OMN-3893: Decouple prefetch contract scan from handler contract paths.

Verifies that _prefetch_config_from_infisical uses the omnibase_infra package root
(not self._contract_paths) and respects the ONEX_NODE_CONTRACTS_DIR env override.
Also validates _config_prefetch_status state transitions.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config

# ContractConfigExtractor and ConfigPrefetcher are lazy-imported inside the
# prefetch method.  To patch them we must target the source module, then
# ensure the lazy import picks up the mock.
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


class TestPrefetchDoesNotUseContractPaths:
    """Verify prefetch scans package root, not self._contract_paths."""

    @pytest.mark.asyncio
    async def test_prefetch_does_not_use_contract_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even when _contract_paths is set to /app/contracts, prefetch should
        resolve the package root from omnibase_infra.__file__."""
        process = _make_process(contract_paths=["/app/contracts"])

        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")

        # Track what paths the extractor receives
        captured_paths: list[list[Path]] = []

        mock_requirements = MagicMock()
        mock_requirements.requirements = []  # empty = no requirements

        mock_extractor = MagicMock()
        mock_extractor.extract_from_paths.side_effect = lambda paths: (
            captured_paths.append(list(paths)),
            mock_requirements,
        )[1]

        with patch(_EXTRACTOR_PATH, return_value=mock_extractor):
            await process._prefetch_config_from_infisical()

        # Verify the extractor was called with paths
        assert len(captured_paths) == 1
        scanned = captured_paths[0]
        assert len(scanned) == 1

        # The scanned path should be the omnibase_infra package root,
        # NOT /app/contracts
        import omnibase_infra as _pkg

        expected_root = Path(_pkg.__file__).parent
        assert scanned[0] == expected_root
        assert str(scanned[0]) != "/app/contracts"


class TestPrefetchRespectsOnexNodeContractsDir:
    """Verify ONEX_NODE_CONTRACTS_DIR env override is honoured."""

    @pytest.mark.asyncio
    async def test_prefetch_respects_onex_node_contracts_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When ONEX_NODE_CONTRACTS_DIR is set, prefetch uses that path."""
        process = _make_process()

        custom_dir = tmp_path / "custom_contracts"
        custom_dir.mkdir()

        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")
        monkeypatch.setenv("ONEX_NODE_CONTRACTS_DIR", str(custom_dir))

        captured_paths: list[list[Path]] = []

        mock_requirements = MagicMock()
        mock_requirements.requirements = []

        mock_extractor = MagicMock()
        mock_extractor.extract_from_paths.side_effect = lambda paths: (
            captured_paths.append(list(paths)),
            mock_requirements,
        )[1]

        with patch(_EXTRACTOR_PATH, return_value=mock_extractor):
            await process._prefetch_config_from_infisical()

        assert len(captured_paths) == 1
        assert captured_paths[0] == [custom_dir]


class TestPrefetchStatusTransitions:
    """Validate _config_prefetch_status transitions through the 5-state vocabulary."""

    def test_initial_status_is_pending(self) -> None:
        process = _make_process()
        assert process._config_prefetch_status == "pending"

    @pytest.mark.asyncio
    async def test_status_skipped_when_no_infisical_addr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = _make_process()
        monkeypatch.delenv("INFISICAL_ADDR", raising=False)

        await process._prefetch_config_from_infisical()
        assert process._config_prefetch_status == "skipped"

    @pytest.mark.asyncio
    async def test_status_degraded_no_requirements(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = _make_process()
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")

        mock_requirements = MagicMock()
        mock_requirements.requirements = []
        mock_requirements.errors = ()  # no extraction errors → degraded_no_requirements

        mock_extractor = MagicMock()
        mock_extractor.extract_from_paths.return_value = mock_requirements

        with patch(_EXTRACTOR_PATH, return_value=mock_extractor):
            await process._prefetch_config_from_infisical()

        assert process._config_prefetch_status == "degraded_no_requirements"

    @pytest.mark.asyncio
    async def test_status_degraded_error_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = _make_process()
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")

        with patch(_EXTRACTOR_PATH, side_effect=RuntimeError("boom")):
            await process._prefetch_config_from_infisical()

        assert process._config_prefetch_status == "degraded_error"

    @pytest.mark.asyncio
    async def test_status_skipped_when_no_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = _make_process()
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")
        monkeypatch.delenv("INFISICAL_CLIENT_ID", raising=False)
        monkeypatch.delenv("INFISICAL_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("INFISICAL_PROJECT_ID", raising=False)

        mock_requirements = MagicMock()
        mock_requirements.requirements = [MagicMock()]
        mock_requirements.errors = ()

        mock_extractor = MagicMock()
        mock_extractor.extract_from_paths.return_value = mock_requirements

        with patch(_EXTRACTOR_PATH, return_value=mock_extractor):
            await process._prefetch_config_from_infisical()

        assert process._config_prefetch_status == "skipped"

    @pytest.mark.asyncio
    async def test_status_ok_on_successful_prefetch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = _make_process()
        monkeypatch.setenv("INFISICAL_ADDR", "http://fake:8080")
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "test-id")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "test-secret")
        monkeypatch.setenv("INFISICAL_PROJECT_ID", "test-project")

        mock_requirements = MagicMock()
        mock_requirements.requirements = [MagicMock()]
        mock_requirements.errors = ()

        mock_extractor = MagicMock()
        mock_extractor.extract_from_paths.return_value = mock_requirements

        mock_result = MagicMock()
        mock_result.success_count = 1
        mock_result.missing = []
        mock_result.errors = {}

        mock_prefetcher = MagicMock()
        mock_prefetcher.prefetch.return_value = mock_result
        mock_prefetcher.apply_to_environment.return_value = 1

        mock_handler = AsyncMock()

        with (
            patch(_EXTRACTOR_PATH, return_value=mock_extractor),
            patch(_PREFETCHER_PATH, return_value=mock_prefetcher),
            patch(_HANDLER_PATH, return_value=mock_handler),
        ):
            await process._prefetch_config_from_infisical()

        assert process._config_prefetch_status == "ok"


class TestNodeContractDiscoveryInvariant:
    """Ensure the package root derivation is stable across install modes."""

    def test_package_root_resolves_to_omnibase_infra(self) -> None:
        """Verify that the package root derivation produces a valid path
        containing node contract YAML files (or at least the package __init__)."""
        import omnibase_infra as _pkg

        package_root = Path(_pkg.__file__).parent
        assert package_root.is_dir()
        assert (package_root / "__init__.py").exists()

    def test_package_root_is_not_runtime_dir(self) -> None:
        """The old code used Path(__file__).parent.parent from within the
        runtime module — that happened to be the package root only by
        accident.  Verify we use the canonical import."""
        import omnibase_infra as _pkg

        package_root = Path(_pkg.__file__).parent
        runtime_parent_parent = Path(
            __import__(
                "omnibase_infra.runtime.service_runtime_host_process",
                fromlist=["service_runtime_host_process"],
            ).__file__
        ).parent.parent
        # Both should resolve to the same directory — but the import-based
        # approach is intentional and stable, not an accidental coincidence.
        assert package_root == runtime_parent_parent

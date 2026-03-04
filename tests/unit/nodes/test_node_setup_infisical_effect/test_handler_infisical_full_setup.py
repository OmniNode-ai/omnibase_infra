# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerInfisicalFullSetup.

Tests:
    - test_full_setup_calls_provision_then_seed_in_order
    - test_skip_identity_skips_provision_script
    - test_dry_run_passes_dry_run_flag_to_seed_script
    - test_provision_failure_sets_success_false_with_error_summary

Invariant I3 — Monkeypatch discipline:
    Patches applied via monkeypatch.setattr(handler_mod, "asyncio", ...)
    where handler_mod is the handler module.

Ticket: OMN-3494
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_setup_infisical_effect.handlers.handler_infisical_full_setup as handler_mod
from omnibase_infra.nodes.node_setup_infisical_effect.handlers.handler_infisical_full_setup import (
    HandlerInfisicalFullSetup,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_container() -> MagicMock:
    container = MagicMock()
    container.config = MagicMock()
    return container


def _make_mock_process(returncode: int = 0, stdout: bytes = b"OK") -> MagicMock:
    """Create a mock subprocess that returns the given returncode and stdout."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, None))
    return proc


def _make_asyncio_mock(side_effects: list[MagicMock]) -> MagicMock:
    """Create a mock asyncio module that returns processes in sequence."""
    mock_asyncio = MagicMock(spec=asyncio)
    call_count: list[int] = [0]

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> MagicMock:
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(side_effects):
            return side_effects[idx]
        return _make_mock_process(returncode=0, stdout=b"OK")

    mock_asyncio.create_subprocess_exec = fake_create_subprocess_exec
    mock_asyncio.subprocess = asyncio.subprocess
    return mock_asyncio


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.unit
class TestHandlerInfisicalFullSetup:
    """Unit tests for HandlerInfisicalFullSetup."""

    @pytest.mark.asyncio
    async def test_full_setup_calls_provision_then_seed_in_order(
        self,
        tmp_path: Path,
    ) -> None:
        """provision-infisical.py runs before seed-infisical.py (skip_identity=False)."""
        # Create fake script files so Path checks succeed
        provision_script = tmp_path / "provision-infisical.py"
        seed_script = tmp_path / "seed-infisical.py"
        provision_script.write_text("# fake provision")
        seed_script.write_text("# fake seed")

        call_order: list[str] = []
        proc_ok = _make_mock_process(returncode=0, stdout=b"OK")

        async def fake_exec(*args: Any, **kwargs: Any) -> MagicMock:
            call_order.append(str(args[1]))
            return proc_ok

        mock_asyncio = MagicMock(spec=asyncio)
        mock_asyncio.create_subprocess_exec = fake_exec
        mock_asyncio.subprocess = asyncio.subprocess

        container = _make_container()
        handler = HandlerInfisicalFullSetup(container)
        await handler.initialize({})

        with patch.object(handler_mod, "asyncio", mock_asyncio):
            output = await handler.execute(
                {
                    "correlation_id": uuid4(),
                    "skip_identity": False,
                    "dry_run": False,
                    "scripts_dir": tmp_path,
                }
            )

        result = output.result
        assert result.success is True
        assert result.status == "completed"
        # provision is called before seed
        assert len(call_order) == 2
        assert "provision-infisical.py" in call_order[0]
        assert "seed-infisical.py" in call_order[1]

    @pytest.mark.asyncio
    async def test_skip_identity_skips_provision_script(
        self,
        tmp_path: Path,
    ) -> None:
        """When skip_identity=True, provision-infisical.py must NOT be called."""
        provision_script = tmp_path / "provision-infisical.py"
        seed_script = tmp_path / "seed-infisical.py"
        provision_script.write_text("# fake provision")
        seed_script.write_text("# fake seed")

        call_order: list[str] = []
        proc_ok = _make_mock_process(returncode=0, stdout=b"OK")

        async def fake_exec(*args: Any, **kwargs: Any) -> MagicMock:
            call_order.append(str(args[1]))
            return proc_ok

        mock_asyncio = MagicMock(spec=asyncio)
        mock_asyncio.create_subprocess_exec = fake_exec
        mock_asyncio.subprocess = asyncio.subprocess

        container = _make_container()
        handler = HandlerInfisicalFullSetup(container)
        await handler.initialize({})

        with patch.object(handler_mod, "asyncio", mock_asyncio):
            output = await handler.execute(
                {
                    "correlation_id": uuid4(),
                    "skip_identity": True,
                    "dry_run": False,
                    "scripts_dir": tmp_path,
                }
            )

        result = output.result
        assert result.success is True
        assert result.status == "completed"
        # Only seed is called (provision is skipped)
        assert len(call_order) == 1
        assert "seed-infisical.py" in call_order[0]

    @pytest.mark.asyncio
    async def test_dry_run_passes_dry_run_flag_to_seed_script(
        self,
        tmp_path: Path,
    ) -> None:
        """When dry_run=True, ``--dry-run`` must be passed to seed-infisical.py."""
        provision_script = tmp_path / "provision-infisical.py"
        seed_script = tmp_path / "seed-infisical.py"
        provision_script.write_text("# fake")
        seed_script.write_text("# fake")

        captured_args: list[tuple[Any, ...]] = []
        proc_ok = _make_mock_process(returncode=0, stdout=b"OK")

        async def fake_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured_args.append(args)
            return proc_ok

        mock_asyncio = MagicMock(spec=asyncio)
        mock_asyncio.create_subprocess_exec = fake_exec
        mock_asyncio.subprocess = asyncio.subprocess

        container = _make_container()
        handler = HandlerInfisicalFullSetup(container)
        await handler.initialize({})

        with patch.object(handler_mod, "asyncio", mock_asyncio):
            output = await handler.execute(
                {
                    "correlation_id": uuid4(),
                    "skip_identity": False,
                    "dry_run": True,
                    "scripts_dir": tmp_path,
                }
            )

        result = output.result
        assert result.success is True
        # seed call is captured_args[1] (after provision at [0])
        assert len(captured_args) == 2
        seed_args = captured_args[1]
        # args are: (python, script_path, extra_args...) → flattened tuple
        assert "--dry-run" in seed_args
        assert "--execute" not in seed_args

    @pytest.mark.asyncio
    async def test_provision_failure_sets_success_false_with_error_summary(
        self,
        tmp_path: Path,
    ) -> None:
        """When provision-infisical.py exits non-zero, success=False with error summary."""
        provision_script = tmp_path / "provision-infisical.py"
        seed_script = tmp_path / "seed-infisical.py"
        provision_script.write_text("# fake")
        seed_script.write_text("# fake")

        proc_fail = _make_mock_process(
            returncode=1, stdout=b"Error: could not connect to Infisical"
        )

        async def fake_exec(*args: Any, **kwargs: Any) -> MagicMock:
            return proc_fail

        mock_asyncio = MagicMock(spec=asyncio)
        mock_asyncio.create_subprocess_exec = fake_exec
        mock_asyncio.subprocess = asyncio.subprocess

        container = _make_container()
        handler = HandlerInfisicalFullSetup(container)
        await handler.initialize({})

        with patch.object(handler_mod, "asyncio", mock_asyncio):
            output = await handler.execute(
                {
                    "correlation_id": uuid4(),
                    "skip_identity": False,
                    "dry_run": False,
                    "scripts_dir": tmp_path,
                }
            )

        result = output.result
        assert result.success is False
        assert result.status == "failed"
        assert result.error is not None
        assert "provision-infisical.py failed" in result.error


__all__: list[str] = ["TestHandlerInfisicalFullSetup"]

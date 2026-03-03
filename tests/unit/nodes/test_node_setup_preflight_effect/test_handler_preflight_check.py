# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerPreflightCheck.

TDD coverage for all 7 preflight checks.

Ticket: OMN-3492
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_setup_preflight_effect.handlers.handler_preflight_check as handler_mod
from omnibase_infra.enums import EnumHandlerType
from omnibase_infra.nodes.node_setup_preflight_effect.handlers.handler_preflight_check import (
    HandlerPreflightCheck,
    _run_all_checks,
)


@pytest.fixture
def mock_container() -> MagicMock:
    return MagicMock()


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerPreflightCheck:
    return HandlerPreflightCheck(mock_container)


def _make_proc_result(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> SimpleNamespace:
    """Build a minimal subprocess.CompletedProcess-like object."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.unit
class TestHandlerPreflightCheck:
    """Unit tests for HandlerPreflightCheck."""

    async def test_handler_type_is_node_handler(
        self, handler: HandlerPreflightCheck
    ) -> None:
        """handler_type property must return NODE_HANDLER."""
        assert handler.handler_type == EnumHandlerType.NODE_HANDLER

    async def test_checks_has_exactly_seven_items_for_minimal(
        self,
        mock_container: MagicMock,
    ) -> None:
        """_run_all_checks() must return exactly 7 results."""
        # Patch all external calls to succeed
        with (
            patch.object(
                handler_mod.subprocess,
                "run",
                return_value=_make_proc_result(stdout="25.0.0"),
            ),
            patch.object(
                handler_mod,
                "_get_python_version_info",
                return_value=(3, 12, 0),
            ),
            patch.object(
                handler_mod,
                "_check_port_free",
                return_value=True,
            ),
            patch.object(
                handler_mod,
                "_omnibase_dir",
                return_value=MagicMock(
                    exists=lambda: True,
                    __str__=lambda self: "/home/.omnibase",
                ),
            ),
            patch("os.access", return_value=True),
            patch("os.environ.get", return_value="secret"),
        ):
            checks = _run_all_checks()

        assert len(checks) == 7

    async def test_docker_version_below_24_fails_check(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Docker version < 24.0.0 must produce a failed docker_version check."""
        with patch.object(
            handler_mod.subprocess,
            "run",
            return_value=_make_proc_result(stdout="23.0.5"),
        ):
            result = handler_mod._check_docker_version()

        assert result.check_key == "docker_version"
        assert result.passed is False
        assert "23.0.5" in result.message

    async def test_port_in_use_fails_port_availability_check(
        self,
        mock_container: MagicMock,
    ) -> None:
        """A port that is in use must produce a failed port_availability check."""
        # _check_port_free returns False meaning port is NOT free (occupied)
        with patch.object(handler_mod, "_check_port_free", return_value=False):
            result = handler_mod._check_port_availability()

        assert result.check_key == "port_availability"
        assert result.passed is False
        assert "in use" in result.message

    async def test_missing_postgres_password_fails_env_check(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Missing POSTGRES_PASSWORD must produce a failed postgres_password_set check."""
        with patch.dict("os.environ", {}, clear=True):
            result = handler_mod._check_postgres_password()

        assert result.check_key == "postgres_password_set"
        assert result.passed is False
        assert "POSTGRES_PASSWORD" in result.message

    async def test_all_checks_pass_returns_passed_true(
        self,
        handler: HandlerPreflightCheck,
    ) -> None:
        """When all 7 checks pass, execute() must return a result with passed=True."""
        from omnibase_core.models.dispatch import ModelHandlerOutput

        with (
            patch.object(
                handler_mod.subprocess,
                "run",
                return_value=_make_proc_result(stdout="25.0.0"),
            ),
            patch.object(
                handler_mod,
                "_get_python_version_info",
                return_value=(3, 12, 0),
            ),
            patch.object(
                handler_mod,
                "_check_port_free",
                return_value=True,
            ),
            patch.object(
                handler_mod,
                "_omnibase_dir",
                return_value=MagicMock(
                    exists=lambda: True,
                    __str__=lambda self: "/home/.omnibase",
                ),
            ),
            patch("os.access", return_value=True),
            patch.dict("os.environ", {"POSTGRES_PASSWORD": "secret"}),
        ):
            await handler.initialize({})
            output = await handler.execute({"correlation_id": uuid4()})

        assert isinstance(output, ModelHandlerOutput)
        from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_effect_output import (
            ModelPreflightEffectOutput,
        )

        assert isinstance(output.result, ModelPreflightEffectOutput)
        assert output.result.passed is True
        assert len(output.result.checks) == 7

    async def test_subprocess_stderr_captured_in_detail_on_failure(
        self,
        mock_container: MagicMock,
    ) -> None:
        """stderr from a failed subprocess call must appear in the check detail field."""
        stderr_msg = (
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
        )
        with patch.object(
            handler_mod.subprocess,
            "run",
            return_value=_make_proc_result(returncode=1, stderr=stderr_msg),
        ):
            result = handler_mod._check_docker_daemon()

        assert result.check_key == "docker_daemon"
        assert result.passed is False
        assert result.detail is not None
        assert "docker.sock" in result.detail

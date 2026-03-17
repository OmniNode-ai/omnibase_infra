# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for verification executor (OMN-5266).

All external calls are mocked -- no real subprocess, network, or filesystem access.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.probes.model_verification_result import ModelVerificationResult
from omnibase_infra.probes.model_verification_spec import ModelVerificationSpec
from omnibase_infra.probes.verification_executor import execute_verification


class TestCommandExit0:
    """Tests for command_exit_0 check type."""

    @pytest.mark.asyncio
    async def test_command_succeeds(self) -> None:
        spec = ModelVerificationSpec(
            check_type="command_exit_0",
            target="echo hello",
        )
        with patch(
            "omnibase_infra.probes.verification_executor.asyncio.create_subprocess_shell"
        ) as mock_proc:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate.return_value = (b"hello\n", b"")
            mock_proc.return_value = proc
            result = await execute_verification(spec)
        assert result.passed is True
        assert result.check_type == "command_exit_0"
        assert result.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_command_fails(self) -> None:
        spec = ModelVerificationSpec(
            check_type="command_exit_0",
            target="false",
        )
        with patch(
            "omnibase_infra.probes.verification_executor.asyncio.create_subprocess_shell"
        ) as mock_proc:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate.return_value = (b"", b"error")
            mock_proc.return_value = proc
            result = await execute_verification(spec)
        assert result.passed is False


class TestFileExists:
    """Tests for file_exists check type."""

    @pytest.mark.asyncio
    async def test_file_exists(self, tmp_path: Path) -> None:
        target = tmp_path / "test.txt"
        target.write_text("content")
        spec = ModelVerificationSpec(
            check_type="file_exists",
            target=str(target),
        )
        result = await execute_verification(spec)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_file_missing(self, tmp_path: Path) -> None:
        spec = ModelVerificationSpec(
            check_type="file_exists",
            target=str(tmp_path / "nonexistent.txt"),
        )
        result = await execute_verification(spec)
        assert result.passed is False


class TestTcpProbe:
    """Tests for tcp_probe check type."""

    @pytest.mark.asyncio
    async def test_tcp_reachable(self) -> None:
        spec = ModelVerificationSpec(
            check_type="tcp_probe",
            target="localhost:5432",
        )
        with patch(
            "omnibase_infra.probes.verification_executor.socket_check",
            return_value=True,
        ):
            result = await execute_verification(spec)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_tcp_unreachable(self) -> None:
        spec = ModelVerificationSpec(
            check_type="tcp_probe",
            target="localhost:5432",
        )
        with patch(
            "omnibase_infra.probes.verification_executor.socket_check",
            return_value=False,
        ):
            result = await execute_verification(spec)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_tcp_invalid_target(self) -> None:
        spec = ModelVerificationSpec(
            check_type="tcp_probe",
            target="no-port-here",
        )
        result = await execute_verification(spec)
        assert result.passed is False
        assert "Invalid target format" in result.message


class TestHttpHealth:
    """Tests for http_health check type."""

    @pytest.mark.asyncio
    async def test_http_healthy(self) -> None:
        spec = ModelVerificationSpec(
            check_type="http_health",
            target="http://localhost:8080/health",
        )
        with patch(
            "omnibase_infra.probes.verification_executor.http_health_check",
            return_value=True,
        ):
            result = await execute_verification(spec)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_http_unhealthy(self) -> None:
        spec = ModelVerificationSpec(
            check_type="http_health",
            target="http://localhost:8080/health",
        )
        with patch(
            "omnibase_infra.probes.verification_executor.http_health_check",
            return_value=False,
        ):
            result = await execute_verification(spec)
        assert result.passed is False


class TestPythonImport:
    """Tests for python_import check type."""

    @pytest.mark.asyncio
    async def test_importable_module(self) -> None:
        spec = ModelVerificationSpec(
            check_type="python_import",
            target="json",
        )
        result = await execute_verification(spec)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_non_importable_module(self) -> None:
        spec = ModelVerificationSpec(
            check_type="python_import",
            target="nonexistent_module_xyz_123",
        )
        result = await execute_verification(spec)
        assert result.passed is False


class TestModelVerificationResult:
    """Tests for ModelVerificationResult model."""

    def test_construct(self) -> None:
        result = ModelVerificationResult(
            passed=True,
            check_type="command_exit_0",
            target="uv --version",
            message="Command succeeded",
            elapsed_ms=42,
        )
        assert result.passed is True
        assert result.elapsed_ms == 42

    def test_round_trip(self) -> None:
        result = ModelVerificationResult(
            passed=False,
            check_type="tcp_probe",
            target="localhost:5432",
            message="Connection refused",
            elapsed_ms=1000,
        )
        data = result.model_dump()
        result2 = ModelVerificationResult.model_validate(data)
        assert result == result2


class TestUnknownCheckType:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_check_type_raises(self) -> None:
        spec = ModelVerificationSpec(
            check_type="unknown_type",
            target="something",
        )
        with pytest.raises(ValueError, match="Unknown check_type"):
            await execute_verification(spec)

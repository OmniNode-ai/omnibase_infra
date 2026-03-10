# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLocalProvision.

Invariants tested:
  I3 — Monkeypatch discipline: patch via handler_mod.asyncio.create_subprocess_exec
       and handler_mod.asyncio.open_connection.
  I4 — Port semantics: OPEN check via asyncio.open_connection.
  I7 — Compose file path: handler validates existence, does not resolve.

Ticket: OMN-3493
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_provision as handler_mod
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_provision import (
    HandlerLocalProvision,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_proc(returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    """Return a mock subprocess that looks like an asyncio.Process."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(b"", stderr))
    proc.returncode = returncode
    return proc


def _make_handler() -> HandlerLocalProvision:
    """Construct a HandlerLocalProvision with a mock container."""
    return HandlerLocalProvision(MagicMock())


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerLocalProvision:
    """Unit tests for HandlerLocalProvision (I3, I4, I7)."""

    async def test_minimal_preset_no_profile_flags_in_command(
        self, tmp_path: Path
    ) -> None:
        """Minimal topology (no compose_profile) must not add --profile flags.

        The generated docker compose command should contain no --profile arguments
        when no service in the topology has a compose_profile set.
        """
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.touch()

        topology = ModelDeploymentTopology.default_minimal()
        handler = _make_handler()
        await handler.initialize({})

        captured_cmd: list[list[str]] = []
        mock_proc = _make_mock_proc(returncode=0)

        async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured_cmd.append(list(args))
            return mock_proc

        # Port poll: immediately return True (port open)
        async def fake_open_connection(
            host: str, port: int
        ) -> tuple[MagicMock, MagicMock]:
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()
            return MagicMock(), writer

        with patch.object(
            handler_mod.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        ):
            with patch.object(
                handler_mod.asyncio, "open_connection", fake_open_connection
            ):
                result = await handler.execute(
                    {
                        "topology": topology,
                        "compose_file_path": str(compose_file),
                        "correlation_id": uuid4(),
                        "max_wait_seconds": 2,
                    }
                )

        assert result is not None
        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        assert "--profile" not in cmd, f"No --profile expected in cmd: {cmd}"

    async def test_standard_preset_adds_secrets_profile_once(
        self, tmp_path: Path
    ) -> None:
        """Standard topology adds exactly one --profile secrets flag.

        The infisical service has compose_profile='secrets'. The command must
        contain '--profile secrets' exactly once even though only one service
        has this profile.
        """
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.touch()

        topology = ModelDeploymentTopology.default_standard()
        handler = _make_handler()
        await handler.initialize({})

        captured_cmd: list[list[str]] = []
        mock_proc = _make_mock_proc(returncode=0)

        async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured_cmd.append(list(args))
            return mock_proc

        async def fake_open_connection(
            host: str, port: int
        ) -> tuple[MagicMock, MagicMock]:
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()
            return MagicMock(), writer

        with patch.object(
            handler_mod.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        ):
            with patch.object(
                handler_mod.asyncio, "open_connection", fake_open_connection
            ):
                result = await handler.execute(
                    {
                        "topology": topology,
                        "compose_file_path": str(compose_file),
                        "correlation_id": uuid4(),
                        "max_wait_seconds": 2,
                    }
                )

        assert result is not None
        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        profile_indices = [i for i, v in enumerate(cmd) if v == "--profile"]
        assert len(profile_indices) == 1, (
            f"Expected exactly 1 --profile flag, got {len(profile_indices)}: {cmd}"
        )
        assert cmd[profile_indices[0] + 1] == "secrets"

    async def test_no_duplicate_profiles_when_two_services_share_profile(
        self, tmp_path: Path
    ) -> None:
        """When two services share the same compose_profile, --profile appears only once.

        Construct a topology with two services both requiring compose_profile='infra'.
        The deduplication logic must produce exactly one '--profile infra' pair.
        """
        from omnibase_core.enums.enum_deployment_mode import EnumDeploymentMode
        from omnibase_core.models.core.model_deployment_topology_local_config import (
            ModelDeploymentTopologyLocalConfig,
        )
        from omnibase_core.models.core.model_deployment_topology_service import (
            ModelDeploymentTopologyService,
        )

        compose_file = tmp_path / "docker-compose.yml"
        compose_file.touch()

        topology = ModelDeploymentTopology(
            schema_version="1.0",
            services={
                "svc_a": ModelDeploymentTopologyService(
                    mode=EnumDeploymentMode.LOCAL,
                    local=ModelDeploymentTopologyLocalConfig(
                        compose_service="svc-a",
                        host_port=9001,
                        compose_profile="infra",
                    ),
                ),
                "svc_b": ModelDeploymentTopologyService(
                    mode=EnumDeploymentMode.LOCAL,
                    local=ModelDeploymentTopologyLocalConfig(
                        compose_service="svc-b",
                        host_port=9002,
                        compose_profile="infra",
                    ),
                ),
            },
        )

        handler = _make_handler()
        await handler.initialize({})

        captured_cmd: list[list[str]] = []
        mock_proc = _make_mock_proc(returncode=0)

        async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured_cmd.append(list(args))
            return mock_proc

        async def fake_open_connection(
            host: str, port: int
        ) -> tuple[MagicMock, MagicMock]:
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()
            return MagicMock(), writer

        with patch.object(
            handler_mod.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        ):
            with patch.object(
                handler_mod.asyncio, "open_connection", fake_open_connection
            ):
                result = await handler.execute(
                    {
                        "topology": topology,
                        "compose_file_path": str(compose_file),
                        "correlation_id": uuid4(),
                        "max_wait_seconds": 2,
                    }
                )

        assert result is not None
        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        profile_indices = [i for i, v in enumerate(cmd) if v == "--profile"]
        assert len(profile_indices) == 1, (
            f"Expected exactly 1 --profile infra (deduped), got {len(profile_indices)}: {cmd}"
        )
        assert cmd[profile_indices[0] + 1] == "infra"

    async def test_failed_subprocess_returns_services_failed(
        self, tmp_path: Path
    ) -> None:
        """A non-zero subprocess returncode yields success=False with services_started=().

        When docker compose exits with a non-zero code the handler must return
        a result with success=False, services_started empty, and a non-empty error.
        """
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.touch()

        topology = ModelDeploymentTopology.default_minimal()
        handler = _make_handler()
        await handler.initialize({})

        mock_proc = _make_mock_proc(
            returncode=1, stderr=b"Error: container already running"
        )

        async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> MagicMock:
            return mock_proc

        with patch.object(
            handler_mod.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        ):
            output = await handler.execute(
                {
                    "topology": topology,
                    "compose_file_path": str(compose_file),
                    "correlation_id": uuid4(),
                    "max_wait_seconds": 2,
                }
            )

        result = output.result
        assert result is not None
        assert result.success is False
        assert result.services_started == ()
        assert result.error is not None
        assert len(result.error) > 0

    async def test_poll_exceeds_max_wait_seconds_returns_not_started(
        self, tmp_path: Path
    ) -> None:
        """When port polling times out, the service appears in the error and success=False.

        Use max_wait_seconds=2 so the polling window gives ~3 attempts before
        timing out. The handler must return success=False with error set.
        """
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.touch()

        topology = ModelDeploymentTopology.default_minimal()
        handler = _make_handler()
        await handler.initialize({})

        mock_proc = _make_mock_proc(returncode=0)

        async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> MagicMock:
            return mock_proc

        # Port always raises OSError (never opens)
        async def fake_open_connection_fail(
            host: str, port: int
        ) -> tuple[MagicMock, MagicMock]:
            raise OSError("Connection refused")

        # Patch asyncio.sleep to no-op so test runs instantly
        async def fast_sleep(delay: float) -> None:
            pass

        with patch.object(
            handler_mod.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        ):
            with patch.object(
                handler_mod.asyncio, "open_connection", fake_open_connection_fail
            ):
                with patch.object(handler_mod.asyncio, "sleep", fast_sleep):
                    output = await handler.execute(
                        {
                            "topology": topology,
                            "compose_file_path": str(compose_file),
                            "correlation_id": uuid4(),
                            "max_wait_seconds": 2,
                        }
                    )

        result = output.result
        assert result is not None
        assert result.success is False
        assert result.services_started == ()
        assert result.error is not None

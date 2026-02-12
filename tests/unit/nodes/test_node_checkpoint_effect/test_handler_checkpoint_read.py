# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerCheckpointRead.

Ticket: OMN-2143
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.enums.enum_checkpoint_phase import EnumCheckpointPhase
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint
from omnibase_infra.models.checkpoint.model_phase_payload_implement import (
    ModelPhasePayloadImplement,
)
from omnibase_infra.nodes.node_checkpoint_effect.handlers.handler_checkpoint_read import (
    HandlerCheckpointRead,
)
from omnibase_infra.nodes.node_checkpoint_effect.handlers.handler_checkpoint_write import (
    HandlerCheckpointWrite,
)


@pytest.fixture
def mock_container() -> MagicMock:
    return MagicMock()


@pytest.fixture
def writer(mock_container: MagicMock) -> HandlerCheckpointWrite:
    return HandlerCheckpointWrite(mock_container)


@pytest.fixture
def reader(mock_container: MagicMock) -> HandlerCheckpointRead:
    return HandlerCheckpointRead(mock_container)


class TestHandlerCheckpointRead:
    """Tests for HandlerCheckpointRead."""

    async def test_read_returns_written_checkpoint(
        self,
        writer: HandlerCheckpointWrite,
        reader: HandlerCheckpointRead,
        tmp_path: Path,
    ) -> None:
        """Round-trip: write then read returns the same checkpoint."""
        await writer.initialize({})
        await reader.initialize({})

        run_id = uuid4()
        checkpoint = ModelCheckpoint(
            run_id=run_id,
            ticket_id="OMN-2143",
            phase=EnumCheckpointPhase.IMPLEMENT,
            timestamp_utc=datetime.now(UTC),
            repo_commit_map={"infra": "deadbeef"},
            attempt_number=1,
            phase_payload=ModelPhasePayloadImplement(
                branch_name="feature-branch",
                commit_sha="deadbeef",
                files_changed=("src/foo.py",),
            ),
        )

        # Write
        write_env: dict[str, object] = {
            "checkpoint": checkpoint,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        await writer.execute(write_env)

        # Read
        read_env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "phase": EnumCheckpointPhase.IMPLEMENT,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await reader.execute(read_env)
        output = result.result

        assert output.success is True
        assert output.checkpoint is not None
        assert output.checkpoint.ticket_id == "OMN-2143"
        assert output.checkpoint.repo_commit_map == {"infra": "deadbeef"}

    async def test_read_returns_latest_attempt(
        self,
        writer: HandlerCheckpointWrite,
        reader: HandlerCheckpointRead,
        tmp_path: Path,
    ) -> None:
        """When multiple attempts exist, read returns the highest attempt."""
        await writer.initialize({})
        await reader.initialize({})

        run_id = uuid4()

        for attempt in (1, 2, 3):
            cp = ModelCheckpoint(
                run_id=run_id,
                ticket_id="OMN-2143",
                phase=EnumCheckpointPhase.LOCAL_REVIEW,
                timestamp_utc=datetime.now(UTC),
                attempt_number=attempt,
                phase_payload={
                    "phase": "local_review",
                    "iteration_count": attempt,
                    "issue_fingerprints": [],
                    "last_clean_sha": f"aabbcc{attempt:04d}",
                },
            )
            write_env: dict[str, object] = {
                "checkpoint": cp,
                "correlation_id": uuid4(),
                "base_dir": str(tmp_path),
            }
            await writer.execute(write_env)

        # Read should return attempt 3
        read_env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "phase": EnumCheckpointPhase.LOCAL_REVIEW,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await reader.execute(read_env)
        assert result.result.success is True
        assert result.result.checkpoint.attempt_number == 3

    async def test_read_missing_directory_returns_error(
        self,
        reader: HandlerCheckpointRead,
        tmp_path: Path,
    ) -> None:
        """Read returns failure when the checkpoint directory does not exist."""
        await reader.initialize({})

        read_env: dict[str, object] = {
            "ticket_id": "OMN-NOEXIST",
            "run_id": uuid4(),
            "phase": EnumCheckpointPhase.IMPLEMENT,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await reader.execute(read_env)
        assert result.result.success is False
        assert "not found" in result.result.error

    async def test_read_handles_invalid_yaml(
        self,
        reader: HandlerCheckpointRead,
        tmp_path: Path,
    ) -> None:
        """Read raises RuntimeHostError when checkpoint file contains invalid YAML."""
        await reader.initialize({})

        run_id = uuid4()
        run_dir = tmp_path / "OMN-2143" / str(run_id)
        run_dir.mkdir(parents=True)

        # Write a corrupt YAML file
        corrupt_file = run_dir / "phase_1_implement_a1.yaml"
        corrupt_file.write_text(": : : [invalid yaml", encoding="utf-8")

        read_env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "phase": EnumCheckpointPhase.IMPLEMENT,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        with pytest.raises(Exception, match=r"Corrupt checkpoint file.*invalid YAML"):
            await reader.execute(read_env)

    async def test_read_handles_empty_yaml(
        self,
        reader: HandlerCheckpointRead,
        tmp_path: Path,
    ) -> None:
        """Read raises RuntimeHostError when checkpoint file is empty (None)."""
        await reader.initialize({})

        run_id = uuid4()
        run_dir = tmp_path / "OMN-2143" / str(run_id)
        run_dir.mkdir(parents=True)

        # Write an empty YAML file (safe_load returns None)
        empty_file = run_dir / "phase_1_implement_a1.yaml"
        empty_file.write_text("", encoding="utf-8")

        read_env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "phase": EnumCheckpointPhase.IMPLEMENT,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        with pytest.raises(Exception, match="expected mapping"):
            await reader.execute(read_env)

    async def test_read_missing_phase_returns_error(
        self,
        writer: HandlerCheckpointWrite,
        reader: HandlerCheckpointRead,
        tmp_path: Path,
    ) -> None:
        """Read returns failure when the requested phase has no checkpoints."""
        await writer.initialize({})
        await reader.initialize({})

        run_id = uuid4()
        # Write implement phase
        cp = ModelCheckpoint(
            run_id=run_id,
            ticket_id="OMN-2143",
            phase=EnumCheckpointPhase.IMPLEMENT,
            timestamp_utc=datetime.now(UTC),
            attempt_number=1,
            phase_payload=ModelPhasePayloadImplement(
                branch_name="branch",
                commit_sha="abc1234",
            ),
        )
        await writer.execute(
            {
                "checkpoint": cp,
                "correlation_id": uuid4(),
                "base_dir": str(tmp_path),
            }
        )

        # Try to read a different phase
        read_env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "phase": EnumCheckpointPhase.CREATE_PR,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await reader.execute(read_env)
        assert result.result.success is False
        assert "No checkpoint found" in result.result.error

    async def test_read_rejects_path_traversal(
        self,
        reader: HandlerCheckpointRead,
        tmp_path: Path,
    ) -> None:
        """Read rejects ticket_id containing path traversal sequences."""
        await reader.initialize({})

        read_env: dict[str, object] = {
            "ticket_id": "../../etc",
            "run_id": uuid4(),
            "phase": EnumCheckpointPhase.IMPLEMENT,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        with pytest.raises(Exception, match="Path traversal detected"):
            await reader.execute(read_env)

    async def test_read_sorts_by_attempt_number_not_name(
        self,
        reader: HandlerCheckpointRead,
        tmp_path: Path,
    ) -> None:
        """Read returns highest attempt even when attempt >= 10."""
        await reader.initialize({})

        run_id = uuid4()
        run_dir = tmp_path / "OMN-2143" / str(run_id)
        run_dir.mkdir(parents=True)

        # Create files with attempts 1, 2, 10 (a10 sorts before a2 lexically)
        for attempt in (1, 2, 10):
            cp = ModelCheckpoint(
                run_id=run_id,
                ticket_id="OMN-2143",
                phase=EnumCheckpointPhase.IMPLEMENT,
                timestamp_utc=datetime.now(UTC),
                attempt_number=attempt,
                phase_payload=ModelPhasePayloadImplement(
                    branch_name="branch",
                    commit_sha="abc1234",
                ),
            )
            filename = f"phase_1_implement_a{attempt}.yaml"
            data = cp.model_dump(mode="json")
            (run_dir / filename).write_text(
                yaml.dump(data, default_flow_style=False), encoding="utf-8"
            )

        read_env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "phase": EnumCheckpointPhase.IMPLEMENT,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await reader.execute(read_env)
        assert result.result.success is True
        # Must return attempt 10, not attempt 2 (which would be the lexicographic last)
        assert result.result.checkpoint.attempt_number == 10

    async def test_rejects_relative_base_dir(
        self,
        reader: HandlerCheckpointRead,
    ) -> None:
        """Read rejects a relative base_dir."""
        await reader.initialize({})

        read_env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": uuid4(),
            "phase": EnumCheckpointPhase.IMPLEMENT,
            "correlation_id": uuid4(),
            "base_dir": "relative/path",
        }

        with pytest.raises(RuntimeHostError, match="base_dir must be an absolute path"):
            await reader.execute(read_env)

    async def test_rejects_base_dir_with_traversal(
        self,
        reader: HandlerCheckpointRead,
    ) -> None:
        """Read rejects a base_dir containing '..' components."""
        await reader.initialize({})

        read_env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": uuid4(),
            "phase": EnumCheckpointPhase.IMPLEMENT,
            "correlation_id": uuid4(),
            "base_dir": "/var/../etc",
        }

        with pytest.raises(
            RuntimeHostError, match=r"must not contain '\.\.' components"
        ):
            await reader.execute(read_env)

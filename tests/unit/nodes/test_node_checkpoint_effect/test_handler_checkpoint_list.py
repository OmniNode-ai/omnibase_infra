# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerCheckpointList.

Ticket: OMN-2143
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_checkpoint_phase import EnumCheckpointPhase
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint
from omnibase_infra.models.checkpoint.model_phase_payload_create_pr import (
    ModelPhasePayloadCreatePr,
)
from omnibase_infra.models.checkpoint.model_phase_payload_implement import (
    ModelPhasePayloadImplement,
)
from omnibase_infra.models.checkpoint.model_phase_payload_local_review import (
    ModelPhasePayloadLocalReview,
)
from omnibase_infra.nodes.node_checkpoint_effect.handlers.handler_checkpoint_list import (
    HandlerCheckpointList,
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
def lister(mock_container: MagicMock) -> HandlerCheckpointList:
    return HandlerCheckpointList(mock_container)


class TestHandlerCheckpointList:
    """Tests for HandlerCheckpointList."""

    async def test_list_empty_returns_empty_tuple(
        self,
        lister: HandlerCheckpointList,
        tmp_path: Path,
    ) -> None:
        """List returns empty tuple when no checkpoints exist."""
        await lister.initialize({})

        env: dict[str, object] = {
            "ticket_id": "OMN-NOEXIST",
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await lister.execute(env)
        assert result.result.success is True
        assert result.result.checkpoints == ()

    async def test_list_returns_all_phases(
        self,
        writer: HandlerCheckpointWrite,
        lister: HandlerCheckpointList,
        tmp_path: Path,
    ) -> None:
        """List returns checkpoints across all phases for a run."""
        await writer.initialize({})
        await lister.initialize({})

        run_id = uuid4()
        phases_and_payloads: list[tuple[EnumCheckpointPhase, object]] = [
            (
                EnumCheckpointPhase.IMPLEMENT,
                ModelPhasePayloadImplement(
                    branch_name="branch",
                    commit_sha="abc1234",
                ),
            ),
            (
                EnumCheckpointPhase.LOCAL_REVIEW,
                ModelPhasePayloadLocalReview(
                    iteration_count=2,
                    last_clean_sha="def5678",
                ),
            ),
            (
                EnumCheckpointPhase.CREATE_PR,
                ModelPhasePayloadCreatePr(
                    pr_url="https://github.com/org/repo/pull/42",
                    pr_number=42,
                    head_sha="aabbccd",
                ),
            ),
        ]

        for phase, payload in phases_and_payloads:
            cp = ModelCheckpoint(
                run_id=run_id,
                ticket_id="OMN-2143",
                phase=phase,
                timestamp_utc=datetime.now(UTC),
                attempt_number=1,
                phase_payload=payload,
            )
            await writer.execute(
                {
                    "checkpoint": cp,
                    "correlation_id": uuid4(),
                    "base_dir": str(tmp_path),
                }
            )

        # List all for this run
        env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await lister.execute(env)
        assert result.result.success is True
        assert len(result.result.checkpoints) == 3

    async def test_list_includes_multiple_attempts(
        self,
        writer: HandlerCheckpointWrite,
        lister: HandlerCheckpointList,
        tmp_path: Path,
    ) -> None:
        """List returns all attempts for a given phase."""
        await writer.initialize({})
        await lister.initialize({})

        run_id = uuid4()

        for attempt in (1, 2):
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
            await writer.execute(
                {
                    "checkpoint": cp,
                    "correlation_id": uuid4(),
                    "base_dir": str(tmp_path),
                }
            )

        env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await lister.execute(env)
        assert result.result.success is True
        assert len(result.result.checkpoints) == 2

    async def test_list_skips_empty_yaml_files(
        self,
        writer: HandlerCheckpointWrite,
        lister: HandlerCheckpointList,
        tmp_path: Path,
    ) -> None:
        """List skips empty or non-dict YAML files without crashing."""
        await writer.initialize({})
        await lister.initialize({})

        run_id = uuid4()

        # Write one valid checkpoint
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

        # Place an empty YAML file alongside the valid one
        run_dir = tmp_path / "OMN-2143" / str(run_id)
        empty_file = run_dir / "phase_1_implement_a0.yaml"
        empty_file.write_text("", encoding="utf-8")

        env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "run_id": run_id,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await lister.execute(env)
        assert result.result.success is True
        # Only the valid checkpoint should be returned; the empty one is skipped
        assert len(result.result.checkpoints) == 1

    async def test_list_without_run_id_scans_all_runs(
        self,
        writer: HandlerCheckpointWrite,
        lister: HandlerCheckpointList,
        tmp_path: Path,
    ) -> None:
        """List without run_id returns checkpoints across all runs."""
        await writer.initialize({})
        await lister.initialize({})

        for _ in range(2):
            cp = ModelCheckpoint(
                run_id=uuid4(),
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

        env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }
        result = await lister.execute(env)
        assert result.result.success is True
        assert len(result.result.checkpoints) == 2

    async def test_rejects_relative_base_dir(
        self,
        lister: HandlerCheckpointList,
    ) -> None:
        """List rejects a relative base_dir."""
        await lister.initialize({})

        env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "correlation_id": uuid4(),
            "base_dir": "relative/path",
        }

        with pytest.raises(RuntimeHostError, match="base_dir must be an absolute path"):
            await lister.execute(env)

    async def test_rejects_base_dir_with_traversal(
        self,
        lister: HandlerCheckpointList,
    ) -> None:
        """List rejects a base_dir containing '..' components."""
        await lister.initialize({})

        env: dict[str, object] = {
            "ticket_id": "OMN-2143",
            "correlation_id": uuid4(),
            "base_dir": "/var/../etc",
        }

        with pytest.raises(
            RuntimeHostError, match=r"must not contain '\.\.' components"
        ):
            await lister.execute(env)

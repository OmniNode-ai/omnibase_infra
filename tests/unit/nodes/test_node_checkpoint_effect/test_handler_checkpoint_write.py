# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerCheckpointWrite.

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
from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint
from omnibase_infra.models.checkpoint.model_phase_payload_implement import (
    ModelPhasePayloadImplement,
)
from omnibase_infra.nodes.node_checkpoint_effect.handlers.handler_checkpoint_write import (
    HandlerCheckpointWrite,
)


@pytest.fixture
def mock_container() -> MagicMock:
    return MagicMock()


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerCheckpointWrite:
    return HandlerCheckpointWrite(mock_container)


@pytest.fixture
def sample_checkpoint() -> ModelCheckpoint:
    return ModelCheckpoint(
        run_id=uuid4(),
        ticket_id="OMN-2143",
        phase=EnumCheckpointPhase.IMPLEMENT,
        timestamp_utc=datetime.now(UTC),
        repo_commit_map={"omnibase_infra": "abc1234"},
        artifact_paths=("src/new_file.py",),
        attempt_number=1,
        phase_payload=ModelPhasePayloadImplement(
            branch_name="jonah/omn-2143-checkpoint-nodes",
            commit_sha="abc1234",
            files_changed=("src/new_file.py",),
        ),
    )


class TestHandlerCheckpointWrite:
    """Tests for HandlerCheckpointWrite."""

    async def test_write_creates_yaml_file(
        self,
        handler: HandlerCheckpointWrite,
        sample_checkpoint: ModelCheckpoint,
        tmp_path: Path,
    ) -> None:
        """Write creates a YAML file at the expected path."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "checkpoint": sample_checkpoint,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }

        result = await handler.execute(envelope)
        output = result.result

        assert output.success is True
        assert output.checkpoint_path is not None

        # Verify file exists
        full_path = tmp_path / output.checkpoint_path
        assert full_path.exists()

        # Verify YAML content round-trips
        raw = yaml.safe_load(full_path.read_text(encoding="utf-8"))
        reloaded = ModelCheckpoint.model_validate(raw)
        assert reloaded.ticket_id == "OMN-2143"
        assert reloaded.phase == EnumCheckpointPhase.IMPLEMENT
        assert reloaded.attempt_number == 1

    async def test_write_preserves_attempt_number(
        self,
        handler: HandlerCheckpointWrite,
        tmp_path: Path,
    ) -> None:
        """Re-runs create new files with incremented attempt_number."""
        await handler.initialize({})
        run_id = uuid4()

        for attempt in (1, 2, 3):
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
            envelope: dict[str, object] = {
                "checkpoint": cp,
                "correlation_id": uuid4(),
                "base_dir": str(tmp_path),
            }
            await handler.execute(envelope)

        # Verify 3 distinct files exist
        run_dir = tmp_path / "OMN-2143" / str(run_id)
        yaml_files = list(run_dir.glob("phase_1_implement_a*.yaml"))
        assert len(yaml_files) == 3

    async def test_write_rejects_absolute_artifact_paths(
        self,
        handler: HandlerCheckpointWrite,
        tmp_path: Path,
    ) -> None:
        """Write rejects checkpoints with absolute artifact paths."""
        await handler.initialize({})

        cp = ModelCheckpoint.model_construct(
            run_id=uuid4(),
            ticket_id="OMN-2143",
            phase=EnumCheckpointPhase.IMPLEMENT,
            timestamp_utc=datetime.now(UTC),
            artifact_paths=("/absolute/path/bad.py",),
            attempt_number=1,
            schema_version="1.0.0",
            repo_commit_map={},
            phase_payload=ModelPhasePayloadImplement(
                branch_name="branch",
                commit_sha="abc1234",
            ),
        )

        envelope: dict[str, object] = {
            "checkpoint": cp,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }

        with pytest.raises(Exception, match="Absolute artifact path forbidden"):
            await handler.execute(envelope)

    async def test_write_requires_checkpoint(
        self,
        handler: HandlerCheckpointWrite,
        tmp_path: Path,
    ) -> None:
        """Write raises when checkpoint is missing from envelope."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }

        with pytest.raises(Exception, match=r"requires.*checkpoint"):
            await handler.execute(envelope)

    async def test_write_accepts_dict_checkpoint(
        self,
        handler: HandlerCheckpointWrite,
        sample_checkpoint: ModelCheckpoint,
        tmp_path: Path,
    ) -> None:
        """Write accepts dict input and validates it via Pydantic."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "checkpoint": sample_checkpoint.model_dump(mode="json"),
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }

        result = await handler.execute(envelope)
        assert result.result.success is True

    async def test_write_rejects_duplicate_attempt(
        self,
        handler: HandlerCheckpointWrite,
        sample_checkpoint: ModelCheckpoint,
        tmp_path: Path,
    ) -> None:
        """Write refuses to overwrite an existing checkpoint (append-only)."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "checkpoint": sample_checkpoint,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }

        # First write succeeds
        result = await handler.execute(envelope)
        assert result.result.success is True

        # Second write with same checkpoint (same attempt_number) fails
        with pytest.raises(Exception, match="Checkpoint already exists"):
            await handler.execute(envelope)

    async def test_write_rejects_path_traversal(
        self,
        handler: HandlerCheckpointWrite,
        tmp_path: Path,
    ) -> None:
        """Write rejects ticket_id containing path traversal sequences."""
        await handler.initialize({})

        cp = ModelCheckpoint(
            run_id=uuid4(),
            ticket_id="../../etc",
            phase=EnumCheckpointPhase.IMPLEMENT,
            timestamp_utc=datetime.now(UTC),
            attempt_number=1,
            phase_payload=ModelPhasePayloadImplement(
                branch_name="branch",
                commit_sha="abc1234",
            ),
        )

        envelope: dict[str, object] = {
            "checkpoint": cp,
            "correlation_id": uuid4(),
            "base_dir": str(tmp_path),
        }

        with pytest.raises(Exception, match="Path traversal detected"):
            await handler.execute(envelope)

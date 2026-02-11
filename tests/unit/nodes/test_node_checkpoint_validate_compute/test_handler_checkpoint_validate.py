# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerCheckpointValidate.

Ticket: OMN-2143
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_checkpoint_phase import EnumCheckpointPhase
from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint
from omnibase_infra.models.checkpoint.model_phase_payload_implement import (
    ModelPhasePayloadImplement,
)
from omnibase_infra.models.checkpoint.model_phase_payload_local_review import (
    ModelPhasePayloadLocalReview,
)
from omnibase_infra.nodes.node_checkpoint_validate_compute.handlers.handler_checkpoint_validate import (
    HandlerCheckpointValidate,
)


@pytest.fixture
def mock_container() -> MagicMock:
    return MagicMock()


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerCheckpointValidate:
    return HandlerCheckpointValidate(mock_container)


def _valid_checkpoint(**overrides: object) -> ModelCheckpoint:
    """Create a valid checkpoint with optional overrides."""
    defaults: dict[str, object] = {
        "run_id": uuid4(),
        "ticket_id": "OMN-2143",
        "phase": EnumCheckpointPhase.IMPLEMENT,
        "timestamp_utc": datetime.now(UTC),
        "repo_commit_map": {"omnibase_infra": "abc1234"},
        "artifact_paths": ("src/foo.py",),
        "attempt_number": 1,
        "phase_payload": ModelPhasePayloadImplement(
            branch_name="feature-branch",
            commit_sha="abc1234",
        ),
    }
    defaults.update(overrides)
    return ModelCheckpoint(**defaults)  # type: ignore[arg-type]


def _invalid_checkpoint(**overrides: object) -> ModelCheckpoint:
    """Create a checkpoint bypassing model validators for handler-level testing."""
    defaults: dict[str, object] = {
        "run_id": uuid4(),
        "ticket_id": "OMN-2143",
        "phase": EnumCheckpointPhase.IMPLEMENT,
        "timestamp_utc": datetime.now(UTC),
        "repo_commit_map": {"omnibase_infra": "abc1234"},
        "artifact_paths": ("src/foo.py",),
        "attempt_number": 1,
        "schema_version": "1.0.0",
        "phase_payload": ModelPhasePayloadImplement(
            branch_name="feature-branch",
            commit_sha="abc1234",
        ),
    }
    defaults.update(overrides)
    return ModelCheckpoint.model_construct(**defaults)  # type: ignore[arg-type]


class TestHandlerCheckpointValidate:
    """Tests for HandlerCheckpointValidate."""

    async def test_valid_checkpoint_passes(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """A well-formed checkpoint passes validation."""
        await handler.initialize({})

        env: dict[str, object] = {
            "checkpoint": _valid_checkpoint(),
            "correlation_id": uuid4(),
        }
        result = await handler.execute(env)
        output = result.result

        assert output.is_valid is True
        assert output.errors == ()

    async def test_absolute_artifact_path_is_error(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Absolute paths in artifact_paths produce an error."""
        await handler.initialize({})

        cp = _invalid_checkpoint(artifact_paths=("/absolute/path.py",))

        env: dict[str, object] = {
            "checkpoint": cp,
            "correlation_id": uuid4(),
        }
        result = await handler.execute(env)
        output = result.result

        assert output.is_valid is False
        assert any("Absolute artifact path" in e for e in output.errors)

    async def test_invalid_commit_sha_is_error(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Non-hex commit SHA produces an error.

        Uses ``model_construct`` to bypass Pydantic field validation so the
        handler's own SHA check is exercised.  The model-level
        ``_validate_commit_shas`` validator would reject this value at
        construction time.
        """
        await handler.initialize({})

        # Build a valid checkpoint, then reconstruct with bad repo_commit_map.
        # We must preserve the original field objects (not serialized dicts)
        # so that model_construct produces a valid-looking instance.
        base = _valid_checkpoint()
        fields = {
            field_name: getattr(base, field_name)
            for field_name in ModelCheckpoint.model_fields
        }
        fields["repo_commit_map"] = {"repo": "not-a-sha!!"}
        cp = ModelCheckpoint.model_construct(**fields)

        env: dict[str, object] = {
            "checkpoint": cp,
            "correlation_id": uuid4(),
        }
        result = await handler.execute(env)
        output = result.result

        assert output.is_valid is False
        assert any("Invalid commit SHA" in e for e in output.errors)

    async def test_phase_payload_mismatch_is_error(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Phase payload phase != header phase produces an error."""
        await handler.initialize({})

        # Header says IMPLEMENT, but payload says "local_review"
        mismatched_payload = ModelPhasePayloadLocalReview.model_construct(
            phase="local_review",
            iteration_count=1,
            last_clean_sha="abc1234",
        )
        cp = _invalid_checkpoint(
            phase=EnumCheckpointPhase.IMPLEMENT,
            phase_payload=mismatched_payload,
        )

        env: dict[str, object] = {
            "checkpoint": cp,
            "correlation_id": uuid4(),
        }
        result = await handler.execute(env)
        output = result.result

        assert output.is_valid is False
        assert any("Phase mismatch" in e for e in output.errors)

    async def test_schema_version_mismatch_is_warning(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Different schema version produces a warning, not error."""
        await handler.initialize({})

        cp = _valid_checkpoint(schema_version="2.0.0")

        env: dict[str, object] = {
            "checkpoint": cp,
            "correlation_id": uuid4(),
        }
        result = await handler.execute(env)
        output = result.result

        assert output.is_valid is True  # Warning, not error
        assert any("Schema version mismatch" in w for w in output.warnings)

    async def test_dict_input_validation(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Handler accepts dict input and validates via Pydantic."""
        await handler.initialize({})

        cp = _valid_checkpoint()
        env: dict[str, object] = {
            "checkpoint": cp.model_dump(mode="json"),
            "correlation_id": uuid4(),
        }
        result = await handler.execute(env)
        assert result.result.is_valid is True

    async def test_invalid_dict_produces_errors(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Invalid dict input produces validation errors."""
        await handler.initialize({})

        env: dict[str, object] = {
            "checkpoint": {"not": "a checkpoint"},
            "correlation_id": uuid4(),
        }
        result = await handler.execute(env)
        assert result.result.is_valid is False
        assert len(result.result.errors) > 0

    async def test_missing_checkpoint_produces_error(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Missing checkpoint field produces an error."""
        await handler.initialize({})

        env: dict[str, object] = {
            "correlation_id": uuid4(),
        }
        result = await handler.execute(env)
        assert result.result.is_valid is False

    async def test_bool_reflects_validity(
        self,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Output __bool__ matches is_valid."""
        await handler.initialize({})

        valid_env: dict[str, object] = {
            "checkpoint": _valid_checkpoint(),
            "correlation_id": uuid4(),
        }
        valid_result = await handler.execute(valid_env)
        assert bool(valid_result.result) is True

        invalid_env: dict[str, object] = {
            "checkpoint": _invalid_checkpoint(artifact_paths=("/abs/path",)),
            "correlation_id": uuid4(),
        }
        invalid_result = await handler.execute(invalid_env)
        assert bool(invalid_result.result) is False

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelDelegationRequest.

Tests cover:
    - Frozen immutability
    - extra="forbid" enforcement
    - Required fields validation
    - task_type literal constraint
    - Serialization roundtrip
    - Default values

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)

pytestmark = [pytest.mark.unit]


def _valid_request(**overrides: object) -> ModelDelegationRequest:
    """Build a valid ModelDelegationRequest with sensible defaults."""
    defaults: dict[str, object] = {
        "prompt": "Write unit tests for auth.py",
        "task_type": "test",
        "correlation_id": uuid4(),
        "emitted_at": datetime.now(tz=UTC),
    }
    defaults.update(overrides)
    return ModelDelegationRequest(**defaults)  # type: ignore[arg-type]


class TestModelDelegationRequestFrozen:
    """Verify ConfigDict(frozen=True) enforcement."""

    def test_frozen_rejects_field_mutation(self) -> None:
        req = _valid_request()
        with pytest.raises(ValidationError):
            req.prompt = "new prompt"  # type: ignore[misc]

    def test_frozen_rejects_new_attribute(self) -> None:
        req = _valid_request()
        with pytest.raises(ValidationError):
            req.new_field = "bad"  # type: ignore[attr-defined]


class TestModelDelegationRequestExtraForbid:
    """Verify extra='forbid' enforcement."""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _valid_request(unknown_field="bad")


class TestModelDelegationRequestRequiredFields:
    """Verify required field validation."""

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelDelegationRequest(
                task_type="test",
                correlation_id=uuid4(),
                emitted_at=datetime.now(tz=UTC),
            )  # type: ignore[call-arg]

    def test_missing_task_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelDelegationRequest(
                prompt="hello",
                correlation_id=uuid4(),
                emitted_at=datetime.now(tz=UTC),
            )  # type: ignore[call-arg]

    def test_missing_correlation_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelDelegationRequest(
                prompt="hello",
                task_type="test",
                emitted_at=datetime.now(tz=UTC),
            )  # type: ignore[call-arg]

    def test_missing_emitted_at_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelDelegationRequest(
                prompt="hello",
                task_type="test",
                correlation_id=uuid4(),
            )  # type: ignore[call-arg]


class TestModelDelegationRequestTaskType:
    """Verify task_type literal constraint."""

    @pytest.mark.parametrize("task_type", ["test", "document", "research"])
    def test_valid_task_types(self, task_type: str) -> None:
        req = _valid_request(task_type=task_type)
        assert req.task_type == task_type

    def test_invalid_task_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(task_type="invalid")


class TestModelDelegationRequestDefaults:
    """Verify default values."""

    def test_max_tokens_default(self) -> None:
        req = _valid_request()
        assert req.max_tokens == 2048

    def test_source_session_id_default_none(self) -> None:
        req = _valid_request()
        assert req.source_session_id is None

    def test_source_file_path_default_none(self) -> None:
        req = _valid_request()
        assert req.source_file_path is None


class TestModelDelegationRequestSerialization:
    """Verify serialization roundtrip."""

    def test_roundtrip(self) -> None:
        req = _valid_request(
            source_session_id="sess-123",
            source_file_path="/home/user/project/auth.py",
            max_tokens=4096,
        )
        data = req.model_dump(mode="json")
        restored = ModelDelegationRequest.model_validate(data)
        assert restored == req

    def test_from_attributes(self) -> None:
        req = _valid_request()
        reconstructed = ModelDelegationRequest.model_validate(req, from_attributes=True)
        assert reconstructed == req

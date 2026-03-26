# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelPRMergedEvent [OMN-6726]."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omnibase_infra.models.github.model_pr_merged_event import ModelPRMergedEvent


@pytest.mark.unit
class TestModelPRMergedEvent:
    """Tests for ModelPRMergedEvent frozen Pydantic model."""

    def _make_event(self, **kwargs: object) -> ModelPRMergedEvent:
        defaults: dict[str, object] = {
            "repo": "OmniNode-ai/omnibase_infra",
            "pr_number": 42,
            "base_ref": "main",
            "head_ref": "feature/test-branch",
            "merge_sha": "abc123def456789",
            "author": "octocat",
            "changed_files": ["src/foo.py", "src/bar.py"],
            "ticket_ids": ["OMN-1234"],
            "title": "feat: add new feature [OMN-1234]",
            "merged_at": None,
        }
        defaults.update(kwargs)
        return ModelPRMergedEvent(**defaults)  # type: ignore[arg-type]

    def test_basic_construction(self) -> None:
        event = self._make_event()
        assert event.repo == "OmniNode-ai/omnibase_infra"
        assert event.pr_number == 42
        assert event.base_ref == "main"
        assert event.head_ref == "feature/test-branch"
        assert event.merge_sha == "abc123def456789"
        assert event.author == "octocat"

    def test_frozen_immutability(self) -> None:
        event = self._make_event()
        with pytest.raises(ValidationError):
            event.repo = "modified"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            self._make_event(unexpected_field="value")

    def test_pr_number_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            self._make_event(pr_number=0)

    def test_changed_files_rejects_none(self) -> None:
        """Passing None for changed_files should raise ValidationError."""
        with pytest.raises(ValidationError):
            self._make_event(changed_files=None)

    def test_changed_files_default_factory(self) -> None:
        """When changed_files is omitted, it defaults to empty list."""
        event = ModelPRMergedEvent(
            repo="OmniNode-ai/omnibase_infra",
            pr_number=1,
            base_ref="main",
            head_ref="feature/x",
            merge_sha="abc",
            author="test",
        )
        assert event.changed_files == []

    def test_ticket_ids_default_factory(self) -> None:
        """When ticket_ids is omitted, it defaults to empty list."""
        event = ModelPRMergedEvent(
            repo="OmniNode-ai/omnibase_infra",
            pr_number=1,
            base_ref="main",
            head_ref="feature/x",
            merge_sha="abc",
            author="test",
        )
        assert event.ticket_ids == []

    def test_merged_at_accepts_datetime(self) -> None:
        now = datetime.now(tz=UTC)
        event = self._make_event(merged_at=now)
        assert event.merged_at == now

    def test_merged_at_accepts_none(self) -> None:
        event = self._make_event(merged_at=None)
        assert event.merged_at is None

    def test_model_dump_json_roundtrip(self) -> None:
        """Verify JSON serialization produces valid JSON."""
        event = self._make_event()
        json_str = event.model_dump_json()
        assert '"repo"' in json_str
        assert '"pr_number"' in json_str
        assert '"merge_sha"' in json_str

    def test_title_default_empty(self) -> None:
        event = ModelPRMergedEvent(
            repo="OmniNode-ai/omnibase_infra",
            pr_number=1,
            base_ref="main",
            head_ref="feature/x",
            merge_sha="abc",
            author="test",
        )
        assert event.title == ""

    def test_required_fields_only(self) -> None:
        """All optional fields have defaults; only required fields needed."""
        event = ModelPRMergedEvent(
            repo="OmniNode-ai/omnibase_infra",
            pr_number=1,
            base_ref="main",
            head_ref="feature/x",
            merge_sha="abc",
            author="test",
        )
        assert event.repo == "OmniNode-ai/omnibase_infra"
        assert event.changed_files == []
        assert event.ticket_ids == []
        assert event.title == ""
        assert event.merged_at is None

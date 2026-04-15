# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for `resolve_project_tracker()`.

Unlike the unit tests in `tests/services/test_resolver.py`, these exercise
the resolver against a real `LocalStubProjectTracker` backing file on disk
and assert the returned instance satisfies the `ProtocolProjectTracker`
behavioral contract end-to-end (connect → create_issue → get_issue → close).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
    LocalStubProjectTracker,
)
from omnibase_infra.services.project_tracker.resolver import resolve_project_tracker

pytestmark = pytest.mark.integration


class TestResolveProjectTrackerIntegration:
    def test_no_token_resolves_to_working_local_stub(self, tmp_path: Path) -> None:
        """End-to-end: resolver → LocalStub → create/get issue round-trip."""
        with patch.dict("os.environ", {}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)

        assert isinstance(tracker, LocalStubProjectTracker)

        async def _roundtrip() -> None:
            await tracker.connect()
            created = await tracker.create_issue(
                title="resolver integration test",
                description="verifies resolver returns a working tracker",
            )
            fetched = await tracker.get_issue(created.identifier)
            assert fetched is not None
            assert fetched.identifier == created.identifier
            assert fetched.title == "resolver integration test"
            await tracker.close()

        asyncio.run(_roundtrip())

        # Backing file should exist and contain the created issue.
        state_file = tmp_path / "project_tracker_stub.json"
        assert state_file.exists()
        assert "resolver integration test" in state_file.read_text()

    def test_fail_soft_returns_working_local_stub(self, tmp_path: Path) -> None:
        """Construction-error path still returns a tracker that can connect/close."""
        with patch.dict("os.environ", {"LINEAR_TOKEN": "bad"}, clear=True):
            tracker = resolve_project_tracker(
                state_root=tmp_path,
                _force_construction_error=True,
            )
        assert isinstance(tracker, LocalStubProjectTracker)

        async def _lifecycle() -> None:
            await tracker.connect()
            await tracker.close()

        asyncio.run(_lifecycle())

    def test_linear_adapter_branch_integration(self, tmp_path: Path) -> None:
        """End-to-end token branch — activates once OMN-8816 lands."""
        pytest.importorskip(
            "omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter",
            reason="AdapterLinearProjectTracker ships in OMN-8816; skip until merged",
        )
        from omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter import (
            AdapterLinearProjectTracker,
        )

        with patch.dict("os.environ", {"LINEAR_TOKEN": "fake"}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)
        assert isinstance(tracker, AdapterLinearProjectTracker)

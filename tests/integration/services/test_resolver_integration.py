# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for `resolve_project_tracker()`.

Unlike the unit tests in ``tests/services/test_resolver.py``, these exercise
the resolver against a real ``LocalStubProjectTracker`` backing file on disk
and assert the returned instance satisfies the ``ProtocolProjectTracker``
behavioral contract end-to-end (connect → create_issue → get_issue → close).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
)
from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
    LocalStubProjectTracker,
)
from omnibase_infra.enums.enum_project_tracker_backend import (
    EnumProjectTrackerBackend,
)
from omnibase_infra.errors import InfraAuthenticationError
from omnibase_infra.services.project_tracker.resolver import resolve_project_tracker

pytestmark = pytest.mark.integration


class TestResolveProjectTrackerIntegration:
    def test_selected_local_stub_is_a_working_tracker(self, tmp_path: Path) -> None:
        """End-to-end: resolver → explicitly selected LocalStub → create/get round-trip."""
        with patch.dict("os.environ", {}, clear=True):
            tracker = resolve_project_tracker(
                state_root=tmp_path, backend=EnumProjectTrackerBackend.LOCAL_STUB
            )

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

    def test_no_key_fails_loud_and_writes_no_stub_state(self, tmp_path: Path) -> None:
        """Linear selected with no key raises, and no stub backing file appears."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(InfraAuthenticationError):
                resolve_project_tracker(state_root=tmp_path)
        assert not (tmp_path / "project_tracker_stub.json").exists()

    def test_linear_adapter_branch_integration(self, tmp_path: Path) -> None:
        """End-to-end token branch — resolver returns the GraphQL adapter."""
        with patch.dict("os.environ", {"LINEAR_TOKEN": "fake"}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)
        assert isinstance(tracker, AdapterLinearGraphQLProjectTracker)

        async def _close() -> None:
            await tracker.close()

        asyncio.run(_close())

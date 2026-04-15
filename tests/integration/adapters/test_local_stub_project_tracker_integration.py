# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for LocalStubProjectTracker."""

import pytest

from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
    LocalStubProjectTracker,
)


@pytest.mark.asyncio
async def test_create_and_get_issue(tmp_path: object) -> None:
    from pathlib import Path

    tracker = LocalStubProjectTracker(state_root=Path(str(tmp_path)))
    await tracker.connect()

    issue = await tracker.create_issue(title="Test issue", description="body")
    assert issue.identifier.startswith("STUB-")
    assert issue.title == "Test issue"

    fetched = await tracker.get_issue(issue.id)
    assert fetched.id == issue.id


@pytest.mark.asyncio
async def test_list_issues_empty(tmp_path: object) -> None:
    from pathlib import Path

    tracker = LocalStubProjectTracker(state_root=Path(str(tmp_path)))
    await tracker.connect()

    issues = await tracker.list_issues()
    assert issues == []


@pytest.mark.asyncio
async def test_update_issue(tmp_path: object) -> None:
    from pathlib import Path

    tracker = LocalStubProjectTracker(state_root=Path(str(tmp_path)))
    await tracker.connect()

    issue = await tracker.create_issue(title="Original", description="desc")
    updated = await tracker.update_issue(
        issue.id, {"title": "Updated", "state": "in_progress"}
    )
    assert updated.title == "Updated"
    assert updated.state == "in_progress"


@pytest.mark.asyncio
async def test_add_comment(tmp_path: object) -> None:
    from pathlib import Path

    tracker = LocalStubProjectTracker(state_root=Path(str(tmp_path)))
    await tracker.connect()

    issue = await tracker.create_issue(title="Issue", description="body")
    comment = await tracker.add_comment(issue.id, "A comment")
    assert comment.body == "A comment"
    assert comment.author == "stub"


@pytest.mark.asyncio
async def test_search_issues(tmp_path: object) -> None:
    from pathlib import Path

    tracker = LocalStubProjectTracker(state_root=Path(str(tmp_path)))
    await tracker.connect()

    await tracker.create_issue(title="Alpha feature", description="x")
    await tracker.create_issue(title="Beta feature", description="y")

    results = await tracker.search_issues("alpha")
    assert len(results) == 1
    assert results[0].title == "Alpha feature"


@pytest.mark.asyncio
async def test_labels_roundtrip(tmp_path: object) -> None:
    from pathlib import Path

    tracker = LocalStubProjectTracker(state_root=Path(str(tmp_path)))
    await tracker.connect()

    issue = await tracker.create_issue(
        title="Labeled", description="d", labels=["bug", "p1"]
    )
    fetched = await tracker.get_issue(issue.id)
    assert fetched.labels == ["bug", "p1"]


@pytest.mark.asyncio
async def test_health_check(tmp_path: object) -> None:
    from pathlib import Path

    tracker = LocalStubProjectTracker(state_root=Path(str(tmp_path)))
    await tracker.connect()

    health = await tracker.health_check()
    assert health.status == "healthy"


@pytest.mark.asyncio
async def test_get_capabilities(tmp_path: object) -> None:
    from pathlib import Path

    tracker = LocalStubProjectTracker(state_root=Path(str(tmp_path)))
    await tracker.connect()

    caps = await tracker.get_capabilities()
    assert "read" in caps
    assert "write" in caps

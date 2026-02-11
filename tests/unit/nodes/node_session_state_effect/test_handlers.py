# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Tests for session state effect handlers.

Verifies:
- Atomic writes with flock on session.json
- Concurrent pipeline isolation
- SessionEnd only clears its own run
- Stale run GC working (4hr TTL)
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumSessionLifecycleState
from omnibase_infra.nodes.node_session_state_effect.handlers import (
    HandlerRunContextRead,
    HandlerRunContextWrite,
    HandlerSessionIndexRead,
    HandlerSessionIndexWrite,
    HandlerStaleRunGC,
)
from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelSessionIndex,
)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Create a temporary state directory."""
    d = tmp_path / "state"
    d.mkdir()
    return d


# ============================================================
# HandlerSessionIndexRead tests
# ============================================================


class TestHandlerSessionIndexRead:
    """Tests for HandlerSessionIndexRead."""

    @pytest.mark.asyncio
    async def test_read_missing_file(self, state_dir: Path) -> None:
        """Reading a non-existent session.json returns default index."""
        handler = HandlerSessionIndexRead(state_dir)
        idx, result = await handler.handle(uuid4())
        assert result.success
        assert idx.active_run_id is None
        assert idx.recent_run_ids == ()

    @pytest.mark.asyncio
    async def test_read_existing_file(self, state_dir: Path) -> None:
        """Reading an existing session.json returns parsed index."""
        data = {
            "active_run_id": "run-1",
            "recent_run_ids": ["run-1", "run-2"],
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        (state_dir / "session.json").write_text(json.dumps(data))

        handler = HandlerSessionIndexRead(state_dir)
        idx, result = await handler.handle(uuid4())
        assert result.success
        assert idx.active_run_id == "run-1"
        assert idx.recent_run_ids == ("run-1", "run-2")

    @pytest.mark.asyncio
    async def test_read_malformed_json(self, state_dir: Path) -> None:
        """Malformed JSON returns default index with error."""
        (state_dir / "session.json").write_text("{invalid json")

        handler = HandlerSessionIndexRead(state_dir)
        idx, result = await handler.handle(uuid4())
        assert not result.success
        assert "PARSE_ERROR" in result.error_code
        assert idx.recent_run_ids == ()


# ============================================================
# HandlerSessionIndexWrite tests
# ============================================================


class TestHandlerSessionIndexWrite:
    """Tests for HandlerSessionIndexWrite."""

    @pytest.mark.asyncio
    async def test_write_creates_file(self, state_dir: Path) -> None:
        """Writing creates session.json."""
        handler = HandlerSessionIndexWrite(state_dir)
        idx = ModelSessionIndex(recent_run_ids=("run-1",))
        result = await handler.handle(idx, uuid4())
        assert result.success

        data = json.loads((state_dir / "session.json").read_text())
        assert data["recent_run_ids"] == ["run-1"]

    @pytest.mark.asyncio
    async def test_write_overwrites(self, state_dir: Path) -> None:
        """Writing overwrites existing session.json atomically."""
        handler = HandlerSessionIndexWrite(state_dir)

        idx1 = ModelSessionIndex(recent_run_ids=("run-1",))
        await handler.handle(idx1, uuid4())

        idx2 = ModelSessionIndex(recent_run_ids=("run-2", "run-1"))
        result = await handler.handle(idx2, uuid4())
        assert result.success

        data = json.loads((state_dir / "session.json").read_text())
        assert data["recent_run_ids"] == ["run-2", "run-1"]

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Writing creates the state directory if it doesn't exist."""
        state_dir = tmp_path / "deep" / "nested" / "state"
        handler = HandlerSessionIndexWrite(state_dir)
        idx = ModelSessionIndex()
        result = await handler.handle(idx, uuid4())
        assert result.success
        assert (state_dir / "session.json").exists()

    @pytest.mark.asyncio
    async def test_concurrent_writes_are_safe(self, state_dir: Path) -> None:
        """Multiple concurrent writes don't corrupt session.json."""
        handler = HandlerSessionIndexWrite(state_dir)

        async def write_index(run_id: str) -> None:
            idx = ModelSessionIndex(recent_run_ids=(run_id,))
            result = await handler.handle(idx, uuid4())
            assert result.success

        # Run 10 concurrent writes
        await asyncio.gather(*(write_index(f"run-{i}") for i in range(10)))

        # File should be valid JSON (one of the writes won)
        data = json.loads((state_dir / "session.json").read_text())
        assert "recent_run_ids" in data


# ============================================================
# HandlerRunContextRead tests
# ============================================================


class TestHandlerRunContextRead:
    """Tests for HandlerRunContextRead."""

    @pytest.mark.asyncio
    async def test_read_missing(self, state_dir: Path) -> None:
        """Reading a non-existent run returns None."""
        handler = HandlerRunContextRead(state_dir)
        ctx, result = await handler.handle("run-missing", uuid4())
        assert result.success
        assert ctx is None

    @pytest.mark.asyncio
    async def test_read_existing(self, state_dir: Path) -> None:
        """Reading an existing run returns parsed context."""
        runs_dir = state_dir / "runs"
        runs_dir.mkdir()
        data = {
            "run_id": "run-1",
            "status": "run_active",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "metadata": {"ticket": "OMN-2117"},
        }
        (runs_dir / "run-1.json").write_text(json.dumps(data))

        handler = HandlerRunContextRead(state_dir)
        ctx, result = await handler.handle("run-1", uuid4())
        assert result.success
        assert ctx is not None
        assert ctx.run_id == "run-1"
        assert ctx.status == EnumSessionLifecycleState.RUN_ACTIVE
        assert ctx.metadata == {"ticket": "OMN-2117"}


# ============================================================
# HandlerRunContextWrite tests
# ============================================================


class TestHandlerRunContextReadPathTraversal:
    """Tests for path traversal rejection in HandlerRunContextRead."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_id",
        ["../etc/passwd", "foo/bar", "foo\\bar", "foo\0bar", ".."],
    )
    async def test_path_traversal_rejected(self, state_dir: Path, bad_id: str) -> None:
        """run_id with path traversal characters returns error result."""
        handler = HandlerRunContextRead(state_dir)
        ctx, result = await handler.handle(bad_id, uuid4())
        assert not result.success
        assert result.error_code == "RUN_CONTEXT_INVALID_ID"
        assert ctx is None


class TestHandlerRunContextWrite:
    """Tests for HandlerRunContextWrite."""

    @pytest.mark.asyncio
    async def test_write_creates_file(self, state_dir: Path) -> None:
        """Writing a run context creates the file."""
        handler = HandlerRunContextWrite(state_dir)
        ctx = ModelRunContext(run_id="run-abc")
        result = await handler.handle(ctx, uuid4())
        assert result.success
        assert (state_dir / "runs" / "run-abc.json").exists()

    @pytest.mark.asyncio
    async def test_roundtrip(self, state_dir: Path) -> None:
        """Written data can be read back identically."""
        writer = HandlerRunContextWrite(state_dir)
        reader = HandlerRunContextRead(state_dir)

        ctx = ModelRunContext(
            run_id="run-rt",
            status=EnumSessionLifecycleState.RUN_ACTIVE,
            metadata={"key": "value"},
        )
        await writer.handle(ctx, uuid4())

        read_ctx, result = await reader.handle("run-rt", uuid4())
        assert result.success
        assert read_ctx is not None
        assert read_ctx.run_id == "run-rt"
        assert read_ctx.status == EnumSessionLifecycleState.RUN_ACTIVE
        assert read_ctx.metadata == {"key": "value"}


# ============================================================
# HandlerStaleRunGC tests
# ============================================================


class TestHandlerStaleRunGC:
    """Tests for HandlerStaleRunGC."""

    @pytest.mark.asyncio
    async def test_gc_removes_stale(self, state_dir: Path) -> None:
        """GC removes run documents older than TTL."""
        runs_dir = state_dir / "runs"
        runs_dir.mkdir()

        # Create a stale run (very old timestamp)
        old = datetime(2020, 1, 1, tzinfo=UTC)
        stale_ctx = ModelRunContext(
            run_id="stale-run",
            created_at=old,
            updated_at=old,
        )
        (runs_dir / "stale-run.json").write_text(stale_ctx.model_dump_json())

        # Create a fresh run
        fresh_ctx = ModelRunContext(run_id="fresh-run")
        (runs_dir / "fresh-run.json").write_text(fresh_ctx.model_dump_json())

        handler = HandlerStaleRunGC(state_dir)
        deleted, result = await handler.handle(uuid4())

        assert result.success
        assert "stale-run" in deleted
        assert "fresh-run" not in deleted
        assert not (runs_dir / "stale-run.json").exists()
        assert (runs_dir / "fresh-run.json").exists()

    @pytest.mark.asyncio
    async def test_gc_removes_malformed(self, state_dir: Path) -> None:
        """GC removes malformed run documents."""
        runs_dir = state_dir / "runs"
        runs_dir.mkdir()
        (runs_dir / "bad.json").write_text("{invalid json")

        handler = HandlerStaleRunGC(state_dir)
        deleted, result = await handler.handle(uuid4())

        assert result.success
        assert "bad" in deleted
        assert not (runs_dir / "bad.json").exists()

    @pytest.mark.asyncio
    async def test_gc_empty_dir(self, state_dir: Path) -> None:
        """GC with no runs directory returns empty list."""
        handler = HandlerStaleRunGC(state_dir)
        deleted, result = await handler.handle(uuid4())
        assert result.success
        assert deleted == []

    @pytest.mark.asyncio
    async def test_gc_custom_ttl(self, state_dir: Path) -> None:
        """GC respects custom TTL setting."""
        runs_dir = state_dir / "runs"
        runs_dir.mkdir()

        # Create a run with a recent but not-quite-now timestamp
        ctx = ModelRunContext(run_id="semi-old")
        (runs_dir / "semi-old.json").write_text(ctx.model_dump_json())

        # With 0-second TTL, everything is stale
        handler = HandlerStaleRunGC(state_dir, ttl_seconds=0.0)
        deleted, result = await handler.handle(uuid4())

        assert result.success
        assert "semi-old" in deleted


# ============================================================
# Concurrent Pipeline Isolation tests
# ============================================================


class TestConcurrentPipelineIsolation:
    """Tests for concurrent pipeline isolation (OMN-2117 DoD)."""

    @pytest.mark.asyncio
    async def test_two_concurrent_pipelines(self, state_dir: Path) -> None:
        """Two concurrent pipelines each get their own runs/{run_id}.json."""
        idx_writer = HandlerSessionIndexWrite(state_dir)
        idx_reader = HandlerSessionIndexRead(state_dir)
        ctx_writer = HandlerRunContextWrite(state_dir)
        ctx_reader = HandlerRunContextRead(state_dir)

        # Pipeline 1 creates a run
        run1 = ModelRunContext(run_id="pipeline-1")
        await ctx_writer.handle(run1, uuid4())

        # Pipeline 2 creates a run
        run2 = ModelRunContext(run_id="pipeline-2")
        await ctx_writer.handle(run2, uuid4())

        # Session index tracks both
        idx = ModelSessionIndex()
        idx = idx.with_run_added("pipeline-1")
        idx = idx.with_run_added("pipeline-2")
        await idx_writer.handle(idx, uuid4())

        # Both runs exist independently
        ctx1, _ = await ctx_reader.handle("pipeline-1", uuid4())
        ctx2, _ = await ctx_reader.handle("pipeline-2", uuid4())
        assert ctx1 is not None
        assert ctx2 is not None
        assert ctx1.run_id == "pipeline-1"
        assert ctx2.run_id == "pipeline-2"

        # Session index contains both
        read_idx, _ = await idx_reader.handle(uuid4())
        assert "pipeline-1" in read_idx.recent_run_ids
        assert "pipeline-2" in read_idx.recent_run_ids

    @pytest.mark.asyncio
    async def test_end_one_pipeline_only_clears_its_own(self, state_dir: Path) -> None:
        """Ending one pipeline only clears its own run document."""
        ctx_writer = HandlerRunContextWrite(state_dir)
        ctx_reader = HandlerRunContextRead(state_dir)
        idx_writer = HandlerSessionIndexWrite(state_dir)
        idx_reader = HandlerSessionIndexRead(state_dir)

        # Both pipelines create runs
        await ctx_writer.handle(ModelRunContext(run_id="p1"), uuid4())
        await ctx_writer.handle(ModelRunContext(run_id="p2"), uuid4())

        idx = ModelSessionIndex()
        idx = idx.with_run_added("p1")
        idx = idx.with_run_added("p2")
        await idx_writer.handle(idx, uuid4())

        # End pipeline 1: update its context and remove from index
        p1_ctx, _ = await ctx_reader.handle("p1", uuid4())
        assert p1_ctx is not None
        ended_ctx = p1_ctx.with_status(EnumSessionLifecycleState.RUN_ENDED)
        await ctx_writer.handle(ended_ctx, uuid4())

        # Update index to remove p1
        read_idx, _ = await idx_reader.handle(uuid4())
        updated_idx = read_idx.with_run_removed("p1")
        await idx_writer.handle(updated_idx, uuid4())

        # Pipeline 2 is still active
        p2_ctx, _ = await ctx_reader.handle("p2", uuid4())
        assert p2_ctx is not None
        assert p2_ctx.run_id == "p2"

        # Pipeline 1's context still exists (but marked ended)
        p1_read, _ = await ctx_reader.handle("p1", uuid4())
        assert p1_read is not None
        assert p1_read.status == EnumSessionLifecycleState.RUN_ENDED

        # Session index only has p2
        final_idx, _ = await idx_reader.handle(uuid4())
        assert "p2" in final_idx.recent_run_ids
        assert "p1" not in final_idx.recent_run_ids

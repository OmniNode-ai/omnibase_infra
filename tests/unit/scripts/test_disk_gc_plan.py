# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Safety tests for the .201 disk-GC removal planner (OMN-13008).

A GC bug that deletes the wrong thing is worse than no GC. These tests pin the
removal-plan invariants of scripts/disk_gc_plan.py WITHOUT touching docker:
the planner is pure (inventory in, plan out), so we can prove it never reaps a
kept repo, a kept tag, an in-use image, or a too-young artifact.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timezone
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "disk_gc_plan", _SCRIPTS / "disk_gc_plan.py"
)
assert _spec and _spec.loader
disk_gc_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(disk_gc_plan)

NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)


def _created(days_ago: float) -> str:
    """A docker CreatedAt string `days_ago` days before NOW."""
    ts = NOW.timestamp() - days_ago * 86400.0
    dt = datetime.fromtimestamp(ts, tz=UTC)
    # docker prints e.g. "2026-05-29 14:03:11 +0000 UTC"
    return dt.strftime("%Y-%m-%d %H:%M:%S +0000 UTC")


KEEP_LIST = {
    "keep_image_repos": ["omninode-runtime", "postgres"],
    "keep_image_tags": ["latest", "rollback"],
    "protect_running": True,
    "superseded_image_keep_generations": 2,
    "min_age_days": 3,
}


def _img(image_id: str, repo: str, tag: str, days_ago: float) -> dict[str, str]:
    return {
        "ID": image_id,
        "Repository": repo,
        "Tag": tag,
        "CreatedAt": _created(days_ago),
    }


@pytest.mark.unit
class TestDiskGcPlanSafety:
    def test_dangling_old_image_is_removed(self) -> None:
        images = [_img("sha-dangle", "<none>", "<none>", days_ago=10)]
        plan = disk_gc_plan.build_plan(KEEP_LIST, images, [], set(), now=NOW)
        assert "sha-dangle" in plan["remove_image_ids"]

    def test_young_dangling_image_is_kept(self) -> None:
        images = [_img("sha-young", "<none>", "<none>", days_ago=1)]
        plan = disk_gc_plan.build_plan(KEEP_LIST, images, [], set(), now=NOW)
        assert plan["remove_image_ids"] == []

    def test_keep_repo_never_removed_regardless_of_age(self) -> None:
        # An old postgres image (keep repo) must never be reaped.
        images = [_img("sha-pg", "postgres", "16", days_ago=400)]
        plan = disk_gc_plan.build_plan(KEEP_LIST, images, [], set(), now=NOW)
        assert "sha-pg" not in plan["remove_image_ids"]

    def test_keep_tag_never_removed(self) -> None:
        images = [_img("sha-latest", "omninode-runtime", "latest", days_ago=400)]
        plan = disk_gc_plan.build_plan(KEEP_LIST, images, [], set(), now=NOW)
        assert "sha-latest" not in plan["remove_image_ids"]

    def test_in_use_image_protected_when_protect_running(self) -> None:
        images = [_img("sha-live", "omninode-runtime", "v0.40.0", days_ago=30)]
        inuse = {"omninode-runtime:v0.40.0"}
        plan = disk_gc_plan.build_plan(KEEP_LIST, images, [], inuse, now=NOW)
        assert "sha-live" not in plan["remove_image_ids"]

    def test_superseded_keeps_newest_n_generations(self) -> None:
        # Five old generations of a kept repo, none in use, none keep-tagged.
        # Keep newest 2, remove the 3 oldest.
        images = [
            _img("g1", "omninode-runtime", "v0.41.0", days_ago=4),
            _img("g2", "omninode-runtime", "v0.40.0", days_ago=10),
            _img("g3", "omninode-runtime", "v0.39.0", days_ago=20),
            _img("g4", "omninode-runtime", "v0.38.0", days_ago=30),
            _img("g5", "omninode-runtime", "v0.37.0", days_ago=40),
        ]
        plan = disk_gc_plan.build_plan(KEEP_LIST, images, [], set(), now=NOW)
        assert "g1" not in plan["remove_image_ids"]  # newest
        assert "g2" not in plan["remove_image_ids"]  # 2nd newest
        assert set(plan["remove_image_ids"]) == {"g3", "g4", "g5"}

    def test_untracked_third_party_repo_is_conservatively_kept(self) -> None:
        # An old non-keep, non-dangling image we don't track: do NOT remove.
        images = [_img("sha-other", "some/random-tool", "1.2.3", days_ago=99)]
        plan = disk_gc_plan.build_plan(KEEP_LIST, images, [], set(), now=NOW)
        assert plan["remove_image_ids"] == []

    def test_stopped_old_container_removed_running_kept(self) -> None:
        containers = [
            {
                "ID": "c-old",
                "State": "exited",
                "Status": "Exited (0) 5 days ago",
                "CreatedAt": _created(5),
            },
            {
                "ID": "c-run",
                "State": "running",
                "Status": "Up 5 days",
                "CreatedAt": _created(5),
            },
            {
                "ID": "c-young",
                "State": "exited",
                "Status": "Exited (0) 1 hour ago",
                "CreatedAt": _created(0.04),
            },
        ]
        plan = disk_gc_plan.build_plan(KEEP_LIST, [], containers, set(), now=NOW)
        assert plan["remove_container_ids"] == ["c-old"]

    def test_empty_inventory_is_noop(self) -> None:
        plan = disk_gc_plan.build_plan(KEEP_LIST, [], [], set(), now=NOW)
        assert plan["remove_image_ids"] == []
        assert plan["remove_container_ids"] == []

    def test_min_age_days_surfaced_in_plan(self) -> None:
        plan = disk_gc_plan.build_plan(KEEP_LIST, [], [], set(), now=NOW)
        assert plan["min_age_days"] == 3

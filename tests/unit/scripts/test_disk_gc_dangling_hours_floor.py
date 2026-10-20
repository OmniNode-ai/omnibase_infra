# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Dangling images are bounded by an hours-scale floor (OMN-20210).

WHY this exists
---------------
On 2026-10-19 the h202 root disk read 97 percent (859G of 937G, 31G free) and
was growing about 189 MB per 90 s. `docker images -f dangling=true` counted
1,617 untagged images: 425 created 2026-10-17, 799 on 2026-10-18 and 391 by
13:00Z on 2026-10-19. Every sampled one carries the compose project label of
the second dev lane, whose deploy agent
rebuilds its 17 services at a roughly hourly cadence, each rebuild re-tags
`:latest`, and the previous generation goes dangling.

The planner kept all of them, because the day-scale `min_age_days: 3` gate ran
before the dangling branch. At that cadence three days of dangling generations
is the whole disk, the same arithmetic OMN-16367 fixed for per-sha tags with
`generation_min_age_hours`. A dangling image has no tag, so no compose file can
name it as a rollback target; it only needs to outlive a build that is still
running, which an hours-scale floor gives it.

The hard guards still win: `keep_image_tags` and `protect_running` run first,
so a dangling image a container references is never removed at any age.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
_spec = importlib.util.spec_from_file_location(
    "disk_gc_plan", _SCRIPTS / "disk_gc_plan.py"
)
assert _spec and _spec.loader
disk_gc_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(disk_gc_plan)

NOW = datetime(2026, 10, 19, 13, 0, 0, tzinfo=UTC)

KEEP_LIST: dict[str, Any] = {
    "keep_image_repos": ["omnibase-infra", "postgres"],
    "keep_image_tags": ["latest", "stable", "rollback"],
    "protect_running": True,
    "superseded_image_keep_generations": 2,
    "min_age_days": 3,
    # The fix under test.
    "dangling_min_age_hours": 6,
}


def _created(hours_ago: float) -> str:
    ts = NOW.timestamp() - hours_ago * 3600.0
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S +0000 UTC")


def _dangling(hours_ago: float, n: int) -> dict[str, Any]:
    return {
        "ID": f"sha256:{n:064x}",
        "Repository": "<none>",
        "Tag": "<none>",
        "CreatedAt": _created(hours_ago),
    }


def _plan(
    images: list[dict[str, Any]],
    inuse_refs: set[str] | None = None,
    **over: Any,
) -> dict[str, Any]:
    return disk_gc_plan.build_plan(
        keep_list={**KEEP_LIST, **over},
        images=images,
        containers=[],
        inuse_refs=inuse_refs or set(),
        now=NOW,
    )


@pytest.mark.unit
class TestDanglingHoursFloor:
    def test_dangling_image_older_than_the_hours_floor_is_removed(self) -> None:
        """RED before the fix: a 12 h old dangling image was kept by min_age_days: 3.

        h202 on 2026-10-19 held 1,617 such images, every one younger than 3 days.
        """
        img = _dangling(12.0, 1)
        plan = _plan([img])
        assert plan["remove_image_ids"] == [img["ID"]], plan["kept_reasons"]

    def test_hourly_dangling_generations_are_reaped_past_the_floor(self) -> None:
        """A day of hourly dev-lane rebuilds: only the last 6 hours survive."""
        images = [_dangling(float(h) + 0.5, h) for h in range(24)]
        removed = set(_plan(images)["remove_image_ids"])
        assert removed == {i["ID"] for i in images[6:]}

    def test_dangling_image_younger_than_the_floor_is_kept(self) -> None:
        """A build that is still running must not lose its intermediate image."""
        img = _dangling(2.0, 2)
        plan = _plan([img])
        assert plan["remove_image_ids"] == []
        assert "dangling_min_age_hours" in plan["kept_reasons"][img["ID"]]

    def test_dangling_image_in_use_is_kept_at_any_age(self) -> None:
        """protect_running still wins over the floor."""
        img = _dangling(48.0, 3)
        plan = _plan([img], inuse_refs={img["ID"]})
        assert plan["remove_image_ids"] == []
        assert "protect_running" in plan["kept_reasons"][img["ID"]]

    def test_tagged_image_still_obeys_min_age_days(self) -> None:
        """The floor is dangling-only: a 12 h tagged generation keeps the day gate."""
        img = {
            "ID": f"sha256:{4:064x}",
            "Repository": "omnibase-infra-lane-runtime-worker",
            "Tag": "previous",
            "CreatedAt": _created(12.0),
        }
        plan = _plan([img])
        assert plan["remove_image_ids"] == []
        assert "min_age_days" in plan["kept_reasons"][img["ID"]]

    def test_absent_key_defaults_to_the_hours_floor(self) -> None:
        """A keep-list without the key still bounds dangling images by hours."""
        keep_list = {
            k: v for k, v in KEEP_LIST.items() if k != "dangling_min_age_hours"
        }
        old, young = _dangling(12.0, 5), _dangling(1.0, 6)
        plan = disk_gc_plan.build_plan(
            keep_list=keep_list,
            images=[old, young],
            containers=[],
            inuse_refs=set(),
            now=NOW,
        )
        assert plan["remove_image_ids"] == [old["ID"]]
        assert disk_gc_plan.DEFAULT_DANGLING_MIN_AGE_HOURS == 6

    def test_committed_keep_list_sets_the_hours_floor(self) -> None:
        """The versioned keep-list the h202 timer reads carries the floor."""
        committed = yaml.safe_load(
            (_REPO / "deploy" / "disk-gc" / "keep-list.yaml").read_text()
        )
        assert committed["dangling_min_age_hours"] == 6
        assert committed["dangling_min_age_hours"] < committed["min_age_days"] * 24

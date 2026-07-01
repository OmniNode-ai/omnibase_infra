# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""PR-state reaping tests for disk_gc_plan.py (OMN-13225).

T1: ghcr.io/omninode-ai/omnibase-infra-runtime:pr-<N> and :sha-* tags are
disposable CI artifacts, not rollback targets. These tests verify that:

  1. A merged-PR pr-* tag is classified REMOVABLE.
  2. An open-PR pr-* tag is KEPT (not removable).
  3. A lookup-error (unknown state) pr-* tag is KEPT (safe default).
  4. A running-lane image is KEPT even when its PR tag is merged.
  5. A keep_image_tags tag is KEPT even when its PR tag is merged.
  6. Dry-run mode: build_plan returns the correct plan without side effects
     (the pr_state_lookup callable is called, but only the plan dict changes
     — no docker rmi or other destructive operations occur).
  7. sha-* tags fall through to age/generation logic when no pr-number exists.
  8. When pr_state_lookup is None, all pr-*/sha-* tags are treated via the
     normal age/generation path (safe default, no network required).

No network calls are made. All tests inject a mock pr_state_lookup.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "disk_gc_plan", _SCRIPTS / "disk_gc_plan.py"
)
assert _spec and _spec.loader
disk_gc_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(disk_gc_plan)

NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=UTC)

KEEP_LIST = {
    "keep_image_repos": ["omninode-runtime", "omnibase-infra"],
    "keep_image_tags": ["latest", "rollback", "stable"],
    "protect_running": True,
    "superseded_image_keep_generations": 2,
    "min_age_days": 3,
}

# A realistic ghcr registry repo prefix for CI images.
GHCR_RUNTIME = "ghcr.io/omninode-ai/omnibase-infra-runtime"


def _created(days_ago: float) -> str:
    """Docker CreatedAt string `days_ago` days before NOW."""
    ts = NOW.timestamp() - days_ago * 86400.0
    dt = datetime.fromtimestamp(ts, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S +0000 UTC")


def _img(
    image_id: str,
    repo: str,
    tag: str,
    days_ago: float = 5.0,
) -> dict[str, str]:
    return {
        "ID": image_id,
        "Repository": repo,
        "Tag": tag,
        "CreatedAt": _created(days_ago),
    }


def _lookup_merged(_pr_number: int) -> str:
    return disk_gc_plan.PR_STATE_MERGED


def _lookup_closed(_pr_number: int) -> str:
    return disk_gc_plan.PR_STATE_CLOSED


def _lookup_open(_pr_number: int) -> str:
    return disk_gc_plan.PR_STATE_OPEN


def _lookup_unknown(_pr_number: int) -> str:
    return disk_gc_plan.PR_STATE_UNKNOWN


@pytest.mark.unit
class TestPrStateClassification:
    """Core DoD: merged-PR tag REMOVABLE; running-lane tag KEPT."""

    def test_merged_pr_tag_is_removable(self) -> None:
        """A pr-<N> tag whose PR is merged must be in remove_image_ids."""
        images = [_img("sha-pr42", GHCR_RUNTIME, "pr-42")]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=_lookup_merged
        )
        assert "sha-pr42" in plan["remove_image_ids"]
        assert "sha-pr42" not in plan["kept_reasons"]

    def test_closed_pr_tag_is_removable(self) -> None:
        """A pr-<N> tag whose PR is closed (not merged) must also be removable."""
        images = [_img("sha-pr99", GHCR_RUNTIME, "pr-99")]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=_lookup_closed
        )
        assert "sha-pr99" in plan["remove_image_ids"]

    def test_open_pr_tag_is_kept(self) -> None:
        """A pr-<N> tag whose PR is still open must NOT be removed."""
        images = [_img("sha-pr7", GHCR_RUNTIME, "pr-7")]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=_lookup_open
        )
        assert "sha-pr7" not in plan["remove_image_ids"]
        assert "sha-pr7" in plan["kept_reasons"]

    def test_unknown_pr_state_is_kept(self) -> None:
        """A pr-<N> tag with lookup error (unknown state) must NOT be removed."""
        images = [_img("sha-pr55", GHCR_RUNTIME, "pr-55")]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=_lookup_unknown
        )
        assert "sha-pr55" not in plan["remove_image_ids"]
        assert "sha-pr55" in plan["kept_reasons"]

    def test_running_lane_image_kept_even_when_pr_merged(self) -> None:
        """Core DoD requirement: protect_running wins over PR-state reaping.

        A running lane container references the image by repo:tag. Even if
        the PR is merged, the image must NOT be removed while a container
        is running.
        """
        tag = "pr-123"
        ref = f"{GHCR_RUNTIME}:{tag}"
        images = [_img("sha-live", GHCR_RUNTIME, tag)]
        inuse = {ref}  # the running-lane container holds this ref
        plan = disk_gc_plan.build_plan(
            KEEP_LIST,
            images,
            [],
            inuse,
            now=NOW,
            pr_state_lookup=_lookup_merged,
        )
        assert "sha-live" not in plan["remove_image_ids"]
        assert "sha-live" in plan["kept_reasons"]
        assert "protect_running" in plan["kept_reasons"]["sha-live"]

    def test_keep_tag_wins_over_pr_state(self) -> None:
        """A tag in keep_image_tags must never be removed, even if PR is merged.

        This is a belt-and-suspenders invariant: keep_image_tags is absolute.
        """
        images = [_img("sha-rollback", GHCR_RUNTIME, "rollback")]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST,
            images,
            [],
            set(),
            now=NOW,
            pr_state_lookup=_lookup_merged,
        )
        assert "sha-rollback" not in plan["remove_image_ids"]
        assert "sha-rollback" in plan["kept_reasons"]

    def test_pr_state_reap_bypasses_age_gate(self) -> None:
        """A merged pr-* tag younger than min_age_days is still REMOVABLE.

        PR-state reaping bypasses the age gate: a just-built CI image whose PR
        is already merged should be reaped immediately, not kept for 3 days.
        """
        images = [_img("sha-fresh", GHCR_RUNTIME, "pr-200", days_ago=0.5)]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=_lookup_merged
        )
        assert "sha-fresh" in plan["remove_image_ids"]


@pytest.mark.unit
class TestPrStateLookupDisabled:
    """When pr_state_lookup is None, pr-*/sha-* tags use normal age/generation logic."""

    def test_no_lookup_pr_tag_falls_through_to_age_path(self) -> None:
        """Without a lookup, a young pr-* tag is kept by the age gate."""
        images = [_img("sha-young-pr", GHCR_RUNTIME, "pr-10", days_ago=1.0)]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=None
        )
        assert "sha-young-pr" not in plan["remove_image_ids"]

    def test_no_lookup_old_pr_tag_follows_generation_logic(self) -> None:
        """Without a lookup, an old pr-* tag on a tracked repo uses generation retention."""
        # 5 old pr-* tags on the tracked repo; keep_generations=2 → oldest 3 removed.
        images = [
            _img(f"sha-g{i}", GHCR_RUNTIME, f"pr-{i}", days_ago=float(4 + i))
            for i in range(1, 6)
        ]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=None
        )
        # Generation logic: sorted by age ascending, keep 2 newest (smallest age).
        # sha-g1 (age 5), sha-g2 (age 6), sha-g3 (age 7), sha-g4 (age 8), sha-g5 (age 9)
        # Keep sha-g1 + sha-g2; remove sha-g3, sha-g4, sha-g5.
        assert "sha-g1" not in plan["remove_image_ids"]
        assert "sha-g2" not in plan["remove_image_ids"]
        assert "sha-g3" in plan["remove_image_ids"]
        assert "sha-g4" in plan["remove_image_ids"]
        assert "sha-g5" in plan["remove_image_ids"]


@pytest.mark.unit
class TestShaTagBehavior:
    """sha-* tags have no PR number — they use the normal age/generation path."""

    def test_sha_tag_no_pr_number_uses_age_path(self) -> None:
        """A sha-* tag falls through to normal age/generation logic (no pr_number)."""
        # Old sha-* tag on tracked repo — falls through to generation logic.
        images = [
            _img("sha-abc", GHCR_RUNTIME, "sha-abc1234", days_ago=10.0),
        ]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=_lookup_merged
        )
        # sha-abc is the only generation of GHCR_RUNTIME; generation keep=2, only 1
        # entry → it is within the keep window.
        assert "sha-abc" not in plan["remove_image_ids"]

    def test_sha_tag_old_beyond_generations_is_removed(self) -> None:
        """Old sha-* tags beyond keep_generations are removed (age/generation path)."""
        # 4 sha-* tags; only 2 kept by generation.
        images = [
            _img("sha-s1", GHCR_RUNTIME, "sha-aaaaaa", days_ago=4.0),
            _img("sha-s2", GHCR_RUNTIME, "sha-bbbbbb", days_ago=8.0),
            _img("sha-s3", GHCR_RUNTIME, "sha-cccccc", days_ago=12.0),
            _img("sha-s4", GHCR_RUNTIME, "sha-dddddd", days_ago=20.0),
        ]
        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=_lookup_merged
        )
        # Sorted by age ascending: sha-s1(4), sha-s2(8), sha-s3(12), sha-s4(20)
        # Keep 2 newest: sha-s1, sha-s2; remove sha-s3, sha-s4.
        assert "sha-s1" not in plan["remove_image_ids"]
        assert "sha-s2" not in plan["remove_image_ids"]
        assert "sha-s3" in plan["remove_image_ids"]
        assert "sha-s4" in plan["remove_image_ids"]


@pytest.mark.unit
class TestDryRunPlanCorrectness:
    """DoD: dry-run plan, when fed a merged PR + running-lane tag, classifies correctly."""

    def test_dry_run_mixed_inventory_classifies_correctly(self) -> None:
        """A mixed inventory (merged CI tag + running lane + keep tag) → correct plan.

        This is the canonical DoD assertion for T1: the dry-run plan must show
        merged/closed-PR ghcr tags as REMOVABLE and all running-lane + keep-list
        rollback tags as KEPT.
        """
        running_tag = "v0.38.3"
        running_ref = f"{GHCR_RUNTIME}:{running_tag}"

        images = [
            # Merged CI tag — should be REMOVABLE.
            _img("sha-ci-pr", GHCR_RUNTIME, "pr-500", days_ago=5.0),
            # Running lane image — KEPT by protect_running.
            _img("sha-running", GHCR_RUNTIME, running_tag, days_ago=10.0),
            # Keep tag — KEPT by keep_image_tags.
            _img("sha-rollback", GHCR_RUNTIME, "rollback", days_ago=30.0),
        ]
        inuse = {running_ref, "sha-running"}

        # Lookup: pr-500 is merged; running_tag is not a pr-tag (no lookup).
        def _lookup(pr_number: int) -> str:
            if pr_number == 500:
                return disk_gc_plan.PR_STATE_MERGED
            return disk_gc_plan.PR_STATE_UNKNOWN

        plan = disk_gc_plan.build_plan(
            KEEP_LIST, images, [], inuse, now=NOW, pr_state_lookup=_lookup
        )

        # Merged CI tag is removable.
        assert "sha-ci-pr" in plan["remove_image_ids"]
        # Running lane image is kept.
        assert "sha-running" not in plan["remove_image_ids"]
        assert "sha-running" in plan["kept_reasons"]
        assert "protect_running" in plan["kept_reasons"]["sha-running"]
        # Keep tag is kept.
        assert "sha-rollback" not in plan["remove_image_ids"]
        assert "sha-rollback" in plan["kept_reasons"]

    def test_lookup_called_with_correct_pr_number(self) -> None:
        """Verify that the lookup is called with the correct PR number from the tag."""
        calls: list[int] = []

        def _recording_lookup(pr_number: int) -> str:
            calls.append(pr_number)
            return disk_gc_plan.PR_STATE_MERGED

        images = [_img("sha-pr77", GHCR_RUNTIME, "pr-77")]
        disk_gc_plan.build_plan(
            KEEP_LIST, images, [], set(), now=NOW, pr_state_lookup=_recording_lookup
        )
        assert calls == [77]

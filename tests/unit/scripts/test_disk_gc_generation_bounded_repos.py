# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Generation-bounded retention for per-sha image families (OMN-16367, DoD item 2).

WHY these exist
---------------
On 2026-09-18 the `.201` bleed was re-measured three times and neither of the two
suspected causes survived contact:

- **It is not build cache.** Three approved `docker builder prune` runs reclaimed
  **0B**, the last one carrying `--all`. `docker buildx du` reads Shared 723.5 GB
  of Total 750 GB, so ~96% of the cache is content shared with image layers in the
  containerd snapshot store, which no builder prune can return while those images
  exist. (On this buildx `--all` means "include internal/frontend images", not the
  classic CLI's "remove all unused cache".)
- **It is not CI proof images.** Exactly **3** proof-pattern images exist on the
  whole host across 792 images. The wave-1 `--rmi local` teardown works, because
  those compose services declare no `image:` field, so `--rmi local` genuinely
  removes what they build.

What is actually accumulating is the **deploy agent's per-sha tags**, rebuilt
roughly hourly and never reaped, counted live:

| images | nominal | repo |
|---|---|---|
| 60 | 219.3 GB | `onex-lab/omninode-runtime` |
| 179 | 205.1 GB | ECR `omnicloud/core` |
| 93 | 106.3 GB | `onex-lab/omnicloud-core` |

Two distinct planner defects let that happen, and this module pins the fix for
both:

1. **Untracked repos are kept forever.** A repo that matches no
   `keep_image_repos` substring falls through to "repo not in keep_image_repos
   (conservative keep)". `omnicloud/core`, `omnicloud-core`,
   `omninode-cloud-migrate` and `omniweb` all miss that list, so ~408 images and
   350+ GB grow without bound.
2. **`min_age_days` defeats a keep-newest-N window at an hourly cadence.** A repo
   that *does* match is still gated behind `min_age_days: 3`, so ~72 generations
   are always too young before "keep the newest 2" can bite. The 60 live
   generations of `onex-lab/omninode-runtime` are that arithmetic.

The fix is the ticket's own DoD item 2, verbatim: a keep-last-N window over
per-sha tag families, excluding anything referenced by a container, never a
blanket prune.

Safety posture: the rule is **opt-in by repo** and **opt-in by tag shape**. A repo
not listed is untouched, and a tag that does not match a build-stamped shape is
untouched. Both failure directions land on KEEP.
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

NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)

# The per-sha families measured on .201, and the repos that must stay untouched.
AGENT_RUNTIME = "onex-lab/omninode-runtime"
AGENT_CLOUD = "onex-lab/omnicloud-core"
# Synthetic account id: the real one is denylisted by the exposed-identifier
# gate (OMN-18024) and the test does not need it -- only the repo suffix is
# load-bearing, because that is what generation_bounded_repos matches on.
ECR_CORE = "000000000000.dkr.ecr.us-east-1.amazonaws.com/omnicloud/core"
STABILITY = "omnibase-infra-stability-test-runtime-worker"


def _created(hours_ago: float) -> str:
    ts = NOW.timestamp() - hours_ago * 3600.0
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S +0000 UTC")


def _img(repo: str, tag: str, hours_ago: float, image_id: str) -> dict[str, Any]:
    return {
        "ID": image_id,
        "Repository": repo,
        "Tag": tag,
        "CreatedAt": _created(hours_ago),
    }


KEEP_LIST: dict[str, Any] = {
    "keep_image_repos": ["omninode-runtime", "postgres"],
    "keep_image_tags": ["latest", "stable", "rollback"],
    "protect_running": True,
    "superseded_image_keep_generations": 2,
    "min_age_days": 3,
    # The fix under test.
    "generation_bounded_repos": ["onex-lab/", "omnicloud/core", "omniweb"],
    "generation_keep": 3,
    "generation_min_age_hours": 6,
}


def _plan(images: list[dict[str, Any]], **over: Any) -> dict[str, Any]:
    keep_list = {**KEEP_LIST, **over}
    return disk_gc_plan.build_plan(
        keep_list=keep_list,
        images=images,
        containers=[],
        inuse_refs=set(),
        now=NOW,
    )


def _agent_generations(
    repo: str, count: int, *, start_hours: float = 6.0
) -> list[dict[str, Any]]:
    """`count` hourly generations of a deploy-agent per-sha family, newest first."""
    return [
        _img(
            repo,
            f"2026091{8 - i // 24}T{i:02d}0000Z-{i:08x}",
            start_hours + i,
            f"sha256:{i:064x}",
        )
        for i in range(count)
    ]


@pytest.mark.unit
class TestUntrackedPerShaFamiliesAreBounded:
    """Defect 1 — a repo outside keep_image_repos must not be kept forever."""

    def test_ecr_per_sha_family_is_reaped_beyond_the_window(self) -> None:
        """RED before the fix: ECR omnicloud/core matches no keep_image_repos substring.

        Live count on .201: 179 images, 205.1 GB, growing without bound.
        """
        images = _agent_generations(ECR_CORE, 10)
        plan = _plan(images)
        removed = set(plan["remove_image_ids"])

        assert len(removed) == 7, (
            "expected 10 generations minus the newest 3 to be reaped; got "
            f"{len(removed)}. Reasons: {plan['kept_reasons']}"
        )
        # The newest three survive as rollback headroom.
        for keeper in images[:3]:
            assert keeper["ID"] not in removed

    def test_untracked_repo_not_listed_is_still_kept_forever(self) -> None:
        """The rule is opt-in. An unlisted third-party repo keeps its old behaviour."""
        images = _agent_generations("docker.io/some/vendor-image", 10)
        plan = _plan(images)
        assert plan["remove_image_ids"] == [], (
            "an unlisted repo was reaped; the generation rule must be opt-in by repo"
        )


@pytest.mark.unit
class TestHourlyCadenceDefeatsTheDayGate:
    """Defect 2 — min_age_days must not gate a keep-newest-N window."""

    def test_generations_younger_than_min_age_days_are_still_bounded(self) -> None:
        """RED before the fix: at hourly cadence every generation is <3d old.

        All 10 generations here are 6-16 hours old, so the existing
        `min_age_days: 3` gate keeps every one of them. That is exactly why 60
        live generations of onex-lab/omninode-runtime exist.
        """
        images = _agent_generations(AGENT_RUNTIME, 10)
        for i in images:
            assert "2026-09-18" in i["CreatedAt"] or "2026-09-17" in i["CreatedAt"]

        plan = _plan(images)
        assert len(plan["remove_image_ids"]) == 7, (
            "the day-scale age gate is still swallowing an hourly cadence; "
            f"kept_reasons={plan['kept_reasons']}"
        )

    def test_freshly_built_generation_is_never_reaped(self) -> None:
        """The hours-scale floor still protects a just-built image."""
        images = [
            _img(AGENT_RUNTIME, "20260918T115900Z-aaaaaaaa", 0.1, "sha256:" + "a" * 64),
            *_agent_generations(AGENT_RUNTIME, 9),
        ]
        plan = _plan(images)
        assert "sha256:" + "a" * 64 not in set(plan["remove_image_ids"]), (
            "an image built minutes ago was reaped; the generation_min_age_hours "
            "floor did not hold"
        )


@pytest.mark.unit
class TestTagShapeOptIn:
    """Only build-stamped tag shapes are reapable; everything else is kept."""

    @pytest.mark.parametrize(
        "tag",
        [
            "20260918T111918Z-e49dea8f",  # <stamp>-<sha>, deploy agent
            "ffd61901-20260918T111918Z",  # <sha>-<stamp>, deploy agent
            "sha-ffd6190",  # bare sha- tag
            "dev-sha-ffd6190",  # prefixed sha- tag, ECR
            "ffd6190171066f231bd6c6725fd53352d1222aa6",  # bare 40-hex, ECR
        ],
        ids=["stamp-sha", "sha-stamp", "sha-tag", "prefixed-sha", "bare-40-hex"],
    )
    def test_build_stamped_shapes_are_reapable(self, tag: str) -> None:
        """Every shape here was read off .201, not invented."""
        images = [
            _img(ECR_CORE, tag, 48.0, "sha256:" + "b" * 64),
            *_agent_generations(ECR_CORE, 5),
        ]
        plan = _plan(images)
        assert "sha256:" + "b" * 64 in set(plan["remove_image_ids"]), (
            f"build-stamped tag {tag!r} was not recognised as a reapable generation"
        )

    @pytest.mark.parametrize("tag", ["dev", "main", "latest", "stable", "rollback"])
    def test_moving_pointers_and_keep_tags_are_never_reaped(self, tag: str) -> None:
        """`dev` and `main` are moving pointers; the keep tags are rollback targets.

        None of them is a build-stamped shape, so the rule must leave them alone
        even though their repo is listed.
        """
        images = [
            _img(ECR_CORE, tag, 480.0, "sha256:" + "c" * 64),
            *_agent_generations(ECR_CORE, 8),
        ]
        plan = _plan(images)
        assert "sha256:" + "c" * 64 not in set(plan["remove_image_ids"]), (
            f"the non-build-stamped tag {tag!r} was reaped"
        )


@pytest.mark.unit
class TestGovernedLanesStayOutOfScope:
    """The stability / judge / lakshman lanes must not be reachable by this rule."""

    def test_stability_lane_preflight_stamps_are_untouched(self) -> None:
        """`preflight-<stamp>` on a stability repo looks stamped but is out of scope.

        The stability lane is the surface the compose-path `stability-proven`
        premise is resolved from, and it is explicitly not this lane's to mutate.
        It is protected structurally: its repo matches no entry in
        `generation_bounded_repos`, so opt-in-by-repo is what keeps it safe.
        """
        images = [
            _img(STABILITY, "preflight-20260916T165814Z", 48.0, "sha256:" + "d" * 64),
            _img(STABILITY, "preflight-20260916T142602Z", 50.0, "sha256:" + "e" * 64),
            _img(STABILITY, "preflight-20260910T203929Z", 200.0, "sha256:" + "f" * 64),
            _img(STABILITY, "latest", 200.0, "sha256:" + "1" * 64),
        ]
        plan = _plan(images)
        assert plan["remove_image_ids"] == [], (
            "a stability-lane image was selected for removal; "
            f"kept_reasons={plan['kept_reasons']}"
        )


@pytest.mark.unit
class TestExistingSafetyInvariantsSurvive:
    """The generation rule must not punch through any existing protection."""

    def test_in_use_image_is_never_reaped(self) -> None:
        images = _agent_generations(AGENT_CLOUD, 10)
        victim = images[-1]
        plan = disk_gc_plan.build_plan(
            keep_list=KEEP_LIST,
            images=images,
            containers=[],
            inuse_refs={f"{victim['Repository']}:{victim['Tag']}"},
            now=NOW,
        )
        assert victim["ID"] not in set(plan["remove_image_ids"]), (
            "protect_running was bypassed by the generation rule"
        )

    def test_keep_image_tags_still_win_over_the_generation_rule(self) -> None:
        images = [
            _img(AGENT_CLOUD, "rollback", 900.0, "sha256:" + "2" * 64),
            *_agent_generations(AGENT_CLOUD, 8),
        ]
        plan = _plan(images)
        assert "sha256:" + "2" * 64 not in set(plan["remove_image_ids"])

    def test_disabling_the_rule_restores_the_previous_behaviour(self) -> None:
        """An empty repo list is the off switch, and it must be a real off switch."""
        images = _agent_generations(ECR_CORE, 10)
        plan = _plan(images, generation_bounded_repos=[])
        assert plan["remove_image_ids"] == []


@pytest.mark.unit
class TestShippedKeepListEnablesTheMeasuredFamilies:
    """The default config must actually cover what was measured on the host."""

    def test_keep_list_lists_the_families_that_are_growing(self) -> None:
        keep_list = yaml.safe_load(
            (_REPO / "deploy" / "disk-gc" / "keep-list.yaml").read_text()
        )
        listed = keep_list.get("generation_bounded_repos") or []
        for measured in ("onex-lab/", "omnicloud/core", "omniweb"):
            assert any(measured in entry for entry in listed), (
                f"{measured!r} was measured growing without bound on .201 but is "
                f"not in generation_bounded_repos: {listed}"
            )

    def test_keep_window_leaves_rollback_headroom(self) -> None:
        keep_list = yaml.safe_load(
            (_REPO / "deploy" / "disk-gc" / "keep-list.yaml").read_text()
        )
        assert keep_list["generation_keep"] >= 2, (
            "a keep window under 2 leaves the deploy agent no rollback target"
        )

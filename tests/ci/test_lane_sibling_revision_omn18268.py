# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The sibling-revision convergence guard (OMN-18268).

WHY A SECOND GUARD RATHER THAN A WIDER FIRST ONE.
``check_dev_lane_staleness.py`` reads ``org.opencontainers.image.revision`` off
the running container. That label is stamped from the OMNIBASE_INFRA commit, and
a sibling-triggered rebuild does not move it -- the lane can be rebuilt five
times at one infra SHA while vendoring five different omnimarket revisions. So
for a sibling trigger that guard answers ``converged`` on its FIRST poll while
proving nothing about the merge that fired it: a false green by construction,
which is the exact failure mode OMN-18200 removed from the compose-dev receipt
and must not be reintroduced here.

WHAT THIS GUARD READS INSTEAD. The image writes
``/app/build-provenance.json`` at build time (``compute_workspace_provenance.py``),
carrying ``per_repo_vcs_provenance.siblings.<repo>.vcs_ref`` -- the commit
``stage_workspace.sh`` actually resolved for that sibling. Read live on
2026-09-14 from ``omninode-runtime-effects``: ``omnimarket`` ->
``65c237cdc914a6c7c1e15de308986ec65c24f309`` (omnimarket#2538) while
``org.opencontainers.image.revision`` was ``732fd291...`` (omnibase_infra).

CONTAINMENT, NOT EQUALITY. The sibling ref is resolved from ``origin/dev`` at
staging time, so a build that starts after a LATER merge legitimately carries a
DESCENDANT of the merge that fired the trigger. Equality would red on a lane that
is more current than asked. ``identical`` and ``ahead`` both mean the lane
carries the merge; ``behind`` and ``diverged`` mean it does not.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "check_lane_sibling_revision.py"
)

# The live bytes read from omninode-runtime-effects on 2026-09-14T10:2xZ.
LIVE_PROVENANCE = {
    "build_source": "workspace",
    "build_time": "2026-09-14T08:56:47Z",
    "infra_vcs_ref": "732fd291a98afeac849a4904670b210096649529",
    "per_repo_vcs_provenance": {
        "siblings": {
            "omnibase_core": {
                "vcs_ref": "cde3aaa3281e47cc9c062b154460157ea5e8f93c",
                "vcs_dirty": False,
                "vcs_branch": "HEAD",
            },
            "omnibase_compat": {
                "vcs_ref": "3b6f801d5d02bc57b4ca71440f71878dc1eb98ab",
                "vcs_dirty": False,
                "vcs_branch": "HEAD",
            },
            "omnimarket": {
                "vcs_ref": "65c237cdc914a6c7c1e15de308986ec65c24f309",
                "vcs_dirty": False,
                "vcs_branch": "HEAD",
            },
        }
    },
}

PR_2538 = "65c237cdc914a6c7c1e15de308986ec65c24f309"
PR_2537 = "4bc2437210939702fa0dc8529362c0a7c231049f"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("_lane_sibling_revision", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_lane_sibling_revision"] = module
    spec.loader.exec_module(module)
    return module


def test_parses_the_sibling_ref_out_of_the_live_manifest() -> None:
    module = _load()
    assert module.parse_sibling_revision(LIVE_PROVENANCE, "omnimarket") == PR_2538


def test_a_sibling_absent_from_the_manifest_fails_closed() -> None:
    """An unlisted sibling is unknown, never 'unchanged'."""
    module = _load()
    with pytest.raises(ValueError, match="omniweb"):
        module.parse_sibling_revision(LIVE_PROVENANCE, "omniweb")


def test_a_dirty_sibling_tree_fails_closed() -> None:
    """``vcs_dirty`` means the recorded SHA does not describe what was built."""
    module = _load()
    payload = {
        "per_repo_vcs_provenance": {
            "siblings": {"omnimarket": {"vcs_ref": PR_2538, "vcs_dirty": True}}
        }
    }
    with pytest.raises(ValueError, match="dirty"):
        module.parse_sibling_revision(payload, "omnimarket")


@pytest.mark.parametrize("status", ["identical", "ahead"])
def test_lane_carrying_the_merge_converges(status: str) -> None:
    module = _load()
    verdict = module.evaluate_sibling_convergence(
        repo="omnimarket",
        lane_revision=PR_2538,
        expected_revision=PR_2537,
        containment=status,
        waited=timedelta(minutes=3),
        wait_timeout=timedelta(minutes=25),
    )
    assert verdict.ok, [f.render() for f in verdict.findings]


@pytest.mark.parametrize("status", ["behind", "diverged"])
def test_lane_without_the_merge_does_not_converge(status: str) -> None:
    module = _load()
    verdict = module.evaluate_sibling_convergence(
        repo="omnimarket",
        lane_revision=PR_2538,
        expected_revision=PR_2537,
        containment=status,
        waited=timedelta(minutes=25),
        wait_timeout=timedelta(minutes=25),
    )
    assert not verdict.ok
    assert any(f.code == "SIBLING_NOT_CONVERGED" for f in verdict.findings)


def test_an_unrecognised_compare_status_fails_closed() -> None:
    """A status this guard has not learned is not a pass."""
    module = _load()
    verdict = module.evaluate_sibling_convergence(
        repo="omnimarket",
        lane_revision=PR_2538,
        expected_revision=PR_2537,
        containment="something_new",
        waited=timedelta(minutes=1),
        wait_timeout=timedelta(minutes=25),
    )
    assert not verdict.ok


def test_the_2026_09_14_measurement_replays_as_a_failure() -> None:
    """The live lane did NOT carry omnimarket#2537; the guard must say so.

    This is the incident itself, not a shape a test author imagined: the parse
    runs over the manifest bytes read from the running container, and #2537
    merged 15 minutes AFTER that image was built, so the compare is ``behind``.
    """
    module = _load()
    lane_revision = module.parse_sibling_revision(LIVE_PROVENANCE, "omnimarket")
    verdict = module.evaluate_sibling_convergence(
        repo="omnimarket",
        lane_revision=lane_revision,
        expected_revision=PR_2537,
        containment="behind",
        waited=timedelta(minutes=25),
        wait_timeout=timedelta(minutes=25),
    )
    assert not verdict.ok


def test_lane_fence_refuses_a_governed_compose_project() -> None:
    """Read-only or not, this guard may only ever look at the dev lane."""
    module = _load()
    with pytest.raises(ValueError, match="lane fence"):
        module.assert_lane_fence(
            compose_project="omnibase-infra-stability-test",
            container="omninode-runtime-effects",
            expected_project="omnibase-infra",
        )

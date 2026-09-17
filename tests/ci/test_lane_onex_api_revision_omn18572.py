# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The onex-api revision convergence guard (OMN-18572).

WHY A THIRD GUARD, AND WHY NEITHER OF THE OTHER TWO ANSWERS THIS.
``check_dev_lane_staleness.py`` reads ``org.opencontainers.image.revision`` off
``omninode-runtime`` -- an **omnibase_infra** commit. ``check_lane_sibling_revision.py``
reads ``/app/build-provenance.json`` out of the runtime image, whose
``per_repo_vcs_provenance.siblings`` lists the repositories ``stage_workspace.sh``
VENDORS. ``omninode_infra`` is in neither: ``onex-api`` is not vendored into the
runtime image at all, it is its OWN image, built by the lab-overlay applier from
``docker/onex-api`` in the omninode_infra overlay tree and run as a separate
service pinned by ``ONEX_API_IMAGE``.

WHAT THIS GUARD READS INSTEAD. The ``onex-api`` container's own
``org.opencontainers.image.revision`` label. ``lab_overlay.build_and_import``
stamps it with the omninode_infra manifest sha the image was built from
(``lab_overlay.py`` ``API_REVISION_LABEL``), and ``repoint_dev_lane_onex_api.py``
already cross-checks that label against the tag's embedded sha8 before pinning.
So the label is the structured provenance fact for this image, exactly as the
provenance manifest is for the runtime one.

Read live on 2026-09-17 from the ``onex-api`` container on the ``.201`` dev lane:
``99fdbd375f3b6c17d564b588f505754160b1d1f2``, on tag
``onex-lab/omnicloud-core:99fdbd37-20260917T110254Z``.

CONTAINMENT, NOT EQUALITY, for the same reason the sibling guard uses it. The
applier builds from the omninode_infra clone's ``origin/dev`` at the moment the
agent reaches it, so a rebuild that starts after a LATER merge legitimately
carries a DESCENDANT of the merge that fired this run. Equality would red on a
lane that is more current than asked -- the OMN-18388 defect, reintroduced.

AN ABSENT LABEL IS UNKNOWN, NEVER UNCHANGED. Images built before OMN-18113 carry
no OCI labels at all, and a guard that read "no label" as "nothing to compare"
would report a lane still running an eight-day-old hand build as converged.
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
    / "check_lane_onex_api_revision.py"
)

#: The live label set read from the `onex-api` container on 2026-09-17, after
#: the OMN-18113 delivery. Kept as bytes-from-the-host rather than a shape a
#: test author invented: a fixture rebuilt by hand proves the verdict logic and
#: nothing about whether the guard can read what the build actually writes.
LIVE_LABELS = {
    "org.opencontainers.image.revision": ("99fdbd375f3b6c17d564b588f505754160b1d1f2"),
    "ai.omninode.image.source-repo": "omninode_infra",
    "com.docker.compose.project": "omnibase-infra",
    "com.docker.compose.service": "onex-api",
}

MERGED = "99fdbd375f3b6c17d564b588f505754160b1d1f2"
PARENT = "f37261c2ada3db7ca5eb194ee15507232f649029"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("_lane_onex_api_revision", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_lane_onex_api_revision"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Reading the revision off the container's labels                             #
# --------------------------------------------------------------------------- #


def test_reads_the_revision_from_the_live_label_set() -> None:
    module = _load()
    assert module.parse_onex_api_revision(LIVE_LABELS) == MERGED


def test_an_absent_revision_label_fails_closed() -> None:
    """A pre-OMN-18113 image carries no labels; that is UNKNOWN, not unchanged."""
    module = _load()
    with pytest.raises(ValueError, match=r"no org\.opencontainers\.image\.revision"):
        module.parse_onex_api_revision({"com.docker.compose.service": "onex-api"})


def test_a_null_label_map_fails_closed() -> None:
    """`docker inspect` renders an unlabelled image's Labels as null, not {}."""
    module = _load()
    with pytest.raises(ValueError):
        module.parse_onex_api_revision(None)


def test_a_source_repo_label_naming_another_repository_fails_closed() -> None:
    """The label must describe an omninode_infra commit, or it is not comparable."""
    module = _load()
    labels = dict(LIVE_LABELS)
    labels["ai.omninode.image.source-repo"] = "omnibase_infra"
    with pytest.raises(ValueError, match="source-repo"):
        module.parse_onex_api_revision(labels)


def test_a_non_sha_revision_is_refused_rather_than_compared() -> None:
    module = _load()
    labels = dict(LIVE_LABELS)
    labels["org.opencontainers.image.revision"] = "dev"
    with pytest.raises(ValueError, match="not a git SHA"):
        module.parse_onex_api_revision(labels)


# --------------------------------------------------------------------------- #
# The convergence verdict                                                      #
# --------------------------------------------------------------------------- #


def _verdict(module: object, containment: str, lane: str = MERGED) -> object:
    return module.evaluate_onex_api_convergence(  # type: ignore[attr-defined]
        lane_revision=lane,
        expected_revision=MERGED,
        containment=containment,
        waited=timedelta(seconds=30),
        wait_timeout=timedelta(minutes=25),
    )


@pytest.mark.parametrize("status", ["identical", "ahead"])
def test_identical_and_ahead_both_mean_the_lane_carries_the_merge(
    status: str,
) -> None:
    """`ahead` is a lane MORE current than asked, which is not a failure."""
    module = _load()
    verdict = _verdict(module, status)
    assert verdict.ok is True
    assert verdict.notes


@pytest.mark.parametrize("status", ["behind", "diverged"])
def test_behind_and_diverged_both_fail(status: str) -> None:
    module = _load()
    verdict = _verdict(module, status, lane=PARENT)
    assert verdict.ok is False
    assert verdict.findings[0].code == "ONEX_API_NOT_CONVERGED"


def test_an_unrecognised_containment_status_fails_closed() -> None:
    """An unmapped compare status is not a pass; it is an unknown."""
    module = _load()
    verdict = _verdict(module, "something-new")
    assert verdict.ok is False
    assert verdict.findings[0].code == "CONTAINMENT_UNKNOWN"


def test_the_failure_names_the_pin_as_the_thing_to_check() -> None:
    """A finding that does not say where to look is a finding nobody acts on."""
    module = _load()
    verdict = _verdict(module, "behind", lane=PARENT)
    detail = verdict.findings[0].detail
    assert "ONEX_API_IMAGE" in detail
    assert PARENT[:12] in detail
    assert MERGED[:12] in detail


# --------------------------------------------------------------------------- #
# The lane fence -- one owner, not a second copy                              #
# --------------------------------------------------------------------------- #


def test_the_lane_fence_refuses_any_project_but_the_dev_lane() -> None:
    """A governed lane must be unreadable by this guard even by container name."""
    module = _load()
    with pytest.raises(ValueError, match="lane fence"):
        module.assert_lane_fence(
            "omnibase-infra-stability-test", "onex-api", "omnibase-infra"
        )


def test_the_fence_and_the_verdict_shape_are_the_sibling_guard_s_own() -> None:
    """Reused, not reimplemented: two copies are two things to drift.

    Asserted on where each symbol's CODE was compiled from, not on object
    identity -- loading the sibling guard a second time by path under a second
    module name produces equal-but-distinct objects, so an identity check here
    would fail against correct code. The positive control is the test above it:
    the fence is proven to FIRE, so this cannot pass against a stub that merely
    re-exports a name.
    """
    module = _load()
    sibling_source = _SCRIPT.with_name("check_lane_sibling_revision.py")

    assert (
        Path(module.assert_lane_fence.__code__.co_filename).resolve()
        == sibling_source.resolve()
    )
    for shape in (module.Finding, module.Verdict):
        assert Path(sys.modules[shape.__module__].__file__).resolve() == (
            sibling_source.resolve()
        )


# --------------------------------------------------------------------------- #
# The CLI surface the workflow spells                                          #
# --------------------------------------------------------------------------- #


def test_the_container_default_is_the_dev_lane_s_onex_api() -> None:
    module = _load()
    assert module.DEFAULT_CONTAINER == "onex-api"
    assert module.DEFAULT_COMPOSE_PROJECT == "omnibase-infra"
    assert module.SOURCE_REPO_SLUG == "OmniNode-ai/omninode_infra"


def test_expect_revision_is_required_and_has_no_default() -> None:
    """A guard that defaults its expectation asserts nothing (rule 8)."""
    module = _load()
    parser = module._build_parser()
    action = next(a for a in parser._actions if a.dest == "expect_revision")
    assert action.required is True
    assert action.default is None

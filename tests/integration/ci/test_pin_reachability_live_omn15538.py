# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live half of the protected-branch pin-reachability gate (OMN-15538).

Static extraction + oracle tests are hermetic and live in
``tests/ci/test_pin_reachability_omn15538.py``. This module holds only the
tests that hit the live GitHub API, under ``tests/integration`` for the same
reasons as its OMN-14941 sibling: the pre-push selector always ignores that
tree, and a red here must not make a branch locally unpushable.

Two things are proven live, and both matter:

1. **Discrimination on the real incident vectors.** The two SHAs that caused
   the 2026-07-30 incidents must resolve UNREACHABLE and their correct
   counterparts REACHABLE. These are the actual pins, not synthetic values —
   proving RED against *exists-but-wrong* rather than against absent (memory
   ``feedback_prove_red_against_exists_but_wrong``). ``879d6fc6`` and
   ``5a907b71`` are both real commits that GitHub still serves; an existence
   probe passes on both, which is exactly why the oracle is compare-based.

2. **This repo's own tree is clean.** Expected GREEN. There is no standing
   authorization for a red here — if it fails, a pin in this repo is not
   durable; re-pin it (OMN-15248 posture: a blessed failure is how a real
   regression gets waved through).

Note on ``5a907b71``'s expected lifetime: it is currently ``ahead`` of core
``dev`` (the live head of ``jonah/omn-15392-evidence-execution-scope``). When
OMN-15392 lands, it becomes ``diverged`` — still UNREACHABLE, so the assertion
below holds across that transition. That is the point: the verdict is stable
while the reason changes, because the pin was never durable in either state.
"""

from __future__ import annotations

import os

import pytest

from scripts.ci.check_pin_reachability import (
    Verdict,
    _Resolver,
    extract_pins,
)
from tests.ci.test_pin_reachability_omn15538 import (
    INCIDENT_A_DEAD_SHA,
    INCIDENT_A_GOOD_SHA,
    INCIDENT_B_DEAD_SHA,
    INCIDENT_B_GOOD_SHA,
    REPO_ROOT,
    WORKFLOWS_DIR,
)

_IN_CI = bool(os.environ.get("CI"))


def _resolve(repo: str, ref: str) -> tuple[Verdict, str]:
    resolution = _Resolver(("dev", "main")).resolve(repo, ref)
    return resolution.verdict, resolution.detail


def _require_determined(verdict: Verdict, detail: str, label: str) -> None:
    """Fail closed in CI, skip only on a developer machine."""
    if verdict is not Verdict.UNDETERMINED:
        return
    message = f"{label}: could not resolve against the live GitHub API ({detail})"
    if _IN_CI:
        pytest.fail(f"{message} — an unresolvable pin is not a passing pin")
    pytest.skip(message)


@pytest.mark.integration
def test_incident_a_workflow_pin_is_unreachable() -> None:
    """The pin that wedged omnibase_infra CI for ~2.5h (OMN-15536).

    ``omnimarket@879d6fc6`` was the head of a PR branch GitHub deleted on
    squash-merge. The shape-only validators in this repo passed on it.
    """
    verdict, detail = _resolve("omnimarket", INCIDENT_A_DEAD_SHA)
    _require_determined(verdict, detail, "incident A dead pin")
    assert verdict is Verdict.UNREACHABLE, (
        f"expected UNREACHABLE for the OMN-15536 wedging pin, got {verdict} ({detail})"
    )


@pytest.mark.integration
def test_incident_a_correct_counterpart_is_reachable() -> None:
    verdict, detail = _resolve("omnimarket", INCIDENT_A_GOOD_SHA)
    _require_determined(verdict, detail, "incident A good pin")
    assert verdict is Verdict.REACHABLE, (
        f"expected REACHABLE for the merged dev squash, got {verdict} ({detail})"
    )


@pytest.mark.integration
def test_incident_b_dependency_pin_is_unreachable() -> None:
    """The dependency pin live on ``omnimarket@dev pyproject.toml``.

    ``omnibase_core@5a907b71`` is the head of an unlanded branch: ``ahead`` of
    dev, ``diverged`` from main. It resolves today and dies on merge. The
    pre-existing uv.lock reachability check (OMN-14449) calls this pin OK,
    because ``git branch -r --contains`` accepts any remote branch including
    the live feature branch.
    """
    verdict, detail = _resolve("omnibase_core", INCIDENT_B_DEAD_SHA)
    _require_determined(verdict, detail, "incident B dead pin")
    assert verdict is Verdict.UNREACHABLE, (
        f"expected UNREACHABLE for the unlanded-branch-head dependency pin, "
        f"got {verdict} ({detail})"
    )


@pytest.mark.integration
def test_incident_b_correct_counterpart_is_reachable() -> None:
    verdict, detail = _resolve("omnibase_core", INCIDENT_B_GOOD_SHA)
    _require_determined(verdict, detail, "incident B good pin")
    assert verdict is Verdict.REACHABLE, (
        f"expected REACHABLE for the merged dev squash, got {verdict} ({detail})"
    )


@pytest.mark.integration
def test_protected_branch_union_is_load_bearing() -> None:
    """A dev-only oracle would false-RED a correct pin in this very repo.

    ``onex_change_control@2dd26ade`` is ``diverged`` from OCC ``dev`` and
    ``behind`` OCC ``main``; this repo pins it in both pyproject.toml and
    uv.lock. Narrowing the oracle to ``dev`` alone turns a durable pin red,
    and a gate that cries wolf gets bypassed.
    """
    dev_only, dev_detail = (
        _Resolver(("dev",))
        .resolve("onex_change_control", "2dd26ade7caaa7131e532473ec9d8a207d0e77ab")
        .verdict,
        "dev-only oracle",
    )
    union, union_detail = _resolve(
        "onex_change_control", "2dd26ade7caaa7131e532473ec9d8a207d0e77ab"
    )
    _require_determined(union, union_detail, "OCC main-reachable pin")
    if dev_only is Verdict.UNDETERMINED:
        pytest.skip(f"{dev_detail}: undetermined")
    assert dev_only is Verdict.UNREACHABLE
    assert union is Verdict.REACHABLE


@pytest.mark.integration
def test_this_repo_has_no_undurable_pins() -> None:
    """Expected GREEN. A failure here means a pin in this repo is a time bomb."""
    pins = extract_pins(
        [WORKFLOWS_DIR, REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"]
    )
    assert pins, "no pins extracted — the extractor is broken, not the tree clean"

    resolver = _Resolver(("dev", "main"))
    bad: list[str] = []
    undetermined: list[str] = []
    for pin in sorted(pins):
        resolution = resolver.resolve(pin.repo, pin.ref)
        located = f"{pin.source}::{pin.locus} -> {pin.repo}@{pin.ref[:12]}"
        if resolution.verdict is Verdict.UNREACHABLE:
            bad.append(f"{located} ({resolution.detail})")
        elif resolution.verdict is Verdict.UNDETERMINED:
            undetermined.append(f"{located} ({resolution.detail})")

    assert not bad, "pins not reachable from dev/main:\n  " + "\n  ".join(bad)
    if undetermined:
        message = "unresolved pins:\n  " + "\n  ".join(undetermined)
        if _IN_CI:
            pytest.fail(f"{message}\n(an unresolvable pin is not a passing pin)")
        pytest.skip(message)

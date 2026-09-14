# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay: the dev lane vendored a stale omnimarket (OMN-18268).

THE INCIDENT, 2026-09-14. Three omnimarket PRs merged to ``dev`` that morning --
#2538 (07:53:57Z), #2537 (09:11:34Z), #2540 (10:09:57Z), all runtime code that
runs inside the ``.201`` effects container. At 10:2xZ the lane's effects
container was healthy, every readiness endpoint answered 200, the Runtime Rebuild
Trigger's last three runs on omnibase_infra were ``success``, and the lane was
vendoring omnimarket at ``65c237cd`` -- #2538. The two later merges were not on
it and nothing anywhere was red.

The reason is not a failure. It is an ABSENCE: omnimarket's own rebuild trigger
had been ``workflow_dispatch``-only since 2026-07-21 (omnimarket#1858,
OMN-14702), so an omnimarket merge published no rebuild command at all, and the
sibling revision was never chosen -- ``stage_workspace.sh`` resolves it from
``origin/dev`` at staging time (OMN-17135), so the lane runs whatever that
pointed at when some OTHER repository's merge last fired a rebuild.

WHY THE EXISTING GUARD CANNOT SEE IT, which is the false green being replaced.
``check_dev_lane_staleness.py`` reads ``org.opencontainers.image.revision``. On
this same container that label read ``732fd291`` -- an omnibase_infra commit,
correct and current. A sibling-only rebuild never moves it, so pointing that
guard at a sibling trigger returns ``converged`` on its first poll while proving
nothing about the merge that fired it.

BOTH ARTIFACTS ARE CAPTURES, NOT RECONSTRUCTIONS. The manifest is the byte output
of ``docker exec omninode-runtime-effects cat /app/build-provenance.json``; the
compare is GitHub's own verbatim answer for the pair. The replay drives the real
guard's real projections over both.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18268"
PROVENANCE = FIXTURES / "build-provenance-omninode-runtime-effects.json.captured"
COMPARE_BEHIND = FIXTURES / "compare-4bc24372-65c237cd.gh-api.json.captured"
COMPARE_AHEAD = FIXTURES / "compare-65c237cd-2f654f8d.gh-api.json.captured"

PROVENANCE_SHA256 = "95a935888fa953848a24e70d20747f46ba7c228f7cacaebb5723824a11324e26"
COMPARE_BEHIND_SHA256 = (
    "f271d06b6255823c2dd9964e1238f0d6a02164368c65883c754b81d899fde919"
)

PR_2537 = "4bc2437210939702fa0dc8529362c0a7c231049f"  # merged 09:11:34Z
PR_2538 = "65c237cdc914a6c7c1e15de308986ec65c24f309"  # merged 07:53:57Z
PR_2540 = "2f654f8d8673b4703b919fd801241e7091d646e6"  # merged 10:09:57Z

_GUARD = REPO_ROOT / "scripts" / "ci" / "check_lane_sibling_revision.py"


def _guard() -> object:
    spec = importlib.util.spec_from_file_location("_omn18268_guard", _GUARD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_omn18268_guard"] = module
    spec.loader.exec_module(module)
    return module


class TestTheCapturesAreTheCaptures:
    """R1: prove the bytes are what they claim before asserting a verdict on them."""

    def test_manifest_bytes_are_unmodified(self) -> None:
        assert hashlib.sha256(PROVENANCE.read_bytes()).hexdigest() == PROVENANCE_SHA256

    def test_compare_bytes_are_unmodified(self) -> None:
        assert (
            hashlib.sha256(COMPARE_BEHIND.read_bytes()).hexdigest()
            == COMPARE_BEHIND_SHA256
        )

    def test_the_capture_exhibits_the_incident_rather_than_describing_it(self) -> None:
        """The lane really did vendor #2538 while its infra label was current."""
        manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        assert (
            manifest["per_repo_vcs_provenance"]["siblings"]["omnimarket"]["vcs_ref"]
            == PR_2538
        )
        assert manifest["infra_vcs_ref"] == "732fd291a98afeac849a4904670b210096649529"
        assert manifest["build_source"] == "workspace"
        # Built at 08:56:47Z -- after #2538 (07:53Z), before #2537 (09:11Z).
        assert manifest["build_time"] == "2026-09-14T08:56:47Z"
        assert json.loads(COMPARE_BEHIND.read_text(encoding="utf-8"))["status"] == (
            "behind"
        )


class TestTheIncident:
    def test_the_real_guard_rejects_the_real_incident_state(self) -> None:
        """Driven through the guard's own projection, over the captured bytes."""
        guard = _guard()
        lane_revision = guard.parse_sibling_revision(
            json.loads(PROVENANCE.read_text(encoding="utf-8")), "omnimarket"
        )
        assert lane_revision == PR_2538
        containment = json.loads(COMPARE_BEHIND.read_text(encoding="utf-8"))["status"]
        verdict = guard.evaluate_sibling_convergence(
            repo="omnimarket",
            lane_revision=lane_revision,
            expected_revision=PR_2537,
            containment=containment,
            waited=timedelta(minutes=25),
            wait_timeout=timedelta(minutes=25),
        )
        assert not verdict.ok
        codes = [finding.code for finding in verdict.findings]
        assert codes == ["SIBLING_NOT_CONVERGED"]
        detail = verdict.findings[0].detail
        # The verdict must name the revision the lane HAS and the one it lacks;
        # reporting only one hides which side of the gap the operator is on.
        assert PR_2538[:12] in detail
        assert PR_2537[:12] in detail

    def test_the_label_the_old_guard_reads_would_have_said_converged(self) -> None:
        """The false green, stated as a fact of this capture rather than a claim.

        The infra revision on this container was current, so a guard reading only
        that label reports the lane converged for a merge it does not carry.
        """
        staleness = importlib.util.spec_from_file_location(
            "_omn18268_staleness",
            REPO_ROOT / "scripts" / "ci" / "check_dev_lane_staleness.py",
        )
        assert staleness is not None and staleness.loader is not None
        module = importlib.util.module_from_spec(staleness)
        sys.modules["_omn18268_staleness"] = module
        staleness.loader.exec_module(module)
        manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        assert module.revisions_match(
            manifest["infra_vcs_ref"], manifest["infra_vcs_ref"]
        )


class TestTheAcceptControl:
    def test_a_lane_carrying_a_descendant_of_the_merge_is_reported_clean(self) -> None:
        """A guard stuck at reject would replay this incident and be worthless.

        Containment, not equality: a rebuild that starts after a later merge
        legitimately vendors a DESCENDANT. GitHub's own answer for
        ``65c237cd...2f654f8d`` is ``ahead``, and that must pass.
        """
        guard = _guard()
        containment = json.loads(COMPARE_AHEAD.read_text(encoding="utf-8"))["status"]
        assert containment == "ahead"
        verdict = guard.evaluate_sibling_convergence(
            repo="omnimarket",
            lane_revision=PR_2540,
            expected_revision=PR_2538,
            containment=containment,
            waited=timedelta(minutes=2),
            wait_timeout=timedelta(minutes=25),
        )
        assert verdict.ok, [finding.render() for finding in verdict.findings]
        assert verdict.notes

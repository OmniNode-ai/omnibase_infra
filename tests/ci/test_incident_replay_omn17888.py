# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17888 incident replay — the dev lane ran 9 commits behind and nothing said so.

The incident, measured not recalled
-----------------------------------
On 2026-09-06 the OMN-17888 publisher fix (omnibase_infra#3234, merge d6d0ec50)
was live and WORKING. Five ``redeploy-start`` commands were published, delivered
and consumed — topic ``onex.cmd.omnimarket.redeploy-start.v1`` moved from
watermark 63 to 68, and the orchestrator consumer group sat Stable at offset 68
with LAG 0. The orchestrator republished ``prod-promotion-gate-evaluate.v1`` on
the same watermark move.

And the dev lane never moved. ``omninode-runtime-effects`` dead-lettered the
downstream ``omnimarket.prod-promotion-gate-evaluated`` event to
``onex.dlq.omnibase-infra.omnimarket.v1`` with
``failure_class=no_dispatcher``, and the boundary terminal was suppressed
(OMN-16812 / OMN-17432). Every downstream topic stayed at 0/0.

So the failure had moved one layer downstream of the one OMN-17888 originally
described, and every existing signal was green through it: the publisher's run
was green, the topic watermark advanced, the consumer group had no lag. The only
fact that was still true and still visible was what the lane was RUNNING —
``org.opencontainers.image.revision=2ea74bc4de76``, nine commits and eight and a
half hours behind ``dev`` head ``116b49143e``.

That is the regression class this guard exists for: false_green. Not a broken
check reporting pass — the ABSENCE of any check that reads the outcome rather
than the mechanism.

What the replay drives
----------------------
The REAL projections (:func:`parse_docker_inspect`, :func:`parse_compare`) over
the verbatim captured bytes of the incident, then the REAL verdict function. So
it proves the guard can read what docker and GitHub actually return, not only
the shapes a test author imagined.

The accept control (``compare-116b49143e-116b49143e``, GitHub's own answer for a
lane that IS on dev head) is what stops a guard that simply always reports stale
from passing this file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.ci.check_dev_lane_staleness import (
    DEFAULT_MAX_AGE,
    DEFAULT_MAX_COMMITS_BEHIND,
    DEV_LANE_COMPOSE_PROJECT,
    DEV_LANE_CONTAINER,
    assert_lane_fence,
    evaluate,
    normalize_revision,
    parse_compare,
    parse_docker_inspect,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn17888"

STALE_COMPARE = FIXTURES / "compare-2ea74bc4de76-116b49143e.gh-api.json.captured"
FRESH_COMPARE = FIXTURES / "compare-116b49143e-116b49143e.gh-api.json.captured"
LANE_INSPECT = FIXTURES / "docker-inspect-omninode-runtime.docker.json.captured"

# Pinned in tests/incident_replays/registry.yaml. Asserted here too so a silent
# edit to a captured artifact fails the replay rather than quietly changing what
# the incident was.
SHA256 = {
    STALE_COMPARE: "57fc91f6f0256cea60118147952fb38389bc701ca7e44e5005cc8aa4f57a9f51",
    FRESH_COMPARE: "787b08cec82ceb68fcd2a9c1e9fd4448771aeb1bfff65f9433123bfea77e9edb",
}

DEV_HEAD = "116b49143eacd65ed95c0eeb7bb9dc483458d466"
# The moment the readback was taken, 2026-09-06T18:49Z.
NOW = datetime(2026, 9, 6, 18, 49, tzinfo=UTC)


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheCapturesAreTheCaptures:
    @pytest.mark.parametrize("path", list(SHA256))
    def test_captured_bytes_match_the_registry_digest(self, path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == SHA256[path], (
            f"{path.name} no longer hashes to the digest recorded in "
            "tests/incident_replays/registry.yaml; a reformatted or edited artifact "
            "is no longer the artifact that failed"
        )

    def test_the_lane_capture_carries_no_container_environment(self) -> None:
        """Sanitization control: docker inspect's Config.Env holds live secrets.

        It was replaced with a single redaction marker at capture time. This test
        is the standing proof that it stays replaced — the guard reads only
        Config.Labels and State.Status, so nothing the replay needs was lost.
        """
        payload = _load(LANE_INSPECT)
        assert isinstance(payload, dict)
        env = payload["Config"]["Env"]
        assert env == [
            "<REDACTED: container environment removed at capture time; "
            "carries live credentials>"
        ]


class TestTheIncident:
    def test_the_real_docker_inspect_bytes_parse_to_the_stale_revision(self) -> None:
        """Proves the guard reads what docker actually returns."""
        lane = parse_docker_inspect(_load(LANE_INSPECT))
        assert lane.revision == "2ea74bc4de76"
        assert lane.compose_project == DEV_LANE_COMPOSE_PROJECT
        assert lane.build_source == "workspace"
        assert lane.state == "running"
        # And the fence passes on the real dev-lane container, so a fence that
        # rejected everything could not masquerade as safety.
        assert_lane_fence(lane, DEV_LANE_CONTAINER, DEV_LANE_COMPOSE_PROJECT)

    def test_the_real_compare_bytes_parse_to_nine_commits_behind(self) -> None:
        """Proves the guard reads what the GitHub compare API actually returns."""
        divergence = parse_compare(_load(STALE_COMPARE), DEV_HEAD)
        assert divergence.status == "ahead"
        assert divergence.commits_behind == 9
        assert divergence.base_sha.startswith("2ea74bc4de76")
        assert divergence.base_committed_at == datetime(
            2026, 9, 6, 10, 13, 54, tzinfo=UTC
        )

    def test_the_real_guard_rejects_the_real_incident_state(self) -> None:
        """The whole point. Green here means the guard is decorative."""
        lane = parse_docker_inspect(_load(LANE_INSPECT))
        divergence = parse_compare(_load(STALE_COMPARE), DEV_HEAD)
        verdict = evaluate(
            lane=lane,
            divergence=divergence,
            now=NOW,
            max_commits_behind=DEFAULT_MAX_COMMITS_BEHIND,
            max_age=DEFAULT_MAX_AGE,
        )
        assert not verdict.ok
        assert [f.code for f in verdict.findings] == ["LANE_STALE"]
        detail = verdict.findings[0].detail
        assert "9 commits behind (bound 3)" in detail
        assert "8h35m old (bound 2h00m)" in detail

    def test_the_deployed_label_is_a_usable_revision_not_a_sentinel(self) -> None:
        """A blank label would have made the incident unprovable rather than red."""
        lane = parse_docker_inspect(_load(LANE_INSPECT))
        assert normalize_revision(lane.revision) == "2ea74bc4de76"


class TestTheAcceptControl:
    """A guard that always reports stale would replay the incident and be useless."""

    def test_a_lane_on_dev_head_is_reported_clean(self) -> None:
        divergence = parse_compare(_load(FRESH_COMPARE), DEV_HEAD)
        assert divergence.status == "identical"
        assert divergence.commits_behind == 0
        lane = parse_docker_inspect(_load(LANE_INSPECT))
        verdict = evaluate(
            lane=lane,
            divergence=divergence,
            # The fresh capture's base commit is dev head itself; evaluated at a
            # point inside the age bound.
            now=divergence.base_committed_at + timedelta(minutes=10),
            max_commits_behind=DEFAULT_MAX_COMMITS_BEHIND,
            max_age=DEFAULT_MAX_AGE,
        )
        assert verdict.ok, verdict.findings

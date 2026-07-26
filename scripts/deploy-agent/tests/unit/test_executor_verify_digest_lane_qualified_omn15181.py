# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-first reproduction + fix coverage for the OMN-15181 verify-digest defect.

Live-reproduced 2026-07-26 on omninode-pc (ledger 10:25Z PREFLIGHT-STOP entry):
``verify_running_image_digest`` docker-inspects the bare literal container
names ``omninode-runtime`` / ``runtime-effects`` (the module-level
``RUNTIME_HEALTH_TARGETS`` constant) for EVERY lane, including prod. No lane
container actually carries that bare name — every non-dev lane container is
qualified via a ``container_name:`` override in its compose overlay (e.g.
``omninode-prod-runtime`` in ``docker-compose.prod.yml``,
``omninode-stability-test-runtime`` in ``docker-compose.stability-test.yml``).
``docker inspect omninode-runtime`` on the prod host therefore always returns
"no such object" and ``deploy_and_verify()`` raises ``DigestMismatchError``
unconditionally for every prod deploy, regardless of whether the actual
recreate+digest-pull succeeded.

The pre-existing ``test_running_digest_match_passes`` /
``test_running_digest_mismatch_fails_closed`` tests in
``test_executor_prod_digest.py`` mock ``_run`` without ever asserting the
docker-inspect argument, so they could not (and did not) catch this — a
mock-everything test that never inspects the command under test is a vacuous
green (see reference_test_the_artifact_that_runs).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import DeployExecutor, lane_config_for

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "c" * 64


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _capture_inspect_target(executor: DeployExecutor, lane: EnumRuntimeLane) -> str:
    """Run verify_running_image_digest and return the docker-inspect argument."""
    captured_cmds: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        captured_cmds.append(cmd)
        return _ok(stdout=_DIGEST + "\n")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        executor.verify_running_image_digest(lane=lane, expected_digest=_DIGEST)

    inspect_cmds = [cmd for cmd in captured_cmds if cmd[:2] == ["docker", "inspect"]]
    assert len(inspect_cmds) == 1, (
        f"expected exactly one docker inspect, got {inspect_cmds}"
    )
    return inspect_cmds[0][-1]


class TestVerifyDigestInspectsLaneQualifiedContainerNames:
    """RED-first: today's code inspects the bare literal name for every lane.

    This reproduces the EXACT live failure mode — verify_running_image_digest
    must dispatch its docker-inspect at the lane-qualified container name, the
    one actually created by that lane's compose overlay
    (``container_name:`` in docker-compose.{stability-test,prod}.yml), never
    the generic bare name that exists on no real lane.
    """

    def test_prod_lane_inspects_prod_qualified_container_name(self) -> None:
        executor = DeployExecutor()
        target = _capture_inspect_target(executor, EnumRuntimeLane.PROD)
        assert target == "omninode-prod-runtime", (
            "prod verify_running_image_digest must docker-inspect the "
            "lane-qualified container name declared as container_name: in "
            f"docker-compose.prod.yml, not a generic bare name; got {target!r}"
        )

    def test_stability_test_lane_inspects_stability_qualified_container_name(
        self,
    ) -> None:
        executor = DeployExecutor()
        target = _capture_inspect_target(executor, EnumRuntimeLane.STABILITY_TEST)
        assert target == "omninode-stability-test-runtime", (
            "stability-test verify_running_image_digest must docker-inspect "
            "the lane-qualified container name declared as container_name: in "
            f"docker-compose.stability-test.yml; got {target!r}"
        )

    def test_dev_lane_still_inspects_base_container_name(self) -> None:
        """Dev lane's bare name IS its real container_name — must be unaffected."""
        executor = DeployExecutor()
        target = _capture_inspect_target(executor, EnumRuntimeLane.DEV)
        assert target == "omninode-runtime"

    def test_lane_config_is_the_single_source_of_truth(self) -> None:
        """verify_running_image_digest must read the name from lane_config_for(),
        never a second hardcoded map — the exact same per-lane config the
        executor already uses to recreate containers (compose_project,
        compose_files, postgres_container)."""
        for lane, expected_name in (
            (EnumRuntimeLane.DEV, "omninode-runtime"),
            (EnumRuntimeLane.STABILITY_TEST, "omninode-stability-test-runtime"),
            (EnumRuntimeLane.PROD, "omninode-prod-runtime"),
        ):
            cfg = lane_config_for(lane)
            assert cfg.runtime_health_targets[0][0] == expected_name

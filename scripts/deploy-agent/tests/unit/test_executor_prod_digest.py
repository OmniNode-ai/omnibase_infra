# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prod digest pinning + fail-closed digest verification for the deploy executor (OMN-12572).

Prod deploys the exact image digest proven in stability-test. The executor must:

1. resolve and PULL the pinned ``image_digest`` rather than rebuilding from a ref,
2. verify the running container image digest equals the requested digest BEFORE
   any health check, failing closed on mismatch,
3. reject a prod request lacking a stability READY digest BEFORE invoking any
   deploy effect (boundary-level guard).

All docker/compose effects are mocked — no real deploy happens.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import (
    DeployExecutor,
    DigestMismatchError,
    ProdStabilityDigestMissingError,
    assert_prod_request_has_stability_digest,
)

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "c" * 64
_OTHER_DIGEST = "sha256:" + "d" * 64


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestProdDeployPullsDigestNotRebuild:
    def test_prod_rebuild_scope_pulls_digest_and_skips_build(self) -> None:
        """Prod must pull the pinned digest and must NOT call _compose_build."""
        executor = DeployExecutor()
        build_calls: list[object] = []
        pull_calls: list[str] = []
        up_calls: list[object] = []

        def fake_build(*args: object, **kwargs: object) -> None:
            build_calls.append((args, kwargs))

        def fake_pull(digest: str, lane: EnumRuntimeLane) -> None:
            pull_calls.append(digest)

        def fake_up(*args: object, **kwargs: object) -> None:
            up_calls.append((args, kwargs))

        executor._compose_build = fake_build  # type: ignore[method-assign]
        executor._pull_pinned_image = fake_pull  # type: ignore[method-assign]
        executor._compose_up = fake_up  # type: ignore[method-assign]

        executor.rebuild_scope(
            Scope.RUNTIME,
            [],
            _noop_phase_update,
            git_sha="",
            lane=EnumRuntimeLane.PROD,
            image_digest=_DIGEST,
        )

        assert pull_calls == [_DIGEST], "prod must pull the pinned digest"
        assert build_calls == [], "prod must NOT rebuild from a ref"
        assert up_calls, "prod must still bring the lane up"

    def test_dev_rebuild_scope_still_builds(self) -> None:
        """Dev lane keeps building from a ref (no digest pin)."""
        executor = DeployExecutor()
        build_calls: list[object] = []
        pull_calls: list[str] = []

        def fake_build(*args: object, **kwargs: object) -> None:
            build_calls.append((args, kwargs))

        def fake_pull(digest: str, lane: EnumRuntimeLane) -> None:
            pull_calls.append(digest)

        executor._compose_build = fake_build  # type: ignore[method-assign]
        executor._pull_pinned_image = fake_pull  # type: ignore[method-assign]
        executor._compose_up = lambda *a, **k: None  # type: ignore[method-assign]

        executor.rebuild_scope(
            Scope.RUNTIME,
            [],
            _noop_phase_update,
            git_sha="abc",
            lane=EnumRuntimeLane.DEV,
        )

        assert build_calls, "dev must build from a ref"
        assert pull_calls == [], "dev must not pull a pinned digest"


class TestRunningDigestVerification:
    def test_running_digest_match_passes(self) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            # docker inspect of the running runtime container returns its image digest
            return _ok(stdout=_DIGEST + "\n")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            # must not raise
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DIGEST
            )

    def test_running_digest_mismatch_fails_closed(self) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            return _ok(stdout=_OTHER_DIGEST + "\n")

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            pytest.raises(DigestMismatchError, match="digest"),
        ):
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DIGEST
            )

    def test_digest_verification_runs_before_health_check(self) -> None:
        """On a digest mismatch, no health check (verify) may proceed."""
        executor = DeployExecutor()
        order: list[str] = []

        def fake_verify_digest(*, lane: EnumRuntimeLane, expected_digest: str) -> None:
            order.append("verify_digest")
            raise DigestMismatchError(
                f"running digest != requested digest ({expected_digest})"
            )

        def fake_verify(
            *, on_phase_update: object, lane: EnumRuntimeLane = EnumRuntimeLane.DEV
        ) -> list[object]:
            order.append("verify_health")
            return []

        executor.verify_running_image_digest = fake_verify_digest  # type: ignore[method-assign]
        executor.verify = fake_verify  # type: ignore[method-assign]

        with pytest.raises(DigestMismatchError):
            executor.deploy_and_verify(
                lane=EnumRuntimeLane.PROD,
                expected_digest=_DIGEST,
                on_phase_update=_noop_phase_update,
            )

        assert order == ["verify_digest"], (
            f"health verify must not run after a digest mismatch; got order {order}"
        )


class TestProdStabilityDigestGuard:
    def test_prod_request_without_stability_digest_is_rejected_before_deploy(
        self,
    ) -> None:
        cmd = ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="ci",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.PROD,
            image_digest=_DIGEST,
        )
        with pytest.raises(ProdStabilityDigestMissingError, match="stability"):
            assert_prod_request_has_stability_digest(cmd, stability_ready_digest=None)

    def test_prod_request_with_mismatched_stability_digest_is_rejected(self) -> None:
        cmd = ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="ci",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.PROD,
            image_digest=_DIGEST,
        )
        with pytest.raises(ProdStabilityDigestMissingError, match="digest"):
            assert_prod_request_has_stability_digest(
                cmd, stability_ready_digest=_OTHER_DIGEST
            )

    def test_prod_request_matching_stability_digest_passes(self) -> None:
        cmd = ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="ci",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.PROD,
            image_digest=_DIGEST,
        )
        # must not raise
        assert_prod_request_has_stability_digest(cmd, stability_ready_digest=_DIGEST)

    def test_non_prod_request_does_not_require_stability_digest(self) -> None:
        for lane in (EnumRuntimeLane.DEV, EnumRuntimeLane.STABILITY_TEST):
            cmd = ModelRebuildRequested(
                correlation_id=uuid4(),
                requested_by="ci",
                scope=Scope.RUNTIME,
                runtime_lane=lane,
            )
            # must not raise even though no stability digest is supplied
            assert_prod_request_has_stability_digest(cmd, stability_ready_digest=None)

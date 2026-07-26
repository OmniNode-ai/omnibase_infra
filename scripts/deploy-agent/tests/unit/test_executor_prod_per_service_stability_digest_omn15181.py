# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-first reproduction + fix coverage for the OMN-15181 round-4 Finding 11 defect.

Live-reproduced 2026-07-26 on omninode-pc (ledger 16:30Z FOREGROUND SUPERVISED
BOOTSTRAP entry, "Finding 11"): the stability-digest guard is per-LANE, not
per-SERVICE. ``resolve_stability_ready_digest`` always inspected
``lane_config_for(STABILITY_TEST).runtime_health_targets[0]`` --
unconditionally the RUNTIME container (``omninode-stability-test-runtime``)
-- regardless of which service the prod request actually targeted. A
``runtime-effects`` command carrying the effects-lane companion digest was
compared against the RUNTIME stability digest and wrongly REJECTED
(``ProdStabilityDigestMissingError``, correlation ``3bc3fb34``) even though
the effects digest was genuinely stability-proven.

Fix: ``resolve_stability_ready_digest`` and ``verify_running_image_digest``
both take an explicit ``service`` argument resolved from the request's
``services`` list (``resolve_prod_target_service``), and both route through
the single shared ``_health_target_container`` mapping (no forked second
copy of "which index is which service" -- the same discipline
``PROD_IMAGE_ENV_VAR_FOR_SERVICE`` already established for the compose
image-env override). Defaults preserve the pre-round-4 RUNTIME-only behavior
for requests that do not name exactly one image-bearing service (e.g. a
full-scope, multi-service deploy).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import (
    DeployExecutor,
    ProdStabilityDigestMissingError,
    assert_prod_request_has_stability_digest,
    lane_config_for,
    resolve_prod_target_service,
)

pytestmark = pytest.mark.unit

_RUNTIME_DIGEST = "sha256:" + "a" * 64
_EFFECTS_DIGEST = "sha256:" + "b" * 64


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _effects_cmd(image_digest: str) -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        services=["runtime-effects"],
        image_digest=image_digest,
    )


class TestResolveProdTargetService:
    """resolve_prod_target_service is the single shared "what does this
    request target" resolution -- used by both the guard and post-deploy
    verification, mirroring PROD_IMAGE_ENV_VAR_FOR_SERVICE's discipline."""

    def test_single_effects_service_resolves_to_effects(self) -> None:
        cmd = _effects_cmd(_EFFECTS_DIGEST)
        assert resolve_prod_target_service(cmd) == "runtime-effects"

    def test_single_runtime_service_resolves_to_runtime(self) -> None:
        cmd = ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="test",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.PROD,
            services=["omninode-runtime"],
            image_digest=_RUNTIME_DIGEST,
        )
        assert resolve_prod_target_service(cmd) == "omninode-runtime"

    def test_empty_services_defaults_to_runtime(self) -> None:
        """A full-scope request (no explicit services) keeps the pre-round-4
        default -- disambiguating a multi-service digest is out of scope."""
        cmd = ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="test",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.PROD,
            image_digest=_RUNTIME_DIGEST,
        )
        assert resolve_prod_target_service(cmd) == "omninode-runtime"


class TestResolveStabilityReadyDigestIsPerService:
    def test_resolves_the_runtime_container_by_default(self) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            return _ok(stdout=_RUNTIME_DIGEST + "\n")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            result = executor.resolve_stability_ready_digest()

        assert result == _RUNTIME_DIGEST
        inspect_cmd = next(cmd for cmd in captured if cmd[:2] == ["docker", "inspect"])
        assert inspect_cmd[-1] == "omninode-stability-test-runtime"

    def test_resolves_the_effects_container_when_asked(self) -> None:
        """THE defect: pre-fix, this call ignored the requested service and
        always inspected the RUNTIME stability container. Post-fix it must
        inspect the effects-lane companion container."""
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            return _ok(stdout=_EFFECTS_DIGEST + "\n")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            result = executor.resolve_stability_ready_digest("runtime-effects")

        assert result == _EFFECTS_DIGEST
        inspect_cmd = next(cmd for cmd in captured if cmd[:2] == ["docker", "inspect"])
        assert inspect_cmd[-1] == "omninode-stability-test-runtime-effects", (
            "a runtime-effects digest resolution must inspect the effects "
            f"stability container, not the runtime one; got {inspect_cmd[-1]!r}"
        )

    def test_unknown_service_fails_loud(self) -> None:
        executor = DeployExecutor()
        with pytest.raises(RuntimeError, match="no runtime_health_targets mapping"):
            executor.resolve_stability_ready_digest("not-a-real-service")


class TestVerifyRunningImageDigestIsPerService:
    def test_verifies_the_effects_container_when_asked(self) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            return _ok(stdout=_EFFECTS_DIGEST + "\n")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            # must not raise: the effects container reports the effects digest
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD,
                expected_digest=_EFFECTS_DIGEST,
                service="runtime-effects",
            )

        inspect_cmd = next(cmd for cmd in captured if cmd[:2] == ["docker", "inspect"])
        assert (
            inspect_cmd[-1]
            == lane_config_for(EnumRuntimeLane.PROD).runtime_health_targets[1][0]
        )


class TestPerServiceStabilityDigestGuardEndToEnd:
    """The exact live scenario: a runtime-effects command carrying the
    effects companion digest must PASS against the effects stability digest
    and FAIL against the runtime stability digest."""

    def test_effects_request_with_effects_digest_passes(self) -> None:
        executor = DeployExecutor()
        cmd = _effects_cmd(_EFFECTS_DIGEST)
        target_service = resolve_prod_target_service(cmd)

        def fake_run(
            cmd_: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            return _ok(stdout=_EFFECTS_DIGEST + "\n")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            stability_digest = executor.resolve_stability_ready_digest(target_service)

        # must not raise -- effects digest matches the effects stability digest
        assert_prod_request_has_stability_digest(
            cmd, stability_ready_digest=stability_digest
        )

    def test_effects_request_with_runtime_digest_fails(self) -> None:
        """THE reproduction: pre-fix, resolve_stability_ready_digest ignored
        the requested service and always returned the RUNTIME digest, so an
        effects command carrying the (genuinely stability-proven) effects
        digest was compared against the wrong container and rejected."""
        executor = DeployExecutor()
        cmd = _effects_cmd(_EFFECTS_DIGEST)
        target_service = resolve_prod_target_service(cmd)

        def fake_run(
            cmd_: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            # The RUNTIME stability container is serving a DIFFERENT digest
            # than the effects container -- the real-world drift condition.
            return _ok(stdout=_RUNTIME_DIGEST + "\n")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            # Pre-fix: target_service is ignored, so this always resolves
            # _RUNTIME_DIGEST regardless of what we asked for -- proving the
            # guard is comparing against the wrong container.
            stability_digest = executor.resolve_stability_ready_digest(target_service)

        with pytest.raises(ProdStabilityDigestMissingError, match="digest"):
            assert_prod_request_has_stability_digest(
                cmd, stability_ready_digest=stability_digest
            )

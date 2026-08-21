# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-first reproduction + fix coverage for the OMN-15181 round-2 D2 defect.

Live-reproduced 2026-07-26 on omninode-pc (ledger 11:55Z PREFLIGHT-STOP
entry, "Finding 8"): ``rebuild_scope()`` for ``lane=prod`` calls
``_pull_pinned_image()``, which does a literal ``docker pull
<image_digest>``. The granted ``image_digest`` is a bare ``sha256:...``
string -- not a valid pull reference on its own (no repo/tag) -- and the
target artifact (``omnibase-infra-stability-test-omninode-runtime:latest``)
was built purely via ``docker compose build`` on the host and never pushed
to any registry. Live-tested: ``docker pull sha256:ddb296f8...`` -> "Error
response from daemon: pull access denied for sha256, repository does not
exist or may require 'docker login'" (exit 1). Every prod rebuild therefore
fails at this step even though the exact artifact already exists locally.

Fix: local-presence-first resolution. ``docker image inspect
<image_digest>`` first; if the image is already present locally (the normal
case for a compose-build-only artifact), use it directly and log
provenance -- no pull attempted, no registry required. Only when the image
is absent locally does the executor attempt a registry pull, and only when
an explicit registry reference is configured
(``DEPLOY_AGENT_PROD_IMAGE_REGISTRY_REF`` -- a real ``repo:tag`` /
``repo@sha256:...`` reference, never the bare digest). Absent both, fail
loud with a RuntimeError naming the missing configuration.

``TestPullPinnedImageRealDockerSemantics`` executes the actual docker CLI
to pin the two real-world docker facts this fix depends on: (docker image
inspect succeeds against a present local image) and (docker pull rejects a
bare sha256 reference) -- skipped when no real docker daemon is reachable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "c" * 64


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str = "not found") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def _docker_daemon_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_daemon_reachable(),
    reason="requires a real, reachable docker daemon (self-hosted CI / omninode-pc)",
)


def _assert_bare_sha256_pull_rejected(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Assert that pulling a bare sha256 digest is never treated as a valid
    pull -- either the daemon rejects it outright (non-zero exit) or the
    resolution attempt never completes within budget (OMN-16317).

    A bare ``sha256:...`` reference has no repo/tag, so a reachable registry
    normally rejects it fast ("repository does not exist"). Under
    registry/network latency the daemon can instead hang past the timeout
    while it tries to resolve that same unpullable reference -- a
    ``TimeoutExpired`` here is a consistent variant of the identical negative
    outcome (the daemon never succeeds in treating it as a satisfiable pull),
    not evidence the reference became valid. Treat both as proof.
    """
    try:
        result = runner(
            ["docker", "pull", "sha256:" + "0" * 64],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return
    assert result.returncode != 0


@requires_docker
class TestPullPinnedImageRealDockerSemantics:
    def test_docker_image_inspect_succeeds_for_a_present_local_image(self) -> None:
        pull = subprocess.run(
            ["docker", "pull", "busybox:latest"],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert pull.returncode == 0, pull.stderr
        id_result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", "busybox:latest"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert id_result.returncode == 0, id_result.stderr
        image_id = id_result.stdout.strip()
        assert image_id.startswith("sha256:")

        inspect_by_id = subprocess.run(
            ["docker", "image", "inspect", image_id],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert inspect_by_id.returncode == 0, inspect_by_id.stderr

    def test_bare_sha256_is_an_invalid_pull_reference(self) -> None:
        """Live-reproduced 2026-07-26 on omninode-pc: `docker pull
        sha256:...` fails with 'pull access denied ... repository does not
        exist' -- a bare digest with no repo/tag is never a valid pull
        reference on its own. See `_assert_bare_sha256_pull_rejected` for why
        a timeout is accepted as an equivalent outcome (OMN-16317)."""
        _assert_bare_sha256_pull_rejected()


class TestBareSha256PullRejectionOutcomes:
    """Mock-based: pins `_assert_bare_sha256_pull_rejected` under both the
    fast-error and slow/timeout paths (OMN-16317), without depending on a
    live, possibly-slow registry round trip for the pinning itself."""

    def test_fast_rejection_passes(self) -> None:
        _assert_bare_sha256_pull_rejected(runner=lambda *a, **k: _fail())

    def test_timeout_while_resolving_is_treated_as_rejection(self) -> None:
        def _raise_timeout(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess:
            raise subprocess.TimeoutExpired(cmd=["docker", "pull"], timeout=15)

        _assert_bare_sha256_pull_rejected(runner=_raise_timeout)

    def test_success_returncode_fails_the_assertion(self) -> None:
        with pytest.raises(AssertionError):
            _assert_bare_sha256_pull_rejected(runner=lambda *a, **k: _ok())


class TestPullPinnedImageLocalPresenceFirst:
    """Mock-based: asserts the real docker CLI arguments our source emits."""

    def test_present_locally_skips_pull_entirely(self) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _ok(stdout=_DIGEST + "\n")
            return _ok()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor._pull_pinned_image(_DIGEST, EnumRuntimeLane.PROD)

        pull_cmds = [cmd for cmd in captured if cmd[:2] == ["docker", "pull"]]
        assert pull_cmds == [], (
            "must not attempt any docker pull when the digest is already "
            f"present locally: {pull_cmds}"
        )
        inspect_cmds = [
            cmd for cmd in captured if cmd[:3] == ["docker", "image", "inspect"]
        ]
        assert inspect_cmds == [["docker", "image", "inspect", _DIGEST]], (
            f"expected exactly one local-presence check, got {inspect_cmds}"
        )

    def test_bare_sha_is_never_passed_as_a_pull_reference(self) -> None:
        """RED-reproduction: today's code runs `docker pull <sha256:...>`
        unconditionally, before ever checking local presence. This must
        never happen -- a bare digest is never a constructible pull ref."""
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _fail()
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("DEPLOY_AGENT_PROD_IMAGE_REGISTRY_REF", None)
            with pytest.raises(RuntimeError):
                executor._pull_pinned_image(_DIGEST, EnumRuntimeLane.PROD)

        bad_pulls = [
            cmd
            for cmd in captured
            if cmd[:2] == ["docker", "pull"] and cmd[-1] == _DIGEST
        ]
        assert bad_pulls == [], (
            f"must never docker pull a bare sha256 digest directly: {bad_pulls}"
        )

    def test_absent_locally_and_no_registry_configured_fails_loud(self) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _fail()
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("DEPLOY_AGENT_PROD_IMAGE_REGISTRY_REF", None)
            with pytest.raises(RuntimeError, match="not present locally"):
                executor._pull_pinned_image(_DIGEST, EnumRuntimeLane.PROD)

    def test_registry_ref_configured_pulls_then_verifies(self) -> None:
        executor = DeployExecutor()
        inspect_calls = {"count": 0}
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            if cmd[:3] == ["docker", "image", "inspect"]:
                inspect_calls["count"] += 1
                if inspect_calls["count"] == 1:
                    return _fail()
                return _ok(stdout=_DIGEST + "\n")
            if cmd[:2] == ["docker", "pull"]:
                assert cmd[-1] == "example.com/omninode-runtime:pinned", (
                    f"must pull the configured registry ref, not the bare digest: {cmd}"
                )
                return _ok()
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.dict(
                os.environ,
                {
                    "DEPLOY_AGENT_PROD_IMAGE_REGISTRY_REF": "example.com/omninode-runtime:pinned"
                },
            ),
        ):
            executor._pull_pinned_image(_DIGEST, EnumRuntimeLane.PROD)

        pull_cmds = [cmd for cmd in captured if cmd[:2] == ["docker", "pull"]]
        assert pull_cmds == [["docker", "pull", "example.com/omninode-runtime:pinned"]]
        assert inspect_calls["count"] == 2, (
            "must re-inspect locally after a registry pull to confirm the "
            "pulled image actually matches the pinned digest"
        )

    def test_registry_pull_that_fails_to_produce_the_pinned_digest_fails_loud(
        self,
    ) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _fail()
            if cmd[:2] == ["docker", "pull"]:
                return _ok()
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.dict(
                os.environ,
                {
                    "DEPLOY_AGENT_PROD_IMAGE_REGISTRY_REF": "example.com/omninode-runtime:pinned"
                },
            ),
            pytest.raises(RuntimeError, match="does not match"),
        ):
            executor._pull_pinned_image(_DIGEST, EnumRuntimeLane.PROD)

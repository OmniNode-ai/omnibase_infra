# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-first reproduction + fix coverage for the OMN-15181 round-2 D1 defect.

Live-reproduced 2026-07-26 on omninode-pc (ledger 11:55Z PREFLIGHT-STOP
entry, "Finding 7"): ``resolve_stability_ready_digest`` (new in round 1) and
``verify_running_image_digest`` (pre-existing) both ran::

    docker inspect --format {{index .RepoDigests 0}} <container_name>

against a CONTAINER object. ``.RepoDigests`` exists only on an IMAGE-inspect
object. Live-tested verbatim against the real
``omninode-stability-test-runtime`` container on omninode-pc: both exact
format strings -> ``docker inspect`` exit 1, "template parsing error ... map
has no entry for key RepoDigests". Net effect: the new
``assert_prod_request_has_stability_digest`` boundary guard (wired as the
FIRST check in ``agent.py._run_deploy`` for any ``runtime_lane=prod``
command) would unconditionally raise before any deploy effect ran.

Fix: both call sites resolve the running container's IMAGE ID via the
CONTAINER-inspect field ``{{.Image}}`` (never ``.RepoDigests``, which is
empty anyway for locally-built images that were never pushed to a registry)
and compare that image ID directly against the expected/granted digest. Both
call sites share one private helper (``_container_image_id``) -- no forked
second implementation.

Class ``TestRepoDigestsOnContainerIsRealDockerFailureMode`` executes the
actual docker CLI against a real, disposable container to pin the underlying
docker semantic (skipped when no real docker daemon is reachable, or when
that daemon cannot produce the base container within budget -- runs on
self-hosted CI / omninode-pc, per feedback_test_the_artifact_that_runs). The
mock-based classes below assert the real command/format-string shape emitted
by our source, not just that some digest was returned.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import DeployExecutor, lane_config_for

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "c" * 64


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


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


_PULL_TIMEOUT_SECONDS = 90
_CREATE_TIMEOUT_SECONDS = 90
_REMOVE_TIMEOUT_SECONDS = 60

# Substrings that identify a `docker create` refusal caused by the base image
# never becoming available locally (registry unreachable / pull timed out),
# as opposed to a genuine docker semantic this fixture should surface.
_IMAGE_UNAVAILABLE_MARKERS = (
    "no such image",
    "not found",
    "manifest unknown",
    "pull access denied",
    "error response from daemon: pull",
)


def _docker(args: list[str], timeout: int) -> subprocess.CompletedProcess[str] | None:
    """Run a docker command, returning ``None`` if it exceeds ``timeout``.

    Every docker call in this fixture already passes ``check=False`` -- a
    non-zero exit is the fixture's business to interpret, not an error. But
    ``subprocess.run`` raises ``TimeoutExpired`` regardless of ``check``, so
    the one failure mode the fixture could not absorb was the one that
    actually fires: docker-daemon and registry contention on the shared
    self-hosted fleet (OMN-15749). Collapsing a timeout to ``None`` puts it
    back under the fixture's control, same reasoning as OMN-16317.
    """
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None


@pytest.fixture
def real_container():
    """A real, disposable container -- exercises actual docker inspect semantics."""
    name = f"omn15181-digest-fixture-{uuid.uuid4().hex[:12]}"
    # Best-effort warm-up: a timeout here is not fatal, the image is often
    # already cached on the runner and `docker create` is the real gate.
    _docker(["docker", "pull", "busybox:latest"], _PULL_TIMEOUT_SECONDS)

    create = _docker(
        ["docker", "create", "--name", name, "busybox:latest"],
        _CREATE_TIMEOUT_SECONDS,
    )
    if create is None:
        pytest.skip(
            "docker create exceeded "
            f"{_CREATE_TIMEOUT_SECONDS}s -- docker daemon contention, not a "
            "statement about docker inspect semantics (OMN-15749)"
        )
    if create.returncode != 0:
        stderr = create.stderr.lower()
        if any(marker in stderr for marker in _IMAGE_UNAVAILABLE_MARKERS):
            pytest.skip(
                "busybox:latest never became available locally: "
                f"{create.stderr.strip()} (OMN-15749)"
            )
        # Any other non-zero exit is a real docker semantic -- surface it.
        raise AssertionError(create.stderr)
    try:
        yield name
    finally:
        # The name is uuid-unique, so a container leaked by a teardown
        # timeout cannot collide with a later run.
        _docker(["docker", "rm", "-f", name], _REMOVE_TIMEOUT_SECONDS)


@requires_docker
class TestRepoDigestsOnContainerIsRealDockerFailureMode:
    """Pins the exact live failure mode this defect round fixes.

    ``.RepoDigests`` exists only on IMAGE-inspect objects; running the
    pre-fix format string against a real CONTAINER always fails with "map
    has no entry for key RepoDigests" (live-reproduced on omninode-pc,
    2026-07-26). This guards against ever re-introducing a container-target
    RepoDigests format string on the theory that some docker version might
    behave differently.
    """

    def test_repo_digests_format_errors_on_a_real_container(
        self, real_container: str
    ) -> None:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{index .RepoDigests 0}}",
                real_container,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode != 0, (
            "RepoDigests must fail on a CONTAINER-inspect target; if this "
            "starts passing, docker semantics changed and the round-2 fix "
            "rationale must be re-examined"
        )
        combined = result.stdout + result.stderr
        assert "RepoDigests" in combined

    def test_image_field_succeeds_on_the_same_real_container(
        self, real_container: str
    ) -> None:
        """The fixed format -- ``{{.Image}}`` -- IS a container-inspect field
        and succeeds where RepoDigests fails, returning the image id."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Image}}", real_container],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith("sha256:")


class TestDigestVerificationUsesContainerImageIdNotRepoDigests:
    """Mock-based: asserts the real docker-inspect format string our source
    emits, not just that a digest value came back (a mock that never
    inspects the format-string argument is a vacuous green -- this exact gap
    let the RepoDigests-on-container defect ship in round 1)."""

    def test_verify_running_image_digest_uses_image_field_not_repodigests(
        self,
    ) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            return _ok(stdout=_DIGEST + "\n")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DIGEST
            )

        inspect_cmd = next(cmd for cmd in captured if cmd[:2] == ["docker", "inspect"])
        fmt = inspect_cmd[inspect_cmd.index("--format") + 1]
        assert "RepoDigests" not in fmt, (
            "must never format against .RepoDigests -- it exists only on "
            f"IMAGE-inspect objects, not on this CONTAINER-inspect call: {fmt!r}"
        )
        assert fmt == "{{.Image}}", (
            f"expected the container-inspect image-id field {{.Image}}, got {fmt!r}"
        )

    def test_resolve_stability_ready_digest_uses_image_field_not_repodigests(
        self,
    ) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            return _ok(stdout=_DIGEST + "\n")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            result = executor.resolve_stability_ready_digest()

        assert result == _DIGEST
        inspect_cmd = next(cmd for cmd in captured if cmd[:2] == ["docker", "inspect"])
        fmt = inspect_cmd[inspect_cmd.index("--format") + 1]
        assert "RepoDigests" not in fmt, (
            f"must never format against .RepoDigests on a container inspect: {fmt!r}"
        )
        assert fmt == "{{.Image}}"

    def test_both_call_sites_share_one_container_image_id_helper(self) -> None:
        """No forked second implementation: both public methods must route
        through the same private helper for resolving a container's image id."""
        import deploy_agent.executor as executor_module

        calls: list[str] = []

        def fake_helper(container_name: str) -> str:
            calls.append(container_name)
            return _DIGEST

        with patch.object(
            executor_module, "_container_image_id", side_effect=fake_helper
        ):
            executor = DeployExecutor()
            resolved = executor.resolve_stability_ready_digest()
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DIGEST
            )

        assert resolved == _DIGEST
        assert calls == [
            lane_config_for(EnumRuntimeLane.STABILITY_TEST).runtime_health_targets[0][
                0
            ],
            lane_config_for(EnumRuntimeLane.PROD).runtime_health_targets[0][0],
        ]

    def test_verify_mismatch_still_fails_closed_via_image_id(self) -> None:
        executor = DeployExecutor()
        other_digest = "sha256:" + "d" * 64

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            return _ok(stdout=other_digest + "\n")

        from deploy_agent.executor import DigestMismatchError

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            pytest.raises(DigestMismatchError, match="does not match"),
        ):
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DIGEST
            )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A derived runtime image is accepted by the deploy agent's digest checks (OMN-22144).

One-runtime plan T18 ports a private service into the runtime as a private node
package that rides in a *derived* image: the public runtime image plus one wheel,
built by the cloud owner's pipeline. T18a asked whether the deploy agent and the
prod gate accept such an image by digest, or whether a mounted wheel (outside the
digest) would be the only way to carry private code.

The digest checks compare one string: the running container's image id
(``docker inspect --format {{.Image}}``) against the requested ``image_digest``.
They read no parent, layer or label field, so a derived image is an image like
any other: it passes when its own id is the one requested and the one running,
and it is refused when the base image id is requested while the derived image
runs (or the reverse). The prod boundary guard applies the same equality between
the requested digest and the digest proven READY before prod, so a derived
image reaches prod only after the same derived digest was proven there.

All docker effects are mocked; no real deploy happens.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import (
    DeployExecutor,
    DigestMismatchError,
    ProdStabilityDigestMissingError,
    assert_prod_request_has_stability_digest,
)

pytestmark = pytest.mark.unit

# Two distinct image ids: the public runtime image, and the image derived from it
# by adding one wheel layer. _FakeDockerDaemon records that lineage (derived's
# parent is base) so the tests can show the check ignores it: only the
# container's ``.Image`` id is ever compared.
_BASE_IMAGE_ID = "sha256:" + "a" * 64
_DERIVED_IMAGE_ID = "sha256:" + "b" * 64
_IMAGE_PARENT = {_DERIVED_IMAGE_ID: _BASE_IMAGE_ID, _BASE_IMAGE_ID: ""}


class _FakeDockerDaemon:
    """A minimal ``docker inspect`` that knows the base -> derived lineage.

    Answers ``--format {{.Image}}`` with the running container's image id and
    ``{{.Parent}}`` with the lineage. Every command is recorded so a test can
    assert the lineage was never asked for.
    """

    def __init__(self, running_image_id: str) -> None:
        self.running_image_id = running_image_id
        self.commands: list[list[str]] = []

    def __call__(
        self, cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(cmd)
        fmt = cmd[cmd.index("--format") + 1] if "--format" in cmd else ""
        if fmt == "{{.Image}}":
            return _ok(self.running_image_id + "\n")
        if fmt == "{{.Parent}}":
            return _ok(_IMAGE_PARENT[self.running_image_id] + "\n")
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr=f"unexpected format {fmt!r}"
        )

    @property
    def formats_queried(self) -> set[str]:
        return {c[c.index("--format") + 1] for c in self.commands if "--format" in c}


def _ok(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _running_image_id(
    image_id: str, captured: list[list[str]] | None = None
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if captured is not None:
            captured.append(cmd)
        return _ok(image_id + "\n")

    return fake_run


def _prod_cmd(image_digest: str) -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="ci",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        image_digest=image_digest,
    )


class TestRunningImageDigestCheckOnADerivedImage:
    def test_derived_image_requested_and_running_is_accepted(self) -> None:
        executor = DeployExecutor()
        with patch(
            "deploy_agent.executor._run",
            side_effect=_running_image_id(_DERIVED_IMAGE_ID),
        ):
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DERIVED_IMAGE_ID
            )

    def test_base_image_requested_while_derived_image_runs_is_refused(self) -> None:
        executor = DeployExecutor()
        with (
            patch(
                "deploy_agent.executor._run",
                side_effect=_running_image_id(_DERIVED_IMAGE_ID),
            ),
            pytest.raises(DigestMismatchError, match="does not match"),
        ):
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_BASE_IMAGE_ID
            )

    def test_derived_image_requested_while_base_image_runs_is_refused(self) -> None:
        executor = DeployExecutor()
        with (
            patch(
                "deploy_agent.executor._run",
                side_effect=_running_image_id(_BASE_IMAGE_ID),
            ),
            pytest.raises(DigestMismatchError, match="does not match"),
        ):
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DERIVED_IMAGE_ID
            )

    def test_check_reads_only_the_image_id_never_a_parent_or_layer_field(self) -> None:
        """Why the check is provenance-blind: it inspects one field, ``.Image``."""
        executor = DeployExecutor()
        captured: list[list[str]] = []
        with patch(
            "deploy_agent.executor._run",
            side_effect=_running_image_id(_DERIVED_IMAGE_ID, captured),
        ):
            executor.verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DERIVED_IMAGE_ID
            )
        inspect_cmd = next(cmd for cmd in captured if cmd[:2] == ["docker", "inspect"])
        assert inspect_cmd[inspect_cmd.index("--format") + 1] == "{{.Image}}"


class TestRunningImageDigestCheckIgnoresImageLineage:
    """The check is by-id equality; a derived image's ancestry earns it nothing."""

    def test_base_id_is_refused_although_the_running_image_descends_from_it(
        self,
    ) -> None:
        daemon = _FakeDockerDaemon(_DERIVED_IMAGE_ID)
        assert _IMAGE_PARENT[_DERIVED_IMAGE_ID] == _BASE_IMAGE_ID  # lineage is real
        with (
            patch("deploy_agent.executor._run", side_effect=daemon),
            pytest.raises(DigestMismatchError, match="does not match"),
        ):
            DeployExecutor().verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_BASE_IMAGE_ID
            )
        assert daemon.formats_queried == {"{{.Image}}"}

    def test_derived_id_is_accepted_without_consulting_the_base(self) -> None:
        daemon = _FakeDockerDaemon(_DERIVED_IMAGE_ID)
        with patch("deploy_agent.executor._run", side_effect=daemon):
            DeployExecutor().verify_running_image_digest(
                lane=EnumRuntimeLane.PROD, expected_digest=_DERIVED_IMAGE_ID
            )
        assert daemon.formats_queried == {"{{.Image}}"}
        assert all(_BASE_IMAGE_ID not in " ".join(c) for c in daemon.commands)


class TestProdBoundaryGuardOnADerivedImage:
    def test_derived_digest_that_ran_in_stability_test_is_accepted(self) -> None:
        # must not raise
        assert_prod_request_has_stability_digest(
            _prod_cmd(_DERIVED_IMAGE_ID), stability_ready_digest=_DERIVED_IMAGE_ID
        )

    def test_derived_digest_when_stability_ran_only_the_base_is_refused(self) -> None:
        with pytest.raises(ProdStabilityDigestMissingError, match="does not equal"):
            assert_prod_request_has_stability_digest(
                _prod_cmd(_DERIVED_IMAGE_ID), stability_ready_digest=_BASE_IMAGE_ID
            )

    def test_base_digest_when_stability_ran_the_derived_image_is_refused(self) -> None:
        with pytest.raises(ProdStabilityDigestMissingError, match="does not equal"):
            assert_prod_request_has_stability_digest(
                _prod_cmd(_BASE_IMAGE_ID), stability_ready_digest=_DERIVED_IMAGE_ID
            )

    def test_derived_digest_with_no_stability_ready_digest_is_refused(self) -> None:
        with pytest.raises(ProdStabilityDigestMissingError, match="stability"):
            assert_prod_request_has_stability_digest(
                _prod_cmd(_DERIVED_IMAGE_ID), stability_ready_digest=None
            )

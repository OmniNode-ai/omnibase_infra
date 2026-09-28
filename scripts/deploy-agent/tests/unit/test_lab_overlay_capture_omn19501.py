# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19501 -- the overlay reads compose-lane inputs captured under the lane lock.

Once the dev lane is released at the compose verdict, the next job may be
building ``omnibase-infra-omninode-runtime:latest``, resetting the deploy-source
clone and recreating ``omninode-runtime`` while this job's k3s overlay is still
applying. Three of the overlay's inputs are compose-lane state:

* the runtime image, which the apply used to ``docker tag`` from ``:latest``;
* the infra migrate image, which it built from the deploy-source clone;
* the compose lane's omnimarket version, which the readback compared against.

Each is now taken by ``capture_compose_inputs`` while the job still holds the
lane lock, and ``apply(capture=...)`` reads only the capture. The extended TLA+
model shows what reading them late costs (``MC_a1_no_capture``:
``OverlayPromotesOwnBuild`` violated): the overlay promotes the NEXT job's build
under THIS job's sha.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.lab_overlay import (
    INFRA_MIGRATE_IMAGE_NAME,
    RUNTIME_IMAGE_NAME,
    ModelLabOverlayCapture,
)

from .test_lab_overlay_omn18200 import SHA, STAMP, FakeRunner, _applier

pytestmark = pytest.mark.unit


@pytest.fixture
def overlay_source(tmp_path: Path) -> Path:
    """A stand-in omninode_infra clone whose archive carries the apply script."""
    source = tmp_path / "omninode_infra"
    (source / "k8s" / "onex-lab").mkdir(parents=True)
    (source / "k8s" / "onex-lab" / "apply_lab_lane.sh").write_text("#!/bin/sh\n")
    return source


class _LaneMovesRunner(FakeRunner):
    """After ``moved`` is set, the compose lane reports a different omnimarket
    version: the next job has recreated ``omninode-runtime``."""

    def __init__(self) -> None:
        super().__init__()
        self.moved = False

    def __call__(self, argv: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        joined = " ".join(argv)
        if "importlib.metadata" in joined and "kubectl" not in joined:
            self.calls.append(list(argv))
            version = "0.4.99" if self.moved else "0.4.61"
            return subprocess.CompletedProcess(list(argv), 0, version, "")
        return super().__call__(argv, **kwargs)


def test_capture_tags_the_runtime_and_builds_infra_migrate_without_importing(
    tmp_path: Path, overlay_source: Path
) -> None:
    """The capture is the part that must run under the lane lock, so it is
    short: one tag, one small build, one version read. No import into k3s, no
    overlay source, no cluster."""
    runner = _LaneMovesRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    capture = applier.capture_compose_inputs(sha=SHA, stamp=STAMP)

    assert capture.runtime_image == f"{RUNTIME_IMAGE_NAME}:{STAMP}-{SHA[:8]}"
    assert (
        capture.infra_migrate_image == f"{INFRA_MIGRATE_IMAGE_NAME}:{STAMP}-{SHA[:8]}"
    )
    assert capture.compose_omnimarket_version == "0.4.61"
    assert capture.pin_error is None
    tags = runner.argv_containing("docker tag")
    assert len(tags) == 1 and tags[0][-1] == capture.runtime_image
    builds = runner.argv_containing("docker build")
    assert len(builds) == 1 and capture.infra_migrate_image in builds[0]
    assert applier._popen.refs == [], "the capture must not import into k3s"
    assert not runner.argv_containing("kubectl")
    assert not runner.argv_containing("k3s ctr")


def test_apply_with_a_capture_reads_nothing_off_the_compose_lane(
    tmp_path: Path, overlay_source: Path
) -> None:
    """RED against the tree without the capture: the apply re-tags
    ``runtime:latest`` and re-reads the compose container when it runs, which
    after the release is the next job's image and the next job's container."""
    runner = _LaneMovesRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    capture = applier.capture_compose_inputs(sha=SHA, stamp=STAMP)
    tags_before = len(runner.argv_containing("docker tag"))
    builds_before = len(runner.argv_containing("docker build"))
    runner.moved = True  # the next job has taken the lane

    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id="cid", capture=capture)

    record = json.loads(path.read_text())
    assert len(runner.argv_containing("docker tag")) == tags_before, (
        "the apply re-tagged runtime:latest after the lane was released"
    )
    later_builds = runner.argv_containing("docker build")[builds_before:]
    assert not [b for b in later_builds if INFRA_MIGRATE_IMAGE_NAME in " ".join(b)], (
        "the apply rebuilt the infra migrate image from the deploy-source clone "
        "after the lane was released"
    )
    compose_reads = [
        c
        for c in runner.argv_containing("importlib.metadata")
        if "kubectl" not in " ".join(c)
    ]
    assert len(compose_reads) == 1, "the compose lane was read again after release"
    version = next(
        c for c in record["checks"] if c["name"] == "runtime_omnimarket_version"
    )
    assert "0.4.61" in version["evidence"] and "0.4.99" not in version["evidence"]
    # Both captured images, and the two overlay-built ones, reach k3s.
    assert capture.runtime_image in applier._popen.refs
    assert capture.infra_migrate_image in applier._popen.refs
    assert len(applier._popen.refs) == 4


def test_a_failed_capture_is_a_failing_images_pinned_check(
    tmp_path: Path, overlay_source: Path
) -> None:
    """A capture never raises into the job: its failure travels to the record
    as a failing check, so the onex-lab-k3s receipt says FAIL, not "missing"."""
    runner = FakeRunner({"docker tag": (1, "no such image")})
    applier = _applier(tmp_path, overlay_source, runner)

    capture = applier.capture_compose_inputs(sha=SHA, stamp=STAMP)
    assert capture.pin_error is not None
    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id="cid", capture=capture)

    record = json.loads(path.read_text())
    pinned = next(c for c in record["checks"] if c["name"] == "images_pinned")
    assert pinned["ok"] is False
    assert "captured" in pinned["evidence"]
    assert not runner.argv_containing("apply_lab_lane.sh")


def test_a_capture_for_another_sha_is_refused(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    capture = ModelLabOverlayCapture(
        sha="e" * 40,
        stamp=STAMP,
        runtime_image=f"{RUNTIME_IMAGE_NAME}:{STAMP}-eeeeeeee",
        infra_migrate_image=f"{INFRA_MIGRATE_IMAGE_NAME}:{STAMP}-eeeeeeee",
        pin_error=None,
        compose_omnimarket_version="0.4.61",
        compose_version_error=None,
    )

    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id="cid", capture=capture)

    record = json.loads(path.read_text())
    pinned = next(c for c in record["checks"] if c["name"] == "images_pinned")
    assert pinned["ok"] is False
    assert "e" * 8 in pinned["evidence"]

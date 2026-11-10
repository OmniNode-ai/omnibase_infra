# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19501 -- the settle reads compose-lane inputs captured under the lane lock.

Once the dev lane is released at the compose verdict, the next job may be
building ``omnibase-infra-omninode-runtime:latest`` and recreating
``omninode-runtime`` while this job's settle still runs. So the runtime image is
tagged by a unique name, and the infra migrate image built, by
``capture_compose_inputs`` while the job still holds the lane lock. The settle's
onex-api build reads only the capture (OMN-23521; its refusals of a capture for
another sha or a failed capture are pinned in the handler's own tests).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.lab_overlay import (
    INFRA_MIGRATE_IMAGE_NAME,
    RUNTIME_IMAGE_NAME,
)

from .test_lab_overlay_omn18200 import SHA, STAMP, FakeRunner, _applier

pytestmark = pytest.mark.unit


@pytest.fixture
def overlay_source(tmp_path: Path) -> Path:
    """A stand-in overlay source clone."""
    source = tmp_path / "overlay-source"
    source.mkdir()
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
    assert not runner.argv_containing("kubectl")
    assert not runner.argv_containing("k3s ctr")

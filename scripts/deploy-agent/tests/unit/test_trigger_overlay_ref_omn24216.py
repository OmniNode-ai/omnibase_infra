# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The manual trigger's --overlay-ref must ride the command it publishes.

OMN-24216. A dev rebuild can carry an overlays repository revision the job
renders and mounts. The flag is dev-lane only and must ride with a rollback
declaration, and the value must be a 40-character lowercase hex sha. These
tests run the REAL script in ``--dry-run`` and validate its REAL output
against the REAL model, exactly the way
tests/unit/test_trigger_payload_matches_model_omn16442.py does.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRollbackDeclaration,
    Scope,
)
from deploy_agent.trigger import build_rebuild_command

TRIGGER = Path(__file__).resolve().parents[2] / "deploy-agent-trigger.sh"

# Not a credential: an arbitrary local value for signing a dry-run envelope
# that is never published anywhere.
_TEST_HMAC = "0" * 64

_SHA = "a" * 40
_DIGEST = "sha256:" + "b" * 64


def _run(
    *args: str, secret: str | None = _TEST_HMAC
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("DEPLOY_AGENT_TRACKING_REF", None)
    env.pop("DEPLOY_AGENT_ALLOWED_LANES", None)
    if secret is None:
        env.pop("DEPLOY_AGENT_HMAC_SECRET", None)
    else:
        env["DEPLOY_AGENT_HMAC_SECRET"] = secret
    return subprocess.run(
        ["bash", str(TRIGGER), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _payload(stdout: str) -> dict[str, Any]:
    """Extract the pretty-printed envelope the script echoes back."""
    start = stdout.index("{")
    end = stdout.rindex("}") + 1
    return json.loads(stdout[start:end])


@pytest.mark.unit
def test_overlay_ref_rides_the_published_command() -> None:
    """A dev rebuild with --overlay-ref publishes a command carrying the sha."""
    proc = _run(
        "--runtime-lane",
        "dev",
        "--git-ref",
        "origin/dev",
        "--overlay-ref",
        _SHA,
        "--rollback-actor",
        "lane-x",
        "--rollback-reason",
        "return to overlays main",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr

    envelope = _payload(proc.stdout)
    command = {k: v for k, v in envelope.items() if k != "_signature"}
    cmd = ModelRebuildRequested.model_validate(command)
    assert cmd.overlay_ref == _SHA
    assert cmd.rollback is not None
    assert cmd.rollback.actor == "lane-x"


@pytest.mark.unit
def test_no_overlay_ref_publishes_empty() -> None:
    """Existing behaviour unchanged: no flag means overlay_ref stays empty."""
    proc = _run(
        "--runtime-lane",
        "dev",
        "--git-ref",
        "origin/dev",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr
    envelope = _payload(proc.stdout)
    command = {k: v for k, v in envelope.items() if k != "_signature"}
    cmd = ModelRebuildRequested.model_validate(command)
    assert cmd.overlay_ref == ""


@pytest.mark.unit
def test_overlay_ref_refused_off_dev_lane() -> None:
    proc = _run(
        "--runtime-lane",
        "prod",
        "--image-digest",
        _DIGEST,
        "--git-ref",
        "origin/dev",
        "--overlay-ref",
        _SHA,
        "--rollback-actor",
        "lane-x",
        "--rollback-reason",
        "return to overlays main",
        "--dry-run",
    )
    assert proc.returncode != 0
    assert "--overlay-ref needs --runtime-lane dev" in proc.stderr


@pytest.mark.unit
def test_overlay_ref_refused_without_rollback() -> None:
    proc = _run(
        "--runtime-lane",
        "dev",
        "--git-ref",
        "origin/dev",
        "--overlay-ref",
        _SHA,
        "--dry-run",
    )
    assert proc.returncode != 0
    assert "--overlay-ref needs --runtime-lane dev with --rollback-actor" in proc.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_sha",
    ["not-a-sha", "a" * 39, "A" * 40],
)
def test_overlay_ref_refuses_malformed_sha(bad_sha: str) -> None:
    proc = _run(
        "--runtime-lane",
        "dev",
        "--git-ref",
        "origin/dev",
        "--overlay-ref",
        bad_sha,
        "--rollback-actor",
        "lane-x",
        "--rollback-reason",
        "return to overlays main",
        "--dry-run",
    )
    assert proc.returncode != 0
    assert "--overlay-ref must be a 40-hex sha" in proc.stderr


@pytest.mark.unit
def test_build_rebuild_command_carries_overlay_ref() -> None:
    """The contract model receives the overlay_ref the operator passed."""
    cmd = build_rebuild_command(
        git_ref="origin/dev",
        runtime_lane=EnumRuntimeLane.DEV,
        scope=Scope.RUNTIME,
        build_source=BuildSource.WORKSPACE,
        requested_by="operator-manual",
        correlation_id=uuid.uuid4(),
        services=[],
        rollback=ModelRollbackDeclaration(
            actor="lane-x", reason="return to overlays main"
        ),
        overlay_ref=_SHA,
    )
    assert cmd.overlay_ref == _SHA

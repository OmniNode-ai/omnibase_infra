# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The manual trigger's envelope must satisfy the agent's own command model.

OMN-16442. ``deploy-agent-trigger.sh`` is the only direct manual path to the
deploy agent, and it silently rotted: ``runtime_lane`` became a required field
of :class:`ModelRebuildRequested` in OMN-12572, the script was never updated,
and because the model is ``extra="forbid"`` the script's ``reason`` key was a
second, independent rejection. Every command it published was refused at
``consumer.poll_and_accept`` with ``invalid_payload`` — before ``self_update``,
before the env-contract validator, before any deploy work ran.

Nothing bound the script to the model, so the drift was invisible for months.
The CI path never caught it either: CI publishes a different envelope to a
different topic (``redeploy-start``, consumed by the redeploy orchestrator) and
does not exercise this script at all.

These tests close that hole by running the REAL script, in ``--dry-run``, and
validating its REAL output against the REAL model. They deliberately do not
re-declare the expected field list: a copy of the schema in the test would rot
in exactly the same way the script did.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.auth import verify_command
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested

TRIGGER = Path(__file__).resolve().parents[2] / "deploy-agent-trigger.sh"

# Not a credential: an arbitrary local value for signing a dry-run envelope
# that is never published anywhere.
_TEST_HMAC = "0" * 64


def _run(
    *args: str, secret: str | None = _TEST_HMAC
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("DEPLOY_AGENT_TRACKING_REF", None)
    # Also cleared so the refusal cases below test what they claim to. The lane
    # is now resolvable from a single-lane DEPLOY_AGENT_ALLOWED_LANES as well as
    # from the flag (see deploy_agent.trigger._resolve_runtime_lane), so an
    # ambient value on the running host would make
    # test_missing_runtime_lane_refuses_rather_than_guessing pass a command
    # instead of a refusal -- a false green on the exact assertion that matters.
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
def test_trigger_dry_run_payload_validates_against_the_real_model() -> None:
    """The regression this file exists for: the script's envelope must parse."""
    proc = _run(
        "--runtime-lane",
        "dev",
        "--git-ref",
        "origin/dev",
        "--reason",
        "binding check",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr

    envelope = _payload(proc.stdout)
    command = {k: v for k, v in envelope.items() if k != "_signature"}

    # No expected-field list here on purpose. model_validate on an
    # extra="forbid" model is the assertion: it fails on a missing required
    # field AND on a stray one, which is exactly the two-sided drift that broke
    # this script.
    cmd = ModelRebuildRequested.model_validate(command)
    assert cmd.runtime_lane is EnumRuntimeLane.DEV
    assert cmd.git_ref == "origin/dev"


@pytest.mark.unit
def test_trigger_envelope_signature_verifies_the_way_the_agent_verifies_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The signature covers the envelope the consumer actually receives.

    Dropping ``reason`` changes the signed bytes, so this pins that the script
    and ``auth.verify_command`` still agree byte-for-byte after the change.
    """
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", _TEST_HMAC)
    proc = _run("--runtime-lane", "dev", "--git-ref", "origin/dev", "--dry-run")
    assert proc.returncode == 0, proc.stderr

    # The echoed payload masks the signature, so re-derive the full envelope the
    # way the consumer does: verify_command over the same field set.
    envelope = _payload(proc.stdout)
    assert envelope["_signature"].endswith("...<masked>")

    import hashlib
    import hmac

    body_dict = {k: v for k, v in envelope.items() if k != "_signature"}
    body = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode()
    expected = hmac.new(_TEST_HMAC.encode(), body, hashlib.sha256).hexdigest()
    assert envelope["_signature"][:8] == expected[:8]
    assert verify_command({**body_dict, "_signature": expected}) is True


@pytest.mark.unit
def test_reason_is_reported_but_never_signed() -> None:
    """``reason`` is operator audit text; the model forbids it in the envelope."""
    proc = _run(
        "--runtime-lane",
        "dev",
        "--git-ref",
        "origin/dev",
        "--reason",
        "some operator note",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr
    assert "some operator note" in proc.stdout, (
        "reason must stay visible to the operator"
    )
    assert "reason" not in _payload(proc.stdout)


@pytest.mark.unit
def test_missing_runtime_lane_refuses_rather_than_guessing() -> None:
    proc = _run("--git-ref", "origin/dev", "--dry-run")
    assert proc.returncode != 0
    assert "--runtime-lane is required" in proc.stderr


@pytest.mark.unit
def test_unknown_runtime_lane_refuses() -> None:
    proc = _run("--runtime-lane", "staging", "--git-ref", "origin/dev", "--dry-run")
    assert proc.returncode != 0
    assert "unknown --runtime-lane" in proc.stderr


@pytest.mark.unit
def test_every_enum_lane_is_accepted_by_the_script() -> None:
    """Positive control for the two refusal tests above.

    Without this, a script that refused *every* lane would still pass them.
    Driven off the enum so a new lane member cannot be added to the model
    without this script learning about it.
    """
    for lane in EnumRuntimeLane:
        args = ["--runtime-lane", lane.value, "--git-ref", "origin/dev", "--dry-run"]
        if lane is EnumRuntimeLane.PROD:
            args += ["--image-digest", "sha256:" + "a" * 64]
        proc = _run(*args)
        assert proc.returncode == 0, f"lane {lane.value} refused: {proc.stderr}"
        command = {k: v for k, v in _payload(proc.stdout).items() if k != "_signature"}
        assert ModelRebuildRequested.model_validate(command).runtime_lane is lane


@pytest.mark.unit
def test_prod_without_digest_refuses_before_signing() -> None:
    """The model refuses this too; the script refuses it first, and says why."""
    proc = _run("--runtime-lane", "prod", "--git-ref", "origin/dev", "--dry-run")
    assert proc.returncode != 0
    assert "requires --image-digest" in proc.stderr


@pytest.mark.unit
def test_unknown_build_source_refuses() -> None:
    proc = _run(
        "--runtime-lane",
        "dev",
        "--git-ref",
        "origin/dev",
        "--build-source",
        "nightly",
        "--dry-run",
    )
    assert proc.returncode != 0
    assert "unknown --build-source" in proc.stderr


@pytest.mark.unit
def test_tracking_ref_still_supplies_the_default_git_ref() -> None:
    """OMN-16442's original fix must survive this change."""
    env_proc = subprocess.run(
        [
            "bash",
            str(TRIGGER),
            "--runtime-lane",
            "dev",
            "--reason",
            "default-ref check",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env={
            k: v
            for k, v in {
                **os.environ,
                "DEPLOY_AGENT_HMAC_SECRET": _TEST_HMAC,
                "DEPLOY_AGENT_TRACKING_REF": "dev",
            }.items()
            if k != "DEPLOY_AGENT_ALLOWED_LANES"
        },
        check=False,
    )
    assert env_proc.returncode == 0, env_proc.stderr
    assert _payload(env_proc.stdout)["git_ref"] == "origin/dev"


@pytest.mark.unit
def test_undeclared_tracking_ref_with_no_git_ref_still_refuses() -> None:
    proc = _run("--runtime-lane", "dev", "--dry-run")
    assert proc.returncode != 0
    assert "DEPLOY_AGENT_TRACKING_REF is not set" in proc.stderr

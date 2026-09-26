# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18573 — the receipt carries the deploy agent's correlation id.

WHAT THIS PINS
--------------
``ModelLabPassReceipt.agent_command_id`` has existed since OMN-17530 and every
compose-dev receipt ever emitted carried ``null`` in it, because no emitter
passed ``--agent-command-id``. The id was knowable the whole time: the
publishing job holds it, and OMN-18573 made it a job output so the convergence
guard could measure the lane's budget from the agent's acceptance of that exact
command.

The readback that found this is the argument for fixing it. Run 35232827031's
receipt carried the correlation id **in the evidence string** of its
``deployed_revision`` check, in prose, while the typed field a machine reads sat
empty. A fact present in prose and absent from the contract is a fact no
consumer can use — the receipt could not be joined to the agent's job record
without a human reading a sentence.

THE FIXTURE IS REAL
-------------------
``OBSERVED_CORRELATION_ID`` and ``OBSERVED_SHA`` are the values the emitting job
actually produced on 2026-09-17T14:26Z, quoted from that receipt's own
``deployed_revision`` evidence. A synthetic uuid would prove the plumbing
accepts a uuid; this proves it accepts the shape the job really emits.

THE TWO LEGAL STATES, AND THE ONE REFUSAL
-----------------------------------------
A workflow expression delivers an unset output as an EMPTY STRING, not as an
absent flag, so both states arrive through the same argv. Empty means the run
published no command and ``None`` is the honest record; a non-uuid is refused,
because a field carrying a value no agent job can be resolved by is worse than
the null it replaced — a reader would believe it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.lab_pass_receipt import (
    main,
    parse_agent_command_id,
)

pytestmark = pytest.mark.unit

# Quoted from the deployed_revision evidence of artifact 10502703690,
# `lab-pass-receipt-compose-dev-78bbba2f9b28cf11ef8dccf45e07a251381700d4`:
# "0h25m since the deploy agent accepted 29dfee72-1fc1-4702-bfd7-96a1534fc632
#  at 2026-09-17T14:26:00.442210+00:00 (budget 1500s from that moment)"
OBSERVED_CORRELATION_ID = "29dfee72-1fc1-4702-bfd7-96a1534fc632"
OBSERVED_SHA = "78bbba2f9b28cf11ef8dccf45e07a251381700d4"

# OMN-18638. BOTH workflows, because the OMN-18573 wiring landed in the direct
# caller only and the reusable — the ONLY path a sibling repository's receipt is
# emitted through — passed no ``--agent-command-id`` at any revision. A test that
# parses one workflow cannot see a defect that lives in the other, which is
# exactly how this reached a month of sibling receipts unnoticed.
#
# The value is the set of job names expected to emit. It is asserted exactly, so
# a NEW emitter in either workflow turns this red rather than inheriting the
# test's silence.
EMITTING_JOBS: dict[Path, set[str]] = {
    REPO_ROOT / ".github/workflows/runtime-rebuild-trigger.yml": {
        "verify-lane-converged",
        "verify-lab-overlay-converged",
    },
    # One emitter, not two: the reusable has no onex-lab job at all. The lab
    # overlay is applied by omnibase_infra's own caller, so there is no second
    # invocation here that could deliberately omit the argument (AC2).
    REPO_ROOT / ".github/workflows/runtime-rebuild-trigger-reusable.yml": {
        "verify-sibling-converged",
    },
}


def _emit(tmp_path: Path, agent_command_id: str | None) -> dict:
    out = tmp_path / "receipt.json"
    argv = [
        "emit",
        "--sha",
        OBSERVED_SHA,
        "--lane",
        "compose-dev",
        "--started-at",
        "2026-09-17T14:27:06Z",
        "--finished-at",
        "2026-09-17T14:51:03Z",
        "--check",
        "ready_main:ok:GET /ready -> 200",
        "--out",
        str(out),
    ]
    if agent_command_id is not None:
        argv.extend(["--agent-command-id", agent_command_id])
    assert main(argv) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_receipt_carries_the_correlation_id_the_job_really_emitted(
    tmp_path: Path,
) -> None:
    """The defect, inverted: the field is NON-NULL on a real job output."""
    payload = _emit(tmp_path, OBSERVED_CORRELATION_ID)
    assert payload["agent_command_id"] == OBSERVED_CORRELATION_ID
    assert payload["agent_command_id"] is not None


def test_an_empty_output_records_null_rather_than_an_empty_string(
    tmp_path: Path,
) -> None:
    """A run that published no command has none, and says so in the field.

    An empty string here would be a third state the contract does not have, and
    a reader checking ``if receipt.agent_command_id:`` would treat it the same
    as null while a reader comparing to ``None`` would not.
    """
    payload = _emit(tmp_path, "")
    assert payload["agent_command_id"] is None


def test_an_omitted_flag_still_records_null(tmp_path: Path) -> None:
    """The onex-lab boot gate's shape: no agent command at all."""
    payload = _emit(tmp_path, None)
    assert payload["agent_command_id"] is None


def test_a_non_uuid_correlation_id_is_refused_not_stored(tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    assert (
        main(
            [
                "emit",
                "--sha",
                OBSERVED_SHA,
                "--lane",
                "compose-dev",
                "--started-at",
                "2026-09-17T14:27:06Z",
                "--finished-at",
                "2026-09-17T14:51:03Z",
                "--check",
                "ready_main:ok:GET /ready -> 200",
                "--agent-command-id",
                "not-a-uuid",
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert not out.exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        (OBSERVED_CORRELATION_ID, OBSERVED_CORRELATION_ID),
        (f"  {OBSERVED_CORRELATION_ID}  ", OBSERVED_CORRELATION_ID),
    ],
)
def test_the_normaliser_has_exactly_two_legal_outputs(
    raw: str | None, expected: str | None
) -> None:
    assert parse_agent_command_id(raw) == expected


@pytest.mark.parametrize("raw", ["not-a-uuid", "29dfee72", "0", "null", "None"])
def test_the_normaliser_refuses_anything_that_is_not_a_uuid(raw: str) -> None:
    with pytest.raises(ValueError, match="not a uuid"):
        parse_agent_command_id(raw)


# ---------------------------------------------------------------------------
# The wiring, pinned. A normaliser no emitter calls changes nothing.
# ---------------------------------------------------------------------------
def _emit_steps(workflow: Path) -> dict[str, dict]:
    model = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    found: dict[str, dict] = {}
    for job_name, job in model["jobs"].items():
        for step in job.get("steps", []):
            if "lab_pass_receipt.py emit" in str(step.get("run", "")):
                found[job_name] = step
    return found


@pytest.mark.parametrize("workflow", sorted(EMITTING_JOBS), ids=lambda p: p.name)
def test_every_emitter_passes_the_correlation_id_from_the_job_output(
    workflow: Path,
) -> None:
    steps = _emit_steps(workflow)
    assert set(steps) == EMITTING_JOBS[workflow], (
        f"the set of lab_pass_receipt emitters in {workflow.name} changed; "
        "decide whether the new one owes the correlation id rather than "
        "letting it inherit this test's silence"
    )
    for job_name, step in steps.items():
        assert "--agent-command-id" in step["run"], f"{workflow.name}:{job_name}"
        assert (
            "needs.trigger-rebuild.outputs.correlation_id"
            in step["env"]["CORRELATION_ID"]
        ), f"{workflow.name}:{job_name}"


@pytest.mark.parametrize("workflow", sorted(EMITTING_JOBS), ids=lambda p: p.name)
def test_the_publishing_job_exports_the_correlation_id_as_an_output(
    workflow: Path,
) -> None:
    """The env reference above resolves to nothing unless the job declares it.

    A `needs.<job>.outputs.<name>` that the producing job never declares is not
    an error in Actions — it evaluates to the empty string, which
    ``parse_agent_command_id`` reads as the honest "no command was published".
    So a missing output would reproduce the exact null this ticket exists to
    fix, silently and with every other assertion still green.
    """
    model = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    outputs = model["jobs"]["trigger-rebuild"].get("outputs", {})
    assert "correlation_id" in outputs, workflow.name
    assert "steps.publish.outputs.correlation_id" in outputs["correlation_id"], (
        workflow.name
    )


@pytest.mark.parametrize("workflow", sorted(EMITTING_JOBS), ids=lambda p: p.name)
def test_the_correlation_id_is_dereferenced_never_interpolated(
    workflow: Path,
) -> None:
    """The same script-injection rule the evidence string already follows."""
    for job_name, step in _emit_steps(workflow).items():
        assert "${{" not in step["run"], f"{workflow.name}:{job_name}"

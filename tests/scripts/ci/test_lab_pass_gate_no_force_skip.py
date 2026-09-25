# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19233 AC5: no force or skip input exists on the staging delivery gate.

Rule 24(b): delivery to staging fails closed without a PASS receipt, with no
force and no skip. This reads the gate step as the delivery workflow declares it
and the script's own parser option strings, and asserts neither declares an
override. It also pins the wiring the rest of OMN-19233 depends on: the own-sha
step requires compose-dev of the PS-1 subject under the 14,400 s bound, and the
compose-dev emitter holds the ``actions: write`` its re-run needs, in a job of
its own and nowhere else.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.lab_pass_receipt import DELIVERY_OVERALL_BOUND_SECONDS, build_parser

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_DELIVER = _ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"
_REBUILD = _ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"

_BANNED = ("force", "skip", "allow", "ignore", "warn", "bypass", "override")


def _workflow(path: Path) -> dict[str, Any]:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(body, dict)
    return body


def _gate_job() -> dict[str, Any]:
    job = _workflow(_DELIVER)["jobs"]["lab-pass-gate"]
    assert isinstance(job, dict)
    return job


def _own_sha_step() -> dict[str, Any]:
    steps = [s for s in _gate_job()["steps"] if s.get("id") == "own-sha-gate"]
    assert len(steps) == 1, "the own-sha gate step must carry id: own-sha-gate"
    return steps[0]


def _subparsers() -> dict[str, argparse.ArgumentParser]:
    action = next(
        a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
    )
    return dict(action.choices)


class TestNoOverride:
    @pytest.mark.parametrize("command", ["gate", "rerun-refused-deliveries"])
    def test_the_parser_declares_no_override_option(self, command: str) -> None:
        parser = _subparsers()[command]
        options = {o for a in parser._actions for o in a.option_strings}
        assert options, f"{command} declares no options at all"
        for option in options:
            for banned in _BANNED:
                assert banned not in option.lower(), (
                    f"{command} {option} reads as an override of the gate"
                )

    def test_the_gate_step_has_no_if_and_no_continue_on_error(self) -> None:
        step = _own_sha_step()
        assert "if" not in step
        assert "continue-on-error" not in step

    def test_the_gate_job_has_no_if_and_no_continue_on_error(self) -> None:
        job = _gate_job()
        assert "continue-on-error" not in job
        assert "if" not in job

    def test_the_gate_step_passes_no_override_flag_or_input(self) -> None:
        run = _own_sha_step()["run"]
        for banned in _BANNED:
            assert f"--{banned}" not in run
        on = _workflow(_DELIVER)[True]  # PyYAML reads the `on:` key as True
        dispatch_inputs = (on.get("workflow_dispatch") or {}).get("inputs") or {}
        for name in dispatch_inputs:
            for banned in _BANNED:
                assert banned not in name.lower(), (
                    f"workflow_dispatch input {name!r} reads as an override"
                )


class TestOwnShaStepRequiresComposeDev:
    def test_the_step_requires_compose_dev_of_the_proof_subject(self) -> None:
        run = _own_sha_step()["run"]
        assert "lab_pass_receipt.py gate" in run
        assert "--require-lane compose-dev" in run
        assert "--resolve-runtime-ancestor" in run
        assert "--detect-pending" in run
        assert f"--overall-bound-seconds {DELIVERY_OVERALL_BOUND_SECONDS}" in run
        assert "--verdict-out" in run
        assert '--run-attempt "${GITHUB_RUN_ATTEMPT}"' in run

    def test_the_named_gate_job_is_the_job_the_first_read_is_measured_on(
        self,
    ) -> None:
        run = _own_sha_step()["run"]
        name = _gate_job()["name"]
        assert f'--gate-job-name "{name}"' in run

    def test_the_job_can_walk_first_parent_history_and_read_labels(self) -> None:
        job = _gate_job()
        assert job["permissions"].get("pull-requests") == "read"
        assert job["permissions"].get("actions") == "read"
        assert "write" not in job["permissions"].values()
        checkout = next(
            s
            for s in job["steps"]
            if str(s.get("uses", "")).startswith("actions/checkout")
            and "repository" not in (s.get("with") or {})
        )
        assert checkout["with"]["fetch-depth"] == 0

    def test_the_verdict_is_uploaded_per_attempt(self) -> None:
        uploads = [
            s
            for s in _gate_job()["steps"]
            if str(s.get("uses", "")).startswith("actions/upload-artifact")
        ]
        assert len(uploads) == 1
        name = uploads[0]["with"]["name"]
        assert name == (
            "lab-pass-gate-verdict-${{ github.run_id }}-${{ github.run_attempt }}"
        )


class TestEmitterRerun:
    def _job(self) -> dict[str, Any]:
        job = _workflow(_REBUILD)["jobs"]["rerun-refused-deliveries"]
        assert isinstance(job, dict)
        return job

    def test_the_rerun_job_holds_actions_write_and_nothing_wider(self) -> None:
        permissions = self._job()["permissions"]
        assert permissions == {"contents": "read", "actions": "write"}

    def test_no_other_job_in_the_emitter_gains_actions_write(self) -> None:
        for name, job in _workflow(_REBUILD)["jobs"].items():
            if name == "rerun-refused-deliveries":
                continue
            permissions = job.get("permissions") or {}
            assert permissions.get("actions") != "write", name

    def test_the_rerun_job_runs_after_the_compose_dev_emitters(self) -> None:
        needs = self._job()["needs"]
        assert "verify-lane-converged" in needs
        assert "reemit-queued-receipts" in needs

    def test_the_rerun_step_uses_the_same_bound_as_the_gate(self) -> None:
        runs = "\n".join(str(s.get("run", "")) for s in self._job()["steps"])
        assert "lab_pass_receipt.py rerun-refused-deliveries" in runs
        assert f"--overall-bound-seconds {DELIVERY_OVERALL_BOUND_SECONDS}" in runs
        assert "--workflow deliver-dev-candidate-to-staging.yml" in runs

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end coverage for the unattributed-route reason (OMN-18306 residual).

The unit module beside this one drives the writer's return value. This one
drives the whole artifact-writing path the customer actually meets — three
files on disk, plus the line printed on stderr — because the reason is
something a person READS, and a reason that is right inside the receipt but
wrong on the terminal line is still wrong where it is looked at.

What the constant asserted, and what was measured against it on 2026-09-15:

* it claimed "every rung this run attempted was refused", for a run that
  attempted nothing — a delegation refused before dispatch in 265 ms with zero
  attempts on its receipt;
* it named no backend, for a run that reached one and was refused there.

Both shapes are exercised here through the real writer, and the fail-closed
property that OMN-18306 closed on — that no route is ever synthesised — is
re-asserted against the files themselves rather than against a return value.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from omnibase_infra.cli.cli_delegate import _write_unattributed_run_files
from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal

pytestmark = pytest.mark.integration

REACHED_A_BACKEND: dict[str, object] = {
    "response": "a partial answer nobody accepted",
    "error_message": "TASK_MISMATCH: failed covers_dependencies",
    "metrics": {"cost_usd": 0.000592},
    "attempts": [
        {
            "tier": "cheap_cloud",
            "backend_id": "cloud-gemini-pro",
            "model_id": "gemini-2.5-flash",
            "failure_class": "rate_limited",
            "acceptance_decision": "climb",
        }
    ],
}

REACHED_NOTHING: dict[str, object] = {
    "response": "",
    "error_message": "acceptance_criteria validation refused the request",
    "metrics": {"cost_usd": 0.0},
    "attempts": [],
}


def _write(
    result: dict[str, object], state_root: Path
) -> tuple[Path, dict[str, object]]:
    run_id = uuid.uuid4()
    _write_unattributed_run_files(
        envelope={
            "run_id": str(run_id),
            "correlation_id": str(uuid.uuid4()),
            "status": "failed",
        },
        # OMN-18569 typed the writer's terminal. The recorded shapes above are
        # unchanged; they are validated into the model the runtime's own
        # terminal validates into, which is a stricter input than the loose
        # dict this used to hand over, not a weaker one.
        result=ModelDelegateTerminal.model_validate(result),
        state_root=state_root,
        prompt="draft two paragraphs of rationale prose",
        task_type="document",
        task_type_resolution="fallback",
    )
    run_dir = state_root / "runs" / str(run_id)
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    return run_dir, receipt


class TestTheCustomerFacingArtifactsAreWritten:
    def test_all_three_files_exist_for_both_shapes(self, tmp_path: Path) -> None:
        """OMN-18306's own property: a failed run is diagnosable on disk."""
        for index, result in enumerate((REACHED_A_BACKEND, REACHED_NOTHING)):
            run_dir, _ = _write(result, tmp_path / str(index))
            for name in ("result.txt", "receipt.json", "run.json"):
                assert (run_dir / name).is_file(), f"{name} missing"

    def test_the_reason_reaches_the_terminal_line_too(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The stderr line is where this is actually read; it must agree."""
        _write(REACHED_A_BACKEND, tmp_path)
        printed = capsys.readouterr().err
        assert "route UNATTRIBUTED" in printed
        assert "cloud-gemini-pro" in printed, (
            "the terminal line carries the same reason as the receipt, or the "
            "customer reads a different story from the one on disk"
        )


class TestTheReasonMatchesWhatHappened:
    def test_a_reached_backend_is_named(self, tmp_path: Path) -> None:
        _, receipt = _write(REACHED_A_BACKEND, tmp_path)
        assert "cloud-gemini-pro" in str(receipt["route_unattributed"])

    def test_reaching_nothing_does_not_claim_refused_rungs(
        self, tmp_path: Path
    ) -> None:
        _, receipt = _write(REACHED_NOTHING, tmp_path)
        reason = str(receipt["route_unattributed"])
        assert "no backend was reached" in reason
        assert "every rung this run attempted was refused" not in reason

    def test_the_cost_and_content_a_run_did_produce_are_preserved(
        self, tmp_path: Path
    ) -> None:
        """A refused run still bills and may still have produced text."""
        run_dir, receipt = _write(REACHED_A_BACKEND, tmp_path)
        assert receipt["cost_usd"] == 0.000592
        assert (run_dir / "result.txt").read_text(
            encoding="utf-8"
        ) == "a partial answer nobody accepted"


class TestAttributionStaysFailClosedOnDisk:
    def test_no_route_is_written_to_any_file(self, tmp_path: Path) -> None:
        """AC3 of OMN-18306, re-asserted against the artifacts themselves."""
        for index, result in enumerate((REACHED_A_BACKEND, REACHED_NOTHING)):
            run_dir, receipt = _write(result, tmp_path / str(index))
            assert receipt["route_attributed"] is False
            run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            assert run_json["route_attributed"] is False
            assert run_json["lane"] is None
            for forbidden in ("backend_id", "model_id", "endpoint"):
                assert forbidden not in receipt
                assert forbidden not in run_json

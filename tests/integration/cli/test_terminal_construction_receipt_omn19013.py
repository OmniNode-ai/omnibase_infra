# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end coverage for the terminal-construction receipt fields (OMN-19013).

The unit suite beside this one drives the refusal's return value. This one
drives the artifact-writing path a person actually reads -- the three files on
disk -- because a construction failure is diagnosed from ``receipt.json``, and
a verdict that is right in the model and absent from the file is still absent
where it is looked at.

Two properties are asserted here, and they are the two halves of OMN-19013:

* a terminal-construction failure is RECORDED: the operational outcome, the
  content verdict and the stable failure reason all reach ``receipt.json``,
  so the row the 0045 view later excludes is traceable to a file;
* a terminal-construction failure is never ACCEPTED: a receipt claiming
  ``completed`` while carrying that outcome is refused rather than counted.

The second is paired with a positive control -- an ordinary completed terminal
that must still pass -- because a refusal that fires on everything would
satisfy the negative assertion just as well while breaking every successful
delegation.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from omnibase_infra.cli.cli_delegate import (
    _require_completed_terminal_evidence,
    _write_unattributed_run_files,
)
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
)
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.integration

# The suite is about carrier shape, not addressing; one neutral in-process
# value, matching the sibling unattributed-reason suite.
_ADDRESSING = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.IN_PROCESS,
    bus="inmemory",
)

# The shape OMN-19013 exists for: a rung answered, the terminal could not be
# constructed from what came back, and the content verdict is deliberately
# undetermined rather than failed -- nobody judged the content.
CONSTRUCTION_FAILED: dict[str, object] = {
    "response": "",
    "status": "failed",
    "operational_outcome": "terminal_construction_failed",
    "content_verdict": "undetermined",
    "terminal_failure_cause": "terminal_construction",
    "terminal_failure_reason": "response payload omitted the attempts array",
    "error_message": "could not construct a delegation terminal from the response",
    "metrics": {"cost_usd": 0.000418},
    "attempts": [
        {
            "tier": "cheap_cloud",
            "backend_id": "cloud-gemini-pro",
            "model_id": "gemini-2.5-flash",
            "failure_class": "malformed_response",
            "acceptance_decision": "climb",
        }
    ],
}

# A legacy recording: neither new field present. It must still be writable and
# still be readable, which is the compatibility half of the 0045 predicate.
LEGACY_NO_OUTCOME_FIELDS: dict[str, object] = {
    "response": "",
    "status": "failed",
    "error_message": "every rung refused",
    "metrics": {"cost_usd": 0.0},
    "attempts": [],
}


def _write(result: dict[str, object], state_root: Path) -> dict[str, object]:
    """Run the real writer and return the receipt it put on disk."""
    run_id = uuid.uuid4()
    _write_unattributed_run_files(
        envelope={
            "run_id": str(run_id),
            "correlation_id": str(uuid.uuid4()),
            "status": result.get("status", "failed"),
        },
        result=ModelDelegateTerminal.model_validate(result),
        state_root=state_root,
        addressing=_ADDRESSING,
        prompt="summarise the rejected payload",
        task_type="document",
        task_type_resolution="fallback",
    )
    run_dir = state_root / "runs" / str(run_id)
    for name in ("result.txt", "receipt.json", "run.json"):
        assert (run_dir / name).is_file(), f"{name} missing"
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    return receipt


class TestTheConstructionFailureReachesTheFile:
    def test_receipt_carries_outcome_verdict_and_reason(self, tmp_path: Path) -> None:
        """All three OMN-19013 fields survive the write, not just the model."""
        receipt = _write(CONSTRUCTION_FAILED, tmp_path)
        assert receipt["operational_outcome"] == "terminal_construction_failed"
        assert receipt["content_verdict"] == "undetermined"
        assert (
            receipt["terminal_failure_reason"]
            == "response payload omitted the attempts array"
        )

    def test_the_reason_is_distinct_from_the_cause(self, tmp_path: Path) -> None:
        """Two fields, two jobs: a stable class and a readable sentence.

        Collapsing them would make the receipt look complete while losing the
        only human-readable statement of what the producer could not parse.
        """
        receipt = _write(CONSTRUCTION_FAILED, tmp_path)
        assert receipt["terminal_failure_cause"] == "terminal_construction"
        assert receipt["terminal_failure_cause"] != receipt["terminal_failure_reason"]

    def test_a_legacy_recording_still_writes_and_reads_null(
        self, tmp_path: Path
    ) -> None:
        """A recording made before OMN-19013 must remain reportable.

        The 0045 predicate includes legacy NULL pairs deliberately; a writer
        that refused them would strand every row already on disk.
        """
        receipt = _write(LEGACY_NO_OUTCOME_FIELDS, tmp_path)
        assert receipt["operational_outcome"] is None
        assert receipt["content_verdict"] is None
        assert receipt["terminal_failure_reason"] is None

    def test_the_cost_a_failed_construction_incurred_is_kept(
        self, tmp_path: Path
    ) -> None:
        """The rung still billed even though nothing could be built from it."""
        receipt = _write(CONSTRUCTION_FAILED, tmp_path)
        assert receipt["cost_usd"] == 0.000418
        assert receipt["route_attributed"] is False


class TestAConstructionFailureIsNeverCountedAsCompleted:
    def test_completed_claiming_construction_failure_is_refused(self) -> None:
        """The core OMN-19013 refusal, through the CLI's own entry point."""
        terminal = ModelDelegateTerminal.model_validate(
            {**CONSTRUCTION_FAILED, "status": "completed", "quality_gate_passed": True}
        )
        with pytest.raises(
            DelegateTerminalUnresolvedError, match="terminal_construction_failed"
        ):
            _require_completed_terminal_evidence(
                terminal,
                require_budget_evidence=False,
                require_contract_evidence=False,
            )

    def test_an_ordinary_completed_terminal_still_passes(self) -> None:
        """Positive control: the refusal is scoped, not blanket.

        Without this, a refusal that raised unconditionally would satisfy the
        assertion above and break every delegation that actually answered.
        """
        terminal = ModelDelegateTerminal.model_validate(
            {
                "response": "an accepted answer",
                "status": "completed",
                "operational_outcome": "completed",
                "content_verdict": "passed",
                "quality_gate_passed": True,
                "attempts": [
                    {
                        "tier": "cheap_cloud",
                        "backend_id": "cloud-gemini-pro",
                        "model_id": "gemini-2.5-flash",
                        "acceptance_decision": "accept",
                    }
                ],
            }
        )
        _require_completed_terminal_evidence(
            terminal,
            require_budget_evidence=False,
            require_contract_evidence=False,
        )

    def test_a_failed_construction_terminal_is_not_refused(self) -> None:
        """Only a COMPLETED claim is refused; recording the failure is the point."""
        terminal = ModelDelegateTerminal.model_validate(CONSTRUCTION_FAILED)
        _require_completed_terminal_evidence(
            terminal,
            require_budget_evidence=False,
            require_contract_evidence=False,
        )

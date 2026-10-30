# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""receipt.json on disk names where a run's execution budget came from (OMN-21604).

The unit suite checks the receipt block. This one runs the real writer and reads
the file a person opens, because a source that is right in the block but
missing from the file is still missing where people look for it.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from omnibase_infra.cli.cli_delegate import _write_unattributed_run_files
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.integration

_ADDRESSING = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.IN_PROCESS,
    bus="inmemory",
)
_OVERLAY_SOURCE = (
    "overlay:/srv/overlays/delegation_budgets/overlay.yaml"
    "#execution_budgets.code_generation"
)
_BUDGET = {
    "requested_timeout_seconds": 300,
    "task_class_timeout_ceiling_seconds": 600,
    "execution_timeout_seconds": 300,
    "terminal_delivery_margin_seconds": 60,
}


def _receipt(terminal: dict[str, object], state_root: Path) -> dict[str, object]:
    run_id = uuid.uuid4()
    _write_unattributed_run_files(
        envelope={
            "run_id": str(run_id),
            "correlation_id": str(uuid.uuid4()),
            "status": "failed",
        },
        result=ModelDelegateTerminal.model_validate(terminal),
        state_root=state_root,
        addressing=_ADDRESSING,
        prompt="write a test",
        task_type="code_generation",
        task_type_resolution="explicit",
    )
    path = state_root / "runs" / str(run_id) / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    return receipt


def test_the_receipt_file_names_the_overlay_the_limit_came_from(
    tmp_path: Path,
) -> None:
    receipt = _receipt(
        {
            "status": "failed",
            "attempts": [],
            "budget_evidence": _BUDGET,
            "budget_source": _OVERLAY_SOURCE,
        },
        tmp_path,
    )

    assert receipt["budget_source"] == _OVERLAY_SOURCE
    assert receipt["budget_evidence"] == _BUDGET


def test_a_terminal_without_a_source_writes_none(tmp_path: Path) -> None:
    receipt = _receipt(
        {"status": "failed", "attempts": [], "budget_evidence": _BUDGET},
        tmp_path,
    )

    assert "budget_source" not in receipt
    assert receipt["budget_evidence"] == _BUDGET

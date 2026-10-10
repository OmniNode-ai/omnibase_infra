# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A ceiling refusal on ``onex delegate --timeout`` names the flag and the field (OMN-19131).

A run with ``--timeout`` above its task class's ceiling ends with a typed
``budget_refusal`` terminal. The failure line used to carry only the runtime's
``requested_timeout_seconds=300`` text, so the operator had to work out that
the field is what ``--timeout`` supplies and what to pass instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.cli.delegate_human_output import render_delegate_outcome

pytestmark = pytest.mark.unit

RUN_ID = "11111111-2222-3333-4444-555555555555"


def _refused_envelope(*, requested: int, ceiling: int) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "correlation_id": "99999999-2222-3333-4444-555555555555",
        "status": "failed",
        "exit_code": 1,
        "result_model": "omnimarket.ModelDelegateSkillFailed",
        "result": {
            "attempts": [],
            "status": "failed",
            "budget_refusal": {
                "reason": "timeout_exceeds_task_class_ceiling",
                "task_type": "document",
                "requested_timeout_seconds": requested,
                "task_class_timeout_ceiling_seconds": ceiling,
            },
            "error_message": (
                "timeout_exceeds_task_class_ceiling: "
                f"requested_timeout_seconds={requested}; task_type=document; "
                f"task_class_timeout_ceiling_seconds={ceiling}"
            ),
        },
    }


def test_ceiling_refusal_names_the_flag_the_field_and_the_ceiling(
    tmp_path: Path,
) -> None:
    outcome = render_delegate_outcome(
        _refused_envelope(requested=300, ceiling=240), state_root=tmp_path
    )
    assert outcome is not None
    assert outcome.succeeded is False
    assert len(outcome.stderr) == 1
    line = outcome.stderr[0]
    assert "\n" not in line
    assert "`--timeout` 300 (`requested_timeout_seconds`)" in line
    assert "240s ceiling for task type `document`" in line
    assert "`--timeout` of 240 or less" in line
    assert RUN_ID in line


def test_failure_without_a_budget_refusal_does_not_mention_the_flag(
    tmp_path: Path,
) -> None:
    envelope = _refused_envelope(requested=300, ceiling=240)
    result = envelope["result"]
    assert isinstance(result, dict)
    del result["budget_refusal"]
    result["error_message"] = "some other failure"
    outcome = render_delegate_outcome(envelope, state_root=tmp_path)
    assert outcome is not None
    assert "--timeout" not in outcome.stderr[0]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` names the delegation it falls back or escalates from (OMN-20606).

A retry of failed work is a new delegation with its own correlation id. On the
h201 dev lane on 2026-10-05, 93 of 94 failed delegations that carried a session
were answered by such a retry, and the answering row named nothing about the
failure it answered. The CLI now writes the relation into the request
``metadata`` map: ``parent_correlation_id``, ``lineage_kind`` (``fallback`` or
``escalation``) and an optional ``parent_failure_cause``. A partial or
malformed lineage is a usage error, never dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import _write_payload, delegate_command
from omnibase_infra.cli.model_delegate_caller import ModelDelegateCaller
from omnibase_infra.models.delegation.model_delegate_lineage import (
    LINEAGE_KINDS,
    ModelDelegateLineage,
)

pytestmark = pytest.mark.unit

_PARENT = "5b0c8a52-9a43-4f3c-a9f1-0d6f2b4e7c11"
_LANE = "deleg-fallback-mark-9143"


def _payload(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    path = _write_payload(
        prompt="Reply with exactly: OK",
        task_type="summarization",
        source="claude-code",
        state_root=tmp_path,
        run_id=uuid4(),
        correlation_id=uuid4(),
        max_tokens=None,
        **kwargs,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_delegate_lineage_the_command_declares_the_three_options() -> None:
    declared = {
        opt for param in delegate_command.params for opt in getattr(param, "opts", ())
    }
    assert {
        "--parent-correlation-id",
        "--lineage-kind",
        "--parent-failure-cause",
    } <= declared
    assert LINEAGE_KINDS == ("fallback", "escalation")


def test_delegate_lineage_is_written_into_the_request_metadata(tmp_path: Path) -> None:
    lineage = ModelDelegateLineage.from_flags(
        _PARENT, "fallback", "provider_quota_exhausted"
    )
    assert lineage is not None
    payload = _payload(
        tmp_path,
        ticket_id="OMN-20606",
        caller=ModelDelegateCaller(lane=_LANE, lane_source="explicit"),
        lineage=lineage.as_metadata(),
    )
    assert payload["metadata"] == {
        "ticket_id": "OMN-20606",
        "caller_lane": _LANE,
        "parent_correlation_id": _PARENT,
        "lineage_kind": "fallback",
        "parent_failure_cause": "provider_quota_exhausted",
    }


def test_delegate_lineage_absent_writes_no_key(tmp_path: Path) -> None:
    assert ModelDelegateLineage.from_flags(None, None, None) is None
    assert "metadata" not in _payload(tmp_path, lineage={})


def test_delegate_lineage_cause_is_optional_and_parent_is_canonical() -> None:
    lineage = ModelDelegateLineage.from_flags(_PARENT.upper(), "escalation", None)
    assert lineage is not None
    assert lineage.as_metadata() == {
        "parent_correlation_id": _PARENT,
        "lineage_kind": "escalation",
    }


@pytest.mark.parametrize(
    ("parent", "kind", "cause", "fragment"),
    [
        (_PARENT, None, None, "go together"),
        (None, "fallback", None, "go together"),
        (None, None, "exit_124", "--parent-failure-cause needs"),
        ("not-a-uuid", "fallback", None, "is not a UUID"),
        (_PARENT, "retry", None, "is not one of fallback, escalation"),
        (_PARENT, "fallback", "Two Words", "is not a cause token"),
    ],
)
def test_delegate_lineage_partial_or_malformed_is_refused_by_name(
    parent: str | None, kind: str | None, cause: str | None, fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        ModelDelegateLineage.from_flags(parent, kind, cause)


def test_delegate_lineage_a_run_cannot_follow_itself() -> None:
    with pytest.raises(ValueError, match="cannot follow itself"):
        ModelDelegateLineage.from_flags(
            _PARENT, "fallback", None, own_correlation_id=UUID(_PARENT)
        )


def test_delegate_lineage_flags_refuse_before_anything_runs() -> None:
    result = CliRunner().invoke(
        delegate_command,
        ["Reply with exactly: OK", "--parent-correlation-id", _PARENT],
    )
    assert result.exit_code == 2
    assert "--lineage-kind" in result.output


def test_delegate_lineage_kind_outside_the_choices_is_a_usage_error() -> None:
    result = CliRunner().invoke(
        delegate_command,
        [
            "Reply with exactly: OK",
            "--parent-correlation-id",
            _PARENT,
            "--lineage-kind",
            "retry",
        ],
    )
    assert result.exit_code == 2
    assert "--lineage-kind" in result.output

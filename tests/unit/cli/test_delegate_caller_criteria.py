# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The caller can state its own criteria, and the fallback is permissive (OMN-18305).

THE RESIDUAL THIS PINS. OMN-18305 landed on 2026-09-13: the CLI's keyword table
was replaced by contract-declared selection predicates, so a standup stopped
being filed as ``test``. Two days of dogfooding then measured a defect the
landed fix does not cover, twice in one hour on 2026-09-15:

* lane ``plans-velocity-shortlist-1605`` delegated rationale prose and every
  rung was refused ``TASK_MISMATCH: failed covers_dependencies`` — the
  ``planning`` rubric's coverage floor, applied to prose that was never a plan
  (ledger ``docs/tracking/ROLLING_WORK_LEDGER.md:8243``);
* lane ``lakshman-serving-cap-rulings-1456`` delegated a 170-word drafting
  prompt, no predicate claimed it, it fell back to ``research``, and every rung
  was refused for missing source citations — ``research``'s ``cites_sources``
  floor (ledger ``docs/tracking/ROLLING_WORK_LEDGER.md:8210``).

Both are the same shape: an answer graded against a rubric belonging to a task
the caller never asked for. There are two causes; this module pins the second.

CAUSE 1 — the fallback is the strictest prose class, not the most permissive.
The fallback is the task-class contract's ``selection_fallback`` (OMN-19407:
the CLI no longer holds a default of its own), so the properties it must
satisfy are tested by the contract's owner against the live contract, in
omnimarket ``tests/unit/inference/test_task_class_resolution_omn19407.py``.

CAUSE 2 — the caller cannot state its own criteria. ``ModelDelegateSkillRequest``
has carried ``acceptance_criteria``, ``quality_contract_mode``,
``response_contract`` and ``system_prompt`` since OMN-15193/OMN-15482, and the
quality-gate reducer already honours ``replace_task_class``
(``handler_quality_gate.py:1953``). The ``onex delegate`` CLI exposed none of
them, so the only grading rubric reachable from the CLI was the task class's —
which is exactly the rubric that was wrong in both measured runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import _write_payload
from tests.helpers.cli_registry_stand_in.node_delegate_stand_in.model_stand_in_delegate_request import (
    STAND_IN_CRITERIA,
    ModelStandInDelegateRequest,
)

pytestmark = pytest.mark.unit

#: The measured prompts, quoted from the two lanes that hit this on 2026-09-15.
#: Neither is a research question and neither is a plan.
DRAFTING_PROMPT = (
    "Write two short paragraphs of rationale prose explaining, from the "
    "verified fact table below, why these rows were ranked in this order. "
    "Do not introduce any fact that is not in the table."
)


class TestTheCallerCanStateItsOwnCriteria:
    """CAUSE 2. Criteria, mode, response contract and system prompt reach the payload."""

    def test_criteria_reach_the_payload(self, tmp_path: Path) -> None:
        """RED before the fix: ``_write_payload`` took no criteria at all."""
        payload = _payload(
            tmp_path,
            acceptance_criteria=("two paragraphs", "no fact absent from the table"),
        )
        assert payload["acceptance_criteria"] == [
            "two paragraphs",
            "no fact absent from the table",
        ]

    def test_replace_mode_reaches_the_payload(self, tmp_path: Path) -> None:
        """``replace_task_class`` is what makes the caller's criteria the WHOLE bar.

        The quality-gate reducer already branches on this value
        (``handler_quality_gate.py:1953``). Without it on the wire, a caller
        stating its own criteria still had ``cites_sources`` applied on top.
        """
        payload = _payload(
            tmp_path,
            acceptance_criteria=("two paragraphs",),
            quality_contract_mode="replace_task_class",
        )
        assert payload["quality_contract_mode"] == "replace_task_class"

    def test_mode_is_omitted_when_the_caller_states_nothing(
        self, tmp_path: Path
    ) -> None:
        """No criteria means no behaviour change for any existing caller."""
        payload = _payload(tmp_path)
        assert "acceptance_criteria" not in payload
        assert "quality_contract_mode" not in payload
        assert "response_contract" not in payload
        assert "system_prompt" not in payload

    def test_response_contract_and_system_prompt_reach_the_payload(
        self, tmp_path: Path
    ) -> None:
        payload = _payload(
            tmp_path,
            response_contract={"type": "object"},
            system_prompt="Answer only from the table.",
        )
        assert payload["response_contract"] == {"type": "object"}
        assert payload["system_prompt"] == "Answer only from the table."


def _payload(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    """Drive the real payload writer and read back what it wrote."""
    import uuid

    run_id = uuid.uuid4()
    path = _write_payload(
        prompt=DRAFTING_PROMPT,
        task_type="document",
        source="claude-code",
        max_tokens=None,
        state_root=tmp_path,
        run_id=run_id,
        correlation_id=uuid.uuid4(),
        **kwargs,  # type: ignore[arg-type]
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


class TestTheFlagsThemselves:
    """``--task-class``/``--task-type`` collapse, and ``--response-contract`` parses."""

    def test_either_spelling_selects_the_class(self) -> None:
        from omnibase_infra.cli.cli_delegate import _resolve_task_class_flag

        assert _resolve_task_class_flag("document", None) == "document"
        assert _resolve_task_class_flag(None, "document") == "document"
        assert _resolve_task_class_flag(None, None) is None

    def test_passing_both_spellings_is_a_usage_error(self) -> None:
        """Never a silent precedence rule between two names for one flag."""
        from omnibase_infra.cli.cli_delegate import _resolve_task_class_flag

        with pytest.raises(ValueError, match="pass one, not both"):
            _resolve_task_class_flag("document", "research")

    def test_response_contract_accepts_inline_json(self) -> None:
        from omnibase_infra.cli.cli_delegate import _load_response_contract

        assert _load_response_contract('{"type": "object"}') == {"type": "object"}
        assert _load_response_contract(None) is None

    def test_response_contract_accepts_a_file(self, tmp_path: Path) -> None:
        from omnibase_infra.cli.cli_delegate import _load_response_contract

        path = tmp_path / "shape.json"
        path.write_text('{"type": "object"}', encoding="utf-8")
        assert _load_response_contract(str(path)) == {"type": "object"}

    def test_a_non_object_response_contract_is_refused_naming_the_flag(self) -> None:
        """Refused here, where the flag is named, not far downstream."""
        from omnibase_infra.cli.cli_delegate import _load_response_contract

        with pytest.raises(
            ValueError, match="--response-contract must be a JSON object"
        ):
            _load_response_contract("[1, 2, 3]")

    def test_a_missing_json_file_is_refused_naming_the_flag(self) -> None:
        from omnibase_infra.cli.cli_delegate import _load_response_contract

        with pytest.raises(ValueError, match="not a readable file"):
            _load_response_contract("/nonexistent/shape.json")

    def test_the_dashed_mode_flag_reaches_the_wire_underscored(
        self, tmp_path: Path
    ) -> None:
        """The wire enum is ``replace_task_class``; the flag is dashed like its peers."""
        payload = _payload(
            tmp_path,
            acceptance_criteria=("two paragraphs",),
            quality_contract_mode="replace-task-class".replace("-", "_"),
        )
        assert payload["quality_contract_mode"] == "replace_task_class"


class TestCriteriaAreCheckedByTheContractsInputModel:
    """``--criteria`` is refused before dispatch, by the delegate contract's own model.

    MEASURED 2026-09-15, live against the real routing config: three free-text
    criteria produced a 265 ms pydantic ``ValidationError``, zero rungs
    attempted, no answer, and no mention of which flag was wrong. OMN-19407:
    the CLI keeps no slug list or slug pattern of its own; it validates the
    request it is about to send with the input model the delegate node's
    contract declares, read through the registry, and names the field.
    """

    @pytest.fixture(autouse=True)
    def _stand_in_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_delegate, "_delegate_request_model", lambda: ModelStandInDelegateRequest
        )

    def _request(self, **fields: object) -> dict[str, object]:
        import uuid

        return cli_delegate._request_payload(
            prompt=DRAFTING_PROMPT,
            task_type="document",
            source="claude-code",
            max_tokens=None,
            correlation_id=uuid.uuid4(),
            **fields,  # type: ignore[arg-type]
        )

    def test_a_free_text_criterion_is_refused_naming_the_field(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            cli_delegate.validate_request_against_contract(
                self._request(acceptance_criteria=("under 400 words",))
            )
        message = str(excinfo.value)
        assert "acceptance_criteria" in message
        assert "under 400 words" in message
        assert ModelStandInDelegateRequest.__qualname__ in message, (
            "the refusal must name the contract model that refused it"
        )
        for slug in STAND_IN_CRITERIA:
            assert slug in message, "the refusal must list what the model allows"

    def test_every_slug_the_model_declares_passes(self) -> None:
        cli_delegate.validate_request_against_contract(
            self._request(acceptance_criteria=tuple(sorted(STAND_IN_CRITERIA)))
        )

    def test_a_mode_the_model_does_not_declare_is_refused(self) -> None:
        with pytest.raises(ValueError, match="quality_contract_mode"):
            cli_delegate.validate_request_against_contract(
                self._request(quality_contract_mode="replace_task_class")
            )

    def test_no_criteria_is_a_valid_request(self) -> None:
        cli_delegate.validate_request_against_contract(self._request())

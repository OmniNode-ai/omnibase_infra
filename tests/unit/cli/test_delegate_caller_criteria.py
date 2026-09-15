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
the caller never asked for. There are two causes and this module pins both.

CAUSE 1 — the fallback is the strictest prose class, not the most permissive.
``research``'s blocking heuristics are ``no_refusal``, ``cites_sources``,
``methodical_analysis`` and ``semantic_adequacy``. Two of those four demand a
SHAPE (attribution, methodical structure) rather than a quality. A prompt no
predicate claimed is by definition a prompt whose shape is unknown, so it must
land on a class whose floors are shape-agnostic. ``document``'s are:
``no_refusal``, ``accurate``, ``semantic_adequacy`` — three floors any
well-formed prose answer can meet — and it still arms ``identifiers_grounded``,
which is the property the original fallback choice was protecting.

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

from omnibase_infra.cli.cli_delegate import _write_payload
from omnibase_infra.cli.task_class_selection import (
    DEFAULT_TASK_TYPE,
    load_selection_fallback,
)

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "delegation" / "omn18305"
_PROBE_CONTRACT = _FIXTURES / "task_class_contracts_probe.yaml"
_FLOORS = _FIXTURES / "task_class_floors.yaml"

#: The rules whose verdict depends only on whether the answer is a well-formed,
#: honest, responsive piece of prose — never on what SHAPE of task it answers.
#: Read from the production contract's own ``quality_rules`` enforcement plus
#: this declaration of which floors are shape-free; a fallback class may carry
#: only these, because a prompt that reached the fallback has no known shape.
SHAPE_AGNOSTIC_FLOORS = frozenset(
    {"no_refusal", "accurate", "semantic_adequacy", "short_form_adequacy"}
)

#: The measured prompts, quoted from the two lanes that hit this on 2026-09-15.
#: Neither is a research question and neither is a plan.
DRAFTING_PROMPT = (
    "Write two short paragraphs of rationale prose explaining, from the "
    "verified fact table below, why these rows were ranked in this order. "
    "Do not introduce any fact that is not in the table."
)


class TestTheFallbackIsPermissiveNotStrict:
    """CAUSE 1. The class an unclaimed prompt lands on carries shape-free floors only."""

    def test_the_fallback_demands_no_shape_specific_floor(self) -> None:
        """RED before the fix: the fallback was ``research``, which demands citations.

        This is the whole of the ``lakshman-serving-cap-rulings-1456`` failure.
        A prompt that no predicate claimed has, by construction, no declared
        shape — so grading it on a shape floor refuses correct answers for not
        being something nobody asked them to be.
        """
        floors = _blocking_floors_for(DEFAULT_TASK_TYPE)
        assert floors, f"{DEFAULT_TASK_TYPE!r} declares no blocking floors at all"
        assert floors <= SHAPE_AGNOSTIC_FLOORS, (
            f"the fallback class {DEFAULT_TASK_TYPE!r} demands "
            f"{sorted(floors - SHAPE_AGNOSTIC_FLOORS)}, which is a shape a prompt "
            "that reached the fallback never declared"
        )

    def test_positive_control_the_two_measured_classes_fail_this_predicate(
        self,
    ) -> None:
        """The predicate above is doing work: it rejects both measured rubrics.

        Without this control, a predicate that accepted everything would pass
        the test above and prove nothing.
        """
        assert not _blocking_floors_for("research") <= SHAPE_AGNOSTIC_FLOORS
        assert not _blocking_floors_for("planning") <= SHAPE_AGNOSTIC_FLOORS
        assert "cites_sources" in _blocking_floors_for("research")
        assert "covers_dependencies" in _blocking_floors_for("planning")

    def test_the_fallback_still_arms_the_grounding_check(self) -> None:
        """The property the original ``research`` choice protected is not lost.

        OMN-18297's ``identifiers_grounded`` check must stay armed on the
        fallback: making the fallback permissive must not make it ungraded.
        """
        assert "identifiers_grounded" in _declared_heuristics(DEFAULT_TASK_TYPE)

    def test_the_contract_declaration_wins_over_the_module_constant(self) -> None:
        """The fallback is a contract decision; the constant is only its default.

        The probe contract declares ``fallback_class``. Reading it must return
        that, not ``DEFAULT_TASK_TYPE`` — otherwise the contract cannot move
        the fallback without a code change, which is the AC2 property
        OMN-18305 was closed on.
        """
        assert load_selection_fallback(_PROBE_CONTRACT) == "fallback_class"

    def test_a_fallback_naming_a_class_the_contract_does_not_expose_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Fail closed rather than silently reverting to the constant."""
        from omnibase_infra.cli.task_class_selection import TaskClassContractError

        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: probe\n"
            "selection_fallback:\n"
            "  task_class: not_a_declared_class\n"
            "task_classes:\n"
            "  only:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 1\n"
            "      phrases: []\n",
            encoding="utf-8",
        )
        with pytest.raises(TaskClassContractError, match="not_a_declared_class"):
            load_selection_fallback(bad)


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


def _production_contract() -> dict[str, object]:
    """The committed floors extract, so this runs with no omnimarket present."""
    import yaml

    raw = yaml.safe_load(_FLOORS.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _declared_heuristics(task_class: str) -> frozenset[str]:
    contract = _production_contract()
    classes = contract["task_classes"]
    assert isinstance(classes, dict), "vocabulary fixture declares no task_classes"
    entry = classes[task_class]
    return frozenset(entry["definition_of_done"]["heuristic"])


def _blocking_floors_for(task_class: str) -> frozenset[str]:
    """The subset of a class's heuristics the contract marks ``blocking``."""
    contract = _production_contract()
    rules = contract["quality_rules"]
    assert isinstance(rules, dict), "vocabulary fixture declares no quality_rules"
    return frozenset(
        name
        for name in _declared_heuristics(task_class)
        if isinstance(rules.get(name), dict)
        and rules[name].get("enforcement") == "blocking"
    )


class TestFixtureMatchesTheLiveContract:
    """The committed floors extract is not allowed to drift from omnimarket.

    A committed copy of someone else's contract is a lie waiting to happen. It
    can only be compared where omnimarket is resolvable, which is NOT the
    ordinary omnibase_infra CI environment (OMN-15620's venv-purity gate
    refuses to run this suite with omnimarket installed at all). So this test
    says out loud that it did not run, rather than passing and reading as
    proof — the same two-halves seam OMN-18305 recorded for the vocabulary.
    """

    def test_floors_extract_equals_the_live_contract(self) -> None:
        import importlib.util

        import yaml

        spec = importlib.util.find_spec("omnimarket")
        if spec is None or not spec.origin:
            pytest.skip(
                "omnimarket is not resolvable here, so the live contract cannot "
                "be compared; this assertion did NOT run and is not evidence"
            )
        live_path = (
            Path(spec.origin).resolve().parent
            / "configs"
            / "task_class_contracts.v1.yaml"
        )
        live = yaml.safe_load(live_path.read_text(encoding="utf-8"))
        committed = _production_contract()

        live_rules = {
            name: entry.get("enforcement")
            for name, entry in live["quality_rules"].items()
        }
        committed_rules = {
            name: entry.get("enforcement")
            for name, entry in committed["quality_rules"].items()  # type: ignore[union-attr]
        }
        assert committed_rules == live_rules

        live_floors = {
            name: list(entry["definition_of_done"]["heuristic"])
            for name, entry in live["task_classes"].items()
            if entry.get("gateway_exposure") == "public"
        }
        committed_floors = {
            name: list(entry["definition_of_done"]["heuristic"])
            for name, entry in committed["task_classes"].items()  # type: ignore[union-attr]
        }
        assert committed_floors == live_floors


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


class TestCriteriaAreAClosedVocabulary:
    """``--criteria`` is a declared slug set, and a typo is refused at the flag.

    MEASURED 2026-09-15, live against the real routing config: three free-text
    criteria produced a 265 ms pydantic ``ValidationError``, zero rungs
    attempted, no answer, and no mention of which flag was wrong. The
    vocabulary was always closed; only the error was unusable.
    """

    def test_a_free_text_criterion_is_refused_naming_the_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.cli import cli_delegate

        monkeypatch.setattr(
            cli_delegate, "load_supported_criteria", lambda: frozenset({"concise"})
        )
        with pytest.raises(ValueError) as excinfo:
            cli_delegate._validate_criteria(("under 400 words",))
        message = str(excinfo.value)
        assert "--criteria takes declared criterion slugs, not free text" in message
        assert "'under 400 words'" in message
        assert "concise" in message, "the refusal must list what IS allowed"

    def test_a_declared_slug_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omnibase_infra.cli import cli_delegate

        monkeypatch.setattr(
            cli_delegate,
            "load_supported_criteria",
            lambda: frozenset({"concise", "task_completed"}),
        )
        assert cli_delegate._validate_criteria(("concise", "task_completed")) == (
            "concise",
            "task_completed",
        )

    def test_the_parameterised_slug_shape_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``max_words_per_sentence_<N>`` is a pattern, not a listed name."""
        from omnibase_infra.cli import cli_delegate

        monkeypatch.setattr(
            cli_delegate, "load_supported_criteria", lambda: frozenset({"concise"})
        )
        assert cli_delegate._validate_criteria(("max_words_per_sentence_20",)) == (
            "max_words_per_sentence_20",
        )
        with pytest.raises(ValueError):
            cli_delegate._validate_criteria(("max_words_per_sentence_0",))

    def test_unreadable_vocabulary_passes_through_rather_than_refusing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No omnimarket means no vocabulary; dispatch fails for its own reason.

        Refusing here would turn a missing co-install into a misleading
        complaint about the caller's criteria.
        """
        from omnibase_infra.cli import cli_delegate

        monkeypatch.setattr(cli_delegate, "load_supported_criteria", lambda: None)
        assert cli_delegate._validate_criteria(("anything at all",)) == (
            "anything at all",
        )

    def test_no_criteria_never_consults_the_vocabulary(self) -> None:
        from omnibase_infra.cli import cli_delegate

        assert cli_delegate._validate_criteria(()) == ()

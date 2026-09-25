# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract-declared task-class selection for ``onex delegate`` (OMN-18305).

THE DEFECT. The CLI carried an ordered keyword table lifted verbatim from
retired skill markdown, matched with a bare ``in`` test over the lowercased
prompt, first rule wins, ``test`` rule first. Two consequences, both measured
live on 2026-09-13:

* the committed 56,593-byte OMN-18297 engineering standup classified as
  ``test``, because the ledger rows it summarised contain the word "test" 46
  times — and ``test`` is not one of the nine classes whose quality bar arms
  ``identifiers_grounded``, so the grounding check and the prose quality band
  never ran on the customer's answer;
* the four-word prompt ``"the latest window"`` classified as ``test``, because
  "latest" contains the substring "test".

Selection is now declared per class in ``task_class_contracts.v1.yaml`` and
read from there. These tests pin the RULES against a hand-written probe
contract so they run in an environment with no omnimarket installed; the live
production vocabulary and predicates are pinned by the drift guard in
``test_cli_delegate.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.cli.task_class_selection import (
    EnumTaskTypeResolution,
    TaskClassContractError,
    load_selectable_task_classes,
    load_task_class_admission,
    resolve_task_type,
)

pytestmark = pytest.mark.unit

_PROBE_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18305"
    / "task_class_contracts_probe.yaml"
)


@pytest.fixture(name="classes")
def _classes() -> tuple[object, ...]:
    return load_selectable_task_classes(_PROBE_CONTRACT)


class TestVocabularyComesFromTheContract:
    def test_only_gateway_exposed_classes_are_selectable(self, classes: tuple) -> None:
        """AC4: the CLI's vocabulary IS the contract's public projection."""
        assert sorted(entry.name for entry in classes) == [
            "fallback_class",
            "long_prose",
            "no_prompt_selects_this",
            "short_keyword",
        ]

    def test_a_class_missing_its_predicate_fails_closed(self, tmp_path: Path) -> None:
        """A class with no declared selection is a contract defect, not a default."""
        broken = tmp_path / "broken.yaml"
        broken.write_text(
            "task_classes:\n  orphan:\n    gateway_exposure: public\n",
            encoding="utf-8",
        )
        with pytest.raises(TaskClassContractError, match="orphan"):
            load_selectable_task_classes(broken)


class TestWordBoundaryMatching:
    def test_latest_does_not_match_test(self, classes: tuple) -> None:
        """AC3, the sharpest statement of the defect."""
        resolution = resolve_task_type(
            "the latest window", explicit=None, classes=classes
        )
        assert resolution.task_type != "short_keyword"
        assert resolution.resolution is EnumTaskTypeResolution.FALLBACK

    def test_a_whole_word_still_matches(self, classes: tuple) -> None:
        resolution = resolve_task_type(
            "write a test for this", explicit=None, classes=classes
        )
        assert resolution.task_type == "short_keyword"
        assert resolution.resolution is EnumTaskTypeResolution.CONTRACT

    def test_contract_phrases_are_case_insensitive(self, tmp_path: Path) -> None:
        contract = tmp_path / "case.yaml"
        contract.write_text(
            "task_classes:\n"
            "  standup:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 50\n"
            "      phrases: ['Standup']\n",
            encoding="utf-8",
        )
        classes = load_selectable_task_classes(contract)

        resolution = resolve_task_type(
            "write the standup", explicit=None, classes=classes
        )

        assert resolution.task_type == "standup"
        assert resolution.resolution is EnumTaskTypeResolution.CONTRACT


class TestShapeOutranksKeywordFrequency:
    def test_a_long_prompt_is_ineligible_for_a_short_class(
        self, classes: tuple
    ) -> None:
        """AC3: 40 occurrences of a keyword do not buy eligibility."""
        prompt = "test " * 40 + "summarise the window"
        resolution = resolve_task_type(prompt, explicit=None, classes=classes)
        assert resolution.task_type == "long_prose"

    def test_a_short_prompt_is_ineligible_for_a_long_class(
        self, classes: tuple
    ) -> None:
        resolution = resolve_task_type("summarise", explicit=None, classes=classes)
        assert resolution.resolution is EnumTaskTypeResolution.FALLBACK

    def test_frequency_never_decides_between_two_eligible_classes(
        self, classes: tuple
    ) -> None:
        """One occurrence of the higher-priority phrase beats many of the lower."""
        prompt = "investigate " * 30 + "summarise this and also more words here"
        resolution = resolve_task_type(prompt, explicit=None, classes=classes)
        assert resolution.task_type == "long_prose"


class TestResolutionIsAnnounced:
    def test_explicit_always_wins(self, classes: tuple) -> None:
        """AC5: an explicit flag is never second-guessed."""
        resolution = resolve_task_type(
            "summarise the window over many many many words here now",
            explicit="short_keyword",
            classes=classes,
        )
        assert resolution.task_type == "short_keyword"
        assert resolution.resolution is EnumTaskTypeResolution.EXPLICIT

    def test_an_unknown_explicit_class_is_refused_by_name(self, classes: tuple) -> None:
        with pytest.raises(TaskClassContractError, match="fallback_class"):
            resolve_task_type("anything", explicit="not_a_class", classes=classes)

    def test_the_fallback_is_named_rather_than_chosen_silently(
        self, classes: tuple
    ) -> None:
        """AMENDED, not deleted (OMN-18305 residual, 2026-09-15).

        This asserted the fallback was the module default or a contract class,
        with ``"research"`` spelled literally. The fallback is now a value the
        CALLER supplies — read from the contract's ``selection_fallback`` — so
        pinning a literal class name here would pin the very coupling the
        parameter removes. What is still worth asserting is the property: the
        fallback is announced, and it is whatever was handed in.
        """
        resolution = resolve_task_type(
            "an unremarkable sentence",
            explicit=None,
            classes=classes,
            fallback="fallback_class",
        )
        assert resolution.resolution is EnumTaskTypeResolution.FALLBACK
        assert resolution.task_type == "fallback_class"
        assert "no declared selection predicate" in resolution.reason

    def test_the_default_fallback_applies_when_the_caller_names_none(
        self, classes: tuple
    ) -> None:
        """The module default is still reachable, and it is the permissive class."""
        from omnibase_infra.cli.task_class_selection import DEFAULT_TASK_TYPE

        resolution = resolve_task_type(
            "an unremarkable sentence", explicit=None, classes=classes
        )
        assert resolution.task_type == DEFAULT_TASK_TYPE

    def test_a_contract_resolution_names_the_deciding_phrase(
        self, classes: tuple
    ) -> None:
        resolution = resolve_task_type(
            "write a test for this", explicit=None, classes=classes
        )
        assert "write a test" in resolution.reason

    def test_an_empty_phrase_list_is_never_selected(self, classes: tuple) -> None:
        """The highest-priority public class declares no phrases; it never wins."""
        resolution = resolve_task_type(
            "summarise the window over many many many words here now",
            explicit=None,
            classes=classes,
        )
        assert resolution.task_type != "no_prompt_selects_this"


class TestExplicitAdmissionCoversEveryDeclaredClass:
    """OMN-13966: an explicit class is checked against the WHOLE contract.

    Auto-selection still reads only the public projection (OMN-18305). An
    explicit ``--task-type`` is a caller naming a class, and every class the
    contract declares is a real Market class (OMN-15651), so the only explicit
    refusals are a name the contract does not declare and a class the contract
    itself declares unroutable.
    """

    @pytest.fixture(name="admission")
    def _admission(self) -> object:
        return load_task_class_admission(_PROBE_CONTRACT)

    def test_admission_partitions_every_declared_class(self, admission) -> None:
        declared = set(
            yaml.safe_load(_PROBE_CONTRACT.read_text(encoding="utf-8"))["task_classes"]
        )
        unavailable = {entry.name for entry in admission.unavailable}
        assert admission.admitted.isdisjoint(unavailable)
        assert admission.admitted | unavailable == declared
        assert unavailable == {"cannot_route_yet"}

    def test_admission_accepts_an_internal_class_by_explicit_name(
        self, classes: tuple, admission
    ) -> None:
        resolution = resolve_task_type(
            "anything at all",
            explicit="never_from_a_prompt",
            classes=classes,
            admission=admission,
        )
        assert resolution.task_type == "never_from_a_prompt"
        assert resolution.resolution is EnumTaskTypeResolution.EXPLICIT
        assert "internal" in resolution.reason

    def test_admission_never_auto_selects_an_internal_class(
        self, classes: tuple, admission
    ) -> None:
        """The internal probe class claims 'summarise' at priority 99; it still never wins."""
        resolution = resolve_task_type(
            "summarise the window over many many many words here now",
            explicit=None,
            classes=classes,
            admission=admission,
        )
        assert resolution.task_type == "long_prose"

    def test_admission_admits_a_newly_declared_class_without_a_code_change(
        self, tmp_path: Path
    ) -> None:
        """No list in this package can fall behind the contract: it is read, not mirrored."""
        contract = tmp_path / "grown.yaml"
        contract.write_text(
            _PROBE_CONTRACT.read_text(encoding="utf-8").replace(
                "task_classes:\n",
                "task_classes:\n"
                "  brand_new_class:\n"
                "    gateway_exposure: internal\n"
                "    selection:\n"
                "      priority: 0\n"
                "      phrases: []\n",
                1,
            ),
            encoding="utf-8",
        )
        resolution = resolve_task_type(
            "anything",
            explicit="brand_new_class",
            classes=load_selectable_task_classes(contract),
            admission=load_task_class_admission(contract),
        )
        assert resolution.task_type == "brand_new_class"

    def test_unavailable_class_refusal_quotes_the_contract(
        self, classes: tuple, admission
    ) -> None:
        with pytest.raises(TaskClassContractError) as refused:
            resolve_task_type(
                "anything",
                explicit="cannot_route_yet",
                classes=classes,
                admission=admission,
            )
        message = str(refused.value)
        assert "unknown" not in message
        for fragment in (
            "cannot_route_yet",
            "pending_capability",
            "probe_capability",
            "PROBE-1 WS-0",
            "No probe tier can serve this class.",
        ):
            assert fragment in message

    def test_unavailable_declaration_missing_a_field_fails_closed(
        self, tmp_path: Path
    ) -> None:
        contract = tmp_path / "half_declared.yaml"
        contract.write_text(
            "task_classes:\n"
            "  half:\n"
            "    gateway_exposure: internal\n"
            "    selection: {priority: 0, phrases: []}\n"
            "    routing_availability:\n"
            "      status: pending_capability\n",
            encoding="utf-8",
        )
        with pytest.raises(TaskClassContractError, match="half"):
            load_task_class_admission(contract)

    def test_unavailable_declaration_with_an_unknown_status_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """A status the CLI was never taught must not be guessed at either way."""
        contract = tmp_path / "new_status.yaml"
        contract.write_text(
            "task_classes:\n"
            "  odd:\n"
            "    gateway_exposure: internal\n"
            "    selection: {priority: 0, phrases: []}\n"
            "    routing_availability:\n"
            "      status: some_future_status\n"
            "      missing_capability: x\n"
            "      tracking: y\n"
            "      reason: z\n",
            encoding="utf-8",
        )
        with pytest.raises(TaskClassContractError, match="odd"):
            load_task_class_admission(contract)

    def test_admission_still_refuses_an_undeclared_class_as_unknown(
        self, classes: tuple, admission
    ) -> None:
        with pytest.raises(TaskClassContractError) as refused:
            resolve_task_type(
                "anything",
                explicit="not_a_class",
                classes=classes,
                admission=admission,
            )
        message = str(refused.value)
        assert "unknown task type 'not_a_class'" in message
        assert "never_from_a_prompt" in message
        assert "cannot_route_yet" in message


class TestDeterminism:
    def test_equal_priorities_resolve_by_class_name(self, tmp_path: Path) -> None:
        contract = tmp_path / "tie.yaml"
        contract.write_text(
            "task_classes:\n"
            "  zebra:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 50\n"
            "      phrases: ['widget']\n"
            "  alpaca:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 50\n"
            "      phrases: ['widget']\n",
            encoding="utf-8",
        )
        classes = load_selectable_task_classes(contract)
        assert (
            resolve_task_type("one widget", explicit=None, classes=classes).task_type
            == "alpaca"
        )

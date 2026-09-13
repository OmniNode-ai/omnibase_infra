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

from omnibase_infra.cli.task_class_selection import (
    EnumTaskTypeResolution,
    TaskClassContractError,
    load_selectable_task_classes,
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
        resolution = resolve_task_type(
            "an unremarkable sentence", explicit=None, classes=classes
        )
        assert resolution.resolution is EnumTaskTypeResolution.FALLBACK
        assert resolution.task_type in {entry.name for entry in classes} | {"research"}
        assert "no declared selection predicate" in resolution.reason

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

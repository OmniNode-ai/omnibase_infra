# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A short summarization request reaches the summarization class (OMN-19140).

THE DEFECT. ``summarization`` declares ``min_words: 120``, and a shape gate is
evaluated before any phrase. A prompt opening ``Summarize in one sentence:``
was therefore structurally ineligible for the class whose whole purpose is
summarising, and fell through to the ``document`` fallback. The OMN-19136
shadow-scoring run traced 15 of its 19 disagreements with the incumbent to
that one line.

WHAT THE FLOOR WAS FOR. OMN-18305, which introduced it, records no reason, so
the reason was measured rather than assumed. Every prompt in the delegation run
records was resolved twice, once with the floor and once without it. Removing
the floor moved 48 of 568 prompts to ``summarization``. The 44 claimed on the
verbs ``summarize`` or ``summarise`` were all summarization requests, and every
one of them opened with that verb. The four claimed on the noun ``summary``
were not: a request to review a one-line summary, a request to classify lanes,
a request to list test cases and a rewrite request. So the floor was keeping
the class's ordinary nouns (``summary``, ``digest``, ``what happened``,
``write up``, ``stand up``) from claiming short prompts that merely mention
them, and it keeps doing that. It also stood between the class and a prompt
too thin to hold anything to summarise.

THE FIX. The floor stays at 120 for every phrase. A new ``short_prompt`` block
admits a prompt below that floor, from its own smaller floor upwards, only
when the prompt OPENS with one of its ``opening_phrases``, the imperative
verbs. An opening phrase must also be one of the class's plain phrases, so the
same request padded past the floor is still claimed.

The falsifier table below is the shadow corpus itself, verbatim, and runs
against the production mirror that ``test_task_class_selection_omn18831.py``
pins to the live contract by digest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.cli.task_class_selection import (
    EnumTaskTypeResolution,
    ModelSelectableTaskClass,
    TaskClassContractError,
    load_selectable_task_classes,
    resolve_task_type,
)

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "delegation"
_MIRROR = _FIXTURES / "omn18831" / "task_class_selection_production_mirror.yaml"
_CORPUS = _FIXTURES / "omn19140" / "shadow_corpus.yaml"

#: The captured 9-word prompt the OMN-19136 report used as its positive control.
_CAPTURED_SHORT = "Summarize in one sentence: the broker accepted this request."

_CORPUS_ROWS: list[dict[str, object]] = yaml.safe_load(
    _CORPUS.read_text(encoding="utf-8")
)["rows"]


@pytest.fixture(name="production")
def _production() -> tuple[ModelSelectableTaskClass, ...]:
    return load_selectable_task_classes(_MIRROR)


def _resolve(prompt: str, classes: tuple[ModelSelectableTaskClass, ...]) -> str:
    return resolve_task_type(prompt, explicit=None, classes=classes).task_type


class TestTheShadowCorpus:
    """AC5: the corpus is the shadow run's own prompts, not a hand-written list."""

    def test_the_corpus_is_the_one_the_report_scored(self) -> None:
        """Positive control on the fixture: 21 rows, 15 of them the floor rows."""
        assert len(_CORPUS_ROWS) == 21
        floor_rows = [
            row
            for row in _CORPUS_ROWS
            if row["incumbent_before"] != "summarization"
            and row["shadow_label"] == "summarization"
        ]
        assert len(floor_rows) == 15
        for row in _CORPUS_ROWS:
            assert len(str(row["prompt"]).split()) == row["words"], row["row"]

    def test_every_floor_row_was_below_the_floor(self) -> None:
        """The 15 are the defect only if length alone kept them out."""
        for row in _CORPUS_ROWS:
            if row["shadow_label"] == "summarization" and (
                row["incumbent_before"] != "summarization"
            ):
                assert int(str(row["words"])) < 120, row["row"]

    @pytest.mark.parametrize(
        "row", _CORPUS_ROWS, ids=[f"row-{row['row']}" for row in _CORPUS_ROWS]
    )
    def test_each_row_resolves_as_required(
        self, production: tuple[ModelSelectableTaskClass, ...], row: dict[str, object]
    ) -> None:
        resolution = resolve_task_type(
            str(row["prompt"]), explicit=None, classes=production
        )
        assert resolution.task_type == row["expected_after"], (
            f"row {row['row']} ({row['why']}): {resolution.reason}"
        )
        if row["expected_after"] == "summarization":
            assert resolution.resolution is EnumTaskTypeResolution.CONTRACT


class TestTheShortRequest:
    """AC1 and AC2: one variable, the word count, and the same answer either side."""

    def test_the_captured_short_request_resolves_to_summarization(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        resolution = resolve_task_type(
            _CAPTURED_SHORT, explicit=None, classes=production
        )
        assert resolution.task_type == "summarization", resolution.reason
        assert resolution.resolution is EnumTaskTypeResolution.CONTRACT
        assert "at the start of a 9-word prompt" in resolution.reason

    def test_the_padded_control_still_resolves_to_summarization(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        padded = _CAPTURED_SHORT + " The broker accepted this request." * 25
        assert len(padded.split()) >= 120
        resolution = resolve_task_type(padded, explicit=None, classes=production)
        assert resolution.task_type == "summarization", resolution.reason
        assert resolution.resolution is EnumTaskTypeResolution.CONTRACT
        assert "at the start" not in resolution.reason

    def test_the_verb_in_second_position_still_counts_when_the_prompt_is_long(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """Above the floor the class claims on presence, exactly as before."""
        long_prompt = "Please summarize this. " + "The broker accepted it. " * 30
        assert _resolve(long_prompt, production) == "summarization"


class TestTheShortRequestIsCountedOnTheRequestOnly:
    """OMN-19140 composed with OMN-19523: the floor counts the request's words.

    Pasted material (a fenced block) is not part of the request, so a short
    request carrying long material is still a short request, admitted by its
    opening verb, and the reason line says so.
    """

    def test_fenced_material_does_not_lift_a_short_request_over_the_floor(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        material = "\n".join(f"line {index} of the pasted log" for index in range(60))
        prompt = f"{_CAPTURED_SHORT}\n\n```\n{material}\n```"
        assert len(prompt.split()) >= 120
        resolution = resolve_task_type(prompt, explicit=None, classes=production)
        assert resolution.task_type == "summarization", resolution.reason
        assert resolution.resolution is EnumTaskTypeResolution.CONTRACT
        assert "at the start of a 9-word prompt" in resolution.reason


class TestTheThinPromptStaysOut:
    """AC4: the negative controls. A prompt with nothing to summarise."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "Summarize this.",
            "Summarize the standup.",
            "Summarize in one sentence:",
            "Summarise what happened today.",
            "summarize the incident for me",
        ],
    )
    def test_a_thin_request_does_not_reach_summarization(
        self, production: tuple[ModelSelectableTaskClass, ...], prompt: str
    ) -> None:
        assert len(prompt.split()) < 6
        resolution = resolve_task_type(prompt, explicit=None, classes=production)
        assert resolution.task_type != "summarization", resolution.reason

    def test_the_thin_floor_is_the_only_difference(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """Control: one more word of subject and the same request is admitted."""
        assert _resolve("summarize the incident for the operator", production) == (
            "summarization"
        )


class TestTheFloorStillDoesItsJob:
    """AC3: what the 120-word floor protected is still protected."""

    @pytest.mark.parametrize(
        ("prompt", "why"),
        [
            (
                "Review this one-line summary for accuracy and reply ACCURATE or "
                "INACCURATE with one sentence of reason: 'A registry workspace now "
                "resolves its delegation transport from its own configuration, so "
                "a fresh workspace dispatches to the shared lane without any "
                "per-user setup.'",
                "the noun names the object under review, not the deliverable",
            ),
            (
                "Add a summary field to the run model.",
                "the noun in a short code-shaped request",
            ),
            (
                "Implement a helper that returns the sha256 digest of a file.",
                "the noun 'digest' in a short code request",
            ),
            (
                "Write a test for the digest helper.",
                "the noun 'digest' in a short test request",
            ),
            (
                "Implement a function that can summarize log lines.",
                "the verb mid-sentence in a short code request",
            ),
            (
                "Tell me what happened with the build yesterday.",
                "'what happened' in a short question",
            ),
        ],
    )
    def test_an_incidental_phrase_does_not_claim_a_short_prompt(
        self,
        production: tuple[ModelSelectableTaskClass, ...],
        prompt: str,
        why: str,
    ) -> None:
        assert len(prompt.split()) < 120
        resolution = resolve_task_type(prompt, explicit=None, classes=production)
        assert resolution.task_type != "summarization", f"{why}: {resolution.reason}"

    def test_the_code_requests_route_to_their_own_classes(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """Positive control: the rows above are refused for the right reason."""
        assert (
            _resolve(
                "Implement a helper that returns the sha256 digest of a file.",
                production,
            )
            == "code_generation"
        )
        assert _resolve("Write a test for the digest helper.", production) == "test"

    def test_the_class_floor_is_still_120_words(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        summarization = next(e for e in production if e.name == "summarization")
        assert summarization.min_words == 120
        assert summarization.short_prompt is not None
        assert summarization.short_prompt.opening_phrases == (
            "summarise",
            "summarize",
        )


class TestTheShortPromptBlockIsRefusedWhenUnusable:
    """A block that could not do what it says fails closed at load."""

    @staticmethod
    def _contract(tmp_path: Path, selection: str) -> Path:
        contract = tmp_path / "short.yaml"
        contract.write_text(
            "task_classes:\n"
            "  summarization:\n"
            "    gateway_exposure: public\n"
            "    selection:\n" + selection,
            encoding="utf-8",
        )
        return contract

    @pytest.mark.parametrize(
        ("selection", "why"),
        [
            (
                "      priority: 80\n"
                "      min_words: 120\n"
                "      phrases: [summary]\n"
                "      short_prompt: {min_words: 6, opening_phrases: [summarize]}\n",
                "an opening phrase the class does not claim at full length",
            ),
            (
                "      priority: 80\n"
                "      phrases: [summarize]\n"
                "      short_prompt: {min_words: 6, opening_phrases: [summarize]}\n",
                "a short-prompt floor on a class with no floor to go below",
            ),
            (
                "      priority: 80\n"
                "      min_words: 120\n"
                "      phrases: [summarize]\n"
                "      short_prompt: {min_words: 120, opening_phrases: [summarize]}\n",
                "a short-prompt floor that is not below the class floor",
            ),
            (
                "      priority: 80\n"
                "      min_words: 120\n"
                "      phrases: [summarize]\n"
                "      short_prompt: {min_words: 6, opening_phrases: []}\n",
                "no opening phrases",
            ),
        ],
    )
    def test_the_block_is_refused(
        self, tmp_path: Path, selection: str, why: str
    ) -> None:
        with pytest.raises(TaskClassContractError):
            load_selectable_task_classes(self._contract(tmp_path, selection))

    def test_a_usable_block_loads(self, tmp_path: Path) -> None:
        """Positive control: the refusals above are about the defects named."""
        classes = load_selectable_task_classes(
            self._contract(
                tmp_path,
                "      priority: 80\n"
                "      min_words: 120\n"
                "      phrases: [summarize]\n"
                "      short_prompt: {min_words: 6, opening_phrases: [summarize]}\n",
            )
        )
        assert _resolve("Summarize in one sentence: it worked.", classes) == (
            "summarization"
        )

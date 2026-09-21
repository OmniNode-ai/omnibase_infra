# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Qualifier-gated selection phrases, and the word senses they recover (OMN-18831).

THE DEFECT. Two of the declared selection phrases are ordinary English before
they are anything technical, and the evaluator had no way to tell the two
senses apart.

* ``"write a"`` claimed a 388-word prose request opening ``"Write a GitHub PR
  body in markdown from these facts."`` for ``code_generation`` (priority 30).
  ``code_generation``'s deterministic floor is ``compiles_without_errors``, so
  five rungs in a row returned correct English and were all refused with
  ``MALFORMED: response does not compile as Python``, scoring 0.433 against a
  bar of 0.850. Run ``21b33edf-32aa-4279-9c1b-52b034c2ee9e``, correlation
  ``13028f3f-0dc3-4eec-8773-78a845696e01``, 2026-09-19T14:28:31Z. The same
  facts reworded to avoid the two words classified as ``document`` and passed
  on the first local rung at 1.0 (run ``06f6b56a-8ded-4594-bf19-70c32fcc43ca``).
* ``"assertion"`` / ``"assertions"`` claimed any prose using the ordinary word
  for a claim, for ``test`` -- whose acceptance is likewise deterministic.

THE FIX, and why it is this one rather than deletion. Deleting the phrases
would regress the genuine requests: ``"write a parser"`` is a code request and
``"add assertions to the auth tests"`` is a test request, and nothing else in
either class's phrase list claims them. AC2 of the ticket asks for either no
bare-substring predicate at all, or a counter-signal the prompt already
carried. The counter-signal chosen is the OBJECT of the verb: a gated phrase
claims a prompt only when one of its class's declared ``qualifiers`` occurs
within ``within_words`` words before or after it. "write a parser" qualifies,
"write a PR body" does not, and no list of prose disqualifiers has to be
guessed at. The qualifier vocabulary is the word sense itself, so it is
declared in the contract and not here.

WHAT THIS FILE PINS. The mechanism, against a hand-written contract; and the
falsifier table, against a mirror of the production predicates. The mirror is
necessary because a routing answer is a statement about every class's
predicate and priority at once, not only the one that used to win -- a table
run against the probe contract next door would prove nothing about the live
vocabulary. The mirror's own fidelity is pinned by the digest below, whose
other half lives in omnimarket.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from omnibase_infra.cli.task_class_selection import (
    DEFAULT_TASK_TYPE,
    EnumTaskTypeResolution,
    ModelSelectableTaskClass,
    load_selectable_task_classes,
    resolve_task_type,
)

pytestmark = pytest.mark.unit

_MIRROR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18831"
    / "task_class_selection_production_mirror.yaml"
)

#: Digest of the canonical public selection projection.
#:
#: THE OTHER HALF OF THIS ASSERTION LIVES IN OMNIMARKET:
#: ``tests/unit/inference/test_task_class_selection_omn18831.py`` computes the
#: same digest over the LIVE contract. Neither suite can import the other's
#: half, so the seam is two pinned halves of one constant: editing the contract
#: turns omnimarket red and prints the new digest, and updating the mirror here
#: is the fix. A digest matching on both sides is what makes the falsifier
#: table below a statement about production rather than about a fixture.
PRODUCTION_SELECTION_DIGEST = (
    "6d43ffef9e8aef89eba02b61da356adec3e74ef53e439a6a819cfa14aaa85819"
)

#: The opening sentence is quoted verbatim from the run's own stderr. The
#: remaining facts are a RECONSTRUCTION -- the ticket records the length (388
#: words) and the shape, not the body -- so the verbatim sentence is asserted
#: on its own as well, where nothing is reconstructed.
_VERBATIM_OPENING = (
    "Write a GitHub PR body in markdown from these facts. "
    "No preamble, no commentary, output only the body."
)

_RECONSTRUCTED_388_WORD_REQUEST = (
    _VERBATIM_OPENING
    + " "
    + (
        "The change lands on the delivery lane and touches one workflow file. "
        "It adds a gate between the candidate boot step and the announcement, "
        "so an announcement cannot go out for a commit nothing has ever run. "
        "The receipt surface is an artifact in the same repository, named after "
        "the commit, and the gate asks for that one name. Absent, unreadable, "
        "malformed and wrong-version all refuse, each naming the commit. There is "
        "no force input and no skip input, and a positive control covers the "
        "refusal so the gate cannot pass by doing nothing. "
    )
    * 6
)

# ---------------------------------------------------------------------------
# The falsifier table.
#
# Every row is (prompt, expected class, why). The PROSE rows are the defect:
# each one carries a gated phrase and no qualifier near it, and each landed on
# a deterministic-acceptance class before this change. The CODE and TEST rows
# are the positive controls: each carries the same gated phrase WITH a
# qualifier, and each must still route exactly where it routed before, which
# is what makes this a word-sense fix rather than a deletion.
# ---------------------------------------------------------------------------

_PROSE_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        _VERBATIM_OPENING,
        "document",
        "the OMN-18831 prompt, verbatim: 'write a' with no code artifact near it",
    ),
    (
        _RECONSTRUCTED_388_WORD_REQUEST,
        "document",
        "the same request at its recorded length, inside code_generation's 600-word gate",
    ),
    (
        "Write an update for the operator covering last night's outage.",
        "document",
        "'write an' before an ordinary noun",
    ),
    (
        "Create a short agenda for tomorrow's planning meeting with the team.",
        "document",
        "'create a' before an ordinary noun",
    ),
    (
        "Build a case for doing the observability work before the launch.",
        "document",
        "'build a' before an ordinary noun",
    ),
    (
        "Generate a list of questions to ask the customer on the call.",
        "document",
        "'generate' before an ordinary noun",
    ),
    (
        "The report's assertions are unsupported by the evidence it cites.",
        "document",
        "'assertions' in its ordinary English sense",
    ),
    (
        "Weigh whether that assertion about the outage holds up.",
        "document",
        "'assertion' in its ordinary English sense",
    ),
    # OMN-19017. Both rows are the captured prompts, abridged to the sentences
    # that carry the phrase, from correlations
    # 04ac362b-8d9d-4208-b251-73a9a090d2a7 and
    # e811f3e5-1ccf-42be-9d88-707ed1923430. Each asked for a list described in
    # words, each was claimed by the bare phrase "test cases" at priority 40,
    # and each was then refused by a class that requires the answer to compile
    # as Python -- deterministically, on every rung, four lanes in one hour.
    (
        # VERBATIM from the receipt, not abridged. An earlier draft of this
        # row shortened the middle and the shortened form routed to `test`,
        # because dropping words moved "binding-class" to within the
        # qualifier window of "test cases" and the qualifier `class` matches
        # inside that hyphenated compound. The abridgement changed the answer,
        # which is the whole reason a captured prompt is quoted rather than
        # summarised. The residual it exposed is pinned as its own row below.
        "Enumerate test cases for one rule. A CI receipt may be re-emitted "
        "only when every one of its failing checks is binding-class, meaning "
        "the run could not bind its observation to the thing under test "
        "rather than the thing under test being unhealthy. Binding-class: "
        "convergence unestablished, probe read a different container "
        "generation. Not binding-class: readiness failed, migrations "
        "missing, consumer lag over bound. List the test cases that "
        "distinguish these two, including the ones most likely to be "
        "forgotten.",
        "document",
        "OMN-19017: 'test cases' asking for prose, no code term near it",
    ),
    (
        "Enumerate RED test cases for a stale-cache defect in a Kafka deploy "
        "agent. Context: the agent has one serial loop and lag sampling "
        "happens only inside the first half of it.",
        "document",
        "OMN-19017: the second captured prompt, same shape",
    ),
    (
        "List the test cases most likely to be forgotten for this rule.",
        "document",
        "OMN-19017: the shortest form of the same request",
    ),
)

_CODE_AND_TEST_ROWS: tuple[tuple[str, str, str], ...] = (
    # OMN-19017 positive controls. Each carries the newly gated phrase WITH a
    # qualifier, so each must still route to the code class. Without these the
    # prose rows above would also pass if the phrase had simply been deleted,
    # which is the failure mode a word-sense fix has to rule out.
    (
        "Add test cases to the pytest module for the trailing comma bug.",
        "test",
        "OMN-19017: 'test cases' qualified by 'pytest'/'module'",
    ),
    (
        "Write test cases for the function that parses the header.",
        "test",
        "OMN-19017: 'test cases' qualified by 'function'",
    ),
    (
        "Extend the test cases and the fixture that covers retries.",
        "test",
        "OMN-19017: 'test cases' qualified by 'fixture'",
    ),
    (
        "Write a parser for the lane manifest file.",
        "code_generation",
        "'write a' qualified by 'parser'",
    ),
    (
        "Write an adapter that speaks the projection protocol.",
        "code_generation",
        "'write an' qualified by 'adapter'",
    ),
    (
        "Create an endpoint that returns the readiness projection.",
        "code_generation",
        "'create a'/'create an' qualified by 'endpoint'",
    ),
    (
        "Build a CLI subcommand that drains the queue.",
        "code_generation",
        "'build a' qualified by 'cli'",
    ),
    (
        "Generate the migration for the new tenant column.",
        "code_generation",
        "'generate' qualified by 'migration'",
    ),
    (
        "Implement the retry policy on the dispatch port.",
        "code_generation",
        "'implement' is unqualified and must keep claiming on its own",
    ),
    (
        "Add assertions to the auth tests.",
        "test",
        "'assertions' qualified by 'tests' -- the row deletion would have broken",
    ),
    (
        "The assertions in this pytest module are checking the wrong field.",
        "test",
        "'assertions' qualified by 'pytest'",
    ),
    (
        "Write a test for the task-class resolver.",
        "test",
        "test outranks code_generation on priority, exactly as before",
    ),
)


@pytest.fixture(name="production")
def _production() -> tuple[ModelSelectableTaskClass, ...]:
    return load_selectable_task_classes(_MIRROR)


def _canonical_projection(contract_path: Path) -> str:
    """Return the canonical public selection projection, as JSON.

    Duplicated verbatim in omnimarket's half. The duplication is the point: a
    shared helper would have to live in one repo and be imported by the other,
    which is the import neither repo can make.
    """
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    projection: dict[str, object] = {}
    for name, entry in raw["task_classes"].items():
        if not isinstance(entry, dict) or entry.get("gateway_exposure") != "public":
            continue
        selection = entry["selection"]
        qualified = selection.get("qualified_phrases")
        projection[str(name)] = {
            "priority": int(selection["priority"]),
            "min_words": selection.get("min_words"),
            "max_words": selection.get("max_words"),
            "phrases": sorted(str(item) for item in (selection.get("phrases") or ())),
            "qualified_phrases": None
            if qualified is None
            else {
                "within_words": int(qualified["within_words"]),
                "phrases": sorted(str(item) for item in qualified["phrases"]),
                "qualifiers": sorted(str(item) for item in qualified["qualifiers"]),
            },
        }
    return json.dumps(projection, sort_keys=True, separators=(",", ":"))


class TestTheMirrorIsTheProductionProjection:
    def test_the_mirror_hashes_to_the_pinned_digest(self) -> None:
        """Half of a two-repo seam; omnimarket hashes the live contract to this."""
        digest = hashlib.sha256(
            _canonical_projection(_MIRROR).encode("utf-8")
        ).hexdigest()
        assert digest == PRODUCTION_SELECTION_DIGEST

    def test_the_mirror_carries_every_public_class(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        assert sorted(entry.name for entry in production) == [
            "code_generation",
            "code_review",
            "complex_reasoning",
            "document",
            "planning",
            "reasoning",
            "refactor",
            "research",
            "review",
            "summarization",
            "test",
        ]

    def test_the_fallback_is_the_prose_class(self) -> None:
        """Every prose row below lands on the fallback, so it must be prose."""
        assert DEFAULT_TASK_TYPE == "document"


class TestTheProseRequestsAreNoLongerClaimed:
    """AC1 and AC2: the misroutes, each as its own row."""

    @pytest.mark.parametrize(
        ("prompt", "expected", "why"),
        _PROSE_ROWS,
        ids=[row[2] for row in _PROSE_ROWS],
    )
    def test_prose_routes_to_a_prose_class(
        self,
        production: tuple[ModelSelectableTaskClass, ...],
        prompt: str,
        expected: str,
        why: str,
    ) -> None:
        resolution = resolve_task_type(prompt, explicit=None, classes=production)
        assert resolution.task_type == expected, f"{why}: {resolution.reason}"

    def test_the_recorded_prompt_reaches_no_deterministic_acceptance_class(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """The specific harm: a prose answer graded on Python compilation."""
        resolution = resolve_task_type(
            _VERBATIM_OPENING, explicit=None, classes=production
        )
        assert resolution.task_type not in {"code_generation", "test", "refactor"}

    def test_a_hyphenated_compound_still_qualifies_the_gated_phrase(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """KNOWN RESIDUAL, pinned deliberately so it is visible, not latent.

        A qualifier is matched on word boundaries, and a hyphen is a word
        boundary, so the qualifier ``class`` matches inside ``binding-class``.
        A prose request carrying such a compound within the qualifier window
        of ``test cases`` is therefore still claimed by the code-floored
        class, even though nothing about it is a code request.

        This is NOT introduced by OMN-19017. The same matching rule already
        governed ``assertion``/``assertions`` under OMN-18831. What OMN-19017
        changes is the exposure: ``test cases`` is a far more common phrase
        than ``assertion``, so the pre-existing imprecision now has more
        surface to act on.

        It is asserted in its CURRENT direction rather than the desired one.
        A test asserting the behaviour we want, over code that does not do it,
        is a failing test, and a skipped test is one nobody reads. Pinned this
        way, the day the matcher learns that a hyphenated compound is not the
        qualifier, this test goes red and names its own successor.

        Both captured prompts pass anyway, because in each the nearest such
        compound falls outside the window. That is proximity rather than
        design, and this row is what stops that distinction being lost.
        """
        leaking = (
            "Enumerate the test cases that distinguish a binding-class "
            "failure from an unhealthy one."
        )

        resolution = resolve_task_type(leaking, explicit=None, classes=production)

        assert resolution.task_type == "test", (
            "the residual this row pins has been fixed; the qualifier no "
            "longer matches inside a hyphenated compound. Delete this test "
            "and move the prompt into the prose rows, where it belongs: "
            f"{resolution.reason}"
        )

    def test_the_same_request_without_the_compound_is_prose(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """The control that proves the row above is about the compound.

        One word different. Without it there is no qualifier in the window
        and the request routes to prose, which is what isolates the cause to
        the hyphenated compound rather than to the sentence, its length or
        its shape.
        """
        resolution = resolve_task_type(
            "Enumerate the test cases that distinguish a binding failure "
            "from an unhealthy one.",
            explicit=None,
            classes=production,
        )

        assert resolution.task_type == "document", resolution.reason

    def test_the_two_word_difference_no_longer_changes_the_class(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """AC1's falsifier pair, stated as the equality it should always have been."""
        with_phrase = resolve_task_type(
            _VERBATIM_OPENING, explicit=None, classes=production
        )
        without_phrase = resolve_task_type(
            "Produce a GitHub PR body in markdown from these facts. "
            "No preamble, no commentary, output only the body.",
            explicit=None,
            classes=production,
        )
        assert with_phrase.task_type == without_phrase.task_type == "document"


class TestTheGenuineRequestsStillRoute:
    """The positive controls. Without these the fix is indistinguishable from
    deleting the phrases, which is the outcome the ticket rules out."""

    @pytest.mark.parametrize(
        ("prompt", "expected", "why"),
        _CODE_AND_TEST_ROWS,
        ids=[row[2] for row in _CODE_AND_TEST_ROWS],
    )
    def test_a_qualified_phrase_still_claims_the_prompt(
        self,
        production: tuple[ModelSelectableTaskClass, ...],
        prompt: str,
        expected: str,
        why: str,
    ) -> None:
        resolution = resolve_task_type(prompt, explicit=None, classes=production)
        assert resolution.task_type == expected, f"{why}: {resolution.reason}"
        assert resolution.resolution is EnumTaskTypeResolution.CONTRACT


class TestTheQualifierRule:
    """The mechanism, against a hand-written contract rather than production."""

    @staticmethod
    def _contract(tmp_path: Path, *, within_words: int = 3) -> Path:
        contract = tmp_path / "qualified.yaml"
        contract.write_text(
            "task_classes:\n"
            "  gated:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 50\n"
            "      phrases: ['implement']\n"
            "      qualified_phrases:\n"
            f"        within_words: {within_words}\n"
            "        phrases: ['write a']\n"
            "        qualifiers: ['parser', 'unit test']\n"
            "  plain:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 10\n"
            "      phrases: ['widget']\n",
            encoding="utf-8",
        )
        return contract

    def test_a_qualifier_after_the_phrase_qualifies_it(self, tmp_path: Path) -> None:
        classes = load_selectable_task_classes(self._contract(tmp_path))
        assert (
            resolve_task_type(
                "write a parser please", explicit=None, classes=classes
            ).task_type
            == "gated"
        )

    def test_a_qualifier_before_the_phrase_qualifies_it(self, tmp_path: Path) -> None:
        """Order is not the signal; proximity is."""
        classes = load_selectable_task_classes(self._contract(tmp_path))
        assert (
            resolve_task_type(
                "update the parser, then write a replacement",
                explicit=None,
                classes=classes,
            ).task_type
            == "gated"
        )

    def test_no_qualifier_leaves_the_phrase_inert(self, tmp_path: Path) -> None:
        classes = load_selectable_task_classes(self._contract(tmp_path))
        resolution = resolve_task_type(
            "write a note to the team", explicit=None, classes=classes
        )
        assert resolution.resolution is EnumTaskTypeResolution.FALLBACK

    def test_a_qualifier_beyond_the_window_does_not_qualify(
        self, tmp_path: Path
    ) -> None:
        """The window is a declared number, and it is enforced."""
        classes = load_selectable_task_classes(self._contract(tmp_path, within_words=3))
        resolution = resolve_task_type(
            "write a note that we will send once the parser lands",
            explicit=None,
            classes=classes,
        )
        assert resolution.resolution is EnumTaskTypeResolution.FALLBACK

    def test_a_later_qualified_occurrence_still_counts(self, tmp_path: Path) -> None:
        """Presence, not first occurrence: one unqualified use must not veto."""
        classes = load_selectable_task_classes(self._contract(tmp_path))
        assert (
            resolve_task_type(
                "write a note now, and later write a parser for it",
                explicit=None,
                classes=classes,
            ).task_type
            == "gated"
        )

    def test_a_multiword_qualifier_matches_on_word_boundaries(
        self, tmp_path: Path
    ) -> None:
        classes = load_selectable_task_classes(self._contract(tmp_path))
        assert (
            resolve_task_type(
                "write a unit test for it", explicit=None, classes=classes
            ).task_type
            == "gated"
        )

    def test_an_unqualified_phrase_in_the_same_class_is_unaffected(
        self, tmp_path: Path
    ) -> None:
        classes = load_selectable_task_classes(self._contract(tmp_path))
        assert (
            resolve_task_type(
                "implement the thing", explicit=None, classes=classes
            ).task_type
            == "gated"
        )

    def test_a_qualifier_inside_the_phrase_itself_does_not_self_qualify(
        self, tmp_path: Path
    ) -> None:
        """A phrase cannot be its own counter-signal."""
        contract = tmp_path / "selfqual.yaml"
        contract.write_text(
            "task_classes:\n"
            "  gated:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 50\n"
            "      phrases: []\n"
            "      qualified_phrases:\n"
            "        within_words: 4\n"
            "        phrases: ['write a parser']\n"
            "        qualifiers: ['parser']\n",
            encoding="utf-8",
        )
        classes = load_selectable_task_classes(contract)
        resolution = resolve_task_type("write a parser", explicit=None, classes=classes)
        assert resolution.resolution is EnumTaskTypeResolution.FALLBACK

    def test_a_contract_with_no_qualified_phrases_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        """The field is optional, so every contract written before it still loads."""
        contract = tmp_path / "old.yaml"
        contract.write_text(
            "task_classes:\n"
            "  plain:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 10\n"
            "      phrases: ['widget']\n",
            encoding="utf-8",
        )
        classes = load_selectable_task_classes(contract)
        assert classes[0].qualified_phrases is None
        assert (
            resolve_task_type("one widget", explicit=None, classes=classes).task_type
            == "plain"
        )


class TestTheContractIsRefusedRatherThanDefaulted:
    @pytest.mark.parametrize(
        ("body", "match"),
        [
            (
                "        within_words: 0\n        phrases: ['x']\n        qualifiers: ['y']\n",
                "within_words",
            ),
            (
                "        within_words: 3\n        phrases: []\n        qualifiers: ['y']\n",
                "phrases",
            ),
            (
                "        within_words: 3\n        phrases: ['x']\n        qualifiers: []\n",
                "qualifiers",
            ),
        ],
        ids=["a zero window", "no gated phrases", "no qualifiers"],
    )
    def test_an_unusable_qualified_block_fails_closed(
        self, tmp_path: Path, body: str, match: str
    ) -> None:
        """A gate that cannot gate is a contract defect, never a silent pass-through.

        Each of these would otherwise degrade to 'the phrase matches
        unconditionally', which is the defect this field exists to remove.
        """
        contract = tmp_path / "broken.yaml"
        contract.write_text(
            "task_classes:\n"
            "  gated:\n"
            "    gateway_exposure: public\n"
            "    selection:\n"
            "      priority: 50\n"
            "      phrases: []\n"
            "      qualified_phrases:\n" + body,
            encoding="utf-8",
        )
        with pytest.raises(Exception, match=match):
            load_selectable_task_classes(contract)

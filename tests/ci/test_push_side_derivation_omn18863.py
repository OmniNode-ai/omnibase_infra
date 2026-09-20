# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The PR-time double derivation, and the wiring that makes it load-bearing.

OMN-18863. The enforcement job derives the application-database TABLE grants
from whichever omnimarket tree it checked out, and which tree that is depends
on the EVENT: a pull request declaring ``Node-Migration-Source-*`` trailers
gets the branch those trailers name, a push to ``dev`` gets the committed pin.
So a vendoring pull request could pass its own required gate and red that same
gate for every other open pull request one push later.

Measured twice in nine minutes on 2026-09-19/20, both times on byte-identical
trees, both times blocking the whole repository until an unrelated lane
diagnosed it. The fix makes the trailered pull request answer the push-side
question as well, so the cost lands on the change that opens the window.

These tests cover the three things that can rot:

* the REFUSAL fires when the pin-side derivation fails, and does not when it
  passes;
* the refusal NAMES the remedy, and names the wrong remedy as wrong, because
  three lanes in one night read the sibling check's text and concluded that a
  plain regeneration was the fix;
* the WIRING is present and conditioned correctly, since a script nothing calls
  enforces nothing.

The live positive controls -- replaying the two pull requests that caused the
outages and showing the new check red on both, and #3867's head green -- are
recorded in the pull request body against real shas. They cannot be unit tests
without vendoring three foreign trees into this repository, which would make
this module the thing it is guarding against.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "assert_push_side_derivation.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_GATE_JOB = "application-database-domain-enforcement"

_PINNED_CONTRACTS = (
    _REPO_ROOT / ".proof-dependencies" / "omnimarket" / "src" / "omnimarket" / "nodes"
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )


def _gate_steps() -> list[dict[str, object]]:
    document = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = document["jobs"][_GATE_JOB]["steps"]
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


class TestTheCheckFailsClosed:
    def test_a_missing_pin_checkout_is_a_failure_not_a_pass(self) -> None:
        """The vacuous-green this entire ticket is about, refused explicitly.

        An absent checkout and a passing derivation are indistinguishable to a
        caller that only reads the exit code, so the script decides rather than
        letting the caller assume.
        """
        result = _run("--pin-contracts-root", str(_REPO_ROOT / "does-not-exist"))
        assert result.returncode == 1
        assert "fails closed" in result.stderr
        assert "was not run is not a derivation that passed" in result.stderr

    def test_the_root_is_required(self) -> None:
        """No default. A default would be a guess about which tree to trust."""
        result = _run()
        assert result.returncode != 0
        assert "--pin-contracts-root" in result.stderr


class TestTheRefusalNamesTheRemedy:
    """The expensive half of the defect was a confident, wrong instruction."""

    def test_it_says_not_to_regenerate(self) -> None:
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "DO NOT regenerate the grants to make this pass" in source

    def test_it_says_what_regenerating_would_break(self) -> None:
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "OMN-18768" in source
        assert "boot crash" in source

    def test_it_names_both_halves_of_the_bridge(self) -> None:
        """A declaration without its expiry entry is how five went stale."""
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "LEGACY_MIGRATION_TABLE_DECLARATIONS" in source
        assert "_INTERIM_ENTRIES" in source

    def test_it_cites_the_precedent_blocks(self) -> None:
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "OMN-18159" in source
        assert "OMN-17426" in source


class TestTheSiblingFailureTextWasCorrectedToo:
    """Three lanes read the OTHER message and were sent the wrong way."""

    def test_the_contract_pin_assertion_warns_against_regeneration(self) -> None:
        text = (
            _REPO_ROOT / "tests" / "ci" / "test_omnimarket_contract_pin.py"
        ).read_text(encoding="utf-8")
        assert "It does NOT mean " in text
        assert "regenerate the grants" in text

    def test_it_still_states_the_ordinary_case(self) -> None:
        """Not a replacement. Advancing the pin IS usually the right answer."""
        text = (
            _REPO_ROOT / "tests" / "ci" / "test_omnimarket_contract_pin.py"
        ).read_text(encoding="utf-8")
        assert "the pin needs advancing" in text


class TestTheWiringIsPresentAndConditioned:
    """A script nothing calls enforces nothing."""

    def test_the_gate_job_invokes_the_script(self) -> None:
        invocations = [
            step
            for step in _gate_steps()
            if "assert_push_side_derivation.py" in str(step.get("run", ""))
        ]
        assert len(invocations) == 1, (
            "the OMN-18863 double derivation must be invoked exactly once by "
            f"{_GATE_JOB}; found {len(invocations)}"
        )

    def test_it_runs_only_on_a_trailered_pull_request(self) -> None:
        """Without trailers both sides already derive from the pin.

        Running it there would be a second identical derivation: slower, and
        it would make the condition look load-bearing when it is not.
        """
        step = next(
            step
            for step in _gate_steps()
            if "assert_push_side_derivation.py" in str(step.get("run", ""))
        )
        assert step.get("if") == "steps.resolve-omnimarket-ref.outputs.ref != 'dev'"

    def test_the_pin_mirror_is_checked_out_under_the_same_condition(self) -> None:
        step = next(
            step
            for step in _gate_steps()
            if str(step.get("with", {}).get("path", "")).endswith(
                ".proof-dependencies/omnimarket-pin"
            )
        )
        assert step.get("if") == "steps.resolve-omnimarket-ref.outputs.ref != 'dev'"
        assert step["with"]["ref"] == "${{ steps.resolve-omnimarket-pin.outputs.sha }}"

    def test_the_pin_resolution_is_unconditional(self) -> None:
        """The trailered path needs the pin sha too, so it cannot be gated.

        This is the one-line change that makes the rest possible, and it is the
        line most likely to be 'tidied' back by someone who reads the original
        condition as an optimisation.
        """
        step = next(
            step for step in _gate_steps() if step.get("id") == "resolve-omnimarket-pin"
        )
        assert "if" not in step

    def test_the_trailer_resolved_derivation_is_untouched(self) -> None:
        """The existing check is the other half and must not be weakened."""
        step = next(
            step
            for step in _gate_steps()
            if str(step.get("with", {}).get("path", "")).endswith(
                ".proof-dependencies/omnimarket"
            )
            and not str(step.get("with", {}).get("path", "")).endswith("-pin")
        )
        assert step["with"]["ref"] == (
            "${{ steps.resolve-omnimarket-ref.outputs.ref == 'dev' "
            "&& steps.resolve-omnimarket-pin.outputs.sha "
            "|| steps.resolve-omnimarket-ref.outputs.ref }}"
        )


@pytest.mark.skipif(
    not _PINNED_CONTRACTS.is_dir(),
    reason=(
        "requires the pinned omnimarket checkout at .proof-dependencies/omnimarket, "
        "which the OMN-15361 enforcement job provides and a bare local run does not"
    ),
)
class TestItPassesOnATreeThatIsInSync:
    def test_the_current_tree_passes_against_its_own_contracts(self) -> None:
        """Negative control. Without it the refusals above could be constant."""
        result = _run("--pin-contracts-root", str(_PINNED_CONTRACTS))
        assert result.returncode == 0, result.stderr
        assert "will not red dev on merge" in result.stdout

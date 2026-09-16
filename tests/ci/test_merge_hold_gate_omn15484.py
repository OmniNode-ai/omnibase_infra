# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15484: the Merge Hold Gate must be wired into a REQUIRED context here.

OMN-15483 built a required check that fails while a PR carries a hold marker, so
the merge sweep cannot land it inside the adversarial-verification window — and
shipped it in ``omnimarket`` only. ``omnibase_infra`` carries incident §C (#2560)
in that ticket's table and had zero coverage.

The fan-out adds a ``merge-hold-gate`` job calling the shared reusable workflow
in omnimarket, plus one entry in ``STRICT_GATE_JOBS``. Two things are proven
here, and they are different in kind:

* **The strict registration behaves as claimed** — by driving this repo's REAL
  ``ci_summary_gate.evaluate()`` over each possible job result. Registration,
  not existence, is the mechanism: the poller's default-deny sweep already
  catches a hold job that FAILS, so a test that only checked ``failure`` would
  pass identically with the registration deleted. The load-bearing vectors are
  ``skipped`` and ``absent``, and each is asserted against its unregistered
  control so the assertion cannot be vacuous.
* **The ci.yml wiring produces the exact string that is registered** — the
  check-run for a reusable call is ``<caller job display name> / <inner job
  display name>``. If either half is renamed the required ``CI Summary`` context
  goes permanently PENDING, which is a repo-wide outage, not a soft failure.

Not proven here, deliberately: the hold vocabulary itself, and that a held title
produces exit 1. Those live in omnimarket — one definition, fleet-wide — and are
re-proven live in THIS repo's CI on every run by the shared workflow's own
self-proof step. Asserting them here would require a local copy of the
vocabulary, which is precisely what OMN-15484 AC1 forbids.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    evaluate,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_YAML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

_JOB_ID = "merge-hold-gate"
_INNER_JOB_ID = "evaluate"
_CONTEXT = f"{_JOB_ID} / {_INNER_JOB_ID}"
_REUSABLE = "OmniNode-ai/omnimarket/.github/workflows/merge-hold-gate-reusable.yml"


def _job(
    name: str, conclusion: str | None, *, status: str = "completed"
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "run_attempt": 1,
    }


def _all_gates_green() -> list[dict[str, object]]:
    """Every strict + skippable gate present and successful."""
    return [_job(g, "success") for g in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)]


def _with_hold(
    conclusion: str | None, *, status: str = "completed"
) -> list[dict[str, object]]:
    """A fully green snapshot in which only the hold gate has the given state."""
    jobs = [j for j in _all_gates_green() if j["name"] != _CONTEXT]
    jobs.append(_job(_CONTEXT, conclusion, status=status))
    return jobs


def _without_hold() -> list[dict[str, object]]:
    """A fully green snapshot in which the hold gate never reported at all."""
    return [j for j in _all_gates_green() if j["name"] != _CONTEXT]


# ---------------------------------------------------------------------------
# AC2 — strict registration, driven through the real evaluator
# ---------------------------------------------------------------------------


class TestStrictRegistration:
    def test_the_hold_gate_is_registered_strict(self) -> None:
        assert _CONTEXT in STRICT_GATE_JOBS
        assert _CONTEXT not in SKIPPABLE_GATE_JOBS, (
            "a skippable slot accepts `skipped` as good, which is the exact "
            "bypass this registration exists to close"
        )

    def test_green_hold_gate_is_success(self) -> None:
        code, report = evaluate(_with_hold("success"))
        assert code == EXIT_SUCCESS, report

    def test_skipped_hold_gate_is_failure(self) -> None:
        """The load-bearing vector.

        The job is unconditional, so a `skipped` conclusion means something went
        wrong — not that the gate legitimately opted out. Unregistered, this is
        SUCCESS (see the control below), and a held PR is required-green.
        """
        code, report = evaluate(_with_hold("skipped"))
        assert code == EXIT_FAILURE, report
        assert _CONTEXT in report

    def test_absent_hold_gate_is_pending_not_success(self) -> None:
        """Deleting the job must not silently re-open the gap.

        PENDING blocks the merge without asserting a false green. Unregistered,
        an absent job is invisible to the poller and CI Summary reports SUCCESS.
        """
        code, report = evaluate(_without_hold())
        assert code == EXIT_PENDING, report
        assert _CONTEXT in report

    @pytest.mark.parametrize("conclusion", ["failure", "cancelled"])
    def test_non_success_conclusions_fail(self, conclusion: str) -> None:
        code, report = evaluate(_with_hold(conclusion))
        assert code == EXIT_FAILURE, report

    def test_incomplete_hold_gate_is_pending(self) -> None:
        """Still running is not yet clear — the completeness anchor holds."""
        code, _ = evaluate(_with_hold(None, status="in_progress"))
        assert code == EXIT_PENDING

    # -- the controls that make the above non-vacuous ------------------------

    def test_control_unregistered_skipped_would_pass(self) -> None:
        """RED-before, expressed as a control rather than a claim.

        Same real evaluator, same snapshot, with the hold gate removed from the
        strict tuple: a skipped hold gate yields SUCCESS. This is the state
        omnibase_infra was in before this PR, and it is what makes the
        `skipped -> FAILURE` assertion above evidence rather than decoration.
        """
        unregistered = tuple(g for g in STRICT_GATE_JOBS if g != _CONTEXT)
        code, _ = evaluate(_with_hold("skipped"), strict_gates=unregistered)
        assert code == EXIT_SUCCESS

    def test_control_unregistered_absent_would_pass(self) -> None:
        unregistered = tuple(g for g in STRICT_GATE_JOBS if g != _CONTEXT)
        code, _ = evaluate(_without_hold(), strict_gates=unregistered)
        assert code == EXIT_SUCCESS

    def test_control_unregistered_failure_still_fails(self) -> None:
        """Why `failure` alone would have been a worthless test.

        The default-deny sweep catches any present+completed non-good job, so
        `failure` fails with OR without the registration. A test suite that only
        covered `failure` would pass unchanged if someone deleted the strict
        entry.
        """
        unregistered = tuple(g for g in STRICT_GATE_JOBS if g != _CONTEXT)
        code, _ = evaluate(_with_hold("failure"), strict_gates=unregistered)
        assert code == EXIT_FAILURE

    def test_the_hold_gate_is_not_soft_allowlisted(self) -> None:
        """The allowlist is prefix-aware, so a caller-segment entry would cover it."""
        from scripts.ci.ci_summary_gate import SOFT_ALLOWLIST, _is_allowlisted

        assert not _is_allowlisted(_CONTEXT, SOFT_ALLOWLIST)


# ---------------------------------------------------------------------------
# The ci.yml wiring must produce exactly the registered string
# ---------------------------------------------------------------------------


class TestCiWiring:
    @staticmethod
    def _workflow() -> dict[str, Any]:
        return dict(yaml.safe_load(_CI_YAML.read_text(encoding="utf-8")))

    @staticmethod
    def _hold_job() -> dict[str, Any]:
        jobs = TestCiWiring._workflow()["jobs"]
        assert _JOB_ID in jobs, f"{_JOB_ID} is not a job in ci.yml"
        return dict(jobs[_JOB_ID])

    def test_the_job_key_is_the_first_segment_of_the_registered_context(self) -> None:
        """A `name:` here would override the job id and unwire the registration.

        The composed check-run is `<caller display name> / <inner display
        name>`; with no `name:` the caller's display name IS its job id. This is
        the trap already documented on `deploy-agent-tests`, made mechanical.
        """
        assert _CONTEXT.split(" / ")[0] == _JOB_ID
        assert "name" not in self._hold_job(), (
            "adding a `name:` changes the check-run context and makes the "
            "required CI Summary gate permanently PENDING"
        )

    def test_the_job_is_unconditional(self) -> None:
        """AC4: no upstream may cascade-skip the hold gate."""
        job = self._hold_job()
        assert "needs" not in job
        assert "if" not in job

    def test_the_job_calls_the_shared_gate(self) -> None:
        """AC1: no local re-implementation and therefore no local vocabulary."""
        job = self._hold_job()
        assert job["uses"].startswith(f"{_REUSABLE}@")
        assert "steps" not in job

    def test_the_declared_context_name_matches_the_registration(self) -> None:
        """AC5 seam: the string is validated in the OTHER repo.

        `context_name` is handed to omnimarket's workflow, which checks it
        against the canonical vocabulary. If it is not the context GitHub really
        mints, the remote guard validates a name that does not exist while the
        real one goes unchecked — and the poller waits forever for a context
        nobody produces.
        """
        assert self._hold_job()["with"]["context_name"] == _CONTEXT

    def test_the_workflow_and_vocabulary_refs_are_the_same(self) -> None:
        """Split-brain guard: gate logic and tokens must be one vintage.

        `uses:` selects the workflow FILE; `vocabulary_ref` selects the SOURCE
        it reads. Drifting them runs vintage-X logic against vintage-Y tokens.
        """
        job = self._hold_job()
        assert job["with"]["vocabulary_ref"] == job["uses"].split("@", 1)[1]

    def test_the_pin_is_immutable_or_mainline(self) -> None:
        """A feature-branch pin breaks this repo when that branch is deleted."""
        ref = self._hold_job()["uses"].split("@", 1)[1]
        assert ref in {"dev", "main"} or re.fullmatch(r"[0-9a-f]{40}", ref), (
            f"the shared gate is pinned at {ref!r}; use a 40-hex SHA or a "
            "mainline ref so a squash-merge branch deletion cannot wedge CI here"
        )

    def test_no_hold_vocabulary_is_declared_in_this_repository(self) -> None:
        """The AC1 property this repo owns, as a fast local echo.

        The enforcing surface is the shared workflow's scan, which runs on every
        CI run here. This test uses the identifier rule only — it deliberately
        carries no token list, so it cannot itself become the second vocabulary
        it exists to forbid.
        """
        offenders: list[str] = []
        for root in (_REPO_ROOT / "src", _REPO_ROOT / "scripts"):
            if not root.is_dir():
                continue
            for path in root.rglob("*.py"):
                for line in path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines():
                    normalized = line.upper().replace("-", "_").replace(" ", "")
                    if "RE.COMPILE" not in normalized:
                        continue
                    if any(
                        fragment in normalized
                        for fragment in ("DO_NOT_MERGE", "HOLD_MARKER")
                    ):
                        offenders.append(f"{path.relative_to(_REPO_ROOT)}: {line!r}")
        assert offenders == [], (
            "a hold vocabulary is declared in this repository; it must live only "
            f"in omnimarket and be read through the shared gate: {offenders}"
        )

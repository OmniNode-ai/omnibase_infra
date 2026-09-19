# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18781 — the behavioural half: in CI, a missing service env FAILS.

The structural ratchet next door proves that no module-level skip is gated on a
variable nobody sets. It cannot prove what happens when one is missing anyway,
and that is the property the whole ticket turns on: a suite selected in CI
without its dependency must go RED, not skip. A structural check alone would
pass a future edit that restored the silent path while keeping every marker in
place.
"""

from __future__ import annotations

import pytest

from tests.helpers.service_env import require_service_env

pytestmark = [pytest.mark.ci]

# pytest.fail / pytest.skip raise these. Named here rather than imported from
# _pytest.outcomes so the test does not reach into a private module.
Failed = pytest.fail.Exception
Skipped = pytest.skip.Exception


def outcome_of() -> tuple[str, str]:
    """Return ``(disposition, message)`` for one require_service_env call.

    Deliberately not ``pytest.raises(Failed)``. A regression that restores the
    silent path raises Skipped, and Skipped escaping a ``pytest.raises`` block
    reports the TEST as skipped — green-looking, and exactly the false-green
    class this ticket exists to remove. Every outcome is classified here
    instead, so a skip is an assertion failure with the word "SKIPPED" in it.
    """
    try:
        require_service_env(**_CALL)
    except Failed as failure:
        return "failed", str(failure)
    except Skipped as skipped:
        return "skipped", str(skipped)
    return "returned", ""


_CALL = {
    "opt_in": "OMN18781_FIXTURE_OPT_IN",
    "endpoint": "OMN18781_FIXTURE_ENDPOINT",
    "workflow": ".github/workflows/ci.yml (service-integration-suites)",
    "service": "FixtureService",
}


def test_ci_without_the_opt_in_fails_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original defect's condition is now a failure carrying its own diagnosis."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv(_CALL["opt_in"], raising=False)

    disposition, message = outcome_of()

    assert disposition == "failed", (
        f"CI with no opt-in must FAIL; got {disposition!r}: {message}"
    )
    assert _CALL["opt_in"] in message
    assert "service-integration-suites" in message
    assert "OMN-18781" in message


def test_ci_with_an_empty_opt_in_is_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set-but-empty variable is the classic way a gate reads as satisfied."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv(_CALL["opt_in"], "")

    disposition, message = outcome_of()

    assert disposition == "failed", (
        f"a set-but-empty opt-in must FAIL in CI; got {disposition!r}: {message}"
    )


def test_ci_with_the_opt_in_set_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provisioned path returns quietly — no skip, no failure."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv(_CALL["opt_in"], "1")

    disposition, message = outcome_of()

    assert disposition == "returned", (
        f"a provisioned run must proceed; got {disposition!r}: {message}"
    )


def test_outside_ci_it_skips_rather_than_forcing_a_laptop_to_run_a_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local developers keep the skip; only CI is held to the stricter bar."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(_CALL["opt_in"], raising=False)

    disposition, message = outcome_of()

    assert disposition == "skipped", (
        f"outside CI this must SKIP, not fail; got {disposition!r}: {message}"
    )
    assert _CALL["opt_in"] in message


@pytest.mark.parametrize("truthy", ["true", "TRUE", "1", "yes"])
def test_every_ci_spelling_this_fleet_uses_is_recognised(
    monkeypatch: pytest.MonkeyPatch, truthy: str
) -> None:
    """A CI detector that recognised only one spelling would skip silently elsewhere."""
    monkeypatch.setenv("CI", truthy)
    monkeypatch.delenv(_CALL["opt_in"], raising=False)

    disposition, message = outcome_of()

    assert disposition == "failed", (
        f"CI spelled {truthy!r} must FAIL with no opt-in; got {disposition!r}: {message}"
    )

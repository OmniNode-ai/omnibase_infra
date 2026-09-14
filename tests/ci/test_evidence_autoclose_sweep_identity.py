# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17664 — the closer's workflow supplies an application identity.

The handler can only write as the application if the job hands it the
application secrets, and it can only refuse a half-configured identity if the
job does not refuse first for a different reason. Both halves live in the
workflow file, so both are asserted here rather than left to review.

What makes this worth a gate rather than a convention: the failure it prevents
is SILENT. A run missing the two application secrets does not fail — it falls
back to ``LINEAR_API_KEY`` and writes flips attributed to a person, which is
exactly the state the ticket exists to leave, and nothing about the run's
outcome says which identity it used. The only observable difference is a log
line and the ``actorId`` on the ticket afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

JOB_ID = "evidence-autoclose-sweep"
SWEEP_STEP_NAME = "Run evidence autoclose sweep"

CLIENT_ID_ENV = "LINEAR_CLOSER_CLIENT_ID"
CLIENT_SECRET_ENV = "LINEAR_CLOSER_CLIENT_SECRET"
API_KEY_ENV = "LINEAR_API_KEY"

# The pre-OMN-17664 preflight, spelled once so its return fails loudly. It
# refused a run whose only missing secret was the personal key — which, once the
# application pair is the preferred identity, refuses the CORRECT configuration.
RETIRED_PREFLIGHT = 'if [ -z "${LINEAR_API_KEY:-}" ]; then'


def _sweep_step() -> dict[str, Any]:
    loaded = yaml.safe_load(SWEEP_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{SWEEP_WORKFLOW} did not parse to a mapping"
    steps = loaded["jobs"][JOB_ID]["steps"]
    for step in steps:
        if isinstance(step, dict) and step.get("name") == SWEEP_STEP_NAME:
            return step
    raise AssertionError(f"step {SWEEP_STEP_NAME!r} is absent from {JOB_ID}")


def test_both_application_secrets_reach_the_sweep_step() -> None:
    """Neither half is useful alone, so both are supplied or the pair is dead."""
    env = _sweep_step()["env"]
    assert env[CLIENT_ID_ENV] == "${{ secrets.LINEAR_CLOSER_CLIENT_ID }}"
    assert env[CLIENT_SECRET_ENV] == "${{ secrets.LINEAR_CLOSER_CLIENT_SECRET }}"


def test_the_personal_key_is_still_supplied_as_the_fallback() -> None:
    """The fallback stays reachable, so an absent application is not an outage."""
    env = _sweep_step()["env"]
    assert env[API_KEY_ENV] == "${{ secrets.LINEAR_API_KEY }}"


def test_preflight_refuses_a_half_configured_application_identity() -> None:
    """Exactly one of the pair is an error, and is never degraded to the key."""
    body = _sweep_step()["run"]
    assert (
        f'[ -n "${{{CLIENT_ID_ENV}:-}}" ] || [ -n "${{{CLIENT_SECRET_ENV}:-}}" ]'
        in body
    ), "the preflight does not detect a half-configured application identity"
    assert "Partial Linear application identity" in body


def test_preflight_still_refuses_a_run_with_no_credential_at_all() -> None:
    body = _sweep_step()["run"]
    assert "No Linear credential is set" in body


def test_preflight_does_not_refuse_a_correctly_configured_application_run() -> None:
    """The retired check treated an absent personal key as fatal on its own.

    With the application pair preferred, that is the configuration the ticket is
    driving toward — so the old unconditional refusal must be gone, not merely
    reordered around.
    """
    body = _sweep_step()["run"]
    assert RETIRED_PREFLIGHT not in body


def test_preflight_names_the_identity_path_it_resolved() -> None:
    """A run that cannot say which identity it used cannot be audited later."""
    body = _sweep_step()["run"]
    assert "Linear identity path: oauth_application" in body
    assert "Linear identity path: personal_api_key" in body


def test_no_secret_value_is_echoed_by_the_preflight() -> None:
    """Only NAMES are printed. A length is a fact about a secret too."""
    body = _sweep_step()["run"]
    for env_name in (CLIENT_SECRET_ENV, API_KEY_ENV, CLIENT_ID_ENV):
        assert f'echo "${{{env_name}}}"' not in body
        assert f"${{#{env_name}}}" not in body

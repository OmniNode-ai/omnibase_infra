# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pre-push ticket-contract leg refuses exactly what ``onex scm pr-create`` refuses (OMN-21642).

FRICTION 2026-10-31T01:38:58Z: the ``prepush-deploy-scope-dod`` hook passed a
ticket contract with ``requirements`` and ``evidence_requirements`` and no
``dod_evidence``, then ``onex scm pr-create`` refused the same contract twice,
first for the missing ``dod_evidence[]`` and then for unbound acceptance
criteria (about 25 lane-minutes). The hook judged only the verifier's check-value
shape guard. pr-create calls ``contract_prerequisite`` from the released
``omnimarket``; the hook now runs that same function at the pin the
repo-evidence caller uses.

The fixtures under ``fixtures/ticket_contract_prerequisite_omn21642/`` come from
a recorded contract, the OMN-21629 contract at omnibase_infra#5481 head
e3a268c96. ``accepted_recorded_omn21629.yaml`` is that file with an SPDX header. The two
planted files are the same text with the ``dod_evidence`` section removed and
with every ``binds_ac`` line removed, the two shapes of the incident.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.ci.check_deploy_scope_dod import (
    _VERIFIER_PYTHON,
    FAIL_NO_CONTRACT,
    PASS_CONTRACTS,
    PREREQUISITE_VERIFIER_VERSION,
    classify_ticket_contracts,
    pinned_verifier_judge,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).parent / "fixtures" / "ticket_contract_prerequisite_omn21642"
_TICKET = "OMN-21629"

_REFUSALS = {
    "planted_no_dod_evidence.yaml": (
        "contracts/OMN-21629.yaml: Contract has empty or missing dod_evidence[] section."
    ),
    "planted_unbound_ac.yaml": (
        "contracts/OMN-21629.yaml: unbound acceptance criteria: AC1, AC2"
    ),
}

#: Runs in the released verifier's environment and prints what pr-create's own
#: module binding returns for each fixture, with whether that binding is the
#: function the hook runs.
_PR_CREATE_PROGRAM = (
    "import json, sys\n"
    "from omnimarket.nodes.node_dod_verify.services import contract_prerequisite as c\n"
    "from omnimarket.nodes.node_github_repo_write_effect.handlers import "
    "handler_github_repo_write as h\n"
    "texts = json.load(sys.stdin)\n"
    "json.dump({'same_function': h.contract_prerequisite is c.contract_prerequisite, "
    "'verdicts': {n: h.contract_prerequisite('OMN-21629', t) "
    "for n, t in texts.items()}}, sys.stdout)\n"
)


def _text(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def released_judge(tmp_path_factory: pytest.TempPathFactory):
    """The released verifier's prerequisite, at the pin the pre-push leg uses."""
    return pinned_verifier_judge(
        PREREQUISITE_VERIFIER_VERSION, tmp_path_factory.mktemp("judge-cwd")
    )


def test_the_incident_contract_with_no_dod_evidence_is_refused(released_judge) -> None:
    decision = classify_ticket_contracts(
        [_TICKET], lambda _t: _text("planted_no_dod_evidence.yaml"), released_judge
    )

    assert decision.exit_code == 1
    assert decision.findings == (_REFUSALS["planted_no_dod_evidence.yaml"],)


def test_the_incident_contract_with_unbound_criteria_is_refused_unbound(
    released_judge,
) -> None:
    decision = classify_ticket_contracts(
        [_TICKET], lambda _t: _text("planted_unbound_ac.yaml"), released_judge
    )

    assert decision.exit_code == 1
    assert decision.findings == (_REFUSALS["planted_unbound_ac.yaml"],)


def test_the_recorded_contract_that_pr_create_accepts_passes(released_judge) -> None:
    decision = classify_ticket_contracts(
        [_TICKET], lambda _t: _text("accepted_recorded_omn21629.yaml"), released_judge
    )

    assert decision.outcome == PASS_CONTRACTS
    assert decision.findings == ()


def test_a_missing_contract_is_still_named_before_the_prerequisite_runs() -> None:
    def never_judge(_contracts):
        raise AssertionError("presence is decided before the prerequisite")

    decision = classify_ticket_contracts([_TICKET], lambda _t: None, never_judge)

    assert decision.outcome == FAIL_NO_CONTRACT


def test_hook_and_pr_create_give_one_verdict_per_fixture_parity(
    released_judge, tmp_path: Path
) -> None:
    """pr-create's own binding of ``contract_prerequisite`` agrees with the hook on every fixture."""
    names = ["accepted_recorded_omn21629.yaml", *_REFUSALS]
    texts = {name: _text(name) for name in names}
    verifier = PREREQUISITE_VERIFIER_VERSION

    ran = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            _VERIFIER_PYTHON,
            "--with",
            f"omnimarket=={verifier}",
            "python",
            "-I",
            "-c",
            _PR_CREATE_PROGRAM,
        ],
        input=json.dumps(texts),
        check=False,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert ran.returncode == 0, ran.stderr[-2000:]
    pr_create = json.loads(ran.stdout)
    assert pr_create["same_function"] is True

    for name in names:
        hook = released_judge({_TICKET: texts[name]})[_TICKET]
        assert hook == pr_create["verdicts"][name], name
    assert pr_create["verdicts"]["accepted_recorded_omn21629.yaml"] is None
    for name, refusal in _REFUSALS.items():
        assert pr_create["verdicts"][name] == refusal

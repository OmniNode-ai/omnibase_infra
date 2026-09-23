# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18254: the failure-rate alert authenticates with the onexbot App, not a PAT.

The alert this repo shipped under OMN-18254/OMN-18322 had never fired. Every
scheduled run it ever had -- 400 of 400, back to the earliest retained one at
2026-09-15T05:08:01Z -- died on the same line, and the sibling zombie-detector
job in the same file stayed green, so the red read as a flaky neighbour rather
than as a dead mechanism.

    gh api repos/OmniNode-ai/omniweb/commits/<sha>/check-runs failed:
    Resource not accessible by personal access token (HTTP 403)

`CROSS_REPO_PAT` is a fine-grained PAT. `omnibase_infra` and `omnimarket` are
public, so their check-runs read succeeds on public access alone; `omniweb` is
PRIVATE and the token carries no Checks permission there. The script reads the
repos in order, so it died on the first private one and never reached the
third.

Two properties keep that from coming back, and they are asserted over the
PARSED workflow rather than over its text, so a reshuffle that preserves the
YAML but loses the property still fails here:

1. the evaluate step's ``GH_TOKEN`` comes from the minted App token;
2. that expression carries NO fallback. A `|| secrets.GITHUB_TOKEN` would not
   fail -- it would authenticate as a token with even less cross-repo reach and
   reproduce the same 403 one repo later. A sweep that cannot read every repo
   it claims to read must fail, never report a clean sweep over the repos it
   could not see.

The third assertion pins the App CHOICE, which is the part that is easy to get
wrong by symmetry: the org has two bot Apps whose secret names differ by one
path segment, and only one of them can serve this script. Read back live on
2026-09-18 from ``gh api /orgs/OmniNode-ai/installations``: `onexbot`
(app id 3342522) carries `administration: read`; `onexbot-occ-writer`
(app id 4361937) does not. This script reads
``branches/{branch}/protection/required_status_checks``, a branch-protection
read, which requires Administration -- so the occ-writer App would mint
successfully and then 403 on the first repo, which is a worse failure than not
minting at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-ci-zombie-detector.yml"
ALERT_JOB = "nonrequired-check-failure-rate"
SCRIPT = REPO_ROOT / "scripts" / "ci" / "nonrequired_check_failure_rate.py"

pytestmark = pytest.mark.unit


def _alert_job_steps() -> list[dict[str, Any]]:
    """The alert job's parsed step list.

    YAML 1.1 reads a bare ``on`` as the boolean ``True``; that key is not used
    here, but the same parser quirk is why every lookup below goes through the
    parsed document rather than through a regex over the file.
    """
    assert WORKFLOW.is_file(), f"the alert workflow is missing at {WORKFLOW}"
    doc: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict), f"unreadable jobs block: {jobs!r}"
    job = jobs.get(ALERT_JOB)
    assert isinstance(job, dict), (
        f"the {ALERT_JOB!r} job is gone from {WORKFLOW.name}; the OMN-18254 "
        "alert has no other home"
    )
    steps = job.get("steps")
    assert isinstance(steps, list), f"the alert job declares no steps: {steps!r}"
    return [s for s in steps if isinstance(s, dict)]


def _step_named(fragment: str) -> dict[str, Any]:
    matches = [
        s
        for s in _alert_job_steps()
        if fragment.lower() in str(s.get("name", "")).lower()
    ]
    assert len(matches) == 1, (
        f"expected exactly one alert-job step whose name contains {fragment!r}, "
        f"found {len(matches)}"
    )
    return matches[0]


def _evaluate_step_token() -> str:
    step = _step_named("Evaluate non-required check failure rates")
    env = step.get("env")
    assert isinstance(env, dict), f"the evaluate step declares no env: {env!r}"
    token = env.get("GH_TOKEN")
    assert isinstance(token, str) and token, (
        f"the evaluate step sets no GH_TOKEN: {token!r}"
    )
    return token


def test_the_alert_authenticates_with_the_minted_app_token() -> None:
    token = _evaluate_step_token()
    assert "steps.app-token.outputs.token" in token, (
        "the alert step must authenticate with the minted onexbot App token. "
        f"It currently reads {token!r}. A personal access token cannot read "
        "check-runs on the private omniweb repository, which is the 403 that "
        "kept this alert from ever firing."
    )


def test_the_token_expression_has_no_fallback() -> None:
    """The property that matters is the ABSENCE of a second alternative.

    A fallback here does not degrade gracefully. It swaps in a token with less
    cross-repo reach and reproduces the same 403 one repository later, with the
    difference that the swap is invisible in the log.
    """
    token = _evaluate_step_token()
    assert "||" not in token, (
        "the GH_TOKEN expression must name exactly one source and fail closed. "
        f"It currently reads {token!r}. `secrets.GITHUB_TOKEN` cannot read the "
        "private repository this sweep covers, so a fallback turns a loud mint "
        "failure into a silent partial sweep."
    )
    assert "secrets.GITHUB_TOKEN" not in token, (
        f"GH_TOKEN must not reference the default token at all: {token!r}"
    )
    assert "secrets.CROSS_REPO_PAT" not in token, (
        "CROSS_REPO_PAT is the credential whose missing Checks permission on "
        f"the private repository caused this outage: {token!r}"
    )


def test_the_mint_step_uses_the_app_that_carries_administration_read() -> None:
    """Pins the App CHOICE, not merely that some App is used.

    Both org Apps mint successfully. Only `onexbot` carries
    `administration: read`, which the branch-protection read below requires, so
    a swap to the occ-writer secrets would fail at the first repo rather than
    at the mint -- the harder failure to attribute.
    """
    step = _step_named("Mint the onexbot App token")
    uses = str(step.get("uses", ""))
    assert uses.startswith("actions/create-github-app-token@"), (
        f"the mint step must use the pinned App-token action: {uses!r}"
    )
    assert "@" in uses and len(uses.split("@")[1].split()[0]) == 40, (
        f"the App-token action must be pinned to a full commit sha: {uses!r}"
    )

    with_block = step.get("with")
    assert isinstance(with_block, dict), f"the mint step declares no `with`: {step!r}"

    app_id = str(with_block.get("app-id", ""))
    private_key = str(with_block.get("private-key", ""))
    assert "ONEXBOT_APP_ID" in app_id and "ONEXBOT_OCC" not in app_id, (
        "the mint step must name the onexbot App id secret, not the occ-writer "
        f"one: {app_id!r}. The occ-writer App lacks administration:read."
    )
    assert "ONEXBOT_APP_PRIVATE_KEY" in private_key, (
        f"the mint step must name the onexbot private-key secret: {private_key!r}"
    )


def test_the_minted_token_requests_every_permission_the_script_reads() -> None:
    """Each requested permission is owed to an endpoint the script actually hits.

    An installation token is downscoped to what it asks for, so a missing
    request here is a 403 at run time, not a warning.
    """
    with_block = _step_named("Mint the onexbot App token").get("with")
    assert isinstance(with_block, dict)

    source = SCRIPT.read_text(encoding="utf-8")
    owed = {
        # branches/{branch}/protection/required_status_checks
        "permission-administration": "/protection/required_status_checks",
        # actions/workflows and .../runs
        "permission-actions": "/actions/workflows",
        # commits/{sha}/check-runs
        "permission-checks": "/check-runs",
        # pulls?state=all
        "permission-pull-requests": "/pulls?state=all",
    }
    for key, endpoint in owed.items():
        assert endpoint in source, (
            f"this test is stale: {SCRIPT.name} no longer reads {endpoint!r}, so "
            f"the {key!r} request may no longer be owed. Re-derive the set."
        )
        assert with_block.get(key) == "read", (
            f"the minted token must request {key}=read; the script reads "
            f"{endpoint!r} and an installation token is downscoped to what it "
            f"asks for. Currently {with_block.get(key)!r}."
        )


def test_the_minted_token_is_scoped_to_exactly_the_repos_the_sweep_reads() -> None:
    """Blast radius: the token may not reach a repository the sweep never reads."""
    with_block = _step_named("Mint the onexbot App token").get("with")
    assert isinstance(with_block, dict)
    declared = {
        line.strip()
        for line in str(with_block.get("repositories", "")).splitlines()
        if line.strip()
    }
    assert declared, (
        "the mint step must pin `repositories:`; this sweep's repo list is a "
        "literal in the run step, so the narrower scope is available"
    )

    run = str(_step_named("Evaluate non-required check failure rates").get("run", ""))
    swept = {
        line.strip().removeprefix("--repo ").strip().rstrip("\\").strip()
        for line in run.splitlines()
        if line.strip().startswith("--repo ")
    }
    assert swept, f"could not read the swept repo list from the run step: {run!r}"
    assert declared == swept, (
        "the minted token's repository scope and the swept repository list must "
        f"be the same set. Token scope {sorted(declared)}, sweep "
        f"{sorted(swept)}. A repo in the sweep but not the scope is a 403; a "
        "repo in the scope but not the sweep is unearned reach."
    )


def test_the_sibling_zombie_detector_keeps_its_own_credential() -> None:
    """Positive control: the two jobs keep separate, separately scoped tokens.

    The detector WRITES (force-cancel) to two repos. Since OMN-19258 it mints
    its own onexbot token with `actions: write` scoped to exactly those two
    repos. This alert job only READS. Without this control, a later edit that
    swept both jobs onto one credential -- or widened this read-only job to
    Actions write -- would pass every assertion above while changing a write
    path nobody reviewed.
    """
    doc: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    detector = doc["jobs"]["detect"]
    detector_steps = [s for s in detector["steps"] if isinstance(s, dict)]
    tokens = [
        str(step["env"]["GH_TOKEN"])
        for step in detector_steps
        if isinstance(step.get("env"), dict) and "GH_TOKEN" in step["env"]
    ]
    assert tokens, "the zombie detector sets no GH_TOKEN; it needs one to cancel runs"
    for token in tokens:
        assert token.strip() == "${{ steps.app-token.outputs.token }}", (
            "the zombie detector must use the token minted in its OWN job "
            f"(OMN-19258): {token!r}"
        )
    detector_mints = [
        s
        for s in detector_steps
        if str(s.get("uses", "")).startswith("actions/create-github-app-token@")
    ]
    assert len(detector_mints) == 1, "the detect job must mint its own token"
    detector_with = detector_mints[0].get("with")
    assert isinstance(detector_with, dict)
    assert detector_with.get("permission-actions") == "write"

    alert_with = _step_named("Mint the onexbot App token").get("with")
    assert isinstance(alert_with, dict)
    assert alert_with.get("permission-actions") != "write", (
        "the alert job only reads; Actions write belongs to the detector alone"
    )

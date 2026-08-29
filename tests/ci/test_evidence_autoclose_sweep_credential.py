# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16832 — the evidence-autoclose sweep must run on a minted App token.

Why this file exists, stated as the failure it prevents rather than as a rule:

``evidence-autoclose-sweep.yml`` resolved its GitHub credential as
``${{ secrets.OMNI_GITHUB_TOKEN || github.token }}`` at six sites. Live readback
on 2026-08-29 proved no secret of that name exists at org, repo, or environment
scope — the only three places ``secrets.*`` resolves from in that job — so every
site had always taken the right-hand branch. ``github.token`` cannot carry
``administration: read`` at all (it is not a grantable scope for the automatic
token), so every ``::pr-live-state`` sub-check the sweep spawns recorded
SKIPPED with cause ``credential_cannot_read_branch_protection``, on every
ticket, in every run — which made the autoclose flip predicate unsatisfiable
from a workflow run for the entire corpus.

The defect was not "the wrong token": it was that a missing credential
degraded *silently* to a weaker one. So the assertions below are about both
halves — the App token is minted AND there is no fallback chain left to hide
behind.
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
MINT_STEP_ID = "app-token"
MINT_ACTION = "actions/create-github-app-token"
APP_TOKEN_EXPR = "${{ steps.app-token.outputs.token }}"

# The retired secret name. Asserted absent from the PARSED document, not from
# the file text: yaml.safe_load drops comments, so the historical note in the
# workflow explaining what was removed does not mask a surviving live use.
RETIRED_SECRET = "OMNI_GITHUB_TOKEN"


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(SWEEP_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{SWEEP_WORKFLOW} did not parse to a mapping"
    return loaded


def _steps() -> list[dict[str, Any]]:
    job = _load_workflow()["jobs"][JOB_ID]
    steps = job["steps"]
    assert isinstance(steps, list) and steps, f"{JOB_ID} declares no steps"
    return steps


def _walk_strings(node: Any) -> list[str]:
    """Every string leaf in the parsed document, comments excluded by the parser."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in _walk_strings(k) + _walk_strings(v)]
    if isinstance(node, list):
        return [s for item in node for s in _walk_strings(item)]
    return []


def test_sweep_mints_an_onexbot_app_token() -> None:
    """The credential is minted from the App, not read from a secret name."""
    mint = [s for s in _steps() if s.get("id") == MINT_STEP_ID]
    assert len(mint) == 1, (
        f"expected exactly one step with id '{MINT_STEP_ID}' in {JOB_ID}; "
        f"found {len(mint)}"
    )
    step = mint[0]

    uses = step.get("uses", "")
    assert uses.startswith(f"{MINT_ACTION}@"), (
        f"the '{MINT_STEP_ID}' step must use {MINT_ACTION}, got {uses!r}"
    )
    assert "@" in uses and not uses.endswith(("@v1", "@main", "@master")), (
        f"{MINT_ACTION} must be pinned to a commit sha, got {uses!r}"
    )

    with_block = step.get("with") or {}
    assert with_block.get("app-id") == "${{ secrets.ONEXBOT_APP_ID }}"
    assert with_block.get("private-key") == "${{ secrets.ONEXBOT_APP_PRIVATE_KEY }}"
    assert with_block.get("owner") == "OmniNode-ai"


def test_minted_token_carries_administration_read() -> None:
    """`administration: read` is the whole point — branch protection is unreadable without it.

    ``github.token`` cannot hold this scope under any configuration, so its
    presence here is what separates a credential that can reach a verdict on
    ``::pr-live-state`` from one that can only SKIP.
    """
    step = next(s for s in _steps() if s.get("id") == MINT_STEP_ID)
    with_block = step.get("with") or {}
    assert with_block.get("permission-administration") == "read", (
        "the minted token must request administration:read; without it every "
        "::pr-live-state sub-check SKIPs with credential_cannot_read_branch_protection"
    )


def test_minted_token_is_read_only() -> None:
    """Least privilege: the sweep's GitHub access is entirely `gh api` GETs.

    The onexbot installation grants checks:write, issues:write and
    pull_requests:write. None are requested here — the handler's only writes go
    to Linear. A `write` permission appearing in this step means either the
    sweep started mutating GitHub (which needs its own review) or the narrowing
    was dropped by accident.
    """
    step = next(s for s in _steps() if s.get("id") == MINT_STEP_ID)
    with_block = step.get("with") or {}
    granted = {
        key: value for key, value in with_block.items() if key.startswith("permission-")
    }
    assert granted, "the mint step must narrow permissions explicitly, not inherit them"
    writes = {k: v for k, v in granted.items() if v != "read"}
    assert not writes, f"the sweep's token must be read-only; found {writes}"


def test_every_github_credential_site_uses_the_minted_token() -> None:
    """No site is left on a different credential.

    A job that mixes credentials hides which one is actually in force — the
    exact condition that let this defect survive six sites and many runs.
    """
    workflow = _load_workflow()
    job = workflow["jobs"][JOB_ID]

    offenders: list[str] = []
    for step in job["steps"]:
        label = step.get("name") or step.get("uses") or "<unnamed step>"
        for container, key in (
            (step.get("env") or {}, "GH_TOKEN"),
            (step.get("with") or {}, "token"),
        ):
            if key in container and container[key] != APP_TOKEN_EXPR:
                offenders.append(f"{label}: {key}={container[key]!r}")

    assert not offenders, (
        "every GH_TOKEN / token: site in the sweep job must be "
        f"{APP_TOKEN_EXPR}; found: {offenders}"
    )

    # And the wiring is actually present, so the assertion above cannot pass
    # vacuously by someone deleting all the sites.
    wired = sum(
        1
        for step in job["steps"]
        for container, key in (
            (step.get("env") or {}, "GH_TOKEN"),
            (step.get("with") or {}, "token"),
        )
        if container.get(key) == APP_TOKEN_EXPR
    )
    assert wired >= 6, f"expected at least 6 wired credential sites, found {wired}"


def test_no_silent_fallback_to_a_weaker_credential() -> None:
    """`a || b` on a credential is the defect class, not a convenience.

    The original expression read as "a stronger token, if configured". It was a
    dead branch over a weaker one, and nothing reported the fall-through. A
    credential that cannot be resolved must fail the job loudly instead.
    """
    workflow = _load_workflow()
    job = workflow["jobs"][JOB_ID]

    fallbacks: list[str] = []
    for step in job["steps"]:
        label = step.get("name") or step.get("uses") or "<unnamed step>"
        for container, key in (
            (step.get("env") or {}, "GH_TOKEN"),
            (step.get("with") or {}, "token"),
        ):
            value = container.get(key)
            if isinstance(value, str) and "||" in value:
                fallbacks.append(f"{label}: {key}={value!r}")

    assert not fallbacks, (
        "a credential must not fall back to another credential — resolve it or "
        f"fail the job. Found: {fallbacks}"
    )


def test_retired_secret_name_has_no_live_reference() -> None:
    """`OMNI_GITHUB_TOKEN` never existed as a secret; it must not read as if it did.

    Checked against the parsed document so the workflow may still *explain* the
    removal in a comment without that explanation satisfying the assertion.
    """
    live = [s for s in _walk_strings(_load_workflow()) if RETIRED_SECRET in s]
    assert not live, (
        f"{RETIRED_SECRET} is not a secret that exists at any scope this job can "
        f"read; remove these live references: {live}"
    )

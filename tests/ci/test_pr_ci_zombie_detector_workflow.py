# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19258: the zombie detector's job cannot go green without having looked.

Until OMN-19258 the `detect` job in `pr-ci-zombie-detector.yml` had two ways to
report success while seeing nothing:

1. A `has_pat` step turned an absent CROSS_REPO_PAT into a warning, a skipped
   detector step and a GREEN job. Dormant while the secret exists, but it is
   the silent-gate shape the 2026-09-20 operator ruling forbids.
2. The detector exited 0 on every completed scan. 68 of 2,904 retained
   scheduled reports (2026-08-25..2026-09-23) carried a rate-limit fetch error
   for BOTH target repos, and every one of those runs was green.

The second is pinned by the script's own tests
(`tests/unit/scripts/ci/test_pr_ci_zombie_detector.py`). These tests pin the
workflow half, over the PARSED document so a reshuffle that keeps the text but
loses the property still fails:

- no step of the detect job is conditional on a credential-presence output;
- the credential step exits non-zero when the secret is empty;
- the detector's GH_TOKEN carries no `||` fallback (a GITHUB_TOKEN fallback
  cannot reach another repo's runs, so it would read nothing, and before
  OMN-19258 that read as a clean sweep);
- the receipt is uploaded even when the detector step fails, and a missing
  receipt is itself an error.

The credential stays CROSS_REPO_PAT, deliberately: force-cancel needs Actions
write on the target repo, and read live on 2026-09-23 both org App
installations carry `actions: read` only. That is pinned as well, so a later
edit that swaps in an App token without the permission change fails here
instead of 403-ing on the first real wedge.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-ci-zombie-detector.yml"

pytestmark = pytest.mark.unit


def _detect_steps() -> list[dict[str, Any]]:
    doc: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = doc["jobs"]["detect"]
    steps = job.get("steps")
    assert isinstance(steps, list), f"the detect job declares no steps: {steps!r}"
    return [s for s in steps if isinstance(s, dict)]


def _step(fragment: str) -> dict[str, Any]:
    matches = [s for s in _detect_steps() if fragment in str(s.get("name", ""))]
    assert len(matches) == 1, (
        f"expected exactly one detect step named like {fragment!r}, found "
        f"{[s.get('name') for s in matches]}"
    )
    return matches[0]


def test_no_detect_step_is_gated_on_a_credential_presence_output() -> None:
    """The removed shape was `if: steps.pat_check.outputs.has_pat == 'true'`."""
    for step in _detect_steps():
        condition = str(step.get("if", ""))
        for marker in ("has_pat", "pat_check", "CROSS_REPO_PAT", "secrets."):
            assert marker not in condition, (
                f"detect step {step.get('name')!r} is conditional on a credential "
                f"({condition!r}); a missing credential must fail the job, never "
                "skip the detector with the job green (OMN-19258)"
            )


def test_no_step_publishes_a_skip_output() -> None:
    for step in _detect_steps():
        run = str(step.get("run", ""))
        assert "GITHUB_OUTPUT" not in run, (
            f"detect step {step.get('name')!r} writes a step output; the only "
            "output this job ever wrote was the skip switch"
        )
        assert "SKIPPED" not in run


def test_the_credential_step_fails_closed_on_an_empty_secret(tmp_path: Path) -> None:
    """Execute the step's own shell with the secret empty, then present.

    Running it is the falsifier: a step that merely *mentions* `exit 1` in a
    branch that never fires would pass a text check and still go green.
    """
    step = _step("Refuse a missing CROSS_REPO_PAT")
    assert str(step["env"]["CROSS_REPO_PAT"]).strip() == (
        "${{ secrets.CROSS_REPO_PAT }}"
    )
    script = tmp_path / "step.sh"
    script.write_text(str(step["run"]), encoding="utf-8")
    base_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}

    empty = subprocess.run(
        ["bash", str(script)],
        env={**base_env, "CROSS_REPO_PAT": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert empty.returncode != 0, "an empty CROSS_REPO_PAT must fail the job"
    assert "::error::" in empty.stdout

    present = subprocess.run(
        ["bash", str(script)],
        env={**base_env, "CROSS_REPO_PAT": "placeholder-not-a-token"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert present.returncode == 0, (
        "positive control: a present secret must pass the check, or the "
        f"assertion above proves nothing ({present.stdout!r} {present.stderr!r})"
    )


def test_the_credential_check_runs_before_the_detector() -> None:
    names = [str(s.get("name", "")) for s in _detect_steps()]
    check = names.index("Refuse a missing CROSS_REPO_PAT")
    run = names.index("Run zombie detector")
    assert check < run


def test_detector_token_has_no_fallback_and_stays_on_the_write_capable_pat() -> None:
    token = str(_step("Run zombie detector")["env"]["GH_TOKEN"])
    assert "||" not in token, (
        f"GH_TOKEN carries a fallback ({token!r}); GITHUB_TOKEN cannot read or "
        "cancel another repo's runs, so a fallback reads nothing"
    )
    assert token.strip() == "${{ secrets.CROSS_REPO_PAT }}", (
        "the detector force-cancels runs, which needs Actions write on the target "
        "repo. Both org App installations carried `actions: read` only when read "
        "on 2026-09-23; moving to an App token needs that permission granted "
        f"first (OMN-19258, OMN-16373). Got {token!r}"
    )


def test_detector_step_is_unconditional() -> None:
    assert "if" not in _step("Run zombie detector")


def test_receipt_uploads_when_the_detector_fails_and_a_missing_one_is_an_error() -> (
    None
):
    names = [str(s.get("name", "")) for s in _detect_steps()]
    for fragment in ("Assert the report is present", "Upload receipt"):
        condition = str(_step(fragment).get("if", ""))
        assert "always()" in condition, (
            f"{fragment!r} must run when the detector exits 1 (its report is "
            f"written before the exit); got if={condition!r}"
        )
    assert names.index("Run zombie detector") < names.index(
        "Assert the report is present and non-empty"
    )
    assert "zombie-detector-report.json" in str(
        _step("Assert the report is present").get("run", "")
    )
    with_block = _step("Upload receipt").get("with")
    assert isinstance(with_block, dict)
    assert with_block.get("if-no-files-found") == "error"
    assert with_block.get("name") == "pr-ci-zombie-detector-report"


def test_no_detect_step_continues_on_error() -> None:
    for step in _detect_steps():
        assert not step.get("continue-on-error"), (
            f"detect step {step.get('name')!r} sets continue-on-error, which turns "
            "a blind or failed detector back into a green job"
        )

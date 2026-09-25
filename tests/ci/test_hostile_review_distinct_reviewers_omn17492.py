# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17492: the hostile reviewer's roster is two different LAB models, and a
reviewer that could not be reached is named on the pull request.

Until this change the roster was ``qwen3-review``, ``qwen3-review-b`` and
``glm-review``. The first two resolve to one endpoint, ``.201:8000``, which
serves one model, so the two-model quorum (OMN-18479) could be met by that
model agreeing with itself. On the OMN-17492 eval (the code-review cases of
knowledge-base-internal#769, run 2026-09-25), the same model on two hosts
agreed on a critical finding in 2 of 8 runs and both agreed findings were
false; the pair with gpt-oss-120b blocked nothing.

The roster is now ``qwen3-review`` (Qwen3.8-27B on .201) and
``gpt-oss-review`` (gpt-oss-120b on the .200 Mac Studio, registered in
omniintelligence under OMN-17492). ``glm-review`` is removed: it is a
third-party cloud model and private diffs go only to lab models (operator,
2026-09-25, OPERATOR-CONSENT ledger row 4842). .200 is always on (operator,
same day), so there is no single-model degraded pass: losing either reviewer
leaves one model, which is no quorum, and the run fails closed as
omnibase_infra#4119 made it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "hostile-reviewer.yml"
REVIEW_JOB = "hostile-review"
ROSTER = ["qwen3-review", "gpt-oss-review"]
ALIASES_OF_201_8000 = ("qwen3-review-b", "deepseek-r1")
NOT_VOTERS = (*ALIASES_OF_201_8000, "glm-review")
CLOUD_REVIEW_ENV = ("LLM_GLM_API_KEY", "LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST")

pytestmark = pytest.mark.unit


def _steps() -> list[dict[str, Any]]:
    with WORKFLOW.open(encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    steps = parsed["jobs"][REVIEW_JOB]["steps"]
    assert isinstance(steps, list)
    return steps


def _step(step_id: str) -> dict[str, Any]:
    matches = [s for s in _steps() if s.get("id") == step_id]
    assert len(matches) == 1, f"expected one step with id {step_id!r}"
    return matches[0]


def _step_named(prefix: str) -> dict[str, Any]:
    matches = [s for s in _steps() if str(s.get("name", "")).startswith(prefix)]
    assert len(matches) == 1, f"expected one step named {prefix!r}"
    return matches[0]


def _cli_models() -> list[str]:
    run = str(_step("review")["run"])
    return re.findall(r"--model\s+([A-Za-z0-9_.-]+)", run)


def test_review_step_runs_the_two_distinct_lab_reviewers_in_order() -> None:
    assert _cli_models() == ROSTER


@pytest.mark.parametrize("alias", NOT_VOTERS)
def test_no_alias_and_no_cloud_model_is_a_voter(alias: str) -> None:
    assert alias not in _cli_models()
    keys = str(_step("preflight")["env"]["REVIEW_MODEL_KEYS"]).split()
    assert alias not in keys


def test_preflight_probes_exactly_the_roster_the_review_runs() -> None:
    keys = str(_step("preflight")["env"]["REVIEW_MODEL_KEYS"]).split()
    assert keys == ROSTER


def _verdict_parser() -> str:
    run = str(_step("review")["run"])
    match = re.search(r"<<'PYEOF'\n(.*?)\n\s*PYEOF\n", run, re.DOTALL)
    assert match, "the review step no longer carries its PYEOF verdict parser"
    lines = match.group(1).splitlines()
    indent = min(len(x) - len(x.lstrip()) for x in lines if x.strip())
    return "\n".join(x[indent:] for x in lines)


def _parse(tmp_path: Path, payload: dict[str, Any]) -> dict[str, str]:
    doc = tmp_path / "result.json"
    doc.write_text(json.dumps(payload), encoding="utf-8")
    script = tmp_path / "parse.py"
    script.write_text(_verdict_parser(), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), str(doc)],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in proc.stdout.splitlines())


def _result(model: str, *, success: bool, error: str | None = None) -> dict[str, Any]:
    return {"model": model, "success": success, "error": error, "findings": []}


def test_a_lost_reviewer_is_named_and_the_run_has_no_verdict(
    tmp_path: Path,
) -> None:
    """One lab model left is no quorum: degraded, which the enforce step fails."""
    unreachable = (
        "unreachable: the TCP reachability probe of gpt-oss-review at "
        "192.168.86.200:8130 failed, so this reviewer was not called"  # onex-allow-internal-ip
    )
    out = _parse(
        tmp_path,
        {
            "models_succeeded": ["qwen3-review"],
            "models_failed": ["gpt-oss-review"],
            "total_findings": 0,
            "results": [
                _result("qwen3-review", success=True),
                _result("gpt-oss-review", success=False, error=unreachable),
            ],
            "quorum": {
                "verdict": "degraded_quorum",
                "blocking_count": 0,
                "warning_count": 0,
            },
        },
    )
    assert out["verdict"] == "degraded"
    assert out["models_succeeded"] == "qwen3-review"
    assert out["models_failed"].startswith("gpt-oss-review (unreachable")


def test_review_step_sends_nothing_to_a_cloud_reviewer() -> None:
    env = _step("review").get("env", {})
    for name in CLOUD_REVIEW_ENV:
        assert name not in env, f"{name} still wired into the review step"
    assert "secrets.LLM_GLM_API_KEY" not in WORKFLOW.read_text(encoding="utf-8")


def test_nothing_lost_names_nothing(tmp_path: Path) -> None:
    """Positive control for the test above."""
    out = _parse(
        tmp_path,
        {
            "models_succeeded": ROSTER,
            "models_failed": [],
            "total_findings": 0,
            "results": [_result(m, success=True) for m in ROSTER],
            "quorum": {"verdict": "passed", "blocking_count": 0, "warning_count": 0},
        },
    )
    assert out["verdict"] == "passed"
    assert out["models_failed"] == ""


def test_summary_comment_renders_the_lost_reviewers() -> None:
    step = _step_named("Post review summary as PR comment")
    assert "steps.review.outputs.models_failed" in str(
        step["env"]["REVIEW_MODELS_FAILED"]
    )
    script = str(step["with"]["script"])
    assert "REVIEW_MODELS_FAILED" in script
    assert "Models unavailable" in script


def test_summary_footer_names_the_roster_that_runs() -> None:
    script = str(_step_named("Post review summary as PR comment")["with"]["script"])
    for key in ROSTER:
        assert key in script
    for alias in NOT_VOTERS:
        assert alias not in script

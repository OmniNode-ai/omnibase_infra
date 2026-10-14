# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Hostile Reviewer reads its voters from the contract overlay (OMN-20910).

Operator RULING 2026-10-10T18:17:37Z: review-voter models are declared in a
contract overlay, never in repository variables or workflow files. Retiring one
endpoint on 2026-10-10 broke this gate on every pull request because the
second voter's URL lived in an Actions variable per repository and a
hard-coded port in this workflow. These tests hold the workflow to passing
the overlay and nothing else, and the ratchet below refuses any voter endpoint,
port, model flag or endpoint variable creeping back into a workflow file.

OMN-20923 (operator RULING 2026-10-10T21:57:38Z): the overlay itself is a
deployment fact and is not in this repository. The workflow fetches it at run
time from the private CI overlay named by the org variable OMNI_CI_OVERLAY_REPO;
with the variable unset the neutral default is "no review voters configured",
and the gate fails closed saying so.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = [pytest.mark.unit]

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
WORKFLOW = _WORKFLOWS / "hostile-reviewer.yml"
OVERLAY_SUFFIX = "config/onex/overlays/node_review_voters_overlay_compute/overlay.yaml"
OVERLAY_ENV = "${{ github.workspace }}/../ci-overlay/" + OVERLAY_SUFFIX
OVERLAY_ARG = '--voters-overlay "$REVIEW_VOTERS_OVERLAY"'

# What a voter endpoint looks like when someone writes one into a workflow:
# a URL with an explicit port, an endpoint variable, a --model flag, or the
# preflight's old model-key list.
_VOTER_ENDPOINT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://[A-Za-z0-9.\-]+:\d{2,5}"),
    re.compile(r"\bLLM_[A-Z0-9_]+_URL\b"),
    re.compile(r"vars\.LLM_"),
    re.compile(r"--model[\s=]"),
    re.compile(r"\bREVIEW_MODEL_KEYS\b"),
)
# The endpoint variables the overlay replaced; no workflow may name them at all.
_RETIRED_ENDPOINT_VARS = ("LLM_LOCAL_STUDIO_PLANNER_URL", "LLM_QWEN3_REVIEW_URL")

# OMN-20910: the voter's model is a field of the overlay, so a workflow names no
# model at all. A ``vars.<anything>MODEL`` reference and a model id literal are
# refused here; ``_env_model_literal_findings`` refuses a MODEL env key bound to
# a literal no pattern knows.
_VOTER_MODEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bvars\.[A-Za-z0-9_]*MODEL", re.IGNORECASE),
    re.compile(
        r"\b(?:claude-(?:opus|sonnet|haiku|fable|instant|\d)|gpt-?\d|o[1-9]-(?:mini|preview)"
        r"|gemini-?\d|qwen-?\d|qwen-|glm-?\d|deepseek-|llama-?\d|mistral-|mixtral-"
        r"|codestral|gemma-?\d|grok-?\d|kimi-)",
        re.IGNORECASE,
    ),
)
# Positive control: the lines this repository's workflow carried before the
# overlay. The ratchet must find every one of them.
_KNOWN_BAD = """\
      LLM_LOCAL_STUDIO_PLANNER_URL: "http://studio.lab.example:8131"
          LLM_LOCAL_STUDIO_PLANNER_URL: ${{ vars.LLM_LOCAL_STUDIO_PLANNER_URL }}
          REVIEW_MODEL_KEYS: "qwen3-review local-studio-planner"
            --model qwen3-review \\
            --model local-studio-planner \\
"""


def _env_model_literal_findings(text: str) -> list[str]:
    """Every job or step ``env:``/``with:`` key naming a MODEL bound to a literal.

    A literal is any non-empty value that is not a ``${{ ... }}`` expression, so
    a bare id that no pattern knows (``REVIEW_MODEL: my-voter``) is refused too.
    A readback such as ``steps.review.outputs.models_succeeded`` is an expression.
    """
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    if not isinstance(parsed, dict) or not isinstance(parsed.get("jobs"), dict):
        return []
    findings: list[str] = []

    def scan(where: str, mapping: object) -> None:
        if not isinstance(mapping, dict):
            return
        for key, value in mapping.items():
            literal = str(value)
            if "MODEL" in str(key).upper() and literal and "${{" not in literal:
                findings.append(f"{where}: {key}={literal}")

    for job_id, job in parsed["jobs"].items():
        scan(f"{job_id}.env", job.get("env"))
        for step in job.get("steps") or []:
            name = str(step.get("name", ""))
            scan(f"{job_id}/{name}.env", step.get("env"))
            scan(f"{job_id}/{name}.with", step.get("with"))
    return findings


def voter_endpoint_findings(text: str) -> list[str]:
    """Every line of ``text`` that names a voter endpoint, port, model or model flag."""
    findings: list[str] = []
    patterns = (*_VOTER_ENDPOINT_PATTERNS, *_VOTER_MODEL_PATTERNS)
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        if any(p.search(line) for p in patterns):
            findings.append(f"{number}: {line.strip()}")
    return findings + _env_model_literal_findings(text)


def _reviewer_workflows() -> list[Path]:
    return [
        path
        for path in sorted(_WORKFLOWS.glob("*.y*ml"))
        if "review_pairing" in path.read_text(encoding="utf-8")
    ]


def _steps() -> list[dict[str, Any]]:
    parsed = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = parsed["jobs"]["hostile-review"]["steps"]
    assert isinstance(steps, list)
    return steps


def _step(prefix: str) -> dict[str, Any]:
    matches = [s for s in _steps() if str(s.get("name", "")).startswith(prefix)]
    assert len(matches) == 1, f"expected one step named {prefix!r}"
    return matches[0]


def test_the_ratchet_finds_every_known_bad_line() -> None:
    """Positive control: a zero below is only evidence if this finds five."""
    assert len(voter_endpoint_findings(_KNOWN_BAD)) == 5


def test_reviewer_workflows_exist() -> None:
    assert WORKFLOW in _reviewer_workflows()


@pytest.mark.parametrize("path", _reviewer_workflows(), ids=lambda p: p.name)
def test_no_reviewer_workflow_names_a_voter_endpoint(path: Path) -> None:
    findings = voter_endpoint_findings(path.read_text(encoding="utf-8"))
    assert findings == [], (
        f"{path.name} names a review voter endpoint, port or model; voters are "
        f"declared only in the overlay (OMN-20910): {findings}"
    )


@pytest.mark.parametrize("name", _RETIRED_ENDPOINT_VARS)
def test_no_workflow_reads_a_retired_endpoint_variable(name: str) -> None:
    for path in sorted(_WORKFLOWS.glob("*.y*ml")):
        assert name not in path.read_text(encoding="utf-8"), path.name


def test_review_reads_the_overlay() -> None:
    assert OVERLAY_ARG in str(_step("Run adversarial review")["run"])


def test_preflight_resolves_targets_from_the_overlay() -> None:
    run = str(_step("Preflight")["run"])
    assert "load_review_voters" in run
    assert '"$REVIEW_VOTERS_OVERLAY"' in run


def test_overlay_path_is_the_ci_overlay_outside_the_workspace() -> None:
    """The roster is read from the CI overlay, never from this pull request's
    checkout: a pull request cannot choose the voters that review it."""
    parsed = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    env = parsed["jobs"]["hostile-review"]["env"]
    assert env["REVIEW_VOTERS_OVERLAY"] == OVERLAY_ENV
    assert "../ci-overlay" in str(_step("Clean stale dependency clones")["run"])


_ROOT = Path(__file__).resolve().parents[2]
_SIBLING_PINS = _ROOT / ".github" / "sibling-pins.yaml"


def test_omniintelligence_is_pinned_to_the_sibling_pin() -> None:
    pin = yaml.safe_load(_SIBLING_PINS.read_text(encoding="utf-8"))["pins"][
        "omniintelligence"
    ]
    assert re.fullmatch(r"[0-9a-f]{40}", pin)
    assert f"checkout --quiet {pin}" in str(_step("Clone omniintelligence")["run"])


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    return out.stdout.splitlines()


def test_this_repository_carries_no_review_voters_overlay() -> None:
    """OMN-20923: no lab roster ships in this repository (positive control:
    the scan finds the schema id in a synthetic overlay text)."""
    marker = "review_voters_overlay.v1"
    assert marker in 'schema_version: "review_voters_overlay.v1"\nvoters: []\n'
    offenders = []
    for rel in _tracked_files():
        path = _ROOT / rel
        if path.suffix not in {".yaml", ".yml"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^schema_version:\s*[\"']?review_voters_overlay\.v1", text, re.M):
            offenders.append(rel)
    assert offenders == []
    assert not (_ROOT / "docker/lane-overlays/hostile-review-voters.yaml").exists()


def test_the_ci_overlay_repository_comes_from_an_org_variable() -> None:
    step = _step("Resolve the CI overlay")
    assert step["env"]["CI_OVERLAY_REPO"] == "${{ vars.OMNI_CI_OVERLAY_REPO }}"
    fetch = _step("Fetch the CI overlay")
    assert fetch["with"]["repository"] == (
        "${{ github.repository_owner }}/${{ steps.ci-overlay.outputs.name }}"
    )
    assert fetch["with"]["persist-credentials"] is False


def test_the_workflow_names_no_network() -> None:
    """The transport's CIDR allowlist is derived from the resolved voters."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b", text)
    parsed = yaml.safe_load(text)
    assert "LLM_ENDPOINT_CIDR_ALLOWLIST" not in parsed["jobs"]["hostile-review"]["env"]
    assert 'LLM_ENDPOINT_CIDR_ALLOWLIST=${ALLOWLIST}" >> "$GITHUB_ENV"' in str(
        _step("Preflight")["run"]
    )


def _run_resolve(repo_var: str, tmp_path: Path) -> tuple[int, str, str]:
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")
    proc = subprocess.run(
        ["bash", "-c", str(_step("Resolve the CI overlay")["run"])],
        env={
            "CI_OVERLAY_REPO": repo_var,
            "GITHUB_OUTPUT": str(output),
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, output.read_text(encoding="utf-8"), proc.stdout


def test_a_fork_with_no_ci_overlay_gets_the_neutral_default(tmp_path: Path) -> None:
    """Unset variable: configured=false and a named "no review voters
    configured" error; the preflight then fails the gate closed with that reason
    before reading any file."""
    rc, out, stdout = _run_resolve("", tmp_path)
    assert rc == 0
    assert "configured=false" in out
    assert "no review voters configured" in stdout
    preflight = str(_step("Preflight")["run"])
    assert preflight.index('"${CI_OVERLAY_CONFIGURED}" != "true"') < preflight.index(
        "PROBE_TARGETS="
    )
    assert "error=${REASON}" in preflight


def test_our_ci_resolves_the_named_overlay_repository(tmp_path: Path) -> None:
    rc, out, _ = _run_resolve("deploy-config", tmp_path)
    assert rc == 0
    assert "configured=true" in out
    assert "name=deploy-config" in out
    rc, _, _ = _run_resolve("owner/repo", tmp_path)
    assert rc == 1


def test_preflight_runs_in_the_installed_environment() -> None:
    """The overlay reader imports omniintelligence.review_pairing, which needs numpy.

    A preflight run from the bare clone (``uv run --no-project``) before the
    install died on ModuleNotFoundError, resolved no voter and reported DEGRADED
    on every run, so the install comes first and the preflight uses its
    environment.
    """
    names = [str(s.get("name", "")) for s in _steps()]
    install = names.index("Install omniintelligence dependencies")
    preflight = next(i for i, n in enumerate(names) if n.startswith("Preflight"))
    assert install < preflight
    run = str(_step("Preflight")["run"])
    assert "--no-project" not in run
    assert "uv run --no-sync python" in run
    install_step = _step("Install omniintelligence dependencies")
    assert "if" not in install_step
    # A failed install fails the job; no step can read a verdict without it.
    assert "continue-on-error" not in install_step


@pytest.mark.live_contact(
    "tests/ci/fixtures/hostile_review_voters_overlay_live_omn20910.json"
)
def test_every_overlay_voter_resolved_to_a_live_single_model_endpoint(
    recorded_response: dict[str, object],
) -> None:
    """The overlay's voters, resolved by omniintelligence's loader, answered live.

    Deliberately not bound to the overlay's current contents: changing a voter
    is an overlay-only edit and must not require a new recording.
    """
    provenance = cast("dict[str, str]", recorded_response["_provenance"])
    # The recording was taken from the overlay path this workflow reads.
    assert OVERLAY_SUFFIX in provenance["source"]
    voters = cast("list[dict[str, object]]", recorded_response["voters"])
    required = [v for v in voters if v["required"] is True]
    assert len(required) >= 2
    assert len({v["endpoint_sha256"] for v in required}) == len(required)
    for voter in voters:
        assert voter["http_status"] == 200, voter["voter_id"]
        assert len(cast("list[str]", voter["served_model_ids"])) == 1, voter
        assert str(voter["resolved_from"]) == "endpoint_url", voter


# --- OMN-20910 widening: the voter's MODEL is a field of the overlay ----------
# These cases drive the same ``voter_endpoint_findings`` entry point as the
# ratchet above, so a pattern missing from it fails them.
_KNOWN_BAD_MODEL = """\
      REVIEW_MODEL: ${{ vars.REVIEW_MODEL }}
          VOTER_B_MODEL_NAME: ${{ vars.VOTER_B_MODEL_NAME }}
      REVIEW_VOTER_MODEL: claude-opus-5-5
          MODEL: qwen3-coder-30b
            run: ./vote --engine gpt-4o
"""
_KNOWN_BAD_ENV_WORKFLOW = """\
jobs:
  hostile-review:
    env:
      REVIEW_VOTER_MODEL: my-voter-model
    steps:
      - name: Run adversarial review
        env:
          MODEL: voter-b
          REVIEW_MODELS: ${{ steps.review.outputs.models_succeeded }}
"""


def test_voter_model_from_overlay_ratchet_finds_every_known_bad_line() -> None:
    """Positive control: a zero on the real workflows is only evidence if this finds five."""
    assert len(voter_endpoint_findings(_KNOWN_BAD_MODEL)) == 5
    # One control per pattern: a count alone survives a broken pattern whenever
    # another pattern matches the same line.
    lines = _KNOWN_BAD_MODEL.splitlines()
    assert _VOTER_MODEL_PATTERNS
    for pattern in _VOTER_MODEL_PATTERNS:
        assert any(pattern.search(line) for line in lines), pattern.pattern


@pytest.mark.parametrize(
    "line",
    [
        "          REVIEW_MODEL: ${{ vars.REVIEW_MODEL }}",
        "          X: ${{ vars.VOTER_B_MODEL_NAME }}",
        "          X: ${{ vars.model }}",
        "          X: ${{ vars.LLM_MODEL_ID }}",
        "          VOTER: claude-sonnet-5-5",
        "          VOTER: Qwen3.8-27B",
        "          VOTER: glm-5.3",
        "          VOTER: gpt-4o",
    ],
)
def test_voter_model_from_overlay_ratchet_refuses_each_shape(line: str) -> None:
    assert voter_endpoint_findings(line + "\n"), line


def test_voter_model_from_overlay_ratchet_ignores_comments_and_overlay_reads() -> None:
    clean = (
        "      # a voter's model lives in the overlay, e.g. vars.X_MODEL is retired\n"
        '          --voters-overlay "$REVIEW_VOTERS_OVERLAY" \\\n'
        "          REVIEW_MODELS: ${{ steps.review.outputs.models_succeeded }}\n"
    )
    assert voter_endpoint_findings(clean) == []


def test_voter_model_from_overlay_env_literal_has_a_positive_control() -> None:
    """A bare id no pattern knows is refused by its ``env:`` key; an expression is not."""
    findings = voter_endpoint_findings(_KNOWN_BAD_ENV_WORKFLOW)
    assert len(findings) == 2, findings
    assert any("REVIEW_VOTER_MODEL=my-voter-model" in f for f in findings)
    assert any("MODEL=voter-b" in f for f in findings)
    assert not any("REVIEW_MODELS" in f for f in findings)


@pytest.mark.parametrize("path", _reviewer_workflows(), ids=lambda p: p.name)
def test_voter_model_from_overlay_no_reviewer_workflow_names_a_model(
    path: Path,
) -> None:
    findings = voter_endpoint_findings(path.read_text(encoding="utf-8"))
    assert findings == [], (
        f"{path.name} names a review voter model, endpoint or port; a voter's "
        f"model is declared only in the overlay (OMN-20910): {findings}"
    )


_THIS_TEST = "tests/ci/test_hostile_reviewer_voters_overlay_omn20910.py"


def test_voter_model_from_overlay_ratchet_is_wired_as_ci_step_and_hook() -> None:
    """The ratchet runs in a job the required ``CI Summary`` context covers, and in pre-commit."""
    ci = yaml.safe_load((_WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    lint_steps = ci["jobs"]["lint"]["steps"]
    assert any(_THIS_TEST in str(step.get("run", "")) for step in lint_steps)
    # CI Summary polls every ci.yml job's conclusion, the lint job included.
    required = yaml.safe_load(
        (_ROOT / ".github" / "required-checks.yaml").read_text(encoding="utf-8")
    )
    assert any(
        gate["name"] == "CI Summary"
        and gate["mode"] == "REQUIRED"
        and gate["workflow"] == "ci.yml"
        for gate in required["gates"]
    )
    hooks = [
        hook
        for repo in yaml.safe_load(
            (_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        )["repos"]
        for hook in repo["hooks"]
    ]
    assert any(_THIS_TEST in str(hook.get("entry", "")) for hook in hooks)

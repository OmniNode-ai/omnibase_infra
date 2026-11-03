# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the Hostile Reviewer workflow's own steps with a fake reviewer and a fake node (OMN-22275).

The ``run:`` text of the review step, the gate-accept step and the enforcement step is
executed as written, in the order the job runs it, with the ``${{ }}`` expressions
substituted and each step's ``if:`` evaluated over the earlier steps' outputs. Only the
world is faked:

* ``uv`` stands in for the reviewer (``cli_review`` writes the recorded document to the
  ``--output`` path the step names) and for the gate-accept node's ``python -m`` entry.
  The node fake checks that the step runs it from the omnimarket clone's source with the
  input the step names (the exact pull request head, the reviewer's result file), reads
  that result file, and then behaves as the node's contract does: a degraded verdict with a
  verified receipt for the exact head is ``pass_agent``, a receipt for another head is
  ``refuse``, a blocked quorum or a blocking finding from a voter that succeeded stays
  ``unchanged``.
* the omnimarket clone is a directory the test creates (with or without the node).

The gate job's script then runs with the root job's conclusion. The result is the
conclusion of the two contexts the workflow reports into CI Summary's external sweep.

Steps outside review, accept and enforce (checkout, overlay, install, preflight, the PR
comment) are not run: they do not decide the verdict, and the preflight is taken as
having found the endpoint reachable.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "hostile-reviewer.yml"

REVIEW_JOB = "hostile-review"
GATE_JOB = "hostile-review-gate"
REPOSITORY = "example-org/example-repo"
PULL_REQUEST = 41
HEAD_SHA = "a" * 40
OTHER_HEAD_SHA = "c" * 40
RESULT_FILE_NAME = "hostile-review-result.json"
NODE_DIR = "src/omnimarket/nodes/node_agent_review_gate_accept_effect"

REVIEWER_DOCUMENTS: dict[str, dict[str, Any]] = {
    "passed": {
        "models_attempted": ["voter-1", "voter-2"],
        "models_succeeded": ["voter-1", "voter-2"],
        "models_failed": [],
        "results": [
            {"model": "voter-1", "success": True, "error": None, "findings": []},
            {"model": "voter-2", "success": True, "error": None, "findings": []},
        ],
        "total_findings": 0,
        "quorum": {"verdict": "passed", "blocking_count": 0, "warning_count": 0},
    },
    "degraded": {
        "models_attempted": ["voter-1", "voter-2"],
        "models_succeeded": [],
        "models_failed": ["voter-1", "voter-2"],
        "results": [
            {"model": "voter-1", "success": False, "error": "503", "findings": []},
            {"model": "voter-2", "success": False, "error": "503", "findings": []},
        ],
        "total_findings": 0,
        "quorum": {
            "verdict": "degraded_quorum",
            "blocking_count": 0,
            "warning_count": 0,
        },
    },
    "degraded_blocking": {
        "models_attempted": ["voter-1", "voter-2"],
        "models_succeeded": ["voter-2"],
        "models_failed": ["voter-1"],
        "results": [
            {"model": "voter-1", "success": False, "error": "503", "findings": []},
            {
                "model": "voter-2",
                "success": True,
                "error": None,
                "findings": [{"severity": "error", "message": "blocking finding"}],
            },
        ],
        "total_findings": 1,
        "quorum": {
            "verdict": "degraded_quorum",
            "blocking_count": 0,
            "warning_count": 0,
        },
    },
    "blocked": {
        "models_attempted": ["voter-1", "voter-2"],
        "models_succeeded": ["voter-1", "voter-2"],
        "models_failed": [],
        "results": [
            {"model": "voter-1", "success": True, "error": None, "findings": []},
            {"model": "voter-2", "success": True, "error": None, "findings": []},
        ],
        "total_findings": 2,
        "quorum": {"verdict": "blocked", "blocking_count": 1, "warning_count": 0},
    },
}

_UV = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
joined = " ".join(argv)
NODE = "omnimarket.nodes.node_agent_review_gate_accept_effect.runtime_gate_accept"
if "cli_review" in joined:
    output = Path(argv[argv.index("--output") + 1])
    output.write_text(Path(os.environ["FAKE_REVIEWER_DOCUMENT"]).read_text())
    sys.exit(int(os.environ.get("FAKE_REVIEWER_EXIT", "0")))
if NODE in argv:
    with open(os.environ["FAKE_NODE_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(argv) + "\\n")
    if os.path.realpath(os.environ.get("PYTHONPATH", "/")) != os.path.realpath(os.environ["FAKE_ACCEPT_SRC"]):
        sys.exit("fake node: not run from the omnimarket clone: " + repr(os.environ.get("PYTHONPATH")))
    if argv[:3] != ["run", "--no-sync", "python"] and argv[:2] != ["--no-sync", "python"]:
        sys.exit("fake node: unexpected uv arguments " + repr(argv))
    flags = argv[argv.index(NODE) + 1:]
    payload = {flags[i][2:].replace("-", "_"): flags[i + 1] for i in range(0, len(flags), 2)}
    payload["pull_request"] = int(payload["pull_request"])
    expected = json.loads(os.environ["FAKE_NODE_EXPECT"])
    if payload != expected:
        sys.exit("fake node: input " + repr(payload) + " is not " + repr(expected))
    if not os.path.exists(payload["voter_result_path"]):
        sys.exit("fake node: the voter result file the step named does not exist")
    voters = json.loads(Path(payload["voter_result_path"]).read_text())
    verdict = voters["quorum"]["verdict"]
    if os.environ.get("FAKE_NODE_EXIT", "0") != "0":
        sys.exit(int(os.environ["FAKE_NODE_EXIT"]))
    receipt_head = os.environ.get("FAKE_RECEIPT_HEAD", "")
    if verdict == "passed":
        decision, reason = "unchanged", "voters_passed"
    elif verdict == "blocked":
        decision, reason = "unchanged", "voters_blocked"
    elif any(
        f.get("severity") == "error"
        for r in voters["results"] if r["success"] for f in r["findings"]
    ):
        decision, reason = "unchanged", "blocking_finding"
    elif not receipt_head:
        decision, reason = "unchanged", "no_receipt"
    elif receipt_head != payload["head_sha"]:
        decision, reason = "refuse", "receipt_for_another_head"
    else:
        decision, reason = "pass_agent", "receipt_verified"
    with open(payload["step_output_path"], "a", encoding="utf-8") as handle:
        handle.write("decision=" + decision + "\\n")
        handle.write("reason=" + reason + "\\n")
        handle.write("receipt_id=" + ("rcpt-1" if decision == "pass_agent" else "") + "\\n")
sys.exit(0)
"""
_SLEEP = "#!/usr/bin/env bash\nexit 0\n"


def load_workflow() -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(WORKFLOW_PATH.read_text()))


def job(name: str) -> dict[str, Any]:
    return cast("dict[str, Any]", load_workflow()["jobs"][name])


def step_named(job_id: str, prefix: str) -> dict[str, Any]:
    found = [
        s for s in job(job_id)["steps"] if str(s.get("name", "")).startswith(prefix)
    ]
    assert len(found) == 1, f"expected one step named {prefix!r}, found {len(found)}"
    return cast("dict[str, Any]", found[0])


def reported_contexts() -> list[str]:
    """The two contexts CI Summary's external sweep reads for this workflow's verdict."""
    return [str(job(REVIEW_JOB)["name"]), str(job(GATE_JOB)["name"])]


@dataclass
class Run:
    """The outcome of one simulated job: each step's exit and every context's result."""

    exits: dict[str, int]
    skipped: list[str]
    outputs: dict[str, dict[str, str]]
    node_calls: list[list[str]]
    contexts: dict[str, bool]

    def green(self, context: str) -> bool:
        return self.contexts[context]


def _evaluate(
    expression: str, outputs: dict[str, dict[str, str]], ctx: dict[str, str]
) -> str:
    for term in (t.strip() for t in expression.split("||")):
        match = re.fullmatch(r"steps\.([\w-]+)\.outputs\.([\w-]+)", term)
        if match:
            value = outputs.get(match.group(1), {}).get(match.group(2), "")
        elif term in ctx:
            value = ctx[term]
        else:
            raise AssertionError(f"unsupported expression term {term!r}")
        if value:
            return value
    return ""


def render(text: str, outputs: dict[str, dict[str, str]], ctx: dict[str, str]) -> str:
    rendered = re.sub(
        r"\$\{\{\s*(.+?)\s*\}\}",
        lambda m: _evaluate(m.group(1), outputs, ctx),
        text,
    )
    assert "${{" not in rendered, f"unsubstituted expression in:\n{rendered}"
    return rendered


def _holds(condition: Any, outputs: dict[str, dict[str, str]]) -> bool:
    """Evaluate an ``if:`` made of ``always()`` and output comparisons joined by ``&&``."""
    if condition is None:
        return True
    for clause in (c.strip() for c in str(condition).split("&&")):
        if clause == "always()":
            continue
        match = re.fullmatch(
            r"steps\.([\w-]+)\.outputs\.([\w-]+) (==|!=) '([^']*)'", clause
        )
        if not match:
            raise AssertionError(f"unsupported if: clause {clause!r}")
        value = outputs.get(match.group(1), {}).get(match.group(2), "")
        if (value == match.group(4)) != (match.group(3) == "=="):
            return False
    return True


def _parse_outputs(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                values[key] = value
    return values


def _bash(
    script: str, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-e", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=cwd,
    )


def simulate(
    work: Path,
    *,
    verdict: str,
    receipt_head: str = "",
    review_step_prefix: str = "Run adversarial review",
    enforce_prefix: str | None = "Enforce hostile-review verdict",
    review_exit: int = 0,
    node_in_clone: bool = True,
    node_exit: int = 0,
) -> Run:
    """Run the job's review, accept and enforcement steps, then the gate job."""
    bin_dir = work / "bin"
    bin_dir.mkdir()
    for name, text in (("uv", _UV), ("sleep", _SLEEP)):
        path = bin_dir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    workspace = work / "workspace"
    workspace.mkdir()
    cwd = work / "omniintelligence"
    cwd.mkdir()
    clone = work / "omnimarket"
    (clone / "src" / "omnimarket").mkdir(parents=True)
    if node_in_clone:
        (clone / NODE_DIR).mkdir(parents=True)
    runner_temp = work / "runner-temp"
    runner_temp.mkdir()
    document = work / "reviewer-document.json"
    document.write_text(json.dumps(REVIEWER_DOCUMENTS[verdict]))
    node_log = work / "node.log"
    result_path = workspace / RESULT_FILE_NAME
    base_env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(work),
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_WORKSPACE": str(workspace),
        "FAKE_REVIEWER_DOCUMENT": str(document),
        "FAKE_REVIEWER_EXIT": str(review_exit),
        "FAKE_RECEIPT_HEAD": receipt_head,
        "FAKE_NODE_LOG": str(node_log),
        "FAKE_NODE_EXIT": str(node_exit),
        "FAKE_ACCEPT_SRC": str(workspace / ".." / "omnimarket" / "src"),
        "PR_NUMBER": str(PULL_REQUEST),
        "REPO": REPOSITORY,
        "HEAD_SHA": HEAD_SHA,
        "REVIEW_VOTERS_OVERLAY": str(work / "overlay.yaml"),
        "LOCAL_LLM_SHARED_SECRET": "unused",
        "GITHUB_TOKEN": "unused",
    }
    ctx = {
        "github.repository": REPOSITORY,
        "github.event.pull_request.number": str(PULL_REQUEST),
        "github.event.pull_request.head.sha": HEAD_SHA,
        "github.workspace": str(workspace),
        "runner.temp": str(runner_temp),
        "secrets.GITHUB_TOKEN": "unused",
        "secrets.LOCAL_LLM_SHARED_SECRET": "unused",
    }
    plan: list[tuple[str, dict[str, Any]]] = [
        ("review", step_named(REVIEW_JOB, review_step_prefix)),
    ]
    accept = [s for s in job(REVIEW_JOB)["steps"] if s.get("id") == "accept"]
    if accept:
        plan.append(("accept", accept[0]))
    if enforce_prefix is not None:
        plan.append(("enforce", step_named(REVIEW_JOB, enforce_prefix)))
    outputs: dict[str, dict[str, str]] = {"preflight": {}}
    exits: dict[str, int] = {}
    skipped: list[str] = []
    failed = False
    for key, step in plan:
        condition = step.get("if")
        runs = _holds(condition, outputs) if condition is not None else not failed
        if not runs:
            skipped.append(key)
            continue
        out_path = work / f"{key}.output"
        summary_path = work / f"{key}.summary"
        env = dict(base_env) | {
            "GITHUB_OUTPUT": str(out_path),
            "GITHUB_STEP_SUMMARY": str(summary_path),
            "FAKE_NODE_EXPECT": json.dumps(
                {
                    "repository": REPOSITORY,
                    "pull_request": PULL_REQUEST,
                    "head_sha": HEAD_SHA,
                    "voter_result_path": str(result_path),
                    "step_output_path": str(out_path),
                    "step_summary_path": str(summary_path),
                }
            ),
        }
        for name, value in (step.get("env") or {}).items():
            env[str(name)] = render(str(value), outputs, ctx)
        script = render(str(step["run"]), outputs, ctx)
        completed = _bash(script, env, cwd)
        exits[key] = completed.returncode
        outputs[str(step.get("id", key))] = _parse_outputs(out_path)
        if completed.returncode != 0:
            failed = True
    result = "failure" if failed else "success"
    gate_ctx = {f"needs.{REVIEW_JOB}.result": result}
    gate_script = render(
        next(str(s["run"]) for s in job(GATE_JOB)["steps"] if "run" in s),
        {},
        gate_ctx,
    )
    node_calls = (
        [json.loads(line) for line in node_log.read_text().splitlines()]
        if node_log.exists()
        else []
    )
    gate = _bash(
        gate_script, base_env | {"GITHUB_STEP_SUMMARY": str(work / "g.sum")}, work
    )
    contexts = {
        str(job(REVIEW_JOB)["name"]): not failed,
        str(job(GATE_JOB)["name"]): gate.returncode == 0,
    }
    return Run(exits, skipped, outputs, node_calls, contexts)

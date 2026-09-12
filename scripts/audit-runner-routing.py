#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Audit GitHub Actions runner routing for trusted OmniNode CI.

The incident this protects against: repo-level GitHub variables drifted
OMNI_TRUSTED_CI_RUNS_ON_JSON back to ["ubuntu-latest"], so trusted CI silently
used GitHub-hosted minutes even though the workflow selector looked correct.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ORG = "OmniNode-ai"
DEFAULT_POLICY = Path("config/runner_routing_policy.yaml")
PUBLIC_PR_RUNNER_VARIABLE = "OMNI_PUBLIC_PR_RUNS_ON_JSON"
TRUSTED_CI_RUNNER_VARIABLE = "OMNI_TRUSTED_CI_RUNS_ON_JSON"
# The GitHub-hosted label a job may not pin directly without a policy exception.
HOSTED_RUNNER_LABEL = "ubuntu-latest"
REQUIRED_CI_RUNNER_VARIABLE = "OMNI_REQUIRED_CI_RUNS_ON_JSON"
FORK_PR_PREDICATE = (
    "github.event_name=='pull_request'&&"
    "github.event.pull_request.head.repo.full_name!=github.repository"
)
DEV_BASE_SHORTCUT = "github.event_name=='pull_request'&&github.base_ref=='dev'"
# OMN-18031: the per-run routing consumer shape. A job whose runs-on resolves
# from a route job's output rather than from the seam expression directly.
ROUTE_CONSUMER_RE = re.compile(r"needs\.([A-Za-z0-9_-]+)\.outputs\.labels")
# OMN-18031: the selector-generation markers. V1 is the inline seam expression
# that 94 job definitions in this repo carry; V2 is a job whose placement comes
# from a route job's output. The marker is a COMMENT, so YAML parsing drops it
# and these two passes are necessarily textual.
SELECTOR_V1_MARKER = "OMNI_RUNNER_SELECTOR_V1"
SELECTOR_V2_MARKER = "OMNI_RUNNER_SELECTOR_V2"
# A marker sits directly above the `runs-on:` it describes. The bound exists so
# an unrelated marker elsewhere in the same job cannot satisfy the requirement.
SELECTOR_MARKER_WINDOW = 24
REQUIRED_OVERRIDE_ACTIVATION_GATE_KEYS: tuple[str, ...] = (
    "sustained_samples",
    "sustained_min_span_seconds",
    "capacity_budget",
    "maintenance_roll_convergence",
    "evidence_companion_fate_isolation",
    "positive_control_acceptance",
)


@dataclass(frozen=True)
class Finding:
    scope: str
    message: str


def _load_policy(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _canonical_json(value: str) -> str:
    return json.dumps(json.loads(value), separators=(",", ":"))


def _require_override_activation_gate(repo_name: str, override: dict[str, Any]) -> None:
    gate = override.get("activation_gate")
    if not isinstance(gate, dict):
        raise ValueError(
            f"repository_overrides[{repo_name}] must carry activation_gate: "
            "future flips need sustained sampling, a capacity budget, maintenance "
            "convergence, evidence-companion fate isolation, and a positive control"
        )
    for key in REQUIRED_OVERRIDE_ACTIVATION_GATE_KEYS:
        if key not in gate:
            raise ValueError(
                f"repository_overrides[{repo_name}].activation_gate is missing "
                f"required key {key!r}"
            )
        value = gate[key]
        if isinstance(value, str) and not value.strip():
            raise ValueError(
                f"repository_overrides[{repo_name}].activation_gate[{key!r}] "
                "must not be blank"
            )
        if value is None:
            raise ValueError(
                f"repository_overrides[{repo_name}].activation_gate[{key!r}] "
                "must not be null"
            )


def _run_gh(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _variables(args: list[str]) -> list[dict[str, Any]]:
    result = _run_gh(
        [
            "variable",
            "list",
            *args,
            "--json",
            "name,value",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh variable list failed: {result.stderr.strip()}")
    values = json.loads(result.stdout or "[]")
    if not isinstance(values, list):
        raise RuntimeError("gh variable list returned a non-list payload")
    return values


def _variable_value(values: list[dict[str, Any]], name: str) -> str | None:
    for item in values:
        if item.get("name") == name:
            value = item.get("value")
            return value if isinstance(value, str) else None
    return None


def audit_github_variables(policy: dict[str, Any]) -> list[Finding]:
    variable = policy["trusted_runner_variable"]
    name = str(variable["name"])
    expected = _canonical_json(str(variable["expected_json"]))
    findings: list[Finding] = []
    org_actual = _variable_value(_variables(["--org", ORG]), name)
    if org_actual is None:
        findings.append(Finding(ORG, f"missing org variable {name}"))
    else:
        try:
            org_normalized = _canonical_json(org_actual)
        except json.JSONDecodeError:
            findings.append(Finding(ORG, f"{name} is not valid JSON: {org_actual!r}"))
            org_normalized = ""
        if org_normalized != expected:
            findings.append(
                Finding(
                    ORG,
                    f"{name} drifted to {org_actual!r}; expected {variable['expected_json']!r}",
                )
            )
    overrides = variable.get("repository_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(
            "trusted_runner_variable.repository_overrides must be a mapping"
        )
    for repo in policy.get("repositories", []):
        repo_name = str(repo)
        # OMN-18031: a repo shadow the policy DECLARES is an asserted value in
        # its own right, not drift from the org value. Without this the audit
        # cannot tell a deliberate per-repo divergence from the silent drift it
        # exists to catch, so a legitimate one produces a permanent red -- and a
        # permanently red audit is the failure mode OMN-16727 recorded, where
        # the job exits on the first finding set and masks every later surface.
        override = overrides.get(repo_name)
        if override is None:
            repo_expected_raw = str(variable["expected_json"])
        else:
            if not isinstance(override, dict):
                raise ValueError(f"repository_overrides[{repo_name}] must be a mapping")
            if not str(override.get("revert_when", "")).strip():
                raise ValueError(
                    f"repository_overrides[{repo_name}] must carry a revert_when: "
                    "a declared divergence with no stated end is indistinguishable "
                    "from drift to the next lane"
                )
            _require_override_activation_gate(repo_name, override)
            repo_expected_raw = str(override["expected_json"])
        repo_expected = _canonical_json(repo_expected_raw)
        actual = _variable_value(_variables(["--repo", f"{ORG}/{repo_name}"]), name)
        if actual is None:
            if override is not None:
                findings.append(
                    Finding(
                        repo_name,
                        f"{name} has no repo shadow but the policy declares an "
                        f"override of {repo_expected_raw!r}; the repo silently "
                        f"inherits the org value instead",
                    )
                )
            continue
        try:
            normalized = _canonical_json(actual)
        except json.JSONDecodeError:
            findings.append(Finding(repo_name, f"{name} is not valid JSON: {actual!r}"))
            continue
        if normalized != repo_expected:
            findings.append(
                Finding(
                    repo_name,
                    f"{name} drifted to {actual!r}; expected {repo_expected_raw!r}",
                )
            )
    return findings


def _workflow_paths(repo_root: Path) -> list[Path]:
    workflow_dir = repo_root / ".github" / "workflows"
    if not workflow_dir.exists():
        return []
    return sorted(
        path
        for path in workflow_dir.iterdir()
        if path.suffix in {".yml", ".yaml"} and path.is_file()
    )


def _normalized_expression(value: str) -> str:
    return re.sub(r"\s+", "", value)


class UnreadableRunsOnError(Exception):
    """A job declares runs-on in a shape this auditor cannot interpret.

    Raised rather than returning an empty label set, because an empty set is
    indistinguishable from a compliant job and would let the gate silently stop
    gating. See ``runs_on_labels``.
    """


def runs_on_labels(runs_on: Any) -> list[str]:
    """Return the runner labels a job's ``runs-on`` value resolves to.

    Reads the PARSED value, never the source text. That is the whole point of
    this helper: YAML resolves a block scalar to a plain string, so
    ``runs-on: >-`` followed by the label on the next line is the same value as
    ``runs-on: ubuntu-latest``, and a line-oriented matcher anchored on
    ``runs-on:`` sees only the ``>-`` and misses the label entirely. The same
    hole swallows a literal block scalar (``|-``) and an ordinary multi-line
    sequence.

    Accepts the three shapes Actions itself accepts -- a string, a sequence of
    strings, and the group/labels mapping -- and raises ``UnreadableRunsOnError`` for
    anything else, naming the type. Callers must not convert that to an empty
    result.
    """
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        if not all(isinstance(item, str) for item in runs_on):
            raise UnreadableRunsOnError(
                f"runs-on sequence contains a non-string entry: {runs_on!r}"
            )
        return list(runs_on)
    if isinstance(runs_on, dict):
        labels = runs_on.get("labels", [])
        if isinstance(labels, str):
            return [labels]
        if isinstance(labels, list) and all(isinstance(i, str) for i in labels):
            return list(labels)
        raise UnreadableRunsOnError(
            f"runs-on mapping has an unreadable labels entry: {labels!r}"
        )
    raise UnreadableRunsOnError(
        f"runs-on is a {type(runs_on).__name__}, which is not a string, "
        "sequence or group/labels mapping"
    )


def is_bare_hosted_pin(labels: list[str]) -> bool:
    """True when the resolved labels are a literal hosted pin, not an expression.

    A value carrying ``${{`` is selector-driven and is judged elsewhere; only a
    label that Actions would use verbatim counts as a bare pin here.
    """
    if any("${{" in label for label in labels):
        return False
    return any(label.strip() == HOSTED_RUNNER_LABEL for label in labels)


def runner_variable_for_event(
    event_name: str,
    head_repository: str | None,
    repository: str,
    *,
    merge_group_variable: str | None = None,
) -> str:
    """Return the runner variable allowed for a workflow event shape.

    The helper models the selector policy for focused regression tests. Workflow
    expressions are audited separately below because Actions evaluates them.
    """
    if event_name == "pull_request" and head_repository != repository:
        return PUBLIC_PR_RUNNER_VARIABLE
    if event_name == "merge_group" and merge_group_variable is not None:
        return merge_group_variable
    return TRUSTED_CI_RUNNER_VARIABLE


SCOPED_VARIABLE_KEYS = (
    "docker_ci_runner_variable",
    "security_scan_runner_variable",
)


def audit_scoped_variables(policy: dict[str, Any]) -> list[Finding]:
    """Audit the narrower routing variables that sit AHEAD of the trusted seam.

    ADDITIVE ON PURPOSE, and separate from ``audit_github_variables`` because the
    trusted seam has per-repo override semantics these do not: a scoped variable
    is set on a closed set of repositories and must be ABSENT everywhere else.

    Why this pass exists (OMN-18205). ``OMNI_DOCKER_CI_RUNS_ON_JSON`` and
    ``OMNI_SECURITY_SCAN_RUNS_ON_JSON`` have been live for weeks -- the first on
    this repository, the second here and on omniclaude -- and were read by no
    rule, no plan enumeration and no audit surface. Ten job definitions in this
    repository resolve their runner through the docker one, ahead of the seam, so
    a silent edit to it moves ten jobs and the seam's own audit reports green.

    The half that catches drift is the ABSENCE assertion. A declared scope
    drifting in value is the obvious case; a NEW shadow appearing on a repository
    that declares none is the invisible one, because nothing else in the estate
    would ever mention it. Both are findings here.
    """
    findings: list[Finding] = []
    audited = [str(r) for r in policy.get("repositories", [])]
    for key in SCOPED_VARIABLE_KEYS:
        declaration = policy.get(key)
        if declaration is None:
            findings.append(
                Finding(
                    "policy",
                    f"{key} is missing from the policy file; this pass audits it "
                    f"by name, so removing the declaration silently stops "
                    f"auditing a live variable",
                )
            )
            continue
        name = str(declaration["name"])
        expected_raw = str(declaration["expected_json"])
        expected = _canonical_json(expected_raw)
        declared = [str(r) for r in declaration["repositories"]]
        org_scope = str(declaration["org_scope"])
        if org_scope != "absent":
            raise ValueError(
                f"{key}.org_scope must be 'absent'; an org-level value for a "
                f"scoped variable would silently govern every repository"
            )

        org_actual = _variable_value(_variables(["--org", ORG]), name)
        if org_actual is not None:
            findings.append(
                Finding(
                    ORG,
                    f"{name} is declared org_scope: absent but an org-level "
                    f"value {org_actual!r} exists; it would govern every repo",
                )
            )

        undeclared = [r for r in declared if r not in audited]
        if undeclared:
            findings.append(
                Finding(
                    "policy",
                    f"{key}.repositories names {undeclared} which are not in the "
                    f"audited repositories list, so they are never read",
                )
            )

        for repo_name in audited:
            actual = _variable_value(_variables(["--repo", f"{ORG}/{repo_name}"]), name)
            if repo_name not in declared:
                if actual is not None:
                    findings.append(
                        Finding(
                            repo_name,
                            f"{name} appeared with value {actual!r} but this "
                            f"repository declares no scope for it; an undeclared "
                            f"shadow governs jobs ahead of the trusted seam",
                        )
                    )
                continue
            if actual is None:
                findings.append(
                    Finding(
                        repo_name,
                        f"{name} is declared for this repository at "
                        f"{expected_raw!r} but no shadow exists, so its jobs fall "
                        f"through to the trusted seam",
                    )
                )
                continue
            try:
                normalized = _canonical_json(actual)
            except json.JSONDecodeError:
                findings.append(
                    Finding(repo_name, f"{name} is not valid JSON: {actual!r}")
                )
                continue
            if normalized != expected:
                findings.append(
                    Finding(
                        repo_name,
                        f"{name} drifted to {actual!r}; expected {expected_raw!r}",
                    )
                )
    return findings


def audit_local_workflows(policy: dict[str, Any], repo_root: Path) -> list[Finding]:
    allowlist = {
        str(item["path"])
        for item in policy.get("hosted_runner_allowlist", [])
        if isinstance(item, dict) and "path" in item
    }
    findings: list[Finding] = []
    for path in _workflow_paths(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        if "pull_request_target" in text:
            findings.append(
                Finding(
                    rel,
                    "pull_request_target is prohibited because untrusted fork code must never reach self-hosted runners",
                )
            )

        workflow = yaml.safe_load(text)
        jobs = workflow.get("jobs", {}) if isinstance(workflow, dict) else {}
        if not isinstance(jobs, dict):
            continue
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue

            # OMN-18031: the bare-hosted-pin check reads the PARSED runs-on
            # value per job. It used to be a regex over the raw file anchored on
            # `runs-on:` with the label on the same line, which a block scalar
            # or a multi-line sequence walks straight past -- a gate that fails
            # open and reads as compliance. A job that delegates with `uses:`
            # has no runs-on of its own and is out of scope: the callee owns
            # placement.
            if "runs-on" in job:
                try:
                    labels = runs_on_labels(job["runs-on"])
                except UnreadableRunsOnError as exc:
                    findings.append(Finding(f"{rel}:{job_name}", str(exc)))
                    continue
                if is_bare_hosted_pin(labels) and rel not in allowlist:
                    findings.append(
                        Finding(
                            f"{rel}:{job_name}",
                            f"bare runs-on: {HOSTED_RUNNER_LABEL} is not allowed; "
                            "use OMNI_RUNNER_SELECTOR_V1 or add an explicit policy exception",
                        )
                    )
            elif "uses" not in job:
                findings.append(
                    Finding(
                        f"{rel}:{job_name}",
                        "job declares neither runs-on nor uses, so its runner placement cannot be audited",
                    )
                )

            runs_on = job.get("runs-on")
            if not isinstance(runs_on, str):
                continue
            expression = _normalized_expression(runs_on)
            if PUBLIC_PR_RUNNER_VARIABLE not in expression:
                continue
            if DEV_BASE_SHORTCUT in expression:
                findings.append(
                    Finding(
                        f"{rel}:{job_name}",
                        "public PR runner selection must not use the pull_request/dev-base shortcut",
                    )
                )
            if FORK_PR_PREDICATE not in expression:
                findings.append(
                    Finding(
                        f"{rel}:{job_name}",
                        "OMNI_PUBLIC_PR_RUNS_ON_JSON is allowed only for pull requests whose head repository differs from github.repository",
                    )
                )
    return findings


def audit_route_wiring(policy: dict[str, Any], repo_root: Path) -> list[Finding]:
    """Audit the OMN-18031 per-run routing consumer shape.

    ADDITIVE ON PURPOSE. This is a new pass inside ``--local-workflows``; it
    changes nothing about the two existing passes. That placement is deliberate
    and was checked against the live workflow rather than assumed: the
    OMN-16727 masking everyone cites is at the WORKFLOW-STEP level --
    ``.github/workflows/runner-routing-audit.yml`` runs ``--local-workflows``
    and ``--github-vars`` as two separate ``run:`` steps, so a failure in the
    first means the second never executes. ``main()`` itself already extends
    one findings list and prints them all, so adding a pass HERE inherits no
    masking. Do not "fix" ``main()`` on the strength of that ticket's title.

    What it enforces, and why each rule is mechanical rather than a review
    convention:

    1. A route consumer must NOT also reference ``OMNI_PUBLIC_PR_RUNS_ON_JSON``.
       Fork isolation lives INSIDE ``runner_route_decision.py`` (step S1),
       deliberately once, rather than being restated at each of the 46
       selector-carrying call sites in this repo where one of them can be
       edited wrong and nothing notices. A call site that re-implements the
       fork branch is re-opening exactly that hole.
    2. A route consumer must actually declare ``needs:`` on the job it reads.
       Without it the expression resolves to nothing, the job fails to
       SCHEDULE, and that does not look like a test failure.
    3. A workflow that consumes a route output must contain a job that
       produces one -- otherwise the consumer is reading a job that was
       deleted or renamed, which is the silent-retirement shape again.
    """
    allowlist = {
        str(item["path"])
        for item in policy.get("hosted_runner_allowlist", [])
        if isinstance(item, dict) and "path" in item
    }
    findings: list[Finding] = []
    for path in _workflow_paths(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        try:
            workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            findings.append(Finding(rel, f"workflow is not parseable YAML: {exc}"))
            continue
        jobs = workflow.get("jobs", {}) if isinstance(workflow, dict) else {}
        if not isinstance(jobs, dict):
            continue

        producers = {
            name
            for name, job in jobs.items()
            if isinstance(job, dict)
            and (
                "runner-route-reusable" in str(job.get("uses", ""))
                or "labels" in (job.get("outputs") or {})
            )
        }

        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            runs_on = job.get("runs-on")
            if not isinstance(runs_on, str):
                continue
            match = ROUTE_CONSUMER_RE.search(_normalized_expression(runs_on))
            if match is None:
                continue
            producer = match.group(1)
            scope = f"{rel}:{job_name}"

            if PUBLIC_PR_RUNNER_VARIABLE in _normalized_expression(runs_on):
                findings.append(
                    Finding(
                        scope,
                        "a per-run route consumer must not also reference "
                        f"{PUBLIC_PR_RUNNER_VARIABLE}; fork isolation is enforced inside "
                        "scripts/ci/runner_route_decision.py (step S1), not restated at "
                        "call sites where it can be edited wrong",
                    )
                )

            needs = job.get("needs")
            declared = (
                [needs]
                if isinstance(needs, str)
                else list(needs)
                if isinstance(needs, list)
                else []
            )
            if producer not in declared:
                findings.append(
                    Finding(
                        scope,
                        f"runs-on reads needs.{producer}.outputs.labels but the job does not "
                        f"declare needs: {producer} -- the expression would resolve to nothing "
                        "and the job would fail to schedule, which does not surface as a failure",
                    )
                )

            if producer not in jobs:
                findings.append(
                    Finding(
                        scope,
                        f"runs-on reads needs.{producer}.outputs.labels but no job named "
                        f"{producer!r} exists in this workflow",
                    )
                )
            elif producers and producer not in producers:
                findings.append(
                    Finding(
                        scope,
                        f"job {producer!r} is consumed as a route producer but declares no "
                        "'labels' output and does not call the route reusable workflow",
                    )
                )

        # A workflow carrying the routing decision must be pinned hosted: a
        # router that queues behind the fleet it routes onto cannot report on
        # saturation.
        if producers and "runner-route" in rel and rel not in allowlist:
            findings.append(
                Finding(
                    rel,
                    "a workflow that produces a routing decision must be listed in "
                    "hosted_runner_allowlist with a reason -- it must not share fate with "
                    "the fleet it routes onto",
                )
            )
    return findings


def _runs_on_blocks(text: str) -> list[tuple[int, str]]:
    """Yield ``(line_index, full_value)`` for every ``runs-on:`` key in ``text``.

    Textual on purpose. The marker these blocks are matched against is a YAML
    COMMENT, and ``yaml.safe_load`` discards comments -- so a pass built on the
    parsed document literally cannot see the thing it must check. Continuation
    lines of a folded scalar are joined in, because the V1 expression spans six
    lines and a line-at-a-time match would miss it.
    """
    lines = text.splitlines()
    blocks: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^(\s*)runs-on:\s*(.*)$", line)
        if match is None:
            continue
        indent = len(match.group(1))
        value = [match.group(2)]
        for follow in lines[idx + 1 :]:
            if not follow.strip():
                continue
            if len(follow) - len(follow.lstrip()) <= indent:
                break
            value.append(follow.strip())
        blocks.append((idx, " ".join(value)))
    return blocks


def _marker_comments_above(lines: list[str], runs_on_index: int) -> list[str]:
    """Comment lines between the job key and ``runs-on:``.

    Walking stops at the first NON-comment line indented less than the
    ``runs-on:`` key -- that is the job key itself. Comment lines are collected
    at any indentation, so the scan is not defeated by a marker written flush
    left, and the window never reaches into the job above.
    """
    runs_on_indent = len(lines[runs_on_index]) - len(lines[runs_on_index].lstrip())
    collected: list[str] = []
    lower = max(0, runs_on_index - SELECTOR_MARKER_WINDOW)
    for idx in range(runs_on_index - 1, lower - 1, -1):
        line = lines[idx]
        stripped = line.strip()
        if stripped.startswith("#"):
            collected.append(stripped)
            continue
        if not stripped:
            continue
        if len(line) - len(line.lstrip()) < runs_on_indent:
            break
    return collected


def audit_selector_markers(repo_root: Path) -> list[Finding]:
    """Audit the OMN-18031 V2 selector marker (additive).

    ADDITIVE AND TEXTUAL. ``audit_route_wiring`` above already enforces the
    STRUCTURE of a route consumer -- the ``needs:`` edge, the producer's
    existence, the fork-isolation prohibition. It cannot enforce the marker,
    because the marker is a comment and the structural pass reads parsed YAML.

    Why a comment is worth a gate at all, when it changes no behaviour: the V1
    marker is how 94 job definitions in this repo are FOUND. Every survey,
    migration and drift audit of runner placement to date -- including the one
    that established this design had zero prior art in the org -- was a grep for
    that string. A V2 consumer that carries no marker is invisible to the same
    grep, so the next survey silently under-counts the very jobs that moved, and
    the first person to notice is whoever is debugging a placement incident.

    Both directions are enforced, because each alone is a half-measure:

    1. A route-consuming ``runs-on:`` must carry the V2 marker above it.
    2. A V2 marker must be followed by a route-consuming ``runs-on:``. Without
       this, a job reverted from V2 back to the seam expression keeps a marker
       that now lies, which is worse than no marker.
    3. A route-consuming ``runs-on:`` must NOT still carry the V1 marker. That
       pairing is a half-finished migration whose comment contradicts its code.
    """
    findings: list[Finding] = []
    for path in _workflow_paths(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        consumer_indices: set[int] = set()
        for idx, value in _runs_on_blocks(text):
            if ROUTE_CONSUMER_RE.search(_normalized_expression(value)) is None:
                continue
            consumer_indices.add(idx)
            comments = "\n".join(_marker_comments_above(lines, idx))
            if SELECTOR_V2_MARKER not in comments:
                findings.append(
                    Finding(
                        f"{rel}:{idx + 1}",
                        "runs-on resolves from a route output but carries no "
                        f"{SELECTOR_V2_MARKER} marker comment; every survey of runner "
                        f"placement in this org is a grep for {SELECTOR_V1_MARKER}, so an "
                        "unmarked consumer is invisible to the next one",
                    )
                )
            if SELECTOR_V1_MARKER in comments:
                findings.append(
                    Finding(
                        f"{rel}:{idx + 1}",
                        f"runs-on resolves from a route output but still carries the "
                        f"{SELECTOR_V1_MARKER} marker; the comment contradicts the "
                        "expression it describes",
                    )
                )

        for idx, line in enumerate(lines):
            if SELECTOR_V2_MARKER not in line:
                continue
            if any(
                idx < consumer <= idx + SELECTOR_MARKER_WINDOW
                for consumer in consumer_indices
            ):
                continue
            findings.append(
                Finding(
                    f"{rel}:{idx + 1}",
                    f"{SELECTOR_V2_MARKER} is declared but no runs-on below it resolves "
                    "from a route output; a marker left behind by a revert states a "
                    "placement the workflow does not have",
                )
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--repo-root", type=Path, default=Path())
    parser.add_argument("--local-workflows", action="store_true")
    parser.add_argument("--github-vars", action="store_true")
    args = parser.parse_args()

    if not args.local_workflows and not args.github_vars:
        args.local_workflows = True
        args.github_vars = True

    policy = _load_policy(args.policy)
    findings: list[Finding] = []
    if args.local_workflows:
        findings.extend(audit_local_workflows(policy, args.repo_root))
        # OMN-18031, additive: the per-run routing consumer shape. Extends the
        # same findings list, so its results print alongside the existing pass
        # rather than replacing or short-circuiting it.
        findings.extend(audit_route_wiring(policy, args.repo_root))
        # OMN-18031, additive: the V2 selector marker. Textual rather than
        # structural, because the marker is a YAML comment the parsed
        # document above does not retain.
        findings.extend(audit_selector_markers(args.repo_root))
    if args.github_vars:
        findings.extend(audit_github_variables(policy))
        # OMN-18205: additive, and inside the SAME step as the seam audit so a
        # finding here cannot be masked by the step-level ordering OMN-16727
        # records -- main() extends one findings list and prints them all.
        findings.extend(audit_scoped_variables(policy))

    if findings:
        for finding in findings:
            print(
                f"::error title=Runner routing drift::{finding.scope}: {finding.message}"
            )
        return 1

    print("Runner routing audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

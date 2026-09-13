# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay the pre-commit hook stack over real merged history (OMN-17474).

WHY THIS EXISTS
    OMN-17474 sorts every hook into three populations and makes exactly one of
    them deletable:

        has caught something                      -> keep
        observed firing, never caught in a window -> candidate, window named
        no signal in either source                -> NOT a candidate

    The third row is load-bearing: silence is not evidence. Separating the
    middle population from the third needs a SECOND source beyond CI history,
    because a hook that blocks locally is never pushed and leaves no CI trace.
    Only 10.4% of distinct hook ids are attributable to CI at all.

    The originally-planned second source was a local firing telemetry window
    (OMN-17473). The 2026-09-09 operator decision recorded on OMN-17470 defers
    it until a supported pre-commit execution-observer API exists and authorises
    no hook wiring, installation, runtime activation, validator-configuration
    change, bus publication or wrapper implementation in the meantime.

    Replay is a second source that needs none of those. Run each hook, in
    check-only mode, against the real changed files of recently merged commits.
    A hook that never fires on any of them is inert for the code this
    organisation actually writes. Nothing is installed into anyone's
    `.git/hooks`, no config is edited, and no sink is activated: this is a
    measurement harness invoked by hand, off-box, against a scratch clone.

WHAT IS MEASURED, PRECISELY
    For each merged commit in a window, the changed-file list is computed and
    the hook stack is run against exactly those paths, once per stage. Every
    hook's own verdict is read from `pre-commit run --verbose`, which prints a
    `- hook id:` and `- duration:` line per hook, so one invocation yields a row
    per hook rather than requiring one invocation per hook id. That is the
    difference between ~600 process spawns and ~36,000.

    FIRED means the hook exited non-zero on that commit's changed files.
    SKIPPED means its `files`/`types` filter matched nothing, which is not a
    run and is never counted in the denominator.

TWO HOOK SOURCES, AND WHY `tip` IS THE DEFAULT
    `--hook-source tip` (default) pins the hook stack -- config, validator
    scripts, environment -- at the branch tip, and restores only the commit's
    CHANGED FILES to their as-of-that-commit content. It answers the question
    the deletion pass actually asks: does TODAY's hook fire on the diffs this
    organisation writes? A hook added last week is measured against the whole
    window rather than against the five days it has existed.

    `--hook-source as-of` checks out the commit wholesale, so both the hook and
    the code are historical. It is higher fidelity per commit and lower coverage
    per hook, and it is offered for cross-checking a specific finding.

THE CONFOUND, STATED RATHER THAN LEFT IMPLICIT
    A merged commit's tree already satisfied whatever hooks were in force when
    it landed. Replaying those same hooks over that same tree is therefore
    biased towards passing BY CONSTRUCTION, and a zero-fire result must never be
    read as "this hook can never fire". What a zero over a real window does
    support is the narrower claim the ticket needs: across N real changes, this
    hook blocked nothing -- which is the middle population, with N stated.

    `tip` mode weakens the bias but does not remove it, since most of the stack
    predates most of the window. The report must carry the window with the
    count, and a deletion decision still requires the CI source to agree.

    The opposite error is the dangerous one. A hook that errors for an
    environment reason exits non-zero and reads as FIRED, which KEEPS it. False
    fires are conservative; false zeros are not. So the harness never suppresses
    stderr, records every non-zero hook's output, and refuses to report a
    repository's zeros at all unless the positive control fired.

POSITIVE CONTROL -- ALWAYS RUN, NEVER OPTIONAL
    Before any commit is replayed, the harness appends known violations to a REAL
    tracked source file -- a hardcoded absolute path (operating rule 6), a
    hardcoded LAN address, a hardcoded event topic, and trailing whitespace --
    and runs the stack against it. At least one hook must fire WITH A FINDING.

    Two details are load-bearing. The target is a real tracked file rather than a
    new root-level one, because hooks scope themselves with `files:` patterns and
    a root-level probe is skipped by most of the stack. And an environment
    failure does not satisfy the control: a run whose only fires are "executable
    not found" has proved that a process can be launched, not that this
    repository's gates can execute, and its zeros are a measurement failure.

EXIT CODES
    0  measurement completed (the window may be partial; the artifact says so)
    1  could not measure -- bad repo, unreadable config, control did not fire
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import yaml

# Stages whose economics this measurement covers. `commit-msg`, `post-checkout`,
# `manual` and friends are deliberately out of scope: the deletion pass is about
# the two tiers that run on every change.
COMMIT_STAGES = frozenset({"pre-commit", "commit"})
PUSH_STAGES = frozenset({"pre-push", "push"})
MEASURED_STAGES = COMMIT_STAGES | PUSH_STAGES

# `pre-commit run --verbose` prints, per hook, a padded status line followed by
# detail lines. The status line's dot run is at least three dots; a parenthesised
# note such as `(no files to check)` may sit between the dots and the verdict.
_STATUS_RE = re.compile(
    r"^(?P<name>.*?)\.{3,}(?:\((?P<note>[^)]*)\))?(?P<status>Passed|Failed|Skipped)\s*$"
)
_HOOK_ID_RE = re.compile(r"^- hook id:\s*(?P<hook_id>\S+)\s*$")
_DURATION_RE = re.compile(r"^- duration:\s*(?P<seconds>[0-9.]+)s?\s*$")
_EXIT_CODE_RE = re.compile(r"^- exit code:\s*(?P<code>-?\d+)\s*$")

# Text a hook emits when it failed because the MACHINE was wrong rather than the
# code. This distinction is the whole reason fire output is captured: on the web
# repository `type-check` fired on 83 of 83 commits with "Executable `pnpm` not
# found", which is a missing toolchain, not 83 caught defects. Classifying it as
# a catch would promote an unmeasured hook to "keep" on no evidence at all.
ENVIRONMENT_FAILURE_MARKERS: tuple[str, ...] = (
    "not found",
    "No such file or directory",
    "command not found",
    "ModuleNotFoundError",
    "ImportError",
    "Permission denied",
    "was not found in the environment",
    "Failed to spawn",
    "error: Distribution not found",
    "Executable `",
)

# Appended to a REAL tracked file to make the control. A standalone file at the
# repository root is a weaker control than it looks: most hooks scope themselves
# with a `files:` pattern such as `^src/`, so a root-level probe is SKIPPED by
# the very hooks whose liveness is in question, and the control then "passes" on
# one incidental whitespace hook while the stack it was meant to prove never ran.
_CONTROL_VIOLATIONS = (
    "\n\n# OMN-17474 replay positive control -- never committed.\n"
    # Split so this module does not itself carry the literals it plants.
    "OMN17474_CONTROL_PATH = " + '"/Users/" + "someone/Code/omni_home"' + "\n"
    "OMN17474_CONTROL_HOST = " + '"192." + "168.86.201:9092"' + "\n"
    "OMN17474_CONTROL_TOPIC = " + '"onex.evt." + "omn17474.control.v1"' + "\n"
    "OMN17474_CONTROL_TRAILING = 1   \n"
)


@dataclass(frozen=True)
class HookRun:
    """One hook's verdict on one commit, as read from pre-commit's own output."""

    hook_id: str
    status: str  # passed | failed | skipped
    duration_s: float | None
    exit_code: int | None
    # The hook's own output, kept only for a fire. This is what separates "the
    # code was wrong" from "the environment was wrong" -- a missing interpreter
    # exits non-zero exactly like a real finding, and without the text the two
    # are indistinguishable in the artifact.
    output: str = ""

    @property
    def fired(self) -> bool:
        return self.status == "failed"

    @property
    def ran(self) -> bool:
        return self.status in {"passed", "failed"}


@dataclass
class CommitRecord:
    """Everything the replay observed for one commit at one stage."""

    commit: str
    committed_at: str
    stage: str
    changed_files: int
    wall_seconds: float
    runs: list[HookRun] = field(default_factory=list)
    error: str | None = None


@dataclass
class HookSummary:
    """Per-hook rollup across the replayed window."""

    hook_id: str
    stage: str
    runs: int
    fires: int
    skips: int
    first_fire_commit: str | None
    last_fire_commit: str | None
    median_duration_s: float | None
    total_duration_s: float


def load_hook_stages(config_path: Path) -> dict[str, str]:
    """Map every declared hook id to the measured stage it runs in.

    A hook-level `stages` beats the file's `default_stages`; that asymmetry is
    why the pre-push tier measured 41 against 29 declared in OMN-17468. Hooks
    that run in neither measured stage are omitted entirely rather than being
    silently attributed to one.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    default_stages = set(raw.get("default_stages") or ["pre-commit"])
    out: dict[str, str] = {}
    for repo in raw.get("repos") or []:
        for hook in repo.get("hooks") or []:
            hook_id = hook.get("id")
            if not hook_id:
                continue
            stages = set(hook.get("stages") or default_stages)
            measured = stages & MEASURED_STAGES
            if not measured:
                continue
            # A hook declared in both tiers is attributed to the commit tier,
            # which is the one whose per-firing cost dominates.
            out[hook_id] = "pre-commit" if measured & COMMIT_STAGES else "pre-push"
    return out


def parse_precommit_output(text: str, output_chars: int = 2000) -> list[HookRun]:
    """Read one `pre-commit run --verbose` transcript into per-hook verdicts.

    The status line carries the hook's NAME, which is not unique and not the id;
    the following `- hook id:` line carries the id. Rows are therefore keyed off
    the id line and inherit the most recent status seen above them, so a hook
    whose own output happens to look like a status line cannot invent a row.
    """
    runs: list[HookRun] = []
    pending_status: str | None = None
    hook_id: str | None = None
    duration: float | None = None
    exit_code: int | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal hook_id, duration, exit_code, pending_status, body
        if hook_id is not None and pending_status is not None:
            status = pending_status.lower()
            # pre-commit reports a fixer that rewrote a file as Failed; that is
            # a fire, and deliberately so -- it would have blocked the commit.
            runs.append(
                HookRun(
                    hook_id=hook_id,
                    status=status,
                    duration_s=duration,
                    exit_code=exit_code,
                    output=(
                        "\n".join(body).strip()[:output_chars]
                        if status == "failed"
                        else ""
                    ),
                )
            )
        hook_id = None
        duration = None
        exit_code = None
        body = []

    for line in text.splitlines():
        stripped = line.rstrip()
        status_match = _STATUS_RE.match(stripped)
        if status_match:
            flush()
            pending_status = status_match.group("status")
            continue
        id_match = _HOOK_ID_RE.match(stripped)
        if id_match:
            if hook_id is not None:
                flush()
            hook_id = id_match.group("hook_id")
            continue
        duration_match = _DURATION_RE.match(stripped)
        if duration_match:
            duration = float(duration_match.group("seconds"))
            continue
        exit_match = _EXIT_CODE_RE.match(stripped)
        if exit_match:
            exit_code = int(exit_match.group("code"))
            continue
        if hook_id is not None and stripped:
            body.append(stripped)
    flush()
    return runs


def classify_fire(output: str) -> str:
    """`environment` when a fire names a broken machine, else `finding`.

    Conservative by design: a fire whose text is empty or unrecognised stays a
    `finding`, because over-classifying as environment would delete a real gate,
    while under-classifying only keeps a hook that may not deserve keeping.
    """
    if not output:
        return "finding"
    lowered = output.lower()
    for marker in ENVIRONMENT_FAILURE_MARKERS:
        if marker.lower() in lowered:
            return "environment"
    return "finding"


def summarise(
    records: Sequence[CommitRecord], hook_stages: dict[str, str]
) -> list[HookSummary]:
    """Roll per-commit verdicts up per hook.

    `runs` counts only commits where the hook actually executed. A hook skipped
    because its file filter matched nothing was never asked a question, and
    counting that as a passing run would manufacture exactly the false zero this
    harness exists to avoid.
    """
    acc: dict[str, dict[str, Any]] = {}
    for record in records:
        for run in record.runs:
            bucket = acc.setdefault(
                run.hook_id,
                {
                    "runs": 0,
                    "fires": 0,
                    "skips": 0,
                    "durations": [],
                    "fire_commits": [],
                    "stage": hook_stages.get(run.hook_id, record.stage),
                },
            )
            if run.status == "skipped":
                bucket["skips"] += 1
                continue
            bucket["runs"] += 1
            if run.duration_s is not None:
                bucket["durations"].append(run.duration_s)
            if run.fired:
                bucket["fires"] += 1
                bucket["fire_commits"].append(record.commit)

    summaries: list[HookSummary] = []
    for hook_id, bucket in acc.items():
        durations = bucket["durations"]
        fire_commits = bucket["fire_commits"]
        summaries.append(
            HookSummary(
                hook_id=hook_id,
                stage=bucket["stage"],
                runs=bucket["runs"],
                fires=bucket["fires"],
                skips=bucket["skips"],
                first_fire_commit=fire_commits[0] if fire_commits else None,
                last_fire_commit=fire_commits[-1] if fire_commits else None,
                median_duration_s=round(median(durations), 3) if durations else None,
                total_duration_s=round(sum(durations), 3),
            )
        )
    summaries.sort(key=lambda s: (-s.fires, -s.runs, s.hook_id))
    return summaries


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def list_commits(
    repo: Path, ref: str, window_days: int, cap: int
) -> list[tuple[str, str]]:
    """Merged commits on `ref`, newest first, as (sha, iso-date) pairs."""
    out = _git(
        repo,
        "log",
        ref,
        f"--since={window_days}.days.ago",
        f"--max-count={cap}",
        "--no-merges",
        "--format=%H\t%cI",
    )
    commits: list[tuple[str, str]] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        sha, iso = line.split("\t", 1)
        commits.append((sha.strip(), iso.strip()))
    return commits


def changed_files(repo: Path, sha: str) -> list[str]:
    """Paths the commit added or modified, excluding deletions.

    A deleted path cannot be handed to `pre-commit run --files`; pre-commit
    would refuse the whole invocation and the commit would read as an error
    rather than as a set of hook verdicts.
    """
    out = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "--root",
        sha,
    )
    files: list[str] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        status, _, rest = line.partition("\t")
        if status.startswith("D"):
            continue
        path = rest.split("\t")[-1].strip()
        if path:
            files.append(path)
    return files


def _run_stage(
    repo: Path,
    stage: str,
    files: Sequence[str],
    env: dict[str, str],
    timeout_s: int,
) -> tuple[str, int]:
    proc = subprocess.run(
        [
            "pre-commit",
            "run",
            "--hook-stage",
            stage,
            "--color",
            "never",
            "--verbose",
            "--files",
            *files,
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
        check=False,
    )
    # stdout and stderr are BOTH kept. Suppressing stderr on a verification
    # sweep is how four consecutive confident false zeros were once reported.
    return proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), proc.returncode


def restore_tip(repo: Path, tip: str) -> None:
    """Return the worktree to the pinned tip, discarding replayed content."""
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--force", tip, "--", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ["git", "-C", str(repo), "reset", "--hard", tip],
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ["git", "-C", str(repo), "clean", "-fdq"],
        capture_output=True,
        text=True,
        check=False,
    )


def pick_control_file(repo: Path) -> str | None:
    """A tracked source file the repository's own hooks actually look at.

    Preference order matches how hooks scope themselves: a `src/` Python module
    first, then any tracked Python file, then any tracked Markdown. Picking the
    most-covered file is what makes the control a real proof of liveness rather
    than a proof that one whitespace hook exists.
    """
    tracked = _git(repo, "ls-files").splitlines()
    predicates: tuple[Callable[[str], bool], ...] = (
        lambda f: f.startswith("src/") and f.endswith(".py"),
        lambda f: f.endswith(".py"),
        lambda f: f.endswith(".md"),
    )
    for predicate in predicates:
        for path in tracked:
            candidate = path.strip()
            if candidate and predicate(candidate) and (repo / candidate).is_file():
                return candidate
    return None


def positive_control(repo: Path, env: dict[str, str], timeout_s: int) -> dict[str, Any]:
    """Plant known violations in a real tracked file and require a real fire.

    The bar is at least one fire that classifies as a `finding`. A control whose
    only fires are environment failures has proved the harness can launch a
    process, not that it can execute this repository's gates -- and a run in that
    state must not have its zeros read as findings.
    """
    target = pick_control_file(repo)
    if target is None:
        return {
            "fired": False,
            "target_file": None,
            "hooks_fired": [],
            "finding_hooks": [],
            "environment_hooks": [],
            "transcript_tail": "no tracked source file to mutate",
        }
    path = repo / target
    original = path.read_bytes()
    try:
        path.write_bytes(original + _CONTROL_VIOLATIONS.encode("utf-8"))
        text, _ = _run_stage(repo, "pre-commit", [target], env, timeout_s)
        runs = parse_precommit_output(text)
        fired = [run for run in runs if run.fired]
        findings = sorted(
            {run.hook_id for run in fired if classify_fire(run.output) == "finding"}
        )
        environment = sorted(
            {run.hook_id for run in fired if classify_fire(run.output) == "environment"}
        )
        return {
            "fired": bool(findings),
            "target_file": target,
            "hooks_fired": sorted({run.hook_id for run in fired}),
            "finding_hooks": findings,
            "environment_hooks": environment,
            "transcript_tail": text[-4000:],
        }
    finally:
        path.write_bytes(original)


def replay(
    repo: Path,
    repo_name: str,
    ref: str,
    window_days: int,
    cap: int,
    budget_s: int,
    hook_source: str,
    per_commit_timeout_s: int,
    env: dict[str, str],
) -> dict[str, Any]:
    tip = _git(repo, "rev-parse", ref).strip()
    config_path = repo / ".pre-commit-config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"{repo_name}: no .pre-commit-config.yaml at {ref}")
    hook_stages = load_hook_stages(config_path)

    control = positive_control(repo, env, per_commit_timeout_s)

    commits = list_commits(repo, ref, window_days, cap)
    records: list[CommitRecord] = []
    started = time.monotonic()
    replayed = 0
    stopped_reason = "window_exhausted"

    for sha, iso in commits:
        if time.monotonic() - started > budget_s:
            stopped_reason = "budget_exhausted"
            break
        files = changed_files(repo, sha)
        if not files:
            continue
        try:
            if hook_source == "as-of":
                _git(repo, "checkout", "--force", sha)
            else:
                # tip mode: today's hook stack, this commit's changed content.
                restore_tip(repo, tip)
                subprocess.run(
                    ["git", "-C", str(repo), "checkout", sha, "--", *files],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            present = [f for f in files if (repo / f).exists()]
            if not present:
                continue
            for stage in ("pre-commit", "pre-push"):
                stage_started = time.monotonic()
                try:
                    text, _ = _run_stage(
                        repo, stage, present, env, per_commit_timeout_s
                    )
                    runs = parse_precommit_output(text)
                    error = None
                except subprocess.TimeoutExpired:
                    runs, error = [], f"timeout after {per_commit_timeout_s}s"
                records.append(
                    CommitRecord(
                        commit=sha,
                        committed_at=iso,
                        stage=stage,
                        changed_files=len(present),
                        wall_seconds=round(time.monotonic() - stage_started, 3),
                        runs=runs,
                        error=error,
                    )
                )
            replayed += 1
        finally:
            restore_tip(repo, tip)

    # Two sample transcripts per firing hook: enough to tell a real finding from
    # a missing interpreter, small enough that the artifact stays readable.
    fire_samples: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for run in record.runs:
            if not run.fired or not run.output:
                continue
            samples = fire_samples.setdefault(run.hook_id, [])
            if len(samples) < 2:
                samples.append(
                    {
                        "commit": record.commit,
                        "exit_code": run.exit_code,
                        "output": run.output,
                    }
                )

    environment_suspect = sorted(
        hook_id
        for hook_id, samples in fire_samples.items()
        if samples
        and all(classify_fire(sample["output"]) == "environment" for sample in samples)
    )

    summaries = summarise(records, hook_stages)
    measured_ids = {s.hook_id for s in summaries}
    observed_window = [iso for _, iso in commits[:replayed]]

    return {
        "schema": "omn17474.hook_replay.v1",
        "repo": repo_name,
        "ref": ref,
        "tip_sha": tip,
        "hook_source": hook_source,
        "window_days": window_days,
        "commit_cap": cap,
        "commits_in_window": len(commits),
        "commits_replayed": replayed,
        "stopped_reason": stopped_reason,
        "window_newest": observed_window[0] if observed_window else None,
        "window_oldest": observed_window[-1] if observed_window else None,
        "wall_seconds": round(time.monotonic() - started, 1),
        "positive_control": control,
        "hooks_declared_measured_stages": len(hook_stages),
        "hooks_observed": len(measured_ids),
        "hooks_declared_never_observed": sorted(set(hook_stages) - measured_ids),
        "summaries": [asdict(s) for s in summaries],
        "zero_fire_hooks": sorted(
            s.hook_id for s in summaries if s.runs > 0 and s.fires == 0
        ),
        "never_ran_hooks": sorted(
            s.hook_id for s in summaries if s.runs == 0 and s.skips > 0
        ),
        "fire_samples": fire_samples,
        # Hooks whose every sampled fire names a broken machine. Their fires are
        # NOT catches and their rows must not be read as evidence of a catch --
        # but neither are they zeros, so they are excluded from both populations
        # and named here so the exclusion is visible rather than silent.
        "environment_suspect_fire_hooks": environment_suspect,
        # Per-commit rows without the captured output, which now lives once per
        # hook in `fire_samples` rather than repeated on every firing commit.
        "records": [
            {
                **asdict(r),
                "runs": [
                    {k: v for k, v in asdict(run).items() if k != "output"}
                    for run in r.runs
                ],
            }
            for r in records
        ],
    }


def render_summary(report: dict[str, Any], limit: int = 25) -> str:
    lines: list[str] = []
    control = report["positive_control"]
    lines.append(
        f"{report['repo']}: replayed {report['commits_replayed']}"
        f"/{report['commits_in_window']} commits over {report['window_days']}d"
        f" ({report['stopped_reason']}, {report['wall_seconds']}s)"
    )
    lines.append(
        "positive control: "
        + (
            f"FIRED on {', '.join(control['finding_hooks'][:6])}"
            f" (target {control['target_file']})"
            if control["fired"]
            else "NO REAL FIRE -- zeros in this run are not evidence"
        )
    )
    lines.append(
        f"hooks declared in measured stages: {report['hooks_declared_measured_stages']}"
        f"; observed: {report['hooks_observed']}"
        f"; zero-fire: {len(report['zero_fire_hooks'])}"
    )
    lines.append("")
    lines.append(f"{'hook id':<52}{'stage':<12}{'runs':>6}{'fires':>7}{'med s':>8}")
    for summary in report["summaries"][:limit]:
        med = summary["median_duration_s"]
        lines.append(
            f"{summary['hook_id']:<52}{summary['stage']:<12}"
            f"{summary['runs']:>6}{summary['fires']:>7}"
            f"{(f'{med:.2f}' if med is not None else '-'):>8}"
        )
    if len(report["summaries"]) > limit:
        lines.append(f"... {len(report['summaries']) - limit} more")
    return "\n".join(lines)


def build_env(extra_path: Iterable[str], pre_commit_home: str | None) -> dict[str, str]:
    import os

    env = dict(os.environ)
    joined = ":".join([*extra_path, env.get("PATH", "")])
    env["PATH"] = joined
    if pre_commit_home:
        env["PRE_COMMIT_HOME"] = pre_commit_home
    # Keep the validator environment pinned at the tip's lockfile while the
    # replayed content moves under it; without this, `uv run` re-syncs on every
    # checkout whose uv.lock differs and the measurement becomes a download.
    env.setdefault("UV_NO_SYNC", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a repository's pre-commit hook stack over merged history (OMN-17474)."
    )
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument(
        "--repo", required=True, help="repository name for the artifact"
    )
    parser.add_argument("--ref", default="origin/dev")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--max-commits", type=int, default=300)
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=5400,
        help="stop cleanly and record a partial window past this wall time",
    )
    parser.add_argument("--per-commit-timeout", type=int, default=900)
    parser.add_argument("--hook-source", choices=("tip", "as-of"), default="tip")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--path-prepend",
        action="append",
        default=[],
        help="directory to prepend to PATH (uv, pre-commit live outside a login PATH on the lab hosts)",
    )
    parser.add_argument("--pre-commit-home", default=None)
    args = parser.parse_args(argv)

    repo = args.repo_path.resolve()
    if not (repo / ".git").exists():
        print(f"error: {repo} is not a git checkout", file=sys.stderr)
        return 1
    if (
        shutil.which("pre-commit", path=":".join([*args.path_prepend, ""]) or None)
        is None
        and shutil.which("pre-commit") is None
    ):
        print("error: pre-commit not on PATH", file=sys.stderr)
        return 1

    env = build_env(args.path_prepend, args.pre_commit_home)
    try:
        report = replay(
            repo=repo,
            repo_name=args.repo,
            ref=args.ref,
            window_days=args.window_days,
            cap=args.max_commits,
            budget_s=args.budget_seconds,
            hook_source=args.hook_source,
            per_commit_timeout_s=args.per_commit_timeout,
            env=env,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(render_summary(report))
    print(f"\nartifact: {args.out}")

    if not report["positive_control"]["fired"]:
        print(
            "error: positive control did not fire; this repository's zeros are a "
            "measurement failure, not a finding",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

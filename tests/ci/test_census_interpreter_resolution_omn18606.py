# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The census names its interpreter; it never inherits one (OMN-18606).

THE DEFECT THIS PINS. `lane-census-refresh.yml` shipped in `omnibase_infra#3710`
and failed on **every** run at the collect step:

    ModuleNotFoundError: No module named 'yaml'   (lane_census_plan.py:76)

`uv sync` installs PyYAML into the project `.venv`, but `setup-python-uv` puts
neither `.venv/bin` on `PATH` nor `VIRTUAL_ENV` in the environment when the
shared CI env is disabled. `lane-census-check.sh` then invoked a bare `python3`,
which on a runner resolves to the SYSTEM interpreter. Two live dispatches failed
identically — runs `35265905466` (`omninode-verify-runner-1`) and `35266125665`
(`omninode-verify-runner-2`), both `host-201`, 2026-09-17.

WHY THE EXISTING TESTS MISSED IT. Every one of them either parsed the workflow
as text or ran the decision module in-process under pytest's own interpreter,
which necessarily has PyYAML. None exercised interpreter RESOLUTION, and that is
the only thing that was broken. A test that runs under the venv can never
observe a bug about not running under the venv.

HOW THESE TESTS AVOID REPEATING THAT MISTAKE. The two behavioural tests below
drive the real script as a subprocess with a real `LANE_CENSUS_PYTHON` and read
its real exit code.

They are hermetic by CONSTRUCTION, not by assumption. `PATH` is a temp directory
holding symlinks to exactly the two external commands the script runs before its
docker check -- `mkdir` and `dirname` -- and nothing else. An earlier revision
used `/usr/bin:/bin` on the reasoning that docker lives in `/usr/local/bin` or
`/opt/homebrew/bin`; that is a macOS fact stated as a universal one. On the CI
runner `docker` IS in that PATH, so the positive control sailed past the docker
check and ran a real census against the runner's own daemon (`lane=ALL,
host=runnervmlun5p, LIVE`, exit 0). Building the PATH up from the commands the
script actually needs cannot be wrong that way: whatever is not linked in is
absent, on every host.

The pair is a discriminator, not a single assertion. One stub fails the PyYAML
import and one passes it; the same invocation must produce two DIFFERENT
refusals. Without the passing stub, the failing one would also pass against a
script that refused everything.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_CENSUS_SH = _REPO / "scripts" / "lane-census-check.sh"
_WORKFLOW = _REPO / ".github" / "workflows" / "lane-census-refresh.yml"

# The only external commands the collector runs before its docker check. Read
# from the script rather than guessed: `mkdir -p "$(dirname "$LOG_FILE")"`.
_PRE_DOCKER_COMMANDS = ("mkdir", "dirname")


def _sealed_path(tmp_path: Path) -> Path:
    """A PATH containing the script's pre-docker needs and provably nothing else.

    `docker` is absent because it was never linked in, which holds on every host
    rather than on the one the author happened to be using.
    """
    sealed = tmp_path / "sealed-bin"
    sealed.mkdir(exist_ok=True)
    for name in _PRE_DOCKER_COMMANDS:
        resolved = shutil.which(name)
        assert resolved, f"{name} is required to drive the collector at all"
        link = sealed / name
        if not link.exists():
            link.symlink_to(resolved)
    assert shutil.which("docker", path=str(sealed)) is None, (
        "the sealed PATH resolves docker; this test would contact a real daemon"
    )
    return sealed


def _stub(path: Path, *, imports_ok: bool) -> Path:
    """An 'interpreter' whose only modelled behaviour is its import check."""
    path.write_text(f"#!/bin/sh\nexit {0 if imports_ok else 1}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _run_census(python: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(_CENSUS_SH), "--snapshot", str(tmp_path / "out.json")],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO,
        env={
            "PATH": str(_sealed_path(tmp_path)),
            "HOME": str(tmp_path),
            "LANE_CENSUS_PYTHON": str(python),
        },
    )


def test_an_interpreter_without_pyyaml_is_refused_by_name(tmp_path: Path) -> None:
    """THE REGRESSION. The exact CI condition, driven end to end.

    Pre-OMN-18606 this reached `lane_census_plan.py` and died with a traceback
    that never named the interpreter, which is why diagnosing it took a live
    dispatch. It must now refuse up front, with exit 3 (missing deps) and the
    offending interpreter printed.
    """
    bad = _stub(tmp_path / "python-without-yaml", imports_ok=False)

    result = _run_census(bad, tmp_path)

    assert result.returncode == 3, (
        f"expected exit 3 (missing deps); got {result.returncode}. "
        f"stderr: {result.stderr.strip()}"
    )
    assert "PyYAML" in result.stderr, result.stderr
    assert str(bad) in result.stderr, (
        "the refusal must NAME the interpreter it rejected — not naming it is "
        f"what made this a live-dispatch diagnosis: {result.stderr.strip()}"
    )
    assert "ModuleNotFoundError" not in result.stderr, (
        "the failure must be the script's own refusal, not a traceback leaking "
        f"out of a sibling script: {result.stderr.strip()}"
    )


def test_an_interpreter_with_pyyaml_gets_past_the_preflight(tmp_path: Path) -> None:
    """POSITIVE CONTROL. Same command, a stub that imports cleanly.

    It must fail on the NEXT precondition (docker, absent from this PATH) rather
    than on PyYAML. Without this, the test above would pass against a script
    that refused every interpreter, and the preflight could be stuck closed.
    """
    good = _stub(tmp_path / "python-with-yaml", imports_ok=True)

    result = _run_census(good, tmp_path)

    assert "PyYAML" not in result.stderr, (
        "an interpreter that imports yaml was refused for PyYAML; the preflight "
        f"is stuck closed: {result.stderr.strip()}"
    )
    assert result.returncode == 3 and "docker not found" in result.stderr, (
        "expected the run to proceed past the interpreter preflight and stop at "
        f"the docker precondition; got {result.returncode}: {result.stderr.strip()}"
    )


def test_a_missing_interpreter_is_refused_rather_than_assumed(tmp_path: Path) -> None:
    """A named interpreter that does not exist must fail, never fall back."""
    result = _run_census(tmp_path / "does-not-exist", tmp_path)
    assert result.returncode == 3
    assert "not found" in result.stderr


def test_the_default_still_names_python3_for_the_host(tmp_path: Path) -> None:
    """The `.201` hourly drop-in runs with no LANE_CENSUS_PYTHON set.

    Its systemd unit passes `PATH=/usr/local/bin:/usr/bin:/bin` and the system
    interpreter there carries PyYAML, so the default must remain `python3`.
    Changing it to something venv-shaped would break the host path that has
    worked for years — the opposite direction of this ticket's defect.
    """
    body = _CENSUS_SH.read_text(encoding="utf-8")
    assert 'LANE_CENSUS_PYTHON="${LANE_CENSUS_PYTHON:-python3}"' in body


def _uncommented_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(raw)
    return out


def test_the_collector_invokes_no_bare_python3(tmp_path: Path) -> None:
    """Source pin. One bare invocation left anywhere reopens the defect.

    Comments are stripped first: this file and the script both discuss `python3`
    at length, and a check that matched prose would pass on the broken script.
    """
    offenders = [
        line
        for line in _uncommented_lines(_CENSUS_SH.read_text(encoding="utf-8"))
        if re.search(r"(?<![\w/\"$-])python3\b", line)
        and "LANE_CENSUS_PYTHON:-python3" not in line
    ]
    assert not offenders, (
        "bare `python3` invocations resolve to whatever the caller's PATH means, "
        f"which is exactly the CI failure OMN-18606 fixed: {offenders}"
    )


def test_the_workflow_names_the_interpreter_it_collects_with() -> None:
    """The job must RESOLVE an interpreter, not hope one is on PATH."""
    body = _WORKFLOW.read_text(encoding="utf-8")
    assert "LANE_CENSUS_PYTHON=" in body, (
        "the collect step does not name an interpreter, so the collector falls "
        "back to the runner's system python3 — the shipped defect"
    )
    assert "uv run python -c 'import sys; print(sys.executable)'" in body, (
        "resolve through `uv run`, matching the other host-201 jobs "
        "(dev-lane-staleness.yml)"
    )


def test_the_workflow_runs_no_bare_python3() -> None:
    """Every interpreter in this job is the resolved one."""
    offenders = [
        line
        for line in _uncommented_lines(_WORKFLOW.read_text(encoding="utf-8"))
        if re.search(r"(?<![\w/\"$-])python3\b", line)
    ]
    assert not offenders, f"bare `python3` in the refresh workflow: {offenders}"


def test_the_shipped_default_is_not_silently_venv_shaped() -> None:
    """Guard against 'fixing' this by hardcoding a venv path into the script.

    That would work in CI and break `.201`, where no project venv exists at the
    path the drop-in runs from. The environment names the interpreter; the
    script carries no opinion about where a venv lives.
    """
    code = _joined_code(_CENSUS_SH.read_text(encoding="utf-8"))

    assert ".venv/bin/python" not in code, (
        "the collector hardcodes a venv path; it must take the interpreter from "
        "LANE_CENSUS_PYTHON so the host path keeps working"
    )

    # An INVOCATION of uv, not a mention. The refusal message quotes
    # `\$(uv run python ...)` as advice to the operator; that backslash is what
    # keeps it text. Matching mentions would fail on the correct script, which
    # is how an over-broad guard gets deleted rather than fixed.
    invocations = [
        line
        for line in code.splitlines()
        if re.search(r"(?<!\\)\$\(\s*uv\s", line) or re.match(r"\s*uv\s", line)
    ]
    assert not invocations, (
        "the collector runs `uv` itself; it must take the interpreter from "
        f"LANE_CENSUS_PYTHON so the .201 host path keeps working: {invocations}"
    )


def _joined_code(text: str) -> str:
    return "\n".join(_uncommented_lines(text))


def test_repo_precedent_for_uv_run_python_on_host_201_still_holds() -> None:
    """Cite the precedent rather than asserting it from memory.

    If `dev-lane-staleness.yml` stops resolving through `uv run python`, the
    justification in this workflow's comment is stale and should be re-read
    rather than trusted.
    """
    precedent = _REPO / ".github" / "workflows" / "dev-lane-staleness.yml"
    assert (
        "uv run python scripts/ci/check_dev_lane_staleness.py"
        in precedent.read_text(encoding="utf-8")
    ), "the cited host-201 precedent for `uv run python` no longer holds"


# --------------------------------------------------------------------------
# Two further defects the first live dispatch exposed (run 35273920602). Both
# are the same class as the interpreter bug: the job inherited something from
# its environment instead of naming it.
# --------------------------------------------------------------------------


def test_drift_does_not_fail_the_collect_step() -> None:
    """Exit 30 is DRIFT — a normal census outcome — and must not fail the job.

    GitHub starts a `run:` step with `bash -e`. The step's own `set -uo pipefail`
    does NOT clear that, so `bash scripts/lane-census-check.sh ...` on its own
    line aborted the step at exit 30 and `rc=$?` never ran. The lab has had five
    warning-severity drift findings all day, so this failed every dispatch while
    the census itself was working perfectly.

    The exit code must be captured in a form `-e` cannot pre-empt.
    """
    body = _WORKFLOW.read_text(encoding="utf-8")
    collect = body[body.index("Collect the census from the lab docker daemon") :]
    collect = collect[: collect.index("- name: Decide")]

    invocation = [
        line
        for line in collect.splitlines()
        if "lane-census-check.sh" in line and "scripts/" in line
    ]
    assert len(invocation) == 1, invocation
    assert "|| rc=" in invocation[0], (
        "the collector's exit code must be captured with `|| rc=$?`; a bare "
        "invocation is aborted by `bash -e` before the next line runs, which "
        f"fails the job on ordinary drift: {invocation[0].strip()}"
    )


def test_the_census_is_attributed_to_the_daemons_host() -> None:
    """The census must name the host it inventoried, not the container it ran in.

    `lane-census-check.sh` defaults `HOST` to `hostname`, which inside the runner
    is the container id — run 35273920602 emitted `host=7f296d05a363`. That id
    changes whenever the runner is recreated, and `host` feeds `alert_key`, so
    a census committed from CI would both name a nonsense host and read as a
    topology change on EVERY run, opening a bump PR each time. That is precisely
    the churn `lane_census_refresh_decision` exists to prevent, arriving through
    a field the decision cannot see is wrong.
    """
    body = _WORKFLOW.read_text(encoding="utf-8")
    assert "LANE_CENSUS_HOST=" in body, (
        "the collect step does not name the census host, so the census is "
        "attributed to the runner container"
    )
    assert "docker info --format '{{.Name}}'" in body, (
        "resolve the host from the daemon being inventoried; any other source "
        "can drift from the thing actually being read"
    )
    assert "LANE-CENSUS-HOST-UNRESOLVED" in body, (
        "an unresolved host must fail the step, not fall back to the container"
    )


def test_the_collector_still_defaults_host_for_the_lab_drop_in() -> None:
    """On `.201` the drop-in sets no LANE_CENSUS_HOST and `hostname` is correct.

    There the script runs ON the host whose lanes it describes, so the default
    is right and must stay. The override belongs to the caller that is NOT on
    that host.
    """
    body = _CENSUS_SH.read_text(encoding="utf-8")
    assert 'HOST="${LANE_CENSUS_HOST:-$(hostname)}"' in body

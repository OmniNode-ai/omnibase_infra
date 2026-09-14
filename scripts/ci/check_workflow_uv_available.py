# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A hosted-runner job may not invoke ``uv`` before it installs it (OMN-18364).

What this exists to prevent
---------------------------
``deliver-dev-candidate-to-staging.yml``'s ``candidate-boot-gate`` job is the
onex-lab lab pass. It emits a ``ModelLabPassReceipt`` whose checks are wired to
step OUTCOMES, and the delivery to staging is refused unless that receipt says
``PASS``. The OMN-18316 step ``Assert the lab rollout deadlines are bounded
under the applier wait`` was inserted into that job ahead of the job's own
``astral-sh/setup-uv`` step and invokes ``uv run``. On a GitHub-hosted image
``uv`` is not on ``PATH``, so the step exited 127 with ``uv: command not
found`` on its very first run and on every run after it::

    /home/runner/work/_temp/….sh: line 3: uv: command not found
    ##[error]Process completed with exit code 127

The receipt then recorded ``rollout_deadline_bounded: fail``. That reads as a
substantive verdict on the rollout-wedge invariant and is nothing of the sort:
it is the check reporting that its own interpreter was missing. Because the
receipt is fail-closed, one such step stopped EVERY candidate reaching staging.
Measured: the last successful delivery run was 2026-09-13T15:03:41Z on head
``8f7417501e1a``; ``4136c7d28`` — the commit that added the step — is the head
of the first failing run, 34771240107 at 17:19:27Z, and nothing was delivered
for the following seventeen hours while the staging deploy ran an
ever-more-stale runtime plane.

This is the SECOND occurrence of that failure class in the same job. The
comment block above that job's ``Install uv`` step already records the first:
``lab_pass_receipt.py`` imports pydantic at module scope, nothing in the job
installed it, and "a gate that has never emitted a passing receipt is not
fail-closed, it is fail-always." That lesson was written down and the next step
added to the job reintroduced it anyway, which is what makes it a ratchet's job
rather than a reviewer's.

Contract
--------
For every job in ``.github/workflows/*.y[a]ml`` that runs on a GitHub-HOSTED
runner, a ``run:`` step invoking ``uv`` or ``uvx`` in command position must be
preceded, in the same job, by a step that puts ``uv`` on ``PATH``:

* ``uses: astral-sh/setup-uv@…``
* a local composite action whose ``action.y[a]ml`` does either of those
* a ``run:`` step that installs it (``curl|wget … astral.sh/uv``, ``pip install
  … uv``)

Exit codes: ``0`` clean, ``1`` violations found (printed one per line).

Scope, and why it is drawn here
-------------------------------
SELF-HOSTED JOBS ARE OUT OF SCOPE, deliberately. The fleet runners are
provisioned with a toolchain, so ``uv`` is on ``PATH`` there without a setup
step, and ~200 steps across this repository rely on that. Flagging them would
make this checker unusable and it would be turned off, which is the failure
mode CLAUDE.md rule 5 describes. The defect is specific to hosted images, where
``uv`` is genuinely absent — and ``shared-env-runner-parity.yml``'s hosted job
asserts exactly that absence as its own positive control.

COMMAND POSITION, not a substring. ``command -v uv`` is a probe FOR uv's
absence, not an invocation of it; a checker that cannot tell those apart would
flag that positive control as a defect.

Wired as a pre-commit hook AND asserted by
``tests/ci/test_workflow_uv_available_omn18364.py`` — per CLAUDE.md rule 5, a
detector that is not a gate is advisory and gets ignored.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: ``uv``/``uvx`` in COMMAND position: at the start of a line, or directly after
#: a shell command separator, optionally behind ``!``, ``sudo`` and any number of
#: leading ``VAR=value`` assignments. This is what separates ``uv run …`` from
#: ``command -v uv`` and from ``--with uv``.
INVOCATION = re.compile(
    r"""(?m)(?:^|[;&|(`]|\$\()\s*(?:!\s*)?(?:sudo\s+)?"""
    r"""(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(uv|uvx)(?=\s)"""
)

#: GitHub-hosted runner label prefixes. A job every one of whose labels starts
#: with one of these is hosted; anything else (a fleet label, an expression that
#: resolves at run time) is treated as self-hosted and left alone.
HOSTED_LABEL = re.compile(r"^(ubuntu|windows|macos)-")

#: A ``run:`` step that puts uv on PATH itself.
INSTALL_IN_RUN = (
    re.compile(r"(?m)^\s*(?:curl|wget)\b[^\n]*astral\.sh/uv"),
    re.compile(r"(?m)\bpip3?\s+install\b[^\n]*\buv\b"),
)

#: The published action, and the marker a local composite action must carry for
#: this checker to accept it as an installer.
SETUP_ACTION = "astral-sh/setup-uv"
INSTALLER_MARKERS = (SETUP_ACTION, "astral.sh/uv")


@dataclass(frozen=True)
class Violation:
    """One hosted-runner step that invokes uv with no installer ahead of it."""

    workflow: str
    job: str
    index: int
    step: str

    def render(self) -> str:
        return (
            f"{self.workflow}: job {self.job!r} step {self.index} "
            f"({self.step}) invokes `uv` on a hosted runner, but no earlier "
            f"step in that job installs it. The step will exit 127 with "
            f"`uv: command not found`."
        )


def _uncommented(script: str) -> str:
    """The script with whole-line ``#`` comments removed.

    A comment naming ``uv`` is documentation, and the comment blocks in this
    repository's workflows are long and quote commands verbatim.
    """
    return "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )


def runs_on_hosted(job: dict[str, Any]) -> bool:
    """True when every label this job requests is a GitHub-hosted image.

    Fail-open by design: an expression, a fleet label, or a group is not
    provably hosted, so the job is left alone rather than flagged on a guess.
    """
    runs_on = job.get("runs-on")
    if isinstance(runs_on, str):
        labels: list[Any] = [runs_on]
    elif isinstance(runs_on, list):
        labels = list(runs_on)
    elif isinstance(runs_on, dict):
        labels = list(runs_on.get("labels") or [])
    else:
        return False
    return bool(labels) and all(
        isinstance(label, str) and HOSTED_LABEL.match(label.strip()) for label in labels
    )


def installs_uv(step: dict[str, Any], repo_root: Path) -> bool:
    """True when this step puts ``uv`` on ``PATH`` for the steps after it."""
    uses = str(step.get("uses") or "")
    if SETUP_ACTION in uses:
        return True
    if uses.startswith("./"):
        action_dir = repo_root / uses[2:]
        for candidate in (action_dir / "action.yml", action_dir / "action.yaml"):
            if not candidate.is_file():
                continue
            text = candidate.read_text(encoding="utf-8")
            if any(marker in text for marker in INSTALLER_MARKERS):
                return True
    script = _uncommented(str(step.get("run") or ""))
    return any(pattern.search(script) for pattern in INSTALL_IN_RUN)


def invokes_uv(step: dict[str, Any]) -> bool:
    """True when this step's script runs ``uv``/``uvx`` as a command."""
    script = str(step.get("run") or "")
    return bool(script) and bool(INVOCATION.search(_uncommented(script)))


def scan_document(
    workflow: str, document: dict[str, Any], repo_root: Path
) -> list[Violation]:
    violations: list[Violation] = []
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return violations
    for job_id, job in jobs.items():
        if not isinstance(job, dict) or not runs_on_hosted(job):
            continue
        available = False
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            if installs_uv(step, repo_root):
                available = True
                continue
            if available or not invokes_uv(step):
                continue
            name = step.get("name") or step.get("id") or "<unnamed>"
            violations.append(
                Violation(
                    workflow=workflow, job=str(job_id), index=index, step=str(name)
                )
            )
    return violations


def workflow_files(workflows_dir: Path) -> list[Path]:
    return sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")])


def scan_paths(paths: list[Path], repo_root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in paths:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            # Parse failures belong to the YAML linter, not to this ratchet.
            continue
        if isinstance(document, dict):
            violations.extend(scan_document(path.name, document, repo_root))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=Path(".github/workflows"),
        help="directory of workflow files to scan (default: .github/workflows)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(),
        help="root the local composite actions in `uses: ./…` resolve against",
    )
    args = parser.parse_args(argv)

    if not args.workflows_dir.is_dir():
        print(
            f"::error::{args.workflows_dir} is not a directory — refusing to "
            "report clean on a scan that inspected nothing.",
            file=sys.stderr,
        )
        return 1

    violations = scan_paths(workflow_files(args.workflows_dir), args.repo_root)
    if not violations:
        return 0

    print(
        "A hosted-runner step invokes `uv` before the job installs it "
        "(OMN-18364). The step exits 127, and where its outcome feeds a "
        "receipt check the receipt records a missing interpreter as a "
        "substantive verdict:",
        file=sys.stderr,
    )
    for violation in violations:
        print(f"  {violation.render()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

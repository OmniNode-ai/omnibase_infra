# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""This sub-project is type-gated, and the gate cannot be removed quietly (OMN-18640).

`scripts/deploy-agent` was type-checked NOWHERE. The repository's mypy leg runs
`mypy src/omnibase_infra`, and this tree is not under `src/`, so for as long as
the deploy agent has existed its strict-mode errors went unreported. There were
thirteen. Most were annotation debt, but two were a real hole: `_run` returned
an unparameterised `subprocess.CompletedProcess`, so `result.stdout` was `Any`
and every caller doing `.strip()` returned `Any` out of a function declared to
return `str`.

Clearing them once is worth nothing on its own. A check that is not a gate is
advisory, and advisory checks are the thing that let thirteen errors accumulate
unseen. So this module pins the two halves the gate is made of, because each
can be deleted independently and neither deletion would fail anything else:

1. the CI job actually runs mypy over this package, from a working directory
   where mypy resolves THIS project's config;
2. this project declares `strict = true`, so the local verdict and the CI
   verdict are the same verdict.

What this module deliberately does NOT do is re-run mypy in-process and assert
zero errors. That would be slow, and it would also be the wrong assertion: the
CI step already fails on a type error, loudly and with the file and line. What
has no other guard is the step's continued existence.

`tests/` is deliberately out of the gate's scope, and the number is recorded
here rather than left as a vague "later": the test tree carries 350 strict
errors across 55 files, measured 2026-09-19. The parent package gates `src/`
and not `tests/` for the same reason, so this matches the repository rather
than inventing a second standard. Widening to `tests/` is a real piece of work,
not a flag flip.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-agent-tests.yml"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _mypy_steps() -> list[dict[str, object]]:
    """Every step in the workflow whose run line invokes mypy."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps: list[dict[str, object]] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []) or []:
            run = str(step.get("run", ""))
            if "mypy" in run:
                steps.append(step)
    return steps


class TestTheGateExists:
    """Each half is independently deletable; neither deletion fails elsewhere."""

    def test_the_ci_job_runs_mypy_over_this_package(self) -> None:
        steps = _mypy_steps()
        assert steps, (
            "no step in deploy-agent-tests.yml invokes mypy: this sub-project is "
            "under no other type gate, because the repository's mypy leg runs "
            "`mypy src/omnibase_infra` and this tree is not under src/"
        )
        assert any("deploy_agent" in str(s.get("run", "")) for s in steps), (
            f"a mypy step exists but does not name this package: {steps}"
        )

    def test_the_mypy_step_runs_where_this_config_resolves(self) -> None:
        """mypy does not walk parents for its config; the cwd decides which one wins."""
        steps = [s for s in _mypy_steps() if "deploy_agent" in str(s.get("run", ""))]
        # Asserted, not assumed: without this the test passes vacuously when the
        # gate has been deleted entirely, which is the case it most needs to
        # catch. Measured against origin/dev while writing it -- two of these
        # five tests passed on a tree with no gate at all, and this was one.
        assert steps, "no mypy step over this package exists to check the cwd of"
        for step in steps:
            run = str(step.get("run", ""))
            cwd = str(step.get("working-directory", ""))
            resolves_here = cwd.endswith("deploy-agent") or "--config-file" in run
            assert resolves_here, (
                "this mypy step neither runs in scripts/deploy-agent nor passes "
                "--config-file, so it reads the PARENT package's [tool.mypy] -- "
                "its mypy_path and its 64-module override list, none of which "
                f"apply here: {step}"
            )

    def test_this_project_declares_strict(self) -> None:
        config = tomllib.loads(PYPROJECT.read_text()).get("tool", {}).get("mypy", {})
        assert config, (
            "scripts/deploy-agent/pyproject.toml declares no [tool.mypy]: the CI "
            "step would then depend on flags spelled in the workflow, and a "
            "developer running mypy locally would get a different verdict"
        )
        assert config.get("strict") is True, (
            f"strict mode is what found the thirteen; got {config!r}"
        )

    def test_mypy_is_a_declared_dev_dependency(self) -> None:
        """A gate whose tool is not pinned in the project is a gate that can vanish."""
        extras = (
            tomllib.loads(PYPROJECT.read_text())
            .get("project", {})
            .get("optional-dependencies", {})
        )
        dev = " ".join(extras.get("dev", []))
        assert "mypy" in dev, f"mypy is not in the dev extra the CI job installs: {dev}"
        assert "types-PyYAML" in dev, (
            "types-PyYAML is what gives yaml.SafeLoader a real type; without it "
            "the compose-ceiling loader subclasses Any, which silences every "
            "override on it rather than merely annoying the author"
        )


class TestNoPerModuleOverridesHaveAppeared:
    """The parent carries an override list because 64 modules predate strict mode."""

    def test_this_project_has_no_override_list(self) -> None:
        """This one starts clean; the first exemption should be a visible decision."""
        config = tomllib.loads(PYPROJECT.read_text()).get("tool", {}).get("mypy", {})
        overrides = config.get("overrides", [])
        assert not overrides, (
            "a per-module mypy override appeared in this sub-project. That is how "
            "the parent accumulated 64 exempt modules. If one is genuinely "
            f"needed, say so in the PR rather than appending quietly: {overrides}"
        )

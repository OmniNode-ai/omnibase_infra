# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's package is inside this repo's type scope (OMN-18640 AC9).

``scripts/deploy-agent`` is a standalone uv sub-project, so it was outside
every declaration of mypy scope this repository makes -- the ``mypy`` step in
``ci.yml``, the same step in the shadow workflow, and the ``files:`` pattern on
the pre-commit hook all named ``src/omnibase_infra`` and nothing else. The code
that runs every deploy on the lab lane therefore had no type gate at all, and
carried ten errors when one was first run by hand on 2026-09-19. Three of them
were functions declared to return ``str`` that were returning ``Any`` read back
out of an untyped ``subprocess`` wrapper.

Scope is declared in three places and this asserts all three, because two of
them agreeing is exactly the state that let the gap exist: a check that reads
only the CI step would pass while a developer's pre-commit run silently skipped
the package.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_AGENT_PACKAGE = "scripts/deploy-agent/deploy_agent"
PROBE_PATHS = (
    "scripts/deploy-agent/deploy_agent/executor.py",
    "scripts/deploy-agent/deploy_agent/agent.py",
    "src/omnibase_infra/__init__.py",
)


def _workflow_mypy_commands(workflow: Path) -> list[str]:
    """Return every ``run:`` script in the workflow that invokes mypy."""
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    commands: list[str] = []
    for job in document["jobs"].values():
        for step in job.get("steps", []) or []:
            script = step.get("run")
            if isinstance(script, str) and "mypy" in script:
                commands.append(script)
    return commands


@pytest.mark.parametrize(
    "workflow_name",
    ["ci.yml", "product-readiness-shadow.yml"],
)
def test_every_ci_mypy_invocation_covers_the_deploy_agent(workflow_name: str) -> None:
    """Read the parsed step, not the file text: a comment mentioning the path
    must not be able to satisfy this."""
    workflow = REPO_ROOT / ".github" / "workflows" / workflow_name
    commands = _workflow_mypy_commands(workflow)
    assert commands, f"{workflow_name} declares no mypy step at all"
    for command in commands:
        runnable = "\n".join(
            line for line in command.splitlines() if not line.strip().startswith("#")
        )
        assert DEPLOY_AGENT_PACKAGE in runnable, (
            f"{workflow_name} runs mypy without the deploy agent in scope: {command!r}"
        )


def test_the_precommit_hook_selects_the_deploy_agent_package() -> None:
    """The ``files:`` pattern is applied the way pre-commit applies it."""
    config = yaml.safe_load(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("id") == "mypy-type-check"
    ]
    assert len(hooks) == 1, "expected exactly one mypy-type-check hook"
    hook = hooks[0]
    files = re.compile(hook["files"])
    exclude = re.compile(hook["exclude"])

    for path in PROBE_PATHS:
        assert files.search(path), f"the hook does not select {path}"
        assert not exclude.search(path), f"the hook excludes {path}"

    # The negative control: the sub-project's own tests stay out, matching the
    # exclusion of this repo's tests/ tree. Without it this test would pass on
    # a pattern that simply matched everything.
    assert not files.search("scripts/deploy-agent/tests/unit/test_kafka_config.py")


def test_the_package_is_not_exempted_by_a_mypy_override() -> None:
    """A per-module override would make the scope above decorative.

    ``pyproject.toml`` carries an override list that relaxes strict flags for
    modules not yet compliant. The deploy agent's package is compliant -- the
    ten errors were fixed rather than waived -- and an entry added here later
    would silence the gate without touching any of the three declarations the
    tests above read.
    """
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy = config["tool"]["mypy"]
    assert mypy["strict"] is True
    for override in mypy.get("overrides", []):
        modules = override["module"]
        if isinstance(modules, str):
            modules = [modules]
        for module in modules:
            assert not module.startswith("deploy_agent"), (
                f"deploy_agent is exempted from strict typing by override {module!r}"
            )

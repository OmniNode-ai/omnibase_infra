# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A local mypy run in the deploy-agent directory gives the CI verdict (OMN-18640).

Follow-up to `#3818`, which put `scripts/deploy-agent/deploy_agent` inside the
ROOT project's type scope. That is the right place for the gate and this does
not move it. This guards a different reader: the one who changes into
`scripts/deploy-agent` and runs `mypy` by hand.

mypy resolves its configuration from `./pyproject.toml` and does NOT walk
parents. So before this, a local run in that directory got no configuration at
all -- in particular no `plugins = ["pydantic.mypy"]` -- and reported three
errors that do not exist, because without the plugin mypy cannot see that a
`mode="before"` field validator is what narrows a `str` onto a Literal field.

Measured on dev at `31d242a71`, same tree, same moment:

    with the plugin     Success: no issues found in 27 source files
    plugin line deleted Found 3 errors in 2 files

A lane acted on those three -- wrote two casts and an annotation against them --
before isolating the cause. So this is a cost already paid once, not a
hypothetical, and the guard is worth its lines.

WHAT THIS PINS, and what it deliberately does not. It pins that the settings
which change a VERDICT agree between the two files. It does not pin the whole
block equal: `mypy_path`, `namespace_packages` and `explicit_package_bases` are
correctly absent from the sub-project, because they point at the root `src/`
tree that is not on its import path. A test asserting byte equality would have
to be weakened the first time one project legitimately needed something the
other did not, and a test that gets weakened is a test nobody trusts.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
AGENT_PYPROJECT = REPO_ROOT / "scripts" / "deploy-agent" / "pyproject.toml"

#: Settings that change a verdict rather than a path. These must agree, or a
#: local run and CI disagree about the same file.
VERDICT_BEARING = ("plugins", "strict", "ignore_missing_imports", "disable_error_code")


def _mypy_config(pyproject: Path) -> dict[str, object]:
    return (
        tomllib.loads(pyproject.read_text(encoding="utf-8"))
        .get("tool", {})
        .get("mypy", {})
    )


def test_the_sub_project_declares_a_mypy_config_at_all() -> None:
    """Absent, a local run silently uses mypy's defaults and invents findings."""
    config = _mypy_config(AGENT_PYPROJECT)
    assert config, (
        "scripts/deploy-agent/pyproject.toml declares no [tool.mypy]. mypy does "
        "not walk parents for its config, so a run in that directory gets no "
        "pydantic plugin and reports three errors that are not real"
    )


@pytest.mark.parametrize("setting", VERDICT_BEARING)
def test_the_verdict_bearing_settings_match_the_root(setting: str) -> None:
    root = _mypy_config(ROOT_PYPROJECT)
    agent = _mypy_config(AGENT_PYPROJECT)
    assert setting in root, f"the root config no longer declares {setting!r}"
    assert agent.get(setting) == root.get(setting), (
        f"{setting!r} differs between the root and the deploy-agent project "
        f"({agent.get(setting)!r} vs {root.get(setting)!r}), so a local mypy run "
        "in scripts/deploy-agent and the CI run over the same files can reach "
        "different verdicts"
    )


def test_the_pydantic_plugin_specifically_is_declared() -> None:
    """Named on its own: this is the one whose absence produced false findings."""
    plugins = _mypy_config(AGENT_PYPROJECT).get("plugins", [])
    assert "pydantic.mypy" in plugins, (
        "without the pydantic plugin, mypy cannot see that a mode='before' "
        f"validator narrows a str onto a Literal field: {plugins!r}"
    )


def test_the_tools_a_local_run_needs_are_declared_dev_dependencies() -> None:
    """The root environment carries both, so CI is green with or without these."""
    extras = (
        tomllib.loads(AGENT_PYPROJECT.read_text(encoding="utf-8"))
        .get("project", {})
        .get("optional-dependencies", {})
    )
    dev = " ".join(extras.get("dev", []))
    assert "mypy" in dev, f"no mypy in the dev extra, so there is nothing to run: {dev}"
    assert "types-PyYAML" in dev, (
        "without the PyYAML stubs a local run reports `Class cannot subclass "
        '"SafeLoader" (has type "Any")` on compose_budget.py, which CI does not '
        f"report because the root environment has them: {dev}"
    )


def test_the_sub_project_declares_no_per_module_overrides() -> None:
    """The root carries a list because 64 of its modules predate strict mode."""
    overrides = _mypy_config(AGENT_PYPROJECT).get("overrides", [])
    assert not overrides, (
        "a per-module mypy override appeared in the deploy-agent project. That "
        "is how the root accumulated 64 exempt modules; the first exemption here "
        f"should be argued in a PR rather than appended quietly: {overrides}"
    )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: the workflows that read onex_change_control now read this repository.

``contract-topic-graph``, ``omni-standards-compliance``, ``pr-ci-zombie-detector``
and ``release`` each read the change-control repository (a checkout at a sibling
pin, a pip install, a run list, a receipt directory). They now read what replaced
it: the graph census drops it from its corpus, the handler-contract check runs
omnibase_core's handler, the zombie detector scans only omnimarket, and the
release resolves its evidence from the in-repo ``contracts/<ticket>.yaml`` alone.
The unused ``onex-change-control`` dev dependency and its sibling pin are gone.

Falsifier: restore any one read and the matching test names it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.validators import contract_topic_graph

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# A read of the change-control repository, in either spelling the workflows use.
OCC_READ = re.compile(r"onex[_-]change[_-]control", re.IGNORECASE)


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _code_lines(name: str) -> list[str]:
    """The workflow's lines that are not YAML comments."""
    return [
        line for line in _text(name).splitlines() if not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize(
    "workflow",
    [
        "contract-topic-graph.yml",
        "omni-standards-compliance.yml",
        "pr-ci-zombie-detector.yml",
        "release.yml",
    ],
)
def test_workflow_reads_no_onex_change_control(workflow: str) -> None:
    offenders = [line for line in _code_lines(workflow) if OCC_READ.search(line)]
    assert not offenders, f"{workflow} still reads onex_change_control: {offenders}"


def test_topic_graph_corpus_has_no_change_control_package() -> None:
    assert "onex_change_control" not in contract_topic_graph.CHECKOUT_PACKAGES
    assert "onex_change_control" not in contract_topic_graph.GRAPH_PACKAGES
    assert "omnimarket" in contract_topic_graph.CHECKOUT_PACKAGES


def test_sibling_pins_carry_no_change_control_pin() -> None:
    pins = yaml.safe_load((ROOT / ".github" / "sibling-pins.yaml").read_text())["pins"]
    assert "onex_change_control" not in pins


def test_handler_contract_compliance_runs_omnibase_core_handler() -> None:
    text = _text("omni-standards-compliance.yml")
    assert (
        "python -m omnibase_core.handlers.handler_arch_handler_contract_compliance"
        in text
    )
    assert "--allowlist-path arch-handler-contract-compliance-allowlist.yaml" in text
    # --no-config: [tool.uv] override-dependencies pins omnibase-core below the
    # release that ships the handler, and would replace this install with it.
    install = re.search(r"uv pip install[^\n]*omnibase-core[^\n]*", text)
    assert install is not None
    assert "--no-config" in install.group(0)
    assert re.search(r"omnibase-core==0\.47\.(4[1-9]|[5-9]\d)", install.group(0))


def test_zombie_detector_scans_only_omnimarket() -> None:
    workflow = yaml.safe_load(_text("pr-ci-zombie-detector.yml"))
    steps = workflow["jobs"]["detect"]["steps"]
    mint = next(s for s in steps if s.get("id") == "app-token")
    repositories = [
        line.strip()
        for line in str(mint["with"]["repositories"]).splitlines()
        if line.strip()
    ]
    assert repositories == ["omnimarket"]
    run = next(s["run"] for s in steps if s.get("id") == "detector")
    assert re.findall(r"--repo (\S+)", run) == ["omnimarket"]


def test_dev_dependencies_carry_no_change_control_package() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert not OCC_READ.search(
        "\n".join(
            line for line in pyproject.splitlines() if not line.lstrip().startswith("#")
        )
    )
    assert 'name = "onex-change-control"' not in lock
    assert "onex_change_control.git" not in lock

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the OMN-18648 live-contact admission node.

Loads the real contract from disk, resolves its handler through the declared
module path, and drives that handler against a real git repository.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.model_ci_live_contact_request import (
    ModelCILiveContactRequest,
)

pytestmark = pytest.mark.integration

CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_ci_live_contact_effect/contract.yaml"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=t@e.invalid",
            *args,
        ],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def test_contract_declares_importable_handler_and_topic() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    entry = contract["handler_routing"]["handlers"][0]["handler"]
    handler_cls = getattr(importlib.import_module(entry["module"]), entry["name"])
    assert callable(handler_cls)
    assert contract["event_bus"]["subscribe_topics"] == [
        "onex.cmd.platform.ci-live-contact-check.v1"
    ]


async def test_handler_refuses_seam_only_ci_change_in_real_repo(tmp_path: Path) -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    entry = contract["handler_routing"]["handlers"][0]["handler"]
    handler = getattr(importlib.import_module(entry["module"]), entry["name"])()

    subprocess.run(
        ["git", "init", "--quiet", str(tmp_path)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    _git(tmp_path, "commit", "--allow-empty", "-m", "base")
    (tmp_path / "scripts/ci").mkdir(parents=True)
    (tmp_path / "scripts/ci/check.py").write_text("print('gate')\n")
    _git(tmp_path, "add", ".")

    result = await handler.handle(ModelCILiveContactRequest(repository=str(tmp_path)))
    assert not result.success
    assert "live-contact" in result.reason


_OLD = "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0"
_NEW = "60178094be3ca82abb6a64498600f7909e247584"
_WORKFLOW = f"""name: Pinned
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@{_OLD} # v7.0.0
        with:
          ref: {_OLD}
"""


@pytest.mark.parametrize(
    ("head", "admitted"),
    [
        (_WORKFLOW.replace(_OLD, _NEW), True),
        (
            _WORKFLOW.replace(_OLD, _NEW) + "      - run: curl https://x.invalid\n",
            False,
        ),
    ],
    ids=["pin-only-admitted", "pin-plus-step-refused"],
)
async def test_handler_decides_pin_only_workflow_edit_from_the_pr_diff(
    tmp_path: Path, head: str, admitted: bool
) -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    entry = contract["handler_routing"]["handlers"][0]["handler"]
    handler = getattr(importlib.import_module(entry["module"]), entry["name"])()

    _git(tmp_path, "init", "--quiet")
    workflow = tmp_path / ".github/workflows/pinned.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(_WORKFLOW)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")
    workflow.write_text(head)
    _git(tmp_path, "commit", "-am", "head")

    result = await handler.handle(
        ModelCILiveContactRequest(repository=str(tmp_path), base_ref="HEAD~1")
    )
    assert result.success is admitted, result
    assert result.changed_tooling == (".github/workflows/pinned.yml",)

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The post-release bump disarms dev after publishing v0.38.75 (OMN-13912).

Read the live manifests and exercise the release train and runtime handler.
At the release commit, dev still carries 0.38.75 and both decisions are armed.
Later bumps must continue to satisfy the same published-version floor.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_release_identity_compute import (
    HandlerReleaseIdentity,
    ModelReleaseIdentityRequest,
)
from scripts.ci.post_release_dev_bump import (
    ACTION_NOOP,
    decide,
    parse_final_version,
    read_project_version,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
RELEASED_TAG = "v0.38.75"


def test_dev_and_editable_lock_are_ahead_of_the_published_v0_38_75() -> None:
    dev_version = read_project_version(PYPROJECT)
    with (REPO_ROOT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    editable = next(
        package
        for package in lock["package"]
        if package["name"] == "omnibase-infra"
        and package["source"] == {"editable": "."}
    )
    assert editable["version"] == dev_version
    assert parse_final_version(dev_version, label="dev") > parse_final_version(
        RELEASED_TAG, label="released"
    ), f"dev {dev_version} is not ahead of published {RELEASED_TAG}"


def test_the_release_train_sees_dev_disarmed_after_v0_38_75() -> None:
    decision = decide(
        dev_version=read_project_version(PYPROJECT), released_version=RELEASED_TAG
    )
    assert decision.action == ACTION_NOOP, decision.reason


def test_runtime_release_identity_accepts_source_changes_after_v0_38_75() -> None:
    decision = HandlerReleaseIdentity().handle(
        ModelReleaseIdentityRequest(
            pyproject_version_raw=read_project_version(PYPROJECT),
            pyproject_path=str(PYPROJECT),
            published_tags=(RELEASED_TAG,),
            changed_files=(
                "src/omnibase_infra/nodes/node_release_identity_compute/"
                "handlers/handler_release_identity.py",
            ),
        )
    )
    assert decision.exit_code == 0, decision.message
    assert decision.reason_code == "version_ahead", decision.message

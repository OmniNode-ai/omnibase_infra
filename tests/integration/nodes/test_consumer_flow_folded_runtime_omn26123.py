# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The C28 docker target reads the shipped policy and inspects only the containers it names."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
    HandlerDockerConsumerFlowTarget,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]


def _folded_project() -> str:
    values = dict(
        line.split("=", 1)
        for line in (REPO_ROOT / "docker" / "runtime-policy.env")
        .read_text(encoding="utf-8")
        .splitlines()
        if "=" in line and not line.startswith("#")
    )
    for key, project in values.items():
        if key.endswith("_COMPOSE_PROJECT") and "effects" in values.get(
            f"{key.removesuffix('_COMPOSE_PROJECT')}_RUNTIME_MAIN_FOLDED_PROFILES", ""
        ):
            return project
    raise AssertionError("the shipped policy folds effects on no lane")


def test_a_folded_lane_never_inspects_the_effects_container(tmp_path: Path) -> None:
    inspected: list[str] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[1] == "inspect":
            inspected.append(argv[-1])
            out = json.dumps(
                [{"Id": argv[-1], "State": {"Status": "running", "StartedAt": "s"}}]
            )
        else:
            out = ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    request = ModelConsumerFlowRequest(
        subject_lane="lane",
        compose_project=_folded_project(),
        base_url="http://127.0.0.1:1",
        scratch=tmp_path,
    )
    HandlerDockerConsumerFlowTarget(repo_root=REPO_ROOT, runner=runner)._observe(
        request
    )
    assert inspected
    assert not any(name.endswith("-runtime-effects") for name in inspected)

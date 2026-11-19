# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The restart-on-merge node, run through its contract (OMN-24677).

Finding 1 of the omnibase_infra#6027 review at 72d8c39490e5: the handler had no
integration test. These run the node contract with the local runtime over the
repository's real unit-drift manifest and tracked units, and over a synthetic host
whose launcher env file is fixed after the first check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.models.model_restart_on_merge import ModelRestartOnMerge

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO_ROOT / "src/omnibase_infra/nodes/node_restart_on_merge_effect/contract.yaml"
)
SETTINGS = (
    "DEPLOY_AGENT_REPO_DIR=/state/deploy-source/omnibase_infra\n"
    "DEPLOY_AGENT_CANONICAL_CLONE=/registry/omnibase_infra\n"
    "DEPLOY_AGENT_LAB_OVERLAY_SOURCE=/registry/overlay-source\n"
)


def _run_node(request: ModelRestartOnMerge, tmp_path: Path) -> ModelRestartOnMerge:
    state = tmp_path / f"runtime-{request.operation}-{len(list(tmp_path.iterdir()))}"
    request_path = state.with_suffix(".json")
    request_path.write_text(request.model_dump_json(), "utf-8")
    runtime = RuntimeLocal(
        workflow_path=CONTRACT,
        state_root=state,
        input_path=request_path,
        backend_overrides={"event_bus": "inmemory"},
        timeout=45,
    )
    runtime.run()
    assert runtime.exit_code == 0
    readback = json.loads((state / "workflow_result.json").read_text("utf-8"))
    return ModelRestartOnMerge.model_validate(readback["handler_result"])


def test_restart_on_merge_node_refuses_the_real_prod_unit(tmp_path: Path) -> None:
    result = _run_node(
        ModelRestartOnMerge(
            unit="deploy-agent.service",
            repo_root=str(REPO_ROOT),
            home=str(tmp_path / "home"),
        ),
        tmp_path,
    )
    assert result.restartable is False
    assert "fenced to prod" in result.reason


def test_restart_on_merge_node_check_host_blocks_until_the_env_file_names_settings(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "deploy").mkdir(parents=True)
    store = tmp_path / "agent.env"
    store.write_text("DEPLOY_AGENT_REPO_DIR=/state/deploy-source/omnibase_infra\n")
    unit = repo / "units" / "agent-dev.service"
    unit.parent.mkdir()
    unit.write_text(
        "Environment=DEPLOY_AGENT_TRACKING_REF=dev\n"
        "Environment=DEPLOY_AGENT_ALLOWED_LANES=dev\n"
        f"Environment=DEPLOY_AGENT_ENV_FILE={store}\n"
    )
    (repo / "deploy" / "unit-drift-manifest.yaml").write_text(
        "units:\n  - name: agent-dev.service\n    tracked: units/agent-dev.service\n"
    )
    home = tmp_path / "home"
    installed = home / ".config" / "systemd" / "user" / "agent-dev.service"
    installed.parent.mkdir(parents=True)
    installed.write_text(unit.read_text())
    request = ModelRestartOnMerge(
        operation="check_host", repo_root=str(repo), home=str(home)
    )

    blocked = _run_node(request, tmp_path)
    assert blocked.restartable is False
    assert [gap.unit for gap in blocked.gaps] == ["agent-dev.service"]
    assert set(blocked.gaps[0].missing) == {
        "DEPLOY_AGENT_CANONICAL_CLONE",
        "DEPLOY_AGENT_LAB_OVERLAY_SOURCE",
    }

    store.write_text(SETTINGS)
    cleared = _run_node(request, tmp_path)
    assert cleared.restartable is True
    assert cleared.gaps == ()

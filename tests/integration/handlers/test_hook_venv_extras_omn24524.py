# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the hook venv reconcile through a real uv-shaped executable and keep installed extras."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import pytest

from omnibase_infra.handlers.hook_venv_reconcile.handler_reconcile_workspace_venvs import (
    HandlerReconcileWorkspaceVenvs,
)
from omnibase_infra.models.model_hook_venv_reconcile_request import (
    ModelHookVenvReconcileRequest,
)
from omnibase_infra.nodes.node_reconcile_workspace_venvs.runtime_reconcile_workspace_venvs import (
    _bootstrap,
)

pytestmark = pytest.mark.integration
_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_reconcile_workspace_venvs/contract.yaml"
)
# Like uv, the executable plans to drop the installed package unless its extra is named.
_UV = """#!/bin/sh
case " $* " in
  *" --extra probeextra "*) echo " Would make no changes" ;;
  *) echo " - probe==1.0" ;;
esac
"""


def test_installed_extra_is_kept_through_a_real_subprocess(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project.optional-dependencies]\nprobeextra = ["probe>=1.0"]\n'
    )
    info = tmp_path / ".venv/lib/python3.12/site-packages/probe.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Name: probe\nVersion: 1.0\n")
    (info / "direct_url.json").write_text(
        json.dumps(
            {
                "url": tmp_path.as_uri(),
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            }
        )
    )
    uv = tmp_path / "uv"
    uv.write_text(_UV)
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)

    handler = asyncio.run(_bootstrap(_CONTRACT)).get_service(
        HandlerReconcileWorkspaceVenvs
    )
    result = handler.handle(
        ModelHookVenvReconcileRequest(
            workspace_root=tmp_path, project=tmp_path, uv_bin=uv
        )
    )
    assert result.status == "already_in_sync"

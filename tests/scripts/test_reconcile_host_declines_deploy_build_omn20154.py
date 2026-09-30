# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""reconcile-host.sh declines while a deploy-agent image build holds its lock (OMN-20154).

On .202 on 2026-09-30 the ``onex`` wrapper's below-floor self-heal ran
reconcile-host.sh in the middle of two deploy-agent builds. Its clone reconcile
force-checked-out ``$OMNI_HOME/omnibase_infra``, which reset the staged
``workspace/sibling-vcs-provenance.json`` to its empty placeholder, and the
agent's second (dev-lane-only) build failed in compute_workspace_provenance.py.

The agent now takes the reconcile-host lock from staging through its last build.
This test pins the half that matters across the language boundary: the REAL
shell script, run against a lock the Python context manager wrote, declines and
touches nothing -- and, as the control, proceeds once the build has released it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_host_lock_holder_omn18608 import (
    _NOTHING_TO_DO,
    EXIT_DECLINED,
    _ready,
)
from tests.scripts.test_reconcile_host_omn17307 import (
    Workspace,
    _run,
    build_workspace,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "deploy-agent"))

from deploy_agent.reconcile_host_lock import (
    hold_reconcile_host_lock,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return build_workspace(tmp_path)


def test_reconcile_host_declines_while_a_deploy_build_holds_the_lock(
    ws: Workspace,
) -> None:
    _ready(ws)
    with hold_reconcile_host_lock(str(ws.root), purpose="deploy-agent image build"):
        proc = _run(ws)
    assert proc.returncode == EXIT_DECLINED, proc.stderr
    assert _NOTHING_TO_DO in proc.stderr
    assert not ws.delegate_witness.exists(), (
        "reconcile-host rewrote the clones while a deploy build held the lock"
    )


def test_reconcile_host_proceeds_after_the_deploy_build_releases(
    ws: Workspace,
) -> None:
    _ready(ws)
    with hold_reconcile_host_lock(str(ws.root), purpose="deploy-agent image build"):
        pass
    proc = _run(ws)
    assert _NOTHING_TO_DO not in proc.stderr, proc.stderr
    assert ws.delegate_witness.exists(), "the reconcile did not proceed"

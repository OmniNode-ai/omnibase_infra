# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reconciler hands the lane-identity module the root it resolved (OMN-19432).

On `.201` a non-interactive ssh session has no OMNI_HOME, so the operator passes
`--omni-home`. The reconciler resolved the root for itself and then ran the
lane-identity module in a child that never saw it, so the module refused with
"neither ONEX_LANE_REGISTRY_ROOT nor OMNI_HOME is set", the hook readback named
every clone unarmed, and the venv delegate exited non-zero even though every
venv had converged.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_lane_hook_arming_omn18260 import (
    _install_lane_identity_stub,
    _set_state,
    _teach_dispatch_python_to_run_programs,
)
from tests.scripts.test_reconcile_workspace_venvs import _SCRIPT, _Workspace

pytestmark = pytest.mark.unit


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    workspace = _Workspace(tmp_path / "omni_home")
    _teach_dispatch_python_to_run_programs(workspace)
    _install_lane_identity_stub(workspace)
    _set_state(workspace)
    return workspace


def test_lane_identity_module_sees_the_root_passed_by_flag(ws: _Workspace) -> None:
    seen = ws.root / "omni-home-seen.log"
    stub = ws.root / "omniclaude" / "scripts" / "lane_identity.py"
    original = stub.read_text(encoding="utf-8")
    stub.write_text(
        original.replace(
            "argv = sys.argv[1:]",
            "argv = sys.argv[1:]\n"
            f"open({str(seen)!r}, 'a').write(os.environ.get('OMNI_HOME', 'UNSET') + '\\n')",
            1,
        ),
        encoding="utf-8",
    )
    env = ws.env()
    env.pop("OMNI_HOME")
    fake_home = ws.root / "fakehome"
    fake_home.mkdir(exist_ok=True)
    env["HOME"] = str(fake_home)
    env["LANE_STUB_STATE"] = str(ws.root / "lane-stub-state.json")
    env["LANE_STUB_LOG"] = str(ws.root / "lane-stub.log")
    env["ONEX_LANE_IDENTITY_SCRIPT"] = str(stub)
    subprocess.run(
        ["bash", str(_SCRIPT), "--omni-home", str(ws.root)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    lines = seen.read_text(encoding="utf-8").splitlines()
    assert lines, "the lane-identity module was never run"
    assert set(lines) == {str(ws.root)}, lines

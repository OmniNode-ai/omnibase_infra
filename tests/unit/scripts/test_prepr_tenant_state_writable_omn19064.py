# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The slot gateway's container user can write the slot tenant credential (OMN-19064).

The slot gateway runs as ``appuser`` (uid 1001) and writes ``credential.env``
into a bind mount of the slot's ``tenant-state`` directory. That directory is
created on the host by ``prepr_verify_lane.sh`` as the operator (uid 1000). At
mode 775 the gateway cannot create the file, so the first live mint on pre-PR
slot 2 (2026-10-20T07:35Z) created the tenant and then failed with no
credential.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VERIFY = ROOT / "scripts/runtime_build/prepr_verify_lane.sh"

pytestmark = pytest.mark.unit


def staging_block() -> str:
    source = VERIFY.read_text()
    match = re.search(
        r'^mkdir -p "\$\{STAGING_ROOT\}" "\$\{SLOT_ENV_DIR\}" "\$\{TENANT_STATE_DIR\}"\n'
        r"(?:.*\n)*?\n",
        source,
        re.M,
    )
    assert match, "prepr_verify_lane.sh must create the staging, env and tenant dirs"
    return match.group(0)


def test_tenant_state_dir_admits_a_non_owner_container_user(tmp_path: Path) -> None:
    staging = tmp_path / "slot-2"
    script = (
        "set -euo pipefail\n"
        f'STAGING_ROOT="{staging}"\n'
        'SLOT_ENV_DIR="${STAGING_ROOT}/.slot-env"\n'
        'TENANT_STATE_DIR="${STAGING_ROOT}/tenant-state"\n' + staging_block()
    )
    subprocess.run(["bash", "-c", script], check=True)

    tenant = (staging / "tenant-state").stat().st_mode
    # A container uid that is neither the owner nor in the group creates the file.
    assert tenant & stat.S_IWOTH and tenant & stat.S_IXOTH
    # Nobody else may list it or remove a credential they do not own.
    assert not tenant & stat.S_IROTH
    assert tenant & stat.S_ISVTX
    # The slot env directory carries secrets and stays private.
    assert stat.S_IMODE((staging / ".slot-env").stat().st_mode) == 0o700

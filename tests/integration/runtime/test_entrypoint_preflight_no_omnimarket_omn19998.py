# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage: the runtime boot preflight never reaches for omnimarket.

OMN-19998: omnimarket depends on omnibase_infra, never the other way round. The
preflight used to import omnimarket to re-derive a missing
``DELEGATION_ROUTING_TIERS_PATH``. This runs the real preflight entry point in a
fresh interpreter where any omnimarket import is refused and recorded, with the
pin pointed at a missing file, and asserts the boot hands off with the pin
unchanged and no omnimarket import attempted.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_BOOT_WITH_OMNIMARKET_REFUSED = """
import sys
from pathlib import Path

marker = Path(sys.argv[1])


class _RefuseOmnimarket:
    def find_spec(self, name, path=None, target=None):
        if name == "omnimarket" or name.startswith("omnimarket."):
            marker.write_text(name, encoding="utf-8")
            raise ImportError(f"omnimarket import refused: {name}")
        return None


sys.meta_path.insert(0, _RefuseOmnimarket())

from omnibase_infra.runtime.entrypoint_preflight import main

raise SystemExit(main(sys.argv[2:]))
"""

_PRINT_TIERS_PIN = (
    "import os; print('PIN=' + os.environ.get('DELEGATION_ROUTING_TIERS_PATH', ''))"
)


def test_preflight_boots_with_missing_tiers_pin_and_never_imports_omnimarket(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "omnimarket_import_attempted"
    missing_pin = tmp_path / "absent" / "routing_tiers.yaml"
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "OMNIBASE_INFRA_DB_URL",
            "OMNIINTELLIGENCE_DB_URL",
            "BIFROST_CONTRACT_PATH",
            "ONEX_SECRET_RESOLVER_CONFIG_PATH",
        }
    }
    env["DELEGATION_ROUTING_TIERS_PATH"] = str(missing_pin)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _BOOT_WITH_OMNIMARKET_REFUSED,
            str(marker),
            sys.executable,
            "-c",
            _PRINT_TIERS_PIN,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not marker.exists(), (
        f"the boot preflight attempted to import {marker.read_text(encoding='utf-8')}"
    )
    assert "[entrypoint] Starting runtime kernel..." in proc.stdout
    assert f"PIN={missing_pin}" in proc.stdout
    assert "re-derive" not in proc.stdout + proc.stderr

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The root reconcile unit must not let root's interpreter write bytecode (OMN-24899).

The unit runs as root and runs the dispatch venv's python for its readbacks. The
`.pyc` files root writes into the owner's venv are root-owned, and the owner's
installer cannot delete them -- which stranded a `.atomic-stage.*` folder in the
dispatch venv on host h201 on 2026-11-16. The setting belongs to the unit's own
`[Service]` section, so it holds for every process the unit starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNIT = (
    _REPO_ROOT
    / "deploy"
    / "maintenance"
    / "systemd"
    / "omninode-workspace-reconcile.service"
)


def test_the_unit_sets_no_bytecode_inside_its_service_section() -> None:
    lines = [line.strip() for line in _UNIT.read_text(encoding="utf-8").splitlines()]
    service = lines.index("[Service]")
    setting = lines.index("Environment=PYTHONDONTWRITEBYTECODE=1")
    assert service < setting, (
        "Environment=PYTHONDONTWRITEBYTECODE=1 must sit in the [Service] section "
        "of the root reconcile unit, or root's python writes bytecode into the "
        "owner's venv again and the owner's installer cannot delete it (OMN-24899)."
    )

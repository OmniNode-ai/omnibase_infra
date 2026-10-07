# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The venv purity refusal names the command the maintenance scheduler runs (OMN-20670).

OMN-20670 replaced ``scripts/reconcile-host.sh`` with the installed
``onex-host-reconcile`` command. The refusal in ``venv_purity`` is the text an
operator acts on, so it must name the same command the deployed scheduler
executes, and must not name the deleted script.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime import venv_purity

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = REPO_ROOT / "deploy" / "maintenance" / "omninode-workspace-reconcile.sh"
REPAIR_COMMAND = "onex-host-reconcile"


@pytest.mark.integration
def test_refusal_and_scheduler_name_the_same_repair_command() -> None:
    assert not (REPO_ROOT / "scripts" / "reconcile-host.sh").exists()
    assert REPAIR_COMMAND in SCHEDULER.read_text(encoding="utf-8")

    source = Path(venv_purity.__file__).read_text(encoding="utf-8")
    assert REPAIR_COMMAND in source
    assert "reconcile-host.sh" not in source

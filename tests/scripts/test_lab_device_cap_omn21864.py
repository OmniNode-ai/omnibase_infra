# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21864: one tailnet login may hold six machine principals on the lab broker.

The lab declaration capped a login at five machines, so the sixth device of the
operator's login was refused. The operator approved raising the cap to six. The local
declaration for the developer-run broker is a separate file and keeps its own cap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import lab_principal_issuer as lpi

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GRANTS = REPO_ROOT / "deploy" / "lab" / "developer-principal-grants.yaml"


def test_lab_declaration_caps_a_login_at_six_machines() -> None:
    declaration = lpi.load_declaration(GRANTS)
    assert declaration.max_devices_per_login == 6

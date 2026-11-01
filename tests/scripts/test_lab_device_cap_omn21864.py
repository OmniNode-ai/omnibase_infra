# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21864: one tailnet login may hold six machine principals on the lab broker.

The lab declaration capped a login at five machines, so the sixth device of the
operator's login was refused (HTTP 409). The operator approved raising the cap to six.
These tests drive the issuer itself with the lab declaration and a fake broker: a login
is issued six distinct devices, the seventh is refused, and a device already held still
rotates at the cap. The local declaration for the developer-run broker is a separate
file and keeps its own cap.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

import pytest

from scripts import lab_principal_issuer as lpi

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GRANTS = REPO_ROOT / "deploy" / "lab" / "developer-principal-grants.yaml"
LOGIN = "dev.person@omninode.ai"


class _Broker(lpi.Broker):
    """A broker that holds users in memory and records nothing else."""

    def __init__(self) -> None:
        self.users: dict[str, str] = {}

    def user_exists(self, name: str) -> bool:
        return name in self.users

    def set_user(self, name: str, password: str, *, exists: bool) -> None:
        self.users[name] = password

    def grant(self, principal: str, grant: lpi.Grant) -> None:
        return None

    def ensure_topic(self, name: str) -> None:
        return None


@pytest.fixture
def issuer(tmp_path: Path) -> lpi.Issuer:
    declaration = lpi.load_declaration(GRANTS)
    return lpi.Issuer(declaration, _Broker(), lpi.Ledger(tmp_path / "issuances.jsonl"))


def test_lab_declaration_caps_a_login_at_six_machines() -> None:
    assert lpi.load_declaration(GRANTS).max_devices_per_login == 6


def test_a_login_is_issued_six_devices_and_the_seventh_is_refused(
    issuer: lpi.Issuer,
) -> None:
    for number in range(6):
        answer = issuer.issue(LOGIN, {"lane": "dev", "device": f"mac-{number}"})
        assert answer["action"] == "created"
    with pytest.raises(lpi.IssuerError) as refused:
        issuer.issue(LOGIN, {"lane": "dev", "device": "mac-6"})
    assert refused.value.status is HTTPStatus.CONFLICT


def test_a_device_already_held_rotates_at_the_cap(issuer: lpi.Issuer) -> None:
    for number in range(6):
        issuer.issue(LOGIN, {"lane": "dev", "device": f"mac-{number}"})
    answer = issuer.issue(LOGIN, {"lane": "dev", "device": "mac-5"})
    assert answer["action"] == "rotated"

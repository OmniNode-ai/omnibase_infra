# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` stamps the tenant declared for the lab host (OMN-18829).

Resolution runs against a real ``lab_run_hosts.yaml`` on disk, with the real
registry reader and no stand-in for it: the host row's tenant wins over the
install identity, and a malformed declaration refuses instead of falling back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    DelegateTenantRefusedError,
    resolve_delegate_tenant,
)

pytestmark = pytest.mark.integration

_HOST_TENANT = "5b8c1f3a-2d4e-4f6a-9b7c-0d1e2f3a4b5c"
_MINTED = "0b2a6f1e-7c1d-4a39-8d55-3e8a4c9f1b27"


def _table(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_declared_lane_host_row_tenant_is_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: _MINTED)
    table = _table(
        tmp_path / "lab_run_hosts.yaml",
        "hosts:\n"
        "  - name: h101\n"
        "    target: ssh://h101.invalid\n"
        f"    tenant_id: {_HOST_TENANT}\n",
    )
    stamp = resolve_delegate_tenant(
        in_process=False,
        environ={"ONEX_LANE_HOST": "h101", "ONEX_LAB_RUN_HOSTS": str(table)},
    )
    assert stamp == _HOST_TENANT


def test_row_without_tenant_refuses_instead_of_reading_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: _MINTED)
    table = _table(
        tmp_path / "lab_run_hosts.yaml",
        "hosts:\n  - name: h101\n    target: ssh://h101.invalid\n",
    )
    with pytest.raises(DelegateTenantRefusedError, match="no tenant_id"):
        resolve_delegate_tenant(
            in_process=False,
            environ={"ONEX_LANE_HOST": "h101", "ONEX_LAB_RUN_HOSTS": str(table)},
        )


def test_no_lab_declaration_falls_through_to_the_install_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: _MINTED)
    assert resolve_delegate_tenant(in_process=False, environ={}) == _MINTED

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26430: the http_mount scan over this repository's real node contracts.

``AsgiMounts.declared`` parses a contract only when its text names
``http_mount``; over the committed ``contract.yaml`` files it reads exactly
those and no other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers.http_mount import asgi_mounts
from omnibase_infra.handlers.http_mount.asgi_mounts import AsgiMounts
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract

pytestmark = pytest.mark.integration

_NODES = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"


def test_declared_over_real_contracts_parses_only_http_mount_namers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered = [
        _parse_contract(
            contract_path=path,
            entry_point_name=path.parent.name,
            package_name="omnibase_infra",
            package_version="0.0.0",
        )
        for path in sorted(_NODES.glob("*/contract.yaml"))
    ]
    assert len(discovered) > 10
    namers = {
        c.contract_path
        for c in discovered
        if "http_mount" in c.contract_path.read_text(encoding="utf-8")
    }
    parsed: list[Path] = []
    real = asgi_mounts.read_contract_yaml

    def counting(path: Path) -> object:
        parsed.append(path)
        return real(path)

    monkeypatch.setattr(asgi_mounts, "read_contract_yaml", counting)

    AsgiMounts.declared(discovered)

    assert set(parsed) == namers
    assert len(parsed) < len(discovered)

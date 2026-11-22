# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26430: the http_mount scan parses only the contracts that declare a mount.

A boot of the dev rollback image's source on lab host h053 under a 0.5 CPU cap
read the stalled-loop answer's ``loop_thread_stack`` as
``read_contract_yaml`` <- ``AsgiMounts.declared`` <- ``AsgiMounts.prepare`` <-
``ServiceHealth.attach_manifest`` for 10.3 s: the kernel's attach parsed the
YAML of every discovered contract on the event loop thread to find the few that
declare ``http_mount``. A contract whose text never names ``http_mount`` cannot
declare one, so it is not parsed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers.http_mount import asgi_mounts
from omnibase_infra.handlers.http_mount.asgi_mounts import AsgiMounts
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)

pytestmark = pytest.mark.unit

_FIXTURES = "tests.unit.handlers.http_mount.asgi_fixtures"
_PLAIN_CONTRACTS = 40


def _contract(tmp_path: Path, index: int, extra: str) -> ModelDiscoveredContract:
    path = tmp_path / f"contract_{index}.yaml"
    path.write_text(
        f"name: node_scan_{index}\n"
        "node_type: EFFECT_GENERIC\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        f"{extra}\n",
        encoding="utf-8",
    )
    return _parse_contract(
        contract_path=path,
        entry_point_name=f"node_scan_{index}",
        package_name="fixture",
        package_version="0.0.0",
    )


def test_declared_parses_only_contracts_that_name_http_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = [_contract(tmp_path, i, "") for i in range(_PLAIN_CONTRACTS)]
    contracts.append(
        _contract(
            tmp_path,
            _PLAIN_CONTRACTS,
            f'http_mount:\n  factory: "{_FIXTURES}:create_app"\n  path_prefix: "/scan"',
        )
    )
    parsed: list[Path] = []
    real = asgi_mounts.read_contract_yaml

    def counting(path: Path) -> object:
        parsed.append(path)
        return real(path)

    monkeypatch.setattr(asgi_mounts, "read_contract_yaml", counting)

    mounts = AsgiMounts.declared(contracts)

    assert [m.path_prefix for m in mounts] == ["/scan"]
    assert parsed == [contracts[-1].contract_path], (
        f"{len(parsed)} contract(s) parsed on the loop thread; only the one that "
        "declares http_mount needs a parse"
    )

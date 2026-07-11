# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for the tenant_scoped_ingress contract opt-in (OMN-14349).

OMN-14208 Path A: a contract-declared boolean that opts a node's
subscribe_topics into topic-prefix tenant derivation. Off by default -- these
tests prove parsing is correct and that an ordinary (non-opted-in) contract is
completely unaffected.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths

pytestmark = pytest.mark.unit


def _write_contract(tmp_path: Path, *, tenant_scoped_ingress_line: str = "") -> Path:
    content = dedent(f"""\
        name: "node_plain_effect"
        node_type: "EFFECT_GENERIC"
        contract_version:
          major: 1
          minor: 0
          patch: 0
        node_version: "1.0.0"
        description: "A plain effect node"
        event_bus:
          subscribe_topics:
            - "onex.evt.platform.plain-input.v1"
          publish_topics:
            - "onex.evt.platform.plain-output.v1"
          {tenant_scoped_ingress_line}
    """)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(content)
    return contract_path


def test_defaults_to_false_when_not_declared(tmp_path: Path) -> None:
    manifest = discover_contracts_from_paths([_write_contract(tmp_path)])

    contract = manifest.contracts[0]
    assert contract.event_bus is not None
    assert contract.event_bus.tenant_scoped_ingress is False


def test_parses_true_when_declared(tmp_path: Path) -> None:
    manifest = discover_contracts_from_paths(
        [
            _write_contract(
                tmp_path, tenant_scoped_ingress_line="tenant_scoped_ingress: true"
            )
        ]
    )

    contract = manifest.contracts[0]
    assert contract.event_bus is not None
    assert contract.event_bus.tenant_scoped_ingress is True


def test_explicit_false_stays_false(tmp_path: Path) -> None:
    manifest = discover_contracts_from_paths(
        [
            _write_contract(
                tmp_path, tenant_scoped_ingress_line="tenant_scoped_ingress: false"
            )
        ]
    )

    contract = manifest.contracts[0]
    assert contract.event_bus is not None
    assert contract.event_bus.tenant_scoped_ingress is False

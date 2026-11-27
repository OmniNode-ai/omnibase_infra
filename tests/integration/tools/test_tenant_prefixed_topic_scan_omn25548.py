# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract scan over real node contracts keeps a tenant-prefixed topic (OMN-25548).

Runs the extractor over a tree holding a copy of a real repo node contract plus a
node declaring the tenant-prefixed proof topic, and asserts the real contracts'
canonical topics still come through unchanged beside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

_PROOF_TOPIC = "tenant-cli-run-h051.onex.evt.omnimarket.pool-usage-verdict-evaluated.v1"
_NODES_ROOT = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"


@pytest.mark.integration
def test_real_node_contracts_scan_keeps_tenant_topic(tmp_path: Path) -> None:
    baseline = ContractTopicExtractor().extract(_NODES_ROOT)
    assert baseline, "the real node contracts declare topics"

    node_dir = tmp_path / "node_tenant"
    node_dir.mkdir()
    (node_dir / "contract.yaml").write_text(
        f'event_bus:\n  publish_topics:\n    - "{_PROOF_TOPIC}"\n', encoding="utf-8"
    )

    scanned = {e.topic: e for e in ContractTopicExtractor().extract(tmp_path)}
    assert set(scanned) == {_PROOF_TOPIC}
    assert scanned[_PROOF_TOPIC].tenant_slug == "cli-run-h051"
    assert all(e.tenant_slug == "" for e in baseline)

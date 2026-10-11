# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-contract reconciliation for typed discovery outcomes (OMN-18713).

Discovery over every contract.yaml shipped under ``src/omnibase_infra/nodes``
must account for each path exactly once: discovered, reported as an error, or
reported as a typed policy skip. A path that silently disappears is the defect
this ticket closes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_NODES_DIR = _ROOT / "src" / "omnibase_infra" / "nodes"


def test_every_real_contract_path_has_one_discovery_outcome() -> None:
    paths = sorted(_NODES_DIR.rglob("contract.yaml"))
    assert paths, f"no contract.yaml found under {_NODES_DIR}"

    manifest = discover_contracts_from_paths(paths)

    outcomes = (
        [c.contract_path for c in manifest.contracts]
        + [e.contract_path for e in manifest.errors]
        + [s.contract_path for s in manifest.skips]
    )
    assert sorted(outcomes) == paths
    assert not [e for e in manifest.errors if e.reason == "parse_error"]

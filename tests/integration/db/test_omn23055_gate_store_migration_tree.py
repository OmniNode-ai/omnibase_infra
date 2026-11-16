# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23055: the vendored gate store files are in the forward tree the runner walks.

Reads the real migration tree and ledger from disk: every relation the grant file
names is created by the create file, and both files are bound in the ledger.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_FORWARD = Path(__file__).resolve().parents[3] / "docker" / "migrations" / "forward"
_NODE = _FORWARD / "nodes" / "node_gate_store_effect"


def test_grant_file_names_only_relations_the_create_file_makes() -> None:
    created = set(
        re.findall(
            r"CREATE TABLE IF NOT EXISTS omninode_internal\.(\w+)",
            (_NODE / "0000_create_gate_store.sql").read_text(encoding="utf-8"),
        )
    )
    granted = set(
        re.findall(
            r"ON omninode_internal\.(\w+)",
            (_NODE / "0001_grant_omninode_runtime_gate_store.sql").read_text(
                encoding="utf-8"
            ),
        )
    )
    assert len(created) == 4
    assert granted == created


def test_ledger_binds_both_vendored_files() -> None:
    ledger = (_FORWARD / "_ledger" / "application-migrations.tsv").read_text(
        encoding="utf-8"
    )
    for path in sorted(_NODE.glob("*.sql")):
        assert f"nodes/node_gate_store_effect/{path.name}\t" in ledger

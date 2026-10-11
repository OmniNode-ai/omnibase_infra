# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Migration wire values and reader grant agree with the writer enum."""

import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
    EnumLedgerChainState,
)

pytestmark = pytest.mark.unit
ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)


def test_migration_and_grant():
    name = "110_add_ledger_chain_chain_state.sql"
    path = ROOT / "docker/migrations/forward" / name
    assert path.exists()
    sql = path.read_text()
    assert "ALTER TABLE public.ledger_chain" in sql
    assert "chain_state TEXT NOT NULL DEFAULT ''" in sql
    check = re.search(r"CHECK\s*\(chain_state IN \((.*?)\)\)", sql, re.S)
    assert check
    assert set(re.findall(r"'([^']*)'", check.group(1))) == {""} | {
        e.value for e in EnumLedgerChainState
    }
    config = yaml.safe_load((ROOT / "config/migration_classes.yaml").read_text())
    assert name in str(config)
    assert (
        "chain_canary_reader:public.ledger_chain:correlation_id,hop,hop_index,replay_green,verifier_verdict,chain_state"
        in (ROOT / "scripts/run-forward-migrations.sh").read_text()
    )


def test_application_sql_gate_checks_service_migration(monkeypatch):
    from scripts.ci import check_application_database_sql as gate

    migration = ROOT / "docker/migrations/forward/110_add_ledger_chain_chain_state.sql"
    monkeypatch.setattr(gate, "changed_sql_paths", lambda *args: (migration,))
    outcome = gate.validate_changed_sql(
        ROOT,
        "HEAD",
        "HEAD",
        ownership_manifest_paths=(
            ROOT / "config/application_database_domain_proof_ownership.yaml",
        ),
    )
    assert outcome.violations == ()
    assert outcome.grandfathered == 0

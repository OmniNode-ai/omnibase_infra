# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20009 vendor identity for the run-locally share and saving-per-run migrations.

Both files are byte-identical copies of omnimarket's node migrations. The
pinned digests below were read from omnimarket at
``3f360db02cd622bdef0b830396e06de6affaf499`` (omnimarket#3406), at
``src/omnimarket/nodes/<node>/migrations/<file>``. A vendored file that drifts
from that source, or a ledger or class row that no longer describes it, fails
here instead of on a lane runner.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"

# (node, file, sha256 of omnimarket's file at the source commit, ledger scope,
# migration class)
_VENDORED = (
    (
        "node_projection_delegation",
        "0055_model_routing_local_call_share.sql",
        "09cb25a37ffc067f3ae73185700168acb3e83ad07974a1d09902fa7a76ff018a",
        "tenant",
        "forward-only",
    ),
    (
        "node_projection_metering_summary",
        "0002_metering_summary_savings_per_measured_run.sql",
        "e3bb212c0301cc8953a62abc7330bdb2b145695e4ae191a5122feb1a9f9cbee5",
        "omninode_internal",
        "expand-only",
    ),
)

_IDS = [entry[1] for entry in _VENDORED]


def _manifest_rows() -> list[list[str]]:
    return [
        row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]


@pytest.mark.parametrize(
    ("node", "name", "sha256", "scope", "klass"), _VENDORED, ids=_IDS
)
def test_vendored_bytes_match_the_omnimarket_source(
    node: str, name: str, sha256: str, scope: str, klass: str
) -> None:
    vendored = _FORWARD / "nodes" / node / name
    assert hashlib.sha256(vendored.read_bytes()).hexdigest() == sha256


@pytest.mark.parametrize(
    ("node", "name", "sha256", "scope", "klass"), _VENDORED, ids=_IDS
)
def test_each_file_has_exactly_one_exact_ledger_row(
    node: str, name: str, sha256: str, scope: str, klass: str
) -> None:
    artifact_path = f"nodes/{node}/{name}"
    matching = [row for row in _manifest_rows() if row[0] == artifact_path]
    assert matching == [
        [
            artifact_path,
            f"node:{node}",
            f"node:{node}",
            scope,
            f"node:{node}:{name}",
            sha256,
        ]
    ]


@pytest.mark.parametrize(
    ("node", "name", "sha256", "scope", "klass"), _VENDORED, ids=_IDS
)
def test_each_file_has_its_migration_class(
    node: str, name: str, sha256: str, scope: str, klass: str
) -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{node}/{name}"] == klass


def test_0055_serves_the_share_and_restores_security_invoker() -> None:
    sql = (
        _FORWARD
        / "nodes"
        / "node_projection_delegation"
        / "0055_model_routing_local_call_share.sql"
    ).read_text(encoding="utf-8")
    assert "CREATE OR REPLACE VIEW projection_delegation_model_routing AS" in sql
    for field in ("'local_call_count'", "'total_call_count'", "'local_call_share'"):
        assert field in sql
    assert (
        "ALTER VIEW projection_delegation_model_routing SET (security_invoker = true);"
        in sql
    )


def test_0002_adds_a_nullable_column_with_no_default() -> None:
    sql = (
        _FORWARD
        / "nodes"
        / "node_projection_metering_summary"
        / "0002_metering_summary_savings_per_measured_run.sql"
    ).read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS savings_per_measured_run_usd TEXT;" in sql
    assert "NOT NULL" not in sql
    assert "DEFAULT" not in sql

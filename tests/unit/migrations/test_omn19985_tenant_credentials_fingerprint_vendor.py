# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19985 vendor identity for the tenant-credentials fingerprint migration.

omnimarket's node_projection_tenant_credentials gains 0005: a nullable
fingerprint prefix and set time on each credential row. The lane runners apply
only what is vendored here, so the file must be the omnimarket bytes exactly,
bound in the application-migration ledger and classified.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_tenant_credentials"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_tenant_inference_credentials.sql"
_FINGERPRINT = "0005_tenant_credentials_fingerprint_and_set_at.sql"
# sha256 of omnimarket's src/omnimarket/nodes/node_projection_tenant_credentials/
# migrations/0005_tenant_credentials_fingerprint_and_set_at.sql at omnimarket#3510
# head 6310b18211a9. Byte-identical: a change on either side must move both.
_SHA256 = "1e1d52343ccdfcb91dab1669ce3cae39982a1064617c0366622d6bffbf44a0eb"


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


def test_vendor_bytes_and_manifest_binding_are_exact() -> None:
    artifact_path = f"nodes/{_NODE}/{_FINGERPRINT}"
    assert hashlib.sha256((_VENDOR / _FINGERPRINT).read_bytes()).hexdigest() == (
        _SHA256
    )
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "tenant",
        f"node:{_NODE}:{_FINGERPRINT}",
        _SHA256,
    ]


def test_migration_class_is_declared_expand_only() -> None:
    """Two nullable ADD COLUMNs are additive: the class checker reports no
    finding for them, so declaring forward-only would make this file a rollback
    barrier for no reason."""
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_FINGERPRINT}"] == "expand-only"


def test_it_only_adds_the_two_nullable_columns() -> None:
    """Amendment 1 (A1.1, A1.5): fingerprint TEXT and set_at TIMESTAMPTZ, both
    nullable with no default or backfill, and no generation_id column. Old rows,
    hosted registrations and revoke-first tombstones read NULL; nothing is
    rewritten, so it applies on every lane without a fence."""
    statements = _statements(_FINGERPRINT)
    assert re.findall(
        r"ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS (\w+) (\w+);", statements
    ) == [
        ("tenant_inference_credentials", "fingerprint", "TEXT"),
        ("tenant_inference_credentials", "set_at", "TIMESTAMPTZ"),
    ]
    assert len([s for s in statements.split(";") if s.strip()]) == 2
    upper = statements.upper()
    for forbidden in (
        "GENERATION_ID",
        "NOT NULL",
        "DEFAULT",
        "DROP",
        "UPDATE",
        "DELETE",
        "TRUNCATE",
        "ROW LEVEL",
    ):
        assert forbidden not in upper
    fenced = yaml.safe_load(
        (_FORWARD / "fenced-node-migrations.yaml").read_text(encoding="utf-8")
    )
    assert f"node:{_NODE}:{_FINGERPRINT}" not in {
        entry["id"] for entry in fenced["fenced_node_migrations"]
    }


def test_the_table_it_alters_exists_before_it_in_runner_order() -> None:
    """run-forward-migrations.sh applies a node's files in lexical sort order, so
    on a fresh database 0005 runs after 0000..0002 but BEFORE 003 and 004.
    That is safe only while 0005 needs nothing but the table 0000 creates."""
    order = sorted(path.name for path in _VENDOR.glob("*.sql"))
    assert order.index(_CREATE) < order.index(_FINGERPRINT)
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS tenant_inference_credentials\s*\(",
        _statements(_CREATE),
    )

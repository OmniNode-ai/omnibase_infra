# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20578 vendor identity for the routing-feedback and alert-channel liveness projections.

omnimarket#3416 adds node_projection_routing_feedback (a public platform table granted
to the tenant projection writer) and omnimarket#3417 adds
node_projection_alert_channel_liveness (an omninode_internal table granted to the
runtime principal). The node-migration vendor-parity gate refuses both PRs until the
byte-identical copies are on this repo's dev, and a runtime that installs the nodes
without the topology grants exits at wiring.
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
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"

_FEEDBACK_NODE = "node_projection_routing_feedback"
_FEEDBACK_CREATE = "0000_create_delegation_routing_feedback.sql"
_FEEDBACK_GRANT = "0001_grant_tenant_projection_writer_delegation_routing_feedback.sql"
_LIVENESS_NODE = "node_projection_alert_channel_liveness"
_LIVENESS_CREATE = "0000_create_alert_channel_liveness_verdicts.sql"
_LIVENESS_GRANT = "0001_grant_omninode_runtime_alert_channel_liveness_verdicts.sql"

_FILES = (
    (_FEEDBACK_NODE, _FEEDBACK_CREATE, "tenant"),
    (_FEEDBACK_NODE, _FEEDBACK_GRANT, "tenant"),
    (_LIVENESS_NODE, _LIVENESS_CREATE, "omninode_internal"),
    (_LIVENESS_NODE, _LIVENESS_GRANT, "omninode_internal"),
)
_SHA256 = {
    _FEEDBACK_CREATE: "43803f20610fb1ddeed49046f6c04656c5a27dfe976f0212ecfa5614f06e60f0",
    _FEEDBACK_GRANT: "048e90f4046ec68a0de942791c5d14c670744c1e801b58438e323af7f4b60c59",
    _LIVENESS_CREATE: "b6dd14f636a520a00d561ec37fad1f00c43af931ec4b7b82ea224824a22808f9",
    _LIVENESS_GRANT: "b15100c7de5b769a8a2f946f2123fa5b101311a58b1e885424586faa242b171b",
}


def _vendored(node: str, filename: str) -> Path:
    return _FORWARD / "nodes" / node / filename


def _statements(node: str, filename: str) -> str:
    return "\n".join(
        line
        for line in _vendored(node, filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.parametrize(("node", "filename", "domain"), _FILES)
def test_vendor_bytes_and_manifest_binding_are_exact(
    node: str, filename: str, domain: str
) -> None:
    artifact_path = f"nodes/{node}/{filename}"
    assert (
        hashlib.sha256(_vendored(node, filename).read_bytes()).hexdigest()
        == _SHA256[filename]
    )
    rows = {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }
    assert rows[artifact_path] == [
        artifact_path,
        f"node:{node}",
        f"node:{node}",
        domain,
        f"node:{node}:{filename}",
        _SHA256[filename],
    ]


@pytest.mark.parametrize(("node", "filename", "domain"), _FILES)
def test_migration_classes_match_the_classifier(
    node: str, filename: str, domain: str
) -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    declared = classes[f"forward/nodes/{node}/{filename}"]
    # The liveness create adds a unique index on the cursor column, which the
    # classifier reads as forward-only. Every other file is additive or a grant.
    expected = "forward-only" if filename == _LIVENESS_CREATE else "expand-only"
    assert declared.split(" #")[0] == expected


def test_routing_feedback_table_and_writer_grant_are_present() -> None:
    create = _statements(_FEEDBACK_NODE, _FEEDBACK_CREATE)
    grants = _statements(_FEEDBACK_NODE, _FEEDBACK_GRANT)
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS public\.delegation_routing_feedback\s*\(", create
    )
    assert "GRANT USAGE ON SCHEMA public TO tenant_projection_writer;" in grants
    assert re.search(
        r"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+"
        r"ON public\.delegation_routing_feedback\s+TO tenant_projection_writer;",
        grants,
    )
    assert "omninode_runtime" not in create + grants
    assert not re.search(r"\bDELETE\b", grants, re.IGNORECASE)


def test_alert_liveness_table_and_runtime_grant_are_present() -> None:
    create = _statements(_LIVENESS_NODE, _LIVENESS_CREATE)
    grants = _statements(_LIVENESS_NODE, _LIVENESS_GRANT)
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS omninode_internal\."
        r"alert_channel_liveness_verdicts\s*\(",
        create,
    )
    assert "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;" in grants
    assert not re.search(
        r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+omninode_runtime\b",
        grants,
        re.IGNORECASE,
    )


@pytest.mark.parametrize("profile", ["local", "onex-dev", "onex-prod"])
def test_both_tables_are_granted_in_the_shipped_topology(profile: str) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile

    text = yaml.safe_dump(
        load_topology_profile(profile).model_dump(mode="json"), sort_keys=True
    )
    assert "delegation_routing_feedback" in text
    assert "alert_channel_liveness_verdicts" in text


@pytest.mark.parametrize(("node", "filename", "domain"), _FILES)
@pytest.mark.parametrize(
    "profile", ["local", "onex-dev", "onex-prod", "stability-test"]
)
def test_migrations_pass_the_application_database_sql_gate(
    node: str, filename: str, domain: str, profile: str
) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _vendored(node, filename).read_text(encoding="utf-8")
    violations = lint_application_database_sql(sql, load_topology_profile(profile))
    assert violations == (), f"{profile}/{filename}: {violations}"


def test_the_sql_gate_is_live_positive_control() -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _vendored(_LIVENESS_NODE, _LIVENESS_CREATE).read_text(encoding="utf-8")
    broken = sql.replace(
        "omninode_internal.alert_channel_liveness_verdicts",
        "tenant.alert_channel_liveness_verdicts",
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()

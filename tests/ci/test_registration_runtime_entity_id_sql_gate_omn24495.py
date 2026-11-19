# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The registration column's SQL passes the application-database SQL gate (OMN-24495).

``registration_projections`` is existing omnibase_infra SERVICE database
authority in the ``public`` schema. The gate has no valid qualified form for it
and no manifest declares it, so migration 117 and the schema file stay
unqualified and sit on the gate's exact-path legacy default-schema list, the
same posture as migrations 106, 110 and 111.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.check_application_database_sql import (
    _is_legacy_default_schema_sql_path,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[2]
_RECORDED = (
    "tests/ci/fixtures/registration_runtime_entity_id_sql_gate_recorded_omn24495.json"
)
_MIGRATION = Path(
    "docker/migrations/forward/117_add_registration_runtime_entity_id.sql"
)
_SCHEMA = Path("src/omnibase_infra/schemas/schema_registration_projection.sql")


@pytest.fixture
def recorded_response() -> dict[str, object]:
    recording: dict[str, object] = json.loads(
        (_ROOT / _RECORDED).read_text(encoding="utf-8")
    )
    return recording


@pytest.mark.live_contact(
    "tests/ci/fixtures/registration_runtime_entity_id_sql_gate_recorded_omn24495.json"
)
def test_recorded_gate_run_fails_qualified_names_and_passes_the_fix(
    recorded_response: dict[str, object],
) -> None:
    before = recorded_response["before_fix"]
    after = recorded_response["after_fix"]
    assert isinstance(before, dict) and isinstance(after, dict)
    assert before["verdict"] == "FAIL"
    violations = before["violations"]
    assert isinstance(violations, list) and len(violations) == 4
    assert all("public.registration_" in v or "'public'" in v for v in violations)
    assert after["verdict"] == "PASS"
    assert "application_database_sql_gate=PASS" in after["output_lines"]


@pytest.mark.parametrize("path", [_MIGRATION, _SCHEMA])
def test_service_database_registration_sql_is_exempt_and_unqualified(
    path: Path,
) -> None:
    assert _is_legacy_default_schema_sql_path(path)
    sql = (_ROOT / path).read_text(encoding="utf-8")
    assert "runtime_entity_id" in sql
    assert "public.registration_" not in sql


def test_exemption_does_not_cover_other_paths() -> None:
    assert not _is_legacy_default_schema_sql_path(
        Path("docker/migrations/forward/118_unlisted.sql")
    )

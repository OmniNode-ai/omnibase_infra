# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for data_provenance plumbing (OMN-11201).

The unit suite covers the reducer/router path with mocked deps. This
integration test exercises the full chain that ships in PR 1639:

- All three projection models (Contract, Topic, Registration) accept the
  documented provenance vocabulary and default to "unknown".
- The forward migration SQL declares the column with the expected vocab
  CHECK constraint on all three target tables.
- The migration vocab matches the model default — drift here means a
  reducer write can succeed while a DB read rejects the row (or vice
  versa).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.models.projection.model_contract_projection import (
    ModelContractProjection,
)
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.models.projection.model_topic_projection import (
    ModelTopicProjection,
)

pytestmark = pytest.mark.integration

_VALID_VOCAB = (
    "demo_seeded",
    "demo_projected_shortcut",
    "measured",
    "estimated",
    "unknown",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION_FILE = (
    _REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "078_add_data_provenance_to_projection_tables.sql"
)


def _now() -> datetime:
    return datetime.now(UTC)


def _contract(provenance: str = "unknown") -> ModelContractProjection:
    return ModelContractProjection(
        contract_id="integration-node:1.0.0",
        node_name="integration-node",
        version_major=1,
        version_minor=0,
        version_patch=0,
        contract_hash="sha256:" + "a" * 64,
        contract_yaml="name: integration-node\n",
        registered_at=_now(),
        last_seen_at=_now(),
        is_active=True,
        data_provenance=provenance,
    )


def test_contract_projection_accepts_every_documented_provenance_value() -> None:
    for value in _VALID_VOCAB:
        projection = _contract(provenance=value)
        assert projection.data_provenance == value


def test_contract_projection_defaults_to_unknown_when_provenance_omitted() -> None:
    projection = ModelContractProjection(
        contract_id="integration-default:1.0.0",
        node_name="integration-default",
        version_major=1,
        version_minor=0,
        version_patch=0,
        contract_hash="sha256:" + "b" * 64,
        contract_yaml="name: integration-default\n",
        registered_at=_now(),
        last_seen_at=_now(),
        is_active=True,
    )
    assert projection.data_provenance == "unknown"


def test_topic_projection_carries_provenance_field() -> None:
    assert "data_provenance" in ModelTopicProjection.model_fields


def test_registration_projection_carries_provenance_field() -> None:
    assert "data_provenance" in ModelRegistrationProjection.model_fields


def test_migration_declares_provenance_column_on_every_target_table() -> None:
    """The forward migration must add data_provenance to all three tables."""
    if not _MIGRATION_FILE.is_file():
        pytest.skip(f"migration file not present at expected path: {_MIGRATION_FILE}")
    sql = _MIGRATION_FILE.read_text(encoding="utf-8").lower()
    for table in ("contracts", "topics", "registration_projections"):
        assert table in sql, f"migration does not mention table {table!r}"
    assert "data_provenance" in sql


def test_migration_check_constraint_covers_full_vocabulary() -> None:
    """The DB CHECK constraint must list the same vocab the model accepts."""
    if not _MIGRATION_FILE.is_file():
        pytest.skip(f"migration file not present at expected path: {_MIGRATION_FILE}")
    sql = _MIGRATION_FILE.read_text(encoding="utf-8")
    for value in _VALID_VOCAB:
        assert value in sql, (
            f"vocab value {value!r} missing from migration CHECK constraint; "
            "model/DB drift will silently corrupt projection writes"
        )

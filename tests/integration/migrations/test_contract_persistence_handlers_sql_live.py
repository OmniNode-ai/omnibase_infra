# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run every real SQL string of node_contract_persistence_effect on PostgreSQL 16.

WHY THIS EXISTS
---------------
``SQL_UPSERT_TOPIC`` in ``handler_postgres_topic_update.py`` carried a Python
style marker comment INSIDE the triple-quoted SQL text. A ``#`` is not an SQL
comment, so PostgreSQL answered every call with ``syntax error at or near "#"``:
the dev runtime logged 640 of them in six minutes. Every existing test of these
handlers drives a mocked pool, so none of them ever handed the SQL text to a
server.

This module closes that: it applies the forward migrations that create the
``contracts`` and ``topics`` tables and their ``data_provenance`` column, then drives each of the six handlers of
the directory through a real asyncpg pool and reads the table state back. A
comment, a typo or a wrong column inside ANY SQL string of the directory turns
the matching step red, because the handler's own ``handle()`` returns
``success=False`` carrying the server's error.

A second, cluster-free test fails when a SQL constant is added to the directory
without a step here, so the next handler cannot dodge the live run.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import asyncpg
import pytest

import omnibase_infra.nodes.node_contract_persistence_effect.handlers as handlers_pkg
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_cleanup_topics import (
    HandlerPostgresCleanupTopics,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_contract_upsert import (
    HandlerPostgresContractUpsert,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_deactivate import (
    HandlerPostgresDeactivate,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_heartbeat import (
    HandlerPostgresHeartbeat,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_mark_stale import (
    HandlerPostgresMarkStale,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_topic_update import (
    HandlerPostgresTopicUpdate,
)
from omnibase_infra.nodes.node_contract_registry_reducer.models import (
    ModelPayloadCleanupTopicReferences,
    ModelPayloadDeactivateContract,
    ModelPayloadMarkStale,
    ModelPayloadUpdateHeartbeat,
    ModelPayloadUpdateTopic,
    ModelPayloadUpsertContract,
)
from tests.integration.migrations.conftest import EphemeralPostgres

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_MIGRATIONS = (
    "001_registration_projection.sql",
    "005_create_contracts_topics.sql",
    "078_add_data_provenance_to_projection_tables.sql",
)

#: Every module-level SQL string of the directory, by the name it carries, and
#: the handler class whose step in ``test_every_handler_sql_runs_on_postgres``
#: executes it.
_SQL_CONSTANT_COVERAGE: dict[str, type] = {
    "handler_postgres_cleanup_topics.SQL_CLEANUP_TOPIC_REFERENCES": (
        HandlerPostgresCleanupTopics
    ),
    "handler_postgres_contract_upsert.SQL_UPSERT_CONTRACT": (
        HandlerPostgresContractUpsert
    ),
    "handler_postgres_deactivate.SQL_DEACTIVATE_CONTRACT": HandlerPostgresDeactivate,
    "handler_postgres_heartbeat._UPDATE_HEARTBEAT_SQL": HandlerPostgresHeartbeat,
    "handler_postgres_mark_stale._MARK_STALE_SQL": HandlerPostgresMarkStale,
    "handler_postgres_topic_update.SQL_UPSERT_TOPIC": HandlerPostgresTopicUpdate,
}

_T0 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
_TOPIC = "onex.evt.platform.contract-registered.v1"  # onex-topic-allow: fixture row for the SQL live proof
_CONTRACT_A = "node-a:1.0.0"
_CONTRACT_B = "node-b:1.0.0"


def _sql_constants() -> dict[str, str]:
    """Every module-level string whose name says it is SQL, in the directory."""
    found: dict[str, str] = {}
    for module_info in pkgutil.iter_modules(handlers_pkg.__path__):
        module = importlib.import_module(f"{handlers_pkg.__name__}.{module_info.name}")
        for attr, value in vars(module).items():
            if isinstance(value, str) and "SQL" in attr.upper():
                found[f"{module_info.name}.{attr}"] = value
    return found


def test_every_sql_constant_of_the_directory_has_a_live_step() -> None:
    """A SQL string added without a live step here fails this test."""
    assert set(_sql_constants()) == set(_SQL_CONSTANT_COVERAGE), (
        "a SQL constant of node_contract_persistence_effect/handlers has no step "
        "in test_every_handler_sql_runs_on_postgres; add a step and list it"
    )


def _ok(step: str, result: ModelBackendResult) -> None:
    assert result.success, (
        f"{step}: PostgreSQL refused the handler's SQL: {result.error}"
    )


def _upsert(
    contract_id: str, node_name: str, seen: datetime
) -> ModelPayloadUpsertContract:
    return ModelPayloadUpsertContract(
        correlation_id=UUID(int=1),
        contract_id=contract_id,
        node_name=node_name,
        version_major=1,
        version_minor=0,
        version_patch=0,
        contract_hash=f"hash-{node_name}",
        contract_yaml={"name": node_name},
        is_active=True,
        registered_at=seen,
        last_seen_at=seen,
    )


def _topic(contract_id: str, node_name: str, seen: datetime) -> ModelPayloadUpdateTopic:
    return ModelPayloadUpdateTopic(
        correlation_id=UUID(int=2),
        topic_suffix=f"{{env}}.{_TOPIC}",
        direction="subscribe",
        contract_id=contract_id,
        node_name=node_name,
        last_seen_at=seen,
    )


@pytest.fixture
async def pool(ephemeral_postgres: EphemeralPostgres) -> AsyncIterator[asyncpg.Pool]:
    for name in _MIGRATIONS:
        applied = ephemeral_postgres.psql(
            "-v", "ON_ERROR_STOP=1", "-f", str(_FORWARD / name)
        )
        assert applied.returncode == 0, applied.stdout + applied.stderr
    created = await asyncpg.create_pool(
        host=ephemeral_postgres.socket_dir,
        port=ephemeral_postgres.port,
        user="postgres",
        database="postgres",
        min_size=1,
        max_size=2,
    )
    assert created is not None
    try:
        yield created
    finally:
        await created.close()


@pytest.mark.ephemeral_pg
@pytest.mark.postgres
async def test_every_handler_sql_runs_on_postgres(pool: asyncpg.Pool) -> None:
    """Each handler's real SQL runs on a real server and leaves the right rows."""
    upsert = HandlerPostgresContractUpsert(pool)
    topic = HandlerPostgresTopicUpdate(pool)
    heartbeat = HandlerPostgresHeartbeat(pool)
    mark_stale = HandlerPostgresMarkStale(pool)
    cleanup = HandlerPostgresCleanupTopics(pool)
    deactivate = HandlerPostgresDeactivate(pool)

    _ok("contract upsert A", await upsert.handle(_upsert(_CONTRACT_A, "node-a", _T0)))
    _ok("contract upsert B", await upsert.handle(_upsert(_CONTRACT_B, "node-b", _T0)))

    # Topic upsert: first sight inserts, a second contract appends, a repeat of
    # the same contract leaves the array unchanged (the `?` containment branch).
    _ok("topic update A", await topic.handle(_topic(_CONTRACT_A, "node-a", _T0)))
    _ok("topic update B", await topic.handle(_topic(_CONTRACT_B, "node-b", _T0)))
    _ok("topic update A again", await topic.handle(_topic(_CONTRACT_A, "node-a", _T0)))
    row = await pool.fetchrow("SELECT topic_suffix, contract_ids FROM topics")
    assert row is not None
    assert row["topic_suffix"] == _TOPIC
    assert sorted(json.loads(row["contract_ids"])) == [_CONTRACT_A, _CONTRACT_B]

    later = _T0 + timedelta(minutes=5)
    _ok(
        "heartbeat",
        await heartbeat.handle(
            ModelPayloadUpdateHeartbeat(
                correlation_id=UUID(int=3),
                contract_id=_CONTRACT_A,
                node_name="node-a",
                last_seen_at=later,
            )
        ),
    )
    assert (
        await pool.fetchval(
            "SELECT last_seen_at FROM contracts WHERE contract_id = $1", _CONTRACT_A
        )
        == later
    )

    # B was last seen at _T0, A at `later`: a cutoff between them stales only B.
    _ok(
        "mark stale",
        await mark_stale.handle(
            ModelPayloadMarkStale(
                correlation_id=UUID(int=4),
                stale_cutoff=_T0 + timedelta(minutes=1),
                checked_at=later,
            )
        ),
    )
    assert await pool.fetchval(
        "SELECT is_active FROM contracts WHERE contract_id = $1", _CONTRACT_A
    )
    assert not await pool.fetchval(
        "SELECT is_active FROM contracts WHERE contract_id = $1", _CONTRACT_B
    )

    _ok(
        "cleanup topic references",
        await cleanup.handle(
            ModelPayloadCleanupTopicReferences(
                correlation_id=UUID(int=5),
                contract_id=_CONTRACT_B,
                node_name="node-b",
                cleaned_at=later,
            )
        ),
    )
    assert json.loads(await pool.fetchval("SELECT contract_ids FROM topics")) == [
        _CONTRACT_A
    ]

    _ok(
        "deactivate",
        await deactivate.handle(
            ModelPayloadDeactivateContract(
                correlation_id=UUID(int=6),
                contract_id=_CONTRACT_A,
                node_name="node-a",
                reason="live proof",
                deactivated_at=later,
            )
        ),
    )
    assert not await pool.fetchval(
        "SELECT is_active FROM contracts WHERE contract_id = $1", _CONTRACT_A
    )

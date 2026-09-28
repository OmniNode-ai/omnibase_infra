# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Adversarial target-separation and resume checks for sim replay."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr, ValidationError

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    _OWNER_PROOF_MINT,
    DelegationOwnerProof,
)
from omnibase_infra.runtime.models.model_execution_graph_read_databases import (
    ModelExecutionGraphReadDatabases,
)
from omnibase_infra.runtime.sim_archive_rehydration_outbox import (
    PostgresSimArchiveRehydrationOutbox,
)
from omnibase_infra.runtime.sim_archive_source_receipt import (
    RawSimArchiveRecord,
    VerifiedSimArchiveRehydrationPlan,
)
from omnibase_infra.runtime.sim_preflight_rehydration import (
    ModelSimPreflightRehydrationConfig,
    rehydrate_verified_sim_preflight_chain,
)
from tests.unit.runtime.test_sim_archive_source_receipt_omn19728 import (
    _fixture,
    _topology,
)


@pytest.fixture(autouse=True)
def _sim_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "sim-202")


def _config(
    *,
    source_analytics: str = "postgresql://readonly@source.example:5432/omnidash_analytics",
    source_ledger: str = "postgresql://readonly@source.example:5432/omnibase_infra",
    source_brokers: str = "source.example:9092",
) -> ModelSimPreflightRehydrationConfig:
    return ModelSimPreflightRehydrationConfig(
        source_databases=ModelExecutionGraphReadDatabases(
            analytics_dsn=SecretStr(source_analytics),
            ledger_dsn=SecretStr(source_ledger),
        ),
        source_kafka=ModelKafkaEventBusConfig(
            bootstrap_servers=source_brokers, environment="source"
        ),
        source_topic_namespace="",
        target_ledger_dsn=SecretStr(
            "postgresql://target@127.0.0.1:65036/omnibase_infra"
        ),
        topology_version=_topology().version,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source_analytics", "source_ledger", "source_brokers"),
    [
        (
            "postgresql://readonly@localhost:65036/omnidash_analytics",
            "postgresql://readonly@source.example:5432/omnibase_infra",
            "source.example:9092",
        ),
        (
            "postgresql://readonly@source.example:5432/omnidash_analytics",
            "postgresql://readonly@localhost:65036/omnibase_infra",
            "source.example:9092",
        ),
        (
            "postgresql://readonly@source.example:5432/omnidash_analytics",
            "postgresql://readonly@source.example:5432/omnibase_infra",
            "source.example:9092,localhost:65092",
        ),
    ],
)
def test_source_coordinates_cannot_alias_loopback_replay_target(
    source_analytics: str,
    source_ledger: str,
    source_brokers: str,
) -> None:
    with pytest.raises(ValidationError):
        _config(
            source_analytics=source_analytics,
            source_ledger=source_ledger,
            source_brokers=source_brokers,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_each_resume_rechecks_fresh_source_before_target_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnibase_infra.runtime import sim_preflight_rehydration as composition

    authority, _verifier, records, rows, _source, _calls = _fixture()
    fresh_records = {record.source_key: record for record in records}
    steps: list[str] = []
    target_pool_opens = 0
    enqueued: list[tuple[str, int, int]] = []

    class FakePool:
        async def close(self) -> None:
            return None

    async def open_pool(*, dsn: str, server_settings: object = None) -> FakePool:
        nonlocal target_pool_opens
        if server_settings is None:
            target_pool_opens += 1
            steps.append("target-pool")
        else:
            assert server_settings == {"default_transaction_read_only": "on"}
            steps.append("source-readonly-pool")
        return FakePool()

    class OwnerReader:
        async def require_owner(
            self, observed_authority: object
        ) -> DelegationOwnerProof:
            assert observed_authority is authority
            steps.append("owner")
            return DelegationOwnerProof(
                correlation_id=authority.correlation_id,
                tenant_id=authority.tenant_id,
                _mint=_OWNER_PROOF_MINT,
            )

    class LedgerReader:
        async def read_full_current(
            self, _authority: object, _owner: object, _read_set: object
        ) -> tuple[object, ...]:
            steps.append("current-ledger")
            return rows

    class FreshSourceReader:
        async def read_exact(
            self, topic: str, partition: int, offset: int
        ) -> RawSimArchiveRecord | None:
            steps.append("fresh-source")
            return fresh_records.get((topic, partition, offset))

    class FakeProducer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def send_and_wait(self, *_args: object, **_kwargs: object) -> None:
            return None

    class FakeOutbox:
        async def require_only_selected_keys(
            self, plans: tuple[VerifiedSimArchiveRehydrationPlan, ...]
        ) -> None:
            assert tuple(plan.source_key for plan in plans) == tuple(
                record.source_key for record in records
            )
            steps.append("target-keys-checked")

        async def enqueue_once(self, plan: VerifiedSimArchiveRehydrationPlan) -> bool:
            enqueued.append(plan.source_key)
            return True

        async def relay_selected_once(
            self, plans: tuple[VerifiedSimArchiveRehydrationPlan, ...]
        ) -> int:
            return len(plans)

    monkeypatch.setattr(
        composition, "verify_live_sim_preflight_target", lambda receipt=None: None
    )
    monkeypatch.setattr(composition.asyncpg, "create_pool", open_pool)
    monkeypatch.setattr(
        composition,
        "PostgresDelegationOwnerReader",
        lambda _pool: OwnerReader(),
    )
    monkeypatch.setattr(
        composition,
        "PostgresSimArchiveSourceLedgerReader",
        lambda _pool: LedgerReader(),
    )
    monkeypatch.setattr(
        composition,
        "KafkaSimArchiveSourceReader",
        lambda *_args, **_kwargs: FreshSourceReader(),
    )
    monkeypatch.setattr(composition, "AIOKafkaProducer", FakeProducer)
    monkeypatch.setattr(
        composition,
        "PostgresSimArchiveRehydrationOutbox",
        lambda *_args, **_kwargs: FakeOutbox(),
    )

    config = _config()
    assert (
        await rehydrate_verified_sim_preflight_chain(
            authority=authority, archive_records=records, config=config
        )
        == 5
    )
    assert target_pool_opens == 1
    assert len(enqueued) == 5
    assert enqueued == [record.source_key for record in records]

    changed = records[0]
    fresh_records[changed.source_key] = replace(changed, value=b"source changed")
    writes_before_resume = len(enqueued)
    with pytest.raises(ValueError, match="archive differs from fresh source receipt"):
        await rehydrate_verified_sim_preflight_chain(
            authority=authority, archive_records=records, config=config
        )

    assert target_pool_opens == 1
    assert len(enqueued) == writes_before_resume
    assert steps.count("owner") == 2
    assert steps.count("current-ledger") == 2
    assert steps.count("fresh-source") > len(records)

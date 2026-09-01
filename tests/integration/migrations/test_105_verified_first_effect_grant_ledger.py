# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real Unix-socket PostgreSQL proof for verifier-gated staging and claims."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import asyncpg
import pytest
from omninode_grant_verifier import signed_executable_grant_v2_vectors

from omnibase_infra.errors.repository import RepositoryExecutionError
from omnibase_infra.runtime.first_effect_ledger import (
    ModelFirstEffectConsumerClaim,
    ModelFirstEffectOutputIdentity,
    ModelFirstEffectStageRequest,
)
from omnibase_infra.runtime.first_effect_ledger.adapter_postgres_verified_first_effect_staging import (
    FirstEffectStagingRejectedError,
    PostgresVerifiedFirstEffectStaging,
)
from omnibase_infra.runtime.first_effect_ledger.composition import (
    build_rsd_verified_first_effect_grant_ingress,
)
from omnibase_infra.runtime.first_effect_ledger.model_expected_output_pin import (
    ModelExpectedOutputPin,
)
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_ROOT = Path(__file__).resolve().parents[3]
_MIGRATIONS = tuple(
    _ROOT / "docker/migrations/forward" / name
    for name in (
        "090_create_delegation_workflow_state.sql",
        "093_add_delegation_workflow_state_outbox_columns.sql",
        "105_create_verified_first_effect_grant_ledger.sql",
        "106_align_verified_first_effect_grant_with_rsd_v2.sql",
    )
)
_ROLLBACK = (
    _ROOT
    / "docker/migrations/rollback/rollback_106_align_verified_first_effect_grant_with_rsd_v2.sql",
    _ROOT
    / "docker/migrations/rollback/rollback_105_create_verified_first_effect_grant_ledger.sql",
)
_TOPIC = "onex.evt.delegation.model-inference-intent.v1"
_CLASS = "ModelInferenceIntent"
_RSD_NOW = datetime(2030, 1, 1, tzinfo=UTC)
_RSD_OUTPUT_PIN = ModelExpectedOutputPin(
    topic="events.public.grant.completed.v2",
    event_class="ModelPublicGrantCompleted",
    index=0,
)


def _apply(ephemeral_postgres: EphemeralPostgres, migration: Path) -> None:
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
    assert result.returncode == 0, result.stderr


def _digest(value: int) -> str:
    return f"{value:064x}"


def _valid_rsd_wire() -> str:
    vectors = signed_executable_grant_v2_vectors()
    wire = vectors["base_wire"]
    assert type(wire) is dict
    return json.dumps(wire)


def _valid_rsd_workflow_identity() -> tuple[str, str]:
    wire = json.loads(_valid_rsd_wire())
    authorization = wire["authorization_material"]
    assert type(authorization) is dict
    material = authorization["grant"]
    assert type(material) is dict
    correlation_id = material["correlation_id"]
    tenant_id = material["tenant_id"]
    assert type(correlation_id) is str and type(tenant_id) is str
    return correlation_id, tenant_id


@dataclass(frozen=True)
class _VerifiedFixture:
    authorization_digest: str
    grant_id: UUID
    grant_envelope_id: UUID
    nonce_digest: str
    request_digest: str
    correlation_id: UUID
    tenant_id: str
    backend_id: str
    rendered_contract_sha256: str
    issuer_key_fingerprint_sha256: str
    retry_disposition: str
    expected_output_topic: str
    expected_output_event_class: str
    expected_output_event_index: int


def _fixture(identity: int) -> _VerifiedFixture:
    return _VerifiedFixture(
        authorization_digest=_digest(identity * 10 + 1),
        grant_id=UUID(int=identity * 10 + 2),
        grant_envelope_id=UUID(int=identity * 10 + 3),
        nonce_digest=_digest(identity * 10 + 4),
        request_digest=_digest(identity * 10 + 5),
        correlation_id=UUID(int=identity),
        tenant_id=f"tenant-{identity}",
        backend_id="backend-a",
        rendered_contract_sha256=_digest(identity * 10 + 6),
        issuer_key_fingerprint_sha256=_digest(identity * 10 + 7),
        retry_disposition="forbidden",
        expected_output_topic=_TOPIC,
        expected_output_event_class=_CLASS,
        expected_output_event_index=0,
    )


def _stage(
    projection: _VerifiedFixture,
) -> ModelFirstEffectStageRequest:
    return ModelFirstEffectStageRequest(
        authorization_digest=projection.authorization_digest,
        expected_ledger_version=0,
        expected_workflow_version=0,
        outbox_body_sha256=_digest(90 + projection.correlation_id.int),
        emitted_output_topic=_TOPIC,
        emitted_output_event_class=_CLASS,
        emitted_output_event_index=0,
        pending_emissions_json=json.dumps(
            [{"class_name": _CLASS, "index": 0, "payload": {"not": "ledger"}}]
        ),
    )


async def _pool(ephemeral_postgres: EphemeralPostgres) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        host=ephemeral_postgres.socket_dir,
        port=ephemeral_postgres.port,
        user="postgres",
        database="postgres",
        min_size=1,
        max_size=2,
    )


async def _adapter(
    ephemeral_postgres: EphemeralPostgres,
) -> PostgresVerifiedFirstEffectStaging:
    pool = await _pool(ephemeral_postgres)

    async def factory() -> asyncpg.Pool:
        return pool

    return PostgresVerifiedFirstEffectStaging(pool_factory=factory)


async def _seed_workflow(
    ephemeral_postgres: EphemeralPostgres,
    projection: _VerifiedFixture,
) -> None:
    await _seed_workflow_identity(
        ephemeral_postgres,
        correlation_id=str(projection.correlation_id),
        tenant_id=projection.tenant_id,
    )


async def _seed_verified_fixture(
    ephemeral_postgres: EphemeralPostgres, fixture: _VerifiedFixture
) -> None:
    pool = await _pool(ephemeral_postgres)
    try:
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO public.first_effect_verified_grant_ledger (
                    authorization_digest, grant_id, grant_envelope_id, nonce_digest,
                    request_digest, correlation_id, tenant_id, backend_id,
                    rendered_contract_sha256, issuer_key_fingerprint_sha256,
                    retry_disposition, expected_output_topic,
                    expected_output_event_class, expected_output_event_index, state
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                    'VERIFIED'
                )
                """,
                fixture.authorization_digest,
                fixture.grant_id,
                fixture.grant_envelope_id,
                fixture.nonce_digest,
                fixture.request_digest,
                str(fixture.correlation_id),
                fixture.tenant_id,
                fixture.backend_id,
                fixture.rendered_contract_sha256,
                fixture.issuer_key_fingerprint_sha256,
                fixture.retry_disposition,
                fixture.expected_output_topic,
                fixture.expected_output_event_class,
                fixture.expected_output_event_index,
            )
    finally:
        await pool.close()


async def _seed_workflow_identity(
    ephemeral_postgres: EphemeralPostgres,
    *,
    correlation_id: str,
    tenant_id: str,
) -> None:
    pool = await _pool(ephemeral_postgres)
    try:
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO public.delegation_workflow_state
                    (correlation_id, tenant_id, state, payload, version)
                VALUES ($1, $2, 'INFERENCE_COMPLETED', '{}'::jsonb, 0)
                """,
                correlation_id,
                tenant_id,
            )
    finally:
        await pool.close()


@pytest.mark.integration
def test_105_applies_and_rollback_removes_verified_grant_boundary(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    for migration in _MIGRATIONS:
        _apply(ephemeral_postgres, migration)
    for rollback in _ROLLBACK:
        _apply(ephemeral_postgres, rollback)
    connection = ephemeral_postgres.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.first_effect_verified_grant_ledger'), "
                "to_regprocedure('public.enforce_verified_first_effect_grant_transition()')"
            )
            assert cursor.fetchone() == (None, None)
    finally:
        connection.close()


@pytest.mark.integration
def test_106_upgrades_105_for_rsd_v2_and_refuses_lossy_rollback(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    for migration in _MIGRATIONS[:-1]:
        _apply(ephemeral_postgres, migration)

    async def scenario() -> None:
        correlation_id, tenant_id = _valid_rsd_workflow_identity()
        await _seed_workflow_identity(
            ephemeral_postgres,
            correlation_id=correlation_id,
            tenant_id=tenant_id,
        )
        authority = await _adapter(ephemeral_postgres)
        try:

            async def ingress_pool_factory() -> asyncpg.Pool:
                return await _pool(ephemeral_postgres)

            ingress = build_rsd_verified_first_effect_grant_ingress(
                pool_factory=ingress_pool_factory, expected_output_pin=_RSD_OUTPUT_PIN
            )
            with pytest.raises(RepositoryExecutionError, match="verified first-effect"):
                await ingress.verify_and_record(_valid_rsd_wire(), now=_RSD_NOW)

            _apply(ephemeral_postgres, _MIGRATIONS[-1])
            verified = await ingress.verify_and_record(_valid_rsd_wire(), now=_RSD_NOW)
            assert verified.retry_disposition == "forbidden"
            assert verified.expected_output_topic == "events.public.grant.completed.v2"
        finally:
            await authority.close()

    asyncio.run(scenario())
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(_ROLLBACK[0]))
    assert result.returncode != 0
    assert "cannot roll back migration 106" in result.stderr


@pytest.mark.integration
def test_105_real_transaction_rollback_concurrent_claim_and_terminal_chain(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    for migration in _MIGRATIONS:
        _apply(ephemeral_postgres, migration)

    async def scenario() -> None:
        projection = _fixture(1)
        await _seed_workflow(ephemeral_postgres, projection)
        authority = await _adapter(ephemeral_postgres)
        try:

            async def ingress_pool_factory() -> asyncpg.Pool:
                return await _pool(ephemeral_postgres)

            await _seed_verified_fixture(ephemeral_postgres, projection)
            verified = await authority.get(
                authorization_digest=projection.authorization_digest
            )
            assert verified is not None
            assert verified.state.value == "VERIFIED"
            staged = await authority.stage(_stage(projection))
            assert staged.state.value == "STAGED"
            assert (
                staged.outbox_envelope_id
                == authority.deterministic_outbox_envelope_id(
                    projection.authorization_digest
                )
            )
            assert staged.outbox_topic == _TOPIC
            assert staged.outbox_event_class == _CLASS
            assert staged.outbox_event_index == 0

            # Closing and rebuilding the adapter simulates a process crash after
            # commit: the durable staged row is recoverable without a duplicate.
            await authority.close()
            authority = await _adapter(ephemeral_postgres)
            assert (
                await authority.get(
                    authorization_digest=projection.authorization_digest
                )
            ).state.value == "STAGED"  # type: ignore[union-attr]

            identity = ModelFirstEffectOutputIdentity(
                authorization_digest=projection.authorization_digest,
                outbox_envelope_id=staged.outbox_envelope_id,
                outbox_body_sha256=staged.outbox_body_sha256,
                output_topic=_TOPIC,
                output_event_class=_CLASS,
                output_event_index=0,
            )
            with pytest.raises(FirstEffectStagingRejectedError):
                await authority.claim_direct_message(
                    ModelFirstEffectConsumerClaim(
                        **identity.model_dump(), expected_ledger_version=staged.version
                    )
                )
            with pytest.raises(FirstEffectStagingRejectedError):
                await authority.record_published_unknown(
                    identity=identity, expected_ledger_version=staged.version
                )
            publishing = await authority.record_publishing_started(
                identity=identity, expected_ledger_version=staged.version
            )
            unknown = await authority.record_published_unknown(
                identity=identity, expected_ledger_version=publishing.version
            )
            assert unknown.state.value == "PUBLISHED_UNKNOWN"
            one = await _adapter(ephemeral_postgres)
            two = await _adapter(ephemeral_postgres)
            try:
                raced = await asyncio.gather(
                    one.claim_direct_message(
                        ModelFirstEffectConsumerClaim(
                            **identity.model_dump(),
                            expected_ledger_version=unknown.version,
                        )
                    ),
                    two.claim_direct_message(
                        ModelFirstEffectConsumerClaim(
                            **identity.model_dump(),
                            expected_ledger_version=unknown.version,
                        )
                    ),
                    return_exceptions=True,
                )
            finally:
                await one.close()
                await two.close()
            winners = [row for row in raced if not isinstance(row, BaseException)]
            assert len(winners) == 1
            assert winners[0].state.value == "CLAIMED"
            assert (
                sum(isinstance(row, FirstEffectStagingRejectedError) for row in raced)
                == 1
            )

            claimed = winners[0]
            terminal = await authority.record_terminal(
                identity=identity,
                expected_ledger_version=claimed.version,
            )
            assert terminal.state.value == "TERMINAL"

            with pytest.raises(FirstEffectStagingRejectedError):
                await authority.claim_direct_message(
                    ModelFirstEffectConsumerClaim(
                        **(
                            identity.model_dump()
                            | {"authorization_digest": _digest(999)}
                        ),
                        expected_ledger_version=staged.version,
                    )
                )

            contradictory_projection = _fixture(3)
            await _seed_workflow(ephemeral_postgres, contradictory_projection)
            await _seed_verified_fixture(ephemeral_postgres, contradictory_projection)
            contradictory_stage = _stage(contradictory_projection).model_copy(
                update={
                    "emitted_output_topic": "onex.evt.other.model-inference-intent.v1"
                }
            )
            with pytest.raises(FirstEffectStagingRejectedError, match="contradicts"):
                await authority.stage(contradictory_stage)
            contradiction = await authority.get(
                authorization_digest=contradictory_projection.authorization_digest
            )
            assert contradiction is not None and contradiction.state.value == "VERIFIED"

            rollback_projection = _fixture(2)
            await _seed_workflow(ephemeral_postgres, rollback_projection)
            pool = await _pool(ephemeral_postgres)
            try:
                async with pool.acquire() as connection:
                    await connection.execute(
                        """
                        CREATE FUNCTION public.fail_first_effect_stage_for_test()
                        RETURNS TRIGGER LANGUAGE plpgsql AS $$
                        BEGIN RAISE EXCEPTION 'forced stage rollback'; END; $$;
                        CREATE TRIGGER aaa_fail_first_effect_stage_for_test
                        BEFORE UPDATE ON public.first_effect_verified_grant_ledger
                        FOR EACH ROW EXECUTE FUNCTION public.fail_first_effect_stage_for_test();
                        """
                    )
            finally:
                await pool.close()
            await _seed_verified_fixture(ephemeral_postgres, rollback_projection)
            with pytest.raises(RepositoryExecutionError, match="stage"):
                await authority.stage(_stage(rollback_projection))
            persisted = await authority.get(
                authorization_digest=rollback_projection.authorization_digest
            )
            assert persisted is not None and persisted.state.value == "VERIFIED"
            pool = await _pool(ephemeral_postgres)
            try:
                async with pool.acquire() as connection:
                    workflow = await connection.fetchrow(
                        "SELECT pending_emissions, version FROM public.delegation_workflow_state WHERE correlation_id = $1",
                        str(rollback_projection.correlation_id),
                    )
                    assert workflow is not None
                    assert workflow["pending_emissions"] is None
                    assert workflow["version"] == 0
            finally:
                await pool.close()
        finally:
            await authority.close()

    asyncio.run(scenario())


@pytest.mark.integration
def test_105_rsd_verified_wire_stages_and_claims_only_its_exact_output_pins(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    for migration in _MIGRATIONS:
        _apply(ephemeral_postgres, migration)

    async def scenario() -> None:
        authority = await _adapter(ephemeral_postgres)
        try:

            async def ingress_pool_factory() -> asyncpg.Pool:
                return await _pool(ephemeral_postgres)

            raw_wire = json.loads(_valid_rsd_wire())
            authorization = raw_wire["authorization_material"]
            assert type(authorization) is dict
            material = authorization["grant"]
            assert type(material) is dict
            correlation_id = material["correlation_id"]
            tenant_id = material["tenant_id"]
            assert type(correlation_id) is str and type(tenant_id) is str
            await _seed_workflow_identity(
                ephemeral_postgres,
                correlation_id=correlation_id,
                tenant_id=tenant_id,
            )
            ingress = build_rsd_verified_first_effect_grant_ingress(
                pool_factory=ingress_pool_factory, expected_output_pin=_RSD_OUTPUT_PIN
            )
            verified = await ingress.verify_and_record(_valid_rsd_wire(), now=_RSD_NOW)
            assert verified.retry_disposition == "forbidden"
            assert verified.expected_output_topic == "events.public.grant.completed.v2"
            assert verified.expected_output_event_class == "ModelPublicGrantCompleted"
            assert verified.expected_output_event_index == 0

            stage_request = ModelFirstEffectStageRequest(
                authorization_digest=verified.authorization_digest,
                expected_ledger_version=verified.version,
                expected_workflow_version=0,
                outbox_body_sha256=_digest(501),
                emitted_output_topic=verified.expected_output_topic,
                emitted_output_event_class=verified.expected_output_event_class,
                emitted_output_event_index=verified.expected_output_event_index,
                pending_emissions_json=json.dumps(
                    [
                        {
                            "class_name": verified.expected_output_event_class,
                            "index": verified.expected_output_event_index,
                            "payload": {"owned": "by-workflow"},
                        }
                    ]
                ),
            )
            staged = await authority.stage(stage_request)
            assert (
                staged.outbox_envelope_id
                == authority.deterministic_outbox_envelope_id(
                    verified.authorization_digest
                )
            )
            assert staged.outbox_topic == verified.expected_output_topic
            assert staged.outbox_event_class == verified.expected_output_event_class
            assert staged.outbox_event_index == verified.expected_output_event_index

            identity = ModelFirstEffectOutputIdentity(
                authorization_digest=verified.authorization_digest,
                outbox_envelope_id=staged.outbox_envelope_id,
                outbox_body_sha256=staged.outbox_body_sha256,
                output_topic=verified.expected_output_topic,
                output_event_class=verified.expected_output_event_class,
                output_event_index=verified.expected_output_event_index,
            )
            publishing = await authority.record_publishing_started(
                identity=identity, expected_ledger_version=staged.version
            )
            unknown = await authority.record_published_unknown(
                identity=identity, expected_ledger_version=publishing.version
            )
            with pytest.raises(FirstEffectStagingRejectedError):
                await authority.claim_direct_message(
                    ModelFirstEffectConsumerClaim(
                        **(
                            identity.model_dump() | {"outbox_body_sha256": _digest(502)}
                        ),
                        expected_ledger_version=unknown.version,
                    )
                )
            still_unknown = await authority.get(
                authorization_digest=verified.authorization_digest
            )
            assert still_unknown is not None
            assert still_unknown.state.value == "PUBLISHED_UNKNOWN"

            claimed = await authority.claim_direct_message(
                ModelFirstEffectConsumerClaim(
                    **identity.model_dump(), expected_ledger_version=unknown.version
                )
            )
            assert claimed.state.value == "CLAIMED"
        finally:
            await authority.close()

    asyncio.run(scenario())

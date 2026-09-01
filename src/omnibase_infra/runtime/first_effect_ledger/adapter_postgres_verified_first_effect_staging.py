# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: S608
"""PostgreSQL-only durable recording boundary for a verified first-effect grant.

This staging/claim scaffold has no publisher, broker client, provider client,
signature parser, cryptographic key, or VERIFIED-row creation path.  The sole
payload write is the workflow row's ``pending_emissions`` field; this ledger
records no payload and confers no effect permission.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from uuid import UUID, uuid5

import asyncpg

from omnibase_infra.errors.repository import RepositoryExecutionError
from omnibase_infra.runtime.first_effect_ledger.enum_verified_first_effect_grant_state import (
    EnumVerifiedFirstEffectGrantState,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    Sha256Digest,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_consumer_claim import (
    ModelFirstEffectConsumerClaim,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_output_identity import (
    ModelFirstEffectOutputIdentity,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_stage_request import (
    ModelFirstEffectStageRequest,
)
from omnibase_infra.runtime.first_effect_ledger.model_verified_first_effect_grant_record import (
    ModelVerifiedFirstEffectGrantRecord,
)

PoolFactory = Callable[[], Awaitable[asyncpg.Pool]]
_TABLE = "public.first_effect_verified_grant_ledger"
_WORKFLOW_TABLE = "public.delegation_workflow_state"
_OUTBOX_NAMESPACE = UUID("1694906c-7366-5bc6-987b-0361832131ce")
_RETURNING = """
authorization_digest, grant_id, grant_envelope_id, nonce_digest, request_digest,
correlation_id, tenant_id, backend_id, rendered_contract_sha256,
issuer_key_fingerprint_sha256, retry_disposition, expected_output_topic,
expected_output_event_class, expected_output_event_index, state,
outbox_envelope_id, outbox_body_sha256, outbox_topic, outbox_event_class,
outbox_event_index, workflow_version, verified_at, staged_at, claimed_at,
publishing_at, published_unknown_at, terminal_at, updated_at, version
"""


class FirstEffectStagingRejectedError(RuntimeError):
    """Raised for an absent, contradictory, stale, or consumed record."""


class PostgresVerifiedFirstEffectStaging:
    """Injected-pool staging/claim repository; it cannot create VERIFIED rows."""

    def __init__(self, *, pool_factory: PoolFactory) -> None:
        self._pool_factory = pool_factory
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                self._pool = await self._pool_factory()
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _record(row: Mapping[str, object]) -> ModelVerifiedFirstEffectGrantRecord:
        return ModelVerifiedFirstEffectGrantRecord.model_validate(dict(row))

    @staticmethod
    def deterministic_outbox_envelope_id(authorization_digest: Sha256Digest) -> UUID:
        """Derive the only permitted first-effect outbox id from authorization."""
        return uuid5(_OUTBOX_NAMESPACE, authorization_digest)

    async def get(
        self, *, authorization_digest: Sha256Digest
    ) -> ModelVerifiedFirstEffectGrantRecord | None:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as connection:
                row = await connection.fetchrow(
                    f"SELECT {_RETURNING} FROM {_TABLE} WHERE authorization_digest = $1",
                    authorization_digest,
                )
        except asyncpg.PostgresError as exc:
            raise RepositoryExecutionError(
                f"verified first-effect grant get failed: {type(exc).__name__}",
                op_name="get_verified_grant",
                table=_TABLE,
            ) from exc
        return None if row is None else self._record(row)

    async def stage(
        self, request: ModelFirstEffectStageRequest
    ) -> ModelVerifiedFirstEffectGrantRecord:
        """Atomically CAS the workflow payload owner and bind the ledger.

        The transaction locks the VERIFIED record, derives its envelope UUID,
        CASes the workflow's single pending batch, then advances the ledger to
        STAGED.  Any refusal or database failure aborts the complete transaction,
        leaving neither a workflow payload nor a ledger transition behind.
        """
        envelope_id = self.deterministic_outbox_envelope_id(
            request.authorization_digest
        )
        try:
            pool = await self._get_pool()
            async with pool.acquire() as connection:
                async with connection.transaction():
                    locked = await connection.fetchrow(
                        f"SELECT {_RETURNING} FROM {_TABLE} "
                        "WHERE authorization_digest = $1 FOR UPDATE",
                        request.authorization_digest,
                    )
                    if locked is None:
                        raise FirstEffectStagingRejectedError(
                            "no verified first-effect grant exists for staging"
                        )
                    record = self._record(locked)
                    if (
                        record.state is not EnumVerifiedFirstEffectGrantState.VERIFIED
                        or record.version != request.expected_ledger_version
                    ):
                        raise FirstEffectStagingRejectedError(
                            "first-effect grant is not the requested VERIFIED version"
                        )
                    if (
                        request.emitted_output_topic != record.expected_output_topic
                        or request.emitted_output_event_class
                        != record.expected_output_event_class
                        or request.emitted_output_event_index
                        != record.expected_output_event_index
                    ):
                        raise FirstEffectStagingRejectedError(
                            "staged output identity contradicts verified grant pins"
                        )
                    workflow = await connection.fetchrow(
                        f"""
                        UPDATE {_WORKFLOW_TABLE}
                           SET pending_emissions = $1::jsonb,
                               in_flight = TRUE,
                               version = version + 1
                         WHERE correlation_id = $2
                           AND tenant_id = $3
                           AND version = $4
                           AND (pending_emissions IS NULL OR pending_emissions = '[]'::jsonb)
                         RETURNING version
                        """,
                        request.pending_emissions_json,
                        str(record.correlation_id),
                        record.tenant_id,
                        request.expected_workflow_version,
                    )
                    if workflow is None:
                        raise FirstEffectStagingRejectedError(
                            "workflow CAS refused first-effect staging"
                        )
                    staged = await connection.fetchrow(
                        f"""
                        UPDATE {_TABLE}
                           SET state = 'STAGED', outbox_envelope_id = $2,
                               outbox_body_sha256 = $3,
                               outbox_topic = $4, outbox_event_class = $5,
                               outbox_event_index = $6,
                               workflow_version = $7, staged_at = NOW(),
                               version = version + 1
                         WHERE authorization_digest = $1
                           AND state = 'VERIFIED'
                           AND version = $8
                         RETURNING {_RETURNING}
                        """,
                        request.authorization_digest,
                        envelope_id,
                        request.outbox_body_sha256,
                        request.emitted_output_topic,
                        request.emitted_output_event_class,
                        request.emitted_output_event_index,
                        workflow["version"],
                        request.expected_ledger_version,
                    )
                    if staged is None:
                        raise FirstEffectStagingRejectedError(
                            "ledger CAS refused first-effect staging"
                        )
        except FirstEffectStagingRejectedError:
            raise
        except asyncpg.PostgresError as exc:
            raise RepositoryExecutionError(
                f"verified first-effect grant stage failed: {type(exc).__name__}",
                op_name="stage_verified_grant",
                table=_TABLE,
            ) from exc
        return self._record(staged)

    async def claim_direct_message(
        self, claim: ModelFirstEffectConsumerClaim
    ) -> ModelVerifiedFirstEffectGrantRecord:
        """Atomically claim one matching message during publish or ambiguity."""
        return await self._transition_matching_output(
            operation="claim_direct_message",
            identity=claim,
            expected_ledger_version=claim.expected_ledger_version,
            from_states=(
                EnumVerifiedFirstEffectGrantState.PUBLISHING,
                EnumVerifiedFirstEffectGrantState.PUBLISHED_UNKNOWN,
            ),
            to_state=EnumVerifiedFirstEffectGrantState.CLAIMED,
            timestamp_column="claimed_at",
        )

    async def record_publishing_started(
        self,
        *,
        identity: ModelFirstEffectOutputIdentity,
        expected_ledger_version: int,
    ) -> ModelVerifiedFirstEffectGrantRecord:
        """Record publisher ownership before a future publisher makes one attempt.

        This is state bookkeeping only.  The future publisher owns the actual
        broker call and must not retry after recording PUBLISHED_UNKNOWN.
        """
        return await self._transition_matching_output(
            operation="record_publishing_started",
            identity=identity,
            expected_ledger_version=expected_ledger_version,
            from_states=(EnumVerifiedFirstEffectGrantState.STAGED,),
            to_state=EnumVerifiedFirstEffectGrantState.PUBLISHING,
            timestamp_column="publishing_at",
        )

    async def record_published_unknown(
        self,
        *,
        identity: ModelFirstEffectOutputIdentity,
        expected_ledger_version: int,
    ) -> ModelVerifiedFirstEffectGrantRecord:
        """Record only an ambiguous result; this adapter did not publish it."""
        return await self._transition_matching_output(
            operation="record_published_unknown",
            identity=identity,
            expected_ledger_version=expected_ledger_version,
            from_states=(EnumVerifiedFirstEffectGrantState.PUBLISHING,),
            to_state=EnumVerifiedFirstEffectGrantState.PUBLISHED_UNKNOWN,
            timestamp_column="published_unknown_at",
        )

    async def record_terminal(
        self,
        *,
        identity: ModelFirstEffectOutputIdentity,
        expected_ledger_version: int,
    ) -> ModelVerifiedFirstEffectGrantRecord:
        """Record terminal state only after the pinned unknown state."""
        return await self._transition_matching_output(
            operation="record_terminal",
            identity=identity,
            expected_ledger_version=expected_ledger_version,
            from_states=(EnumVerifiedFirstEffectGrantState.CLAIMED,),
            to_state=EnumVerifiedFirstEffectGrantState.TERMINAL,
            timestamp_column="terminal_at",
        )

    async def _transition_matching_output(
        self,
        *,
        operation: str,
        identity: ModelFirstEffectOutputIdentity,
        expected_ledger_version: int,
        from_states: tuple[EnumVerifiedFirstEffectGrantState, ...],
        to_state: EnumVerifiedFirstEffectGrantState,
        timestamp_column: str,
    ) -> ModelVerifiedFirstEffectGrantRecord:
        # timestamp_column is a private fixed literal from this module, never a caller value.
        sql = f"""
        UPDATE {_TABLE}
           SET state = $2, {timestamp_column} = NOW(), version = version + 1
         WHERE authorization_digest = $1
           AND state = ANY($3::text[])
           AND version = $4
           AND outbox_envelope_id = $5
           AND outbox_body_sha256 = $6
           AND outbox_topic = $7
           AND outbox_event_class = $8
           AND outbox_event_index = $9
         RETURNING {_RETURNING}
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as connection:
                row = await connection.fetchrow(
                    sql,
                    identity.authorization_digest,
                    to_state.value,
                    [state.value for state in from_states],
                    expected_ledger_version,
                    identity.outbox_envelope_id,
                    identity.outbox_body_sha256,
                    identity.output_topic,
                    identity.output_event_class,
                    identity.output_event_index,
                )
        except asyncpg.PostgresError as exc:
            raise RepositoryExecutionError(
                f"verified first-effect grant {operation} failed: {type(exc).__name__}",
                op_name=operation,
                table=_TABLE,
            ) from exc
        if row is None:
            raise FirstEffectStagingRejectedError(
                "first-effect direct message is absent, contradictory, or already claimed"
            )
        return self._record(row)

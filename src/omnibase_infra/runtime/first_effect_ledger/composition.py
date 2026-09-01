# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: S608
"""Non-authorizing composition guard for external first-effect publishing.

``PostgresFirstEffectLedger`` deliberately records observations only. The
canonical outbox lives in the state-IO row and needs the producer-owned event
payload, so this package cannot manufacture a parallel publish table, grant an
effect permission, or publish. Production composition must inject an external
implementation of this protocol that commits its ledger transition and
canonical outbox intent in the same database transaction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Protocol, runtime_checkable

import asyncpg
from omninode_grant_verifier import (
    ExecutableGrantVerificationError,
    SignedExecutableGrantV2,
    verify_signed_executable_grant_v2,
)

from omnibase_infra.errors.repository import RepositoryExecutionError
from omnibase_infra.runtime.first_effect_ledger.model_expected_output_pin import (
    ModelExpectedOutputPin,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    Sha256Digest,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_ledger_record import (
    ModelFirstEffectLedgerRecord,
)
from omnibase_infra.runtime.first_effect_ledger.model_verified_first_effect_grant_record import (
    ModelVerifiedFirstEffectGrantRecord,
)
from omnibase_infra.runtime.first_effect_ledger.protocol_signed_first_effect_grant_ingress import (
    ProtocolSignedFirstEffectGrantIngress,
)


class FirstEffectPublishCompositionError(RuntimeError):
    """Raised when a publisher lacks a transactional first-effect outbox."""


class FirstEffectGrantIngressRejectedError(RuntimeError):
    """Raised when fixed-anchor verification or deployment output pins reject a wire."""


PoolFactory = Callable[[], Awaitable[asyncpg.Pool]]
_TABLE = "public.first_effect_verified_grant_ledger"
_RETURNING = """authorization_digest, grant_id, grant_envelope_id, nonce_digest,
request_digest, correlation_id, tenant_id, backend_id, rendered_contract_sha256,
issuer_key_fingerprint_sha256, retry_disposition, expected_output_topic,
expected_output_event_class, expected_output_event_index, state,
outbox_envelope_id, outbox_body_sha256, outbox_topic, outbox_event_class,
outbox_event_index, workflow_version, verified_at, staged_at, claimed_at,
publishing_at, published_unknown_at, terminal_at, updated_at, version"""


class FixedRsdVerifiedFirstEffectGrantIngress:
    """Private verifier path; callers only receive its raw-wire protocol."""

    def __init__(
        self,
        *,
        pool_factory: PoolFactory,
        expected_output_pin: ModelExpectedOutputPin,
        owns_pool: bool,
    ) -> None:
        self._pool_factory = pool_factory
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()
        self._expected_output_pin = expected_output_pin
        self._owns_pool = owns_pool

    async def verify_and_record(
        self,
        wire: object,
        *,
        now: datetime,
    ) -> ModelVerifiedFirstEffectGrantRecord:
        try:
            if isinstance(wire, (str, bytes, bytearray)):
                grant = verify_signed_executable_grant_v2(wire, now=now)
            else:
                raise FirstEffectGrantIngressRejectedError(
                    "RSD signed executable grant wire has an unsupported type"
                )
        except ExecutableGrantVerificationError as exc:
            raise FirstEffectGrantIngressRejectedError(
                "RSD signed executable grant was rejected"
            ) from exc
        material = grant.authorization_material.grant
        actual_pin = (
            material.expected_output_topic,
            material.expected_output_event_class,
            material.expected_output_event_index,
        )
        configured_pin = (
            self._expected_output_pin.topic,
            self._expected_output_pin.event_class,
            self._expected_output_pin.index,
        )
        if actual_pin != configured_pin:
            raise FirstEffectGrantIngressRejectedError(
                "RSD signed grant output pin contradicts deployment output pin"
            )
        authorization = grant.authorization_material
        sql = f"""INSERT INTO {_TABLE} (
            authorization_digest, grant_id, grant_envelope_id, nonce_digest,
            request_digest, correlation_id, tenant_id, backend_id,
            rendered_contract_sha256, issuer_key_fingerprint_sha256,
            retry_disposition, expected_output_topic, expected_output_event_class,
            expected_output_event_index, state
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, 'VERIFIED')
        ON CONFLICT DO NOTHING RETURNING {_RETURNING}"""
        material = authorization.grant
        try:
            if self._pool is None:
                async with self._pool_lock:
                    if self._pool is None:
                        self._pool = await self._pool_factory()
            async with self._pool.acquire() as connection:
                row = await connection.fetchrow(
                    sql,
                    grant.authorization_digest,
                    material.grant_id,
                    material.envelope_id,
                    material.nonce_sha256,
                    material.pins.request_sha256,
                    str(material.correlation_id),
                    material.tenant_id,
                    material.backend_id,
                    material.pins.rendered_contract_sha256,
                    authorization.issuer_key_fingerprint_sha256,
                    material.posture.retry_disposition,
                    material.expected_output_topic,
                    material.expected_output_event_class,
                    material.expected_output_event_index,
                )
        except asyncpg.PostgresError as exc:
            raise RepositoryExecutionError(
                f"verified first-effect grant insert failed: {type(exc).__name__}",
                op_name="verify_and_record",
                table=_TABLE,
            ) from exc
        if row is None:
            raise FirstEffectGrantIngressRejectedError(
                "verified grant identity is already bound"
            )
        return ModelVerifiedFirstEffectGrantRecord.model_validate(dict(row))

    async def close(self) -> None:
        """Close only a pool explicitly owned by this composition instance."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
        self._pool = None


@runtime_checkable
class ProtocolTransactionalFirstEffectOutbox(Protocol):
    """External transactional-outbox contract; this package does not publish.

    ``stage_first_effect`` must atomically bind a deterministic canonical-outbox
    intent to the ledger transition. The intent's uniqueness and linkage are
    owned by that outbox implementation, not by the legacy observation ledger.
    """

    async def stage_first_effect(
        self,
        *,
        authorization_digest: Sha256Digest,
        expected_version: int,
    ) -> ModelFirstEffectLedgerRecord: ...


def require_transactional_first_effect_outbox(
    candidate: object,
) -> ProtocolTransactionalFirstEffectOutbox:
    """Reject a recording adapter where external transactional composition is required."""
    if not isinstance(candidate, ProtocolTransactionalFirstEffectOutbox):
        raise FirstEffectPublishCompositionError(
            "first-effect publishing requires an injected transactional canonical-outbox "
            "stager; PostgresFirstEffectLedger is observation-only"
        )
    return candidate


def build_rsd_verified_first_effect_grant_ingress(
    *,
    pool_factory: PoolFactory,
    expected_output_pin: ModelExpectedOutputPin,
    owns_pool: bool = False,
) -> ProtocolSignedFirstEffectGrantIngress:
    """Construct the only verifier ingress from trusted composition.

    ``expected_output_pin`` is immutable deployment configuration supplied by a
    canonical overlay, never by an individual request.  It is matched against
    the verified signed grant before the payload-free projection is recorded.
    The raw recorder remains internal to this construction boundary.
    """

    return FixedRsdVerifiedFirstEffectGrantIngress(
        pool_factory=pool_factory,
        expected_output_pin=expected_output_pin,
        owns_pool=owns_pool,
    )


__all__ = [
    "FirstEffectPublishCompositionError",
    "FirstEffectGrantIngressRejectedError",
    "ProtocolTransactionalFirstEffectOutbox",
    "build_rsd_verified_first_effect_grant_ingress",
    "require_transactional_first_effect_outbox",
]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read recorded envelopes and write derived chain verdicts (OMN-16964)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import yaml

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.chain_replay import (
    assemble_replay_and_verify,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
    ModelDeclaredChainHop,
    ModelDelegationTerminalPayload,
    ModelLedgerChainRow,
    ModelObservedHop,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

HANDLER_ID_DELEGATION_CHAIN_LEDGER = "delegation-chain-ledger-handler"
_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "contract.yaml"

_SQL_READ_OBSERVED = """
SELECT
    topic,
    envelope_id::text AS envelope_id,
    onex_headers ->> 'parent_message_id' AS parent_envelope_id,
    correlation_id::text AS correlation_id
FROM public.event_ledger
WHERE correlation_id = $1::uuid
  AND topic = ANY($2::text[])
ORDER BY COALESCE(event_timestamp, ledger_written_at), partition, kafka_offset
"""

_SQL_UPSERT_ROW = """
INSERT INTO public.ledger_chain (
    correlation_id,
    hop_index,
    hop,
    replay_green,
    verifier_verdict,
    observed_topic,
    envelope_id,
    parent_envelope_id,
    replay_detail,
    verifier_detail,
    recorded_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
ON CONFLICT (correlation_id, hop_index) DO UPDATE SET
    hop = EXCLUDED.hop,
    replay_green = EXCLUDED.replay_green,
    verifier_verdict = EXCLUDED.verifier_verdict,
    observed_topic = EXCLUDED.observed_topic,
    envelope_id = EXCLUDED.envelope_id,
    parent_envelope_id = EXCLUDED.parent_envelope_id,
    replay_detail = EXCLUDED.replay_detail,
    verifier_detail = EXCLUDED.verifier_detail,
    recorded_at = NOW()
"""


def _parse_declared_topology(
    topology_raw: object,
) -> tuple[ModelDeclaredChainHop, ...]:
    """Turn the contract's ``chain_topology`` into declared hops, or refuse.

    OMN-18419. Every refusal here is a contract that cannot be graded against,
    and a topology that cannot be graded against must fail LOUDLY at construction
    rather than quietly produce a chain the replay rates on a relation it
    invented. The four refusals are the four ways a parent relation can be
    unresolvable: a hop named twice (which parent does an edge close against?),
    a parent naming a topic that is not a declared hop, no head at all (every
    hop caused by another is a cycle), and more than one head (two chains, not
    one).
    """
    if not isinstance(topology_raw, list) or not topology_raw:
        raise RuntimeError("chain-writer contract must declare a non-empty topology")

    hops: list[ModelDeclaredChainHop] = []
    for entry in topology_raw:
        if not isinstance(entry, Mapping):
            raise RuntimeError(
                "chain_topology entries must be mappings carrying `topic` and "
                f"`parent`; got {entry!r}. The bare-topic list form cannot "
                "express a branch, which is the OMN-18419 defect."
            )
        try:
            hops.append(ModelDeclaredChainHop.model_validate(dict(entry)))
        except Exception as exc:
            raise RuntimeError(
                f"invalid chain_topology entry {entry!r}: {exc}"
            ) from exc

    topics = [hop.topic for hop in hops]
    duplicates = sorted({topic for topic in topics if topics.count(topic) > 1})
    if duplicates:
        raise RuntimeError(
            f"chain_topology declares {duplicates!r} more than once; a parent "
            "relation resolved by topic cannot say which occurrence it means"
        )

    declared = set(topics)
    dangling = sorted(
        {
            hop.parent
            for hop in hops
            if hop.parent is not None and hop.parent not in declared
        }
    )
    if dangling:
        raise RuntimeError(
            f"chain_topology names parent topic(s) {dangling!r} that are not "
            "declared hops; an edge to an undeclared hop can never close"
        )

    heads = [hop.topic for hop in hops if hop.parent is None]
    if not heads:
        raise RuntimeError(
            "chain_topology declares no head (every hop names a parent), which "
            "is a cycle rather than a chain"
        )
    if len(heads) > 1:
        raise RuntimeError(
            f"chain_topology declares {len(heads)} heads {heads!r}; one "
            "correlation's chain has exactly one head"
        )
    return tuple(hops)


def _load_contract_settings() -> tuple[tuple[ModelDeclaredChainHop, ...], int, float]:
    """Load the declared topology and bounded settle policy from the contract."""
    try:
        raw = yaml.safe_load(  # yaml-safe-load-ok: trusted package contract
            _CONTRACT_PATH.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read chain-writer contract: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RuntimeError("chain-writer contract root must be a mapping")

    topology = _parse_declared_topology(raw.get("chain_topology"))

    writer_raw = raw.get("writer")
    writer = writer_raw if isinstance(writer_raw, Mapping) else {}
    attempts_raw = writer.get("settle_attempts", 1)
    delay_raw = writer.get("settle_delay_ms", 0)
    if not isinstance(attempts_raw, int) or attempts_raw < 1:
        raise RuntimeError("writer.settle_attempts must be a positive integer")
    if not isinstance(delay_raw, int) or delay_raw < 0:
        raise RuntimeError("writer.settle_delay_ms must be a non-negative integer")
    return topology, attempts_raw, delay_raw / 1000


class HandlerDelegationChainLedger:
    """Definition-B handler called by the delegation completion topic."""

    def __init__(
        self,
        container: ModelONEXContainer,
        db_dsn: str | None = None,
        *,
        declared_chain: Sequence[ModelDeclaredChainHop] | None = None,
        settle_attempts: int | None = None,
        settle_delay_seconds: float | None = None,
    ) -> None:
        contract_chain, contract_attempts, contract_delay = _load_contract_settings()
        self._db_handler = HandlerDb(container)
        self._db_dsn = db_dsn.strip() if db_dsn else ""
        self._declared_chain = (
            contract_chain if declared_chain is None else tuple(declared_chain)
        )
        self._settle_attempts = settle_attempts or contract_attempts
        self._settle_delay_seconds = (
            contract_delay
            if settle_delay_seconds is None
            else max(0.0, settle_delay_seconds)
        )
        self._initialized = False
        self._db_init_lock = asyncio.Lock()

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        config_dsn = config.get("dsn")
        if isinstance(config_dsn, str) and config_dsn.strip():
            self._db_dsn = config_dsn.strip()
        await self._ensure_db_ready()

    async def shutdown(self) -> None:
        if self._initialized:
            await self._db_handler.shutdown()
        self._initialized = False

    async def _ensure_db_ready(self) -> None:
        if self._initialized:
            return
        async with self._db_init_lock:
            if self._initialized:
                return
            if not self._db_dsn:
                raise self._runtime_error(
                    "Missing PostgreSQL DSN for delegation-chain persistence",
                    "delegation_chain.connect",
                )
            await self._db_handler.initialize({"dsn": self._db_dsn})
            self._initialized = True

    async def handle(
        self, request: ModelDelegationTerminalPayload
    ) -> ModelHandlerOutput[None]:
        """Assemble and persist one chain from the typed terminal payload.

        The write result is recorded on this node's own persistence path
        (``public.ledger_chain``, via ``_persist_rows``) and nothing consumes a
        published summary of it (OMN-16964 comment, 2026-09-15T11:52:22Z), so
        this returns an effect output with no events. Returning a typed ``.result`` here
        would have the auto-wiring boundary append it to ``output_events``
        with no ``publish_topics`` declared to resolve a result applier for
        it -- the exact OMN-18390 mechanism that dead-lettered every
        successful dispatch as ``UndeliverableDispatchOutputError``.
        """
        correlation_id = self._required_uuid(
            request.correlation_id, "terminal payload correlation_id"
        )
        input_envelope_id = uuid4()
        await self._ensure_db_ready()

        observed: tuple[ModelObservedHop, ...] = ()
        declared_topics = self._declared_topics()
        for attempt in range(self._settle_attempts):
            observed = await self._read_observed(correlation_id)
            observed_topics = {hop.topic for hop in observed}
            if all(topic in observed_topics for topic in declared_topics):
                break
            if attempt + 1 < self._settle_attempts and self._settle_delay_seconds:
                await asyncio.sleep(self._settle_delay_seconds)

        rows = assemble_replay_and_verify(
            correlation_id, observed, self._declared_chain
        )
        if not rows:
            # OMN-18398: `_persist_rows` is a loop over `rows`, so an empty row
            # set writes NOTHING and the effect output below would report that
            # as success -- indistinguishable from a complete write. On the
            # .201 compose dev lane that was every dispatch: `public.ledger_chain`
            # held zero rows while the handler reported `status=success`. A
            # write that cannot happen is a typed failure, never a success.
            raise self._runtime_error(
                "no delegation-chain evidence to persist: public.event_ledger "
                f"carries no row for correlation {correlation_id} on any "
                f"declared chain topic {list(self._declared_topics())!r} after "
                f"{self._settle_attempts} settle attempt(s); no ledger_chain "
                "row can be written, and reporting success here would be "
                "indistinguishable from a complete write",
                "delegation_chain.write",
                correlation_id,
            )
        await self._persist_rows(rows)

        return ModelHandlerOutput.for_effect(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_DELEGATION_CHAIN_LEDGER,
        )

    def _declared_topics(self) -> tuple[str, ...]:
        """The declared hop topics, in declaration order.

        The read filter and the settle predicate want topics; the replay wants
        the parent relation. Deriving the topic list here keeps the declaration
        a single object rather than two lists free to disagree.
        """
        return tuple(hop.topic for hop in self._declared_chain)

    async def _read_observed(
        self, correlation_id: UUID
    ) -> tuple[ModelObservedHop, ...]:
        response = await self._db_handler.execute(
            {
                "operation": "db.query",
                "payload": {
                    "sql": _SQL_READ_OBSERVED,
                    "parameters": [str(correlation_id), list(self._declared_topics())],
                },
                "correlation_id": str(correlation_id),
            }
        )
        if response.result is None:
            raise self._runtime_error(
                "event_ledger query returned no result",
                "delegation_chain.read",
                correlation_id,
            )
        return tuple(
            ModelObservedHop(
                topic=self._required_text(row, "topic"),
                envelope_id=self._required_uuid(
                    row.get("envelope_id"), "event_ledger.envelope_id"
                ),
                parent_envelope_id=self._optional_uuid(row.get("parent_envelope_id")),
                correlation_id=self._optional_uuid(row.get("correlation_id")),
            )
            for row in response.result.payload.rows
        )

    async def _persist_rows(self, rows: Sequence[ModelLedgerChainRow]) -> None:
        for row in rows:
            response = await self._db_handler.execute(
                {
                    "operation": "db.execute",
                    "payload": {
                        "sql": _SQL_UPSERT_ROW,
                        "parameters": [
                            str(row.correlation_id),
                            row.hop_index,
                            row.hop,
                            row.replay_green,
                            row.verifier_verdict.value,
                            row.observed_topic,
                            str(row.envelope_id) if row.envelope_id else "",
                            (
                                str(row.parent_envelope_id)
                                if row.parent_envelope_id
                                else ""
                            ),
                            row.replay_detail,
                            row.verifier_detail,
                        ],
                    },
                    "correlation_id": str(row.correlation_id),
                }
            )
            if response.result is None:
                raise self._runtime_error(
                    f"ledger_chain upsert returned no result for hop {row.hop_index}",
                    "delegation_chain.write",
                    row.correlation_id,
                )
            # OMN-18398: the statement ran, but "ran" is not "wrote". The upsert
            # always touches exactly one row (INSERT, or the ON CONFLICT UPDATE),
            # so a zero count means the row this handler claims to have recorded
            # is not in the relation.
            if response.result.payload.row_count < 1:
                raise self._runtime_error(
                    f"ledger_chain upsert affected no row for hop {row.hop_index} "
                    f"({row.hop!r}); the statement executed but the row is not "
                    "in public.ledger_chain",
                    "delegation_chain.write",
                    row.correlation_id,
                )

    @staticmethod
    def _required_text(row: Mapping[str, object], key: str) -> str:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{key} is missing or unreadable")
        return value

    @staticmethod
    def _required_uuid(value: object, field: str) -> UUID:
        parsed = HandlerDelegationChainLedger._optional_uuid(value)
        if parsed is None:
            raise RuntimeError(f"{field} is missing or unreadable")
        return parsed

    @staticmethod
    def _optional_uuid(value: object) -> UUID | None:
        if value is None or value == "":
            return None
        if isinstance(value, UUID):
            return value
        try:
            return UUID(str(value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid UUID value: {value!r}") from exc

    @staticmethod
    def _runtime_error(
        message: str, operation: str, correlation_id: UUID | None = None
    ) -> RuntimeHostError:
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation=operation,
        )
        return RuntimeHostError(message, context=context)


__all__ = [
    "HANDLER_ID_DELEGATION_CHAIN_LEDGER",
    "HandlerDelegationChainLedger",
]

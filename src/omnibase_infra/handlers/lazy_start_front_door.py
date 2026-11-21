# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lazy-start front door: a dormant contract starts on its first request (OMN-21592).

Plan task A4, section 3.11.5 and the lazy-start lifecycle of 3.11.9. A contract
outside the lane's boot set is dormant: nothing of it is imported, registered,
provisioned or attached at boot. The first request that names it arrives here;
the front door runs the start transaction (build, then provision, confirm and
attach) and then delivers the held request to the node's own dispatcher.
Concurrent first requests share one start; a request that finds the node
serving builds nothing. A start that fails is rolled back (no dispatcher and no
subscription is left; a topic it created stays) and is retried within the
policy's retry budget, and a later request starts it again. A request not
answered inside the policy's budget is answered ``START_TIMEOUT`` and stays
held: it is delivered once the start completes, never dropped.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection

from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.models.lazy_start import (
    ModelLazyStartPolicy,
    ModelLazyStartRequest,
    ModelLazyStartResult,
)
from omnibase_infra.protocols.protocol_lazy_start_transaction import (
    ProtocolLazyStartTransaction,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

__all__: list[str] = ["LazyStartFrontDoor"]

logger = logging.getLogger(__name__)


class LazyStartFrontDoor:
    """Start a dormant contract on its first request, once, then deliver."""

    def __init__(
        self,
        transaction: ProtocolLazyStartTransaction,
        dormant_contracts: Collection[str],
        policy: ModelLazyStartPolicy,
    ) -> None:
        self._transaction = transaction
        self._dormant = frozenset(dormant_contracts)
        self._policy = policy
        self._serving: set[str] = set()
        self._starts: dict[str, asyncio.Task[None]] = {}
        self._held: set[asyncio.Task[ModelLazyStartResult]] = set()
        self.held_results: list[ModelLazyStartResult] = []

    @property
    def serving(self) -> frozenset[str]:
        """The contracts this front door has started."""
        return frozenset(self._serving)

    async def handle(self, request: ModelLazyStartRequest) -> ModelLazyStartResult:
        """Deliver ``request`` to its node, starting the node first when dormant."""
        name = request.target_contract
        if name not in self._dormant:
            # A5 owns the full refusal set (NOT_ADMITTED, HASH_DRIFT); nothing is built.
            return _result(request, EnumLazyStartOutcome.CONTRACT_NOT_ANNOUNCED)
        if name in self._serving:
            return await self._deliver(request)
        start = self._start_of(name)
        try:
            await asyncio.wait_for(
                asyncio.shield(start), timeout=self._policy.budget_seconds
            )
        except TimeoutError:
            held = asyncio.create_task(self._deliver_when_started(request, start))
            self._held.add(held)
            held.add_done_callback(self._held.discard)
            return _result(
                request,
                EnumLazyStartOutcome.START_TIMEOUT,
                held=True,
                detail=f"start exceeded {self._policy.budget_seconds}s; request held",
            )
        except Exception as exc:
            logger.exception(
                "Lazy start: request for '%s' answered START_FAILED (OMN-21592)",
                name,
            )
            return _result(
                request,
                EnumLazyStartOutcome.START_FAILED,
                detail=sanitize_error_message(exc),
            )
        return await self._deliver(request)

    def _start_of(self, name: str) -> asyncio.Task[None]:
        """The one in-flight start for ``name``; a failed start is replaced."""
        start = self._starts.get(name)
        if start is None or (start.done() and start.exception() is not None):
            start = asyncio.create_task(self._start(name), name=f"lazy-start-{name}")
            self._starts[name] = start
        return start

    async def _start(self, name: str) -> None:
        attempts = 1 + self._policy.max_retries
        for attempt in range(1, attempts + 1):
            try:
                await self._transaction.build(name)
                await self._transaction.attach(name)
            except Exception:
                await self._transaction.rollback(name)
                logger.exception(
                    "Lazy start: contract '%s' did not start (attempt %d of %d) "
                    "(OMN-21592)",
                    name,
                    attempt,
                    attempts,
                )
                if attempt == attempts:
                    raise
                continue
            self._serving.add(name)
            logger.info(
                "Lazy start: contract '%s' started on its first request (attempt %d) "
                "(OMN-21592)",
                name,
                attempt,
            )
            return

    async def _deliver_when_started(
        self, request: ModelLazyStartRequest, start: asyncio.Task[None]
    ) -> ModelLazyStartResult:
        try:
            await start
        except Exception as exc:
            logger.exception(
                "Lazy start: held request for '%s' answered START_FAILED (OMN-21592)",
                request.target_contract,
            )
            result = _result(
                request,
                EnumLazyStartOutcome.START_FAILED,
                held=True,
                detail=sanitize_error_message(exc),
            )
        else:
            result = await self._deliver(request)
        self.held_results.append(result)
        return result

    async def _deliver(self, request: ModelLazyStartRequest) -> ModelLazyStartResult:
        answer = await self._transaction.deliver(
            request.target_contract, request.topic, request.envelope
        )
        return _result(request, EnumLazyStartOutcome.DELIVERED, answer=answer)

    async def drain_held(self) -> tuple[ModelLazyStartResult, ...]:
        """Wait for every held request to be answered (shutdown and tests)."""
        if self._held:
            await asyncio.gather(*tuple(self._held), return_exceptions=True)
        return tuple(self.held_results)


def _result(
    request: ModelLazyStartRequest,
    outcome: EnumLazyStartOutcome,
    *,
    held: bool = False,
    answer: object = None,
    detail: str = "",
) -> ModelLazyStartResult:
    return ModelLazyStartResult(
        target_contract=request.target_contract,
        correlation_id=request.correlation_id,
        outcome=outcome,
        held=held,
        answer=answer,
        detail=detail,
    )

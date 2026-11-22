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

A first request wakes the whole closure of the contract it names (OMN-26396): that
contract plus every dormant contract it publishes to, transitively, in one stable
order (by name). Each contract has one start in flight at a time, shared by every
wake that needs it, so two workflows that share a contract start it once and the
second waits on the first. A contract is committed as serving only when the wake
that needed it completes; a wake that fails rolls back every contract it started
that no live wake still holds and that no completed wake committed, and keeps what
another workflow already serves. A topic a failed wake created is never deleted.
A bus message addressed to a dormant consumer wakes it through ``wake_for_topic``
before the message is published. A wake that fails there fails that contract only
(OMN-21753): the failure is logged with the contract's name and kept in
``wake_failures``, the wake is rolled back as for any start, and the publish
goes on, so a publisher that happens to write to a dormant consumer's topic (the
runtime's own heartbeat among them) is never failed by that consumer's start.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.handlers.lazy_start_closure import LazyStartClosureGraph
from omnibase_infra.models.lazy_start import (
    ModelLazyStartPolicy,
    ModelLazyStartRequest,
    ModelLazyStartResult,
)
from omnibase_infra.protocols.protocol_lazy_start_transaction import (
    ProtocolLazyStartTransaction,
)
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

__all__: list[str] = ["LazyStartFrontDoor"]

logger = logging.getLogger(__name__)


class LazyStartFrontDoor:
    """Start a dormant contract's closure on its first request, once, then deliver."""

    def __init__(
        self,
        transaction: ProtocolLazyStartTransaction,
        dormant_contracts: Mapping[str, ModelDiscoveredContract],
        policy: ModelLazyStartPolicy,
        wired_contracts: Mapping[str, ModelDiscoveredContract] | None = None,
    ) -> None:
        self._transaction = transaction
        self._dormant = frozenset(dormant_contracts)
        self._graph = LazyStartClosureGraph(
            {**(wired_contracts or {}), **dormant_contracts}
        )
        self._policy = policy
        self._serving: set[str] = set()
        self._starts: dict[str, asyncio.Task[None]] = {}
        self._wakes: dict[str, asyncio.Task[None]] = {}
        self._holders: dict[str, set[str]] = {}
        self._rollbacks: dict[str, asyncio.Task[None]] = {}
        self._held: set[asyncio.Task[ModelLazyStartResult]] = set()
        self.held_results: list[ModelLazyStartResult] = []
        self.wake_failures: dict[str, str] = {}
        """Contract name to why its wake for a published message failed (OMN-21753)."""

    @property
    def serving(self) -> frozenset[str]:
        """The contracts whose wake completed: started and committed."""
        return frozenset(self._serving)

    async def handle(self, request: ModelLazyStartRequest) -> ModelLazyStartResult:
        """Deliver ``request`` to its node, starting the node first when dormant."""
        name = request.target_contract
        if name not in self._dormant:
            # A5 owns the full refusal set (NOT_ADMITTED, HASH_DRIFT); nothing is built.
            return _result(request, EnumLazyStartOutcome.CONTRACT_NOT_ANNOUNCED)
        if name in self._serving:
            return await self._deliver(request)
        wake = self._wake_of(name)
        try:
            await asyncio.wait_for(
                asyncio.shield(wake), timeout=self._policy.budget_seconds
            )
        except TimeoutError:
            held = asyncio.create_task(self._deliver_when_started(request, wake))
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

    async def wake_for_topic(self, topic: str) -> None:
        """Wake every dormant consumer of ``topic``, before a message is published to it.

        Returns at once when no dormant consumer is left. Waits on the wake in
        flight when there is one (the message is held, never published into a
        topic whose consumer is still starting). A wake that fails, or does not
        finish inside the policy's budget, fails that contract only (OMN-21753):
        it is logged with the contract's name and kept in ``wake_failures``, and
        this returns so the publish goes on. The start itself has already rolled
        the contract back; a later message wakes it again.
        """
        for name in self._graph.consumers_of(topic):
            if name in self._dormant and name not in self._serving:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._wake_of(name)),
                        timeout=self._policy.budget_seconds,
                    )
                except Exception as exc:
                    detail = sanitize_error_message(exc)
                    self.wake_failures[name] = detail
                    logger.exception(
                        "Lazy start: contract '%s' did not wake for a message to "
                        "'%s'; the publish goes on and only this contract is "
                        "affected (OMN-21753)",
                        name,
                        topic,
                    )

    def _wake_of(self, name: str) -> asyncio.Task[None]:
        """The one in-flight wake of ``name``'s closure; a failed wake is replaced."""
        wake = self._wakes.get(name)
        if wake is None or _failed(wake):
            wake = asyncio.create_task(self._wake(name), name=f"lazy-wake-{name}")
            self._wakes[name] = wake
        return wake

    async def _wake(self, target: str) -> None:
        held: list[str] = []
        try:
            for member in self._graph.closure_of(target):
                if member not in self._dormant or member in self._serving:
                    continue
                start = self._start_of(member)
                self._holders.setdefault(member, set()).add(target)
                held.append(member)
                await asyncio.shield(start)
        except BaseException:
            await self._abandon(target, held)
            raise
        self._serving.update(held)
        for member in held:
            self._holders[member].discard(target)
            self.wake_failures.pop(member, None)
        logger.info(
            "Lazy start: closure of '%s' woke %d contract(s) (OMN-26396)",
            target,
            len(held),
        )

    async def _abandon(self, target: str, held: list[str]) -> None:
        """Roll back what this wake started and nothing else needs; keep the rest."""
        doomed: list[str] = []
        for member in reversed(held):
            holders = self._holders[member]
            holders.discard(target)
            start = self._starts.get(member)
            if (
                not holders
                and member not in self._serving
                and start is not None
                and start.done()
                and not _failed(start)
            ):
                doomed.append(member)
        for member in doomed:
            del self._starts[member]
            self._rollbacks[member] = asyncio.create_task(self._roll_back(member))
        if doomed:
            await asyncio.gather(*(self._rollbacks[m] for m in doomed))

    async def _roll_back(self, name: str) -> None:
        try:
            await self._transaction.rollback(name)
        except Exception:
            logger.exception(
                "Lazy start: rollback of '%s' after a failed closure wake raised "
                "(OMN-26396)",
                name,
            )
        finally:
            del self._rollbacks[name]

    def _start_of(self, name: str) -> asyncio.Task[None]:
        """The one in-flight start for ``name``; a failed start is replaced."""
        start = self._starts.get(name)
        if start is None or _failed(start):
            start = asyncio.create_task(self._start(name), name=f"lazy-start-{name}")
            self._starts[name] = start
        return start

    async def _start(self, name: str) -> None:
        rollback = self._rollbacks.get(name)
        if rollback is not None:
            await rollback
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


def _failed(task: asyncio.Task[None]) -> bool:
    return task.done() and (task.cancelled() or task.exception() is not None)


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

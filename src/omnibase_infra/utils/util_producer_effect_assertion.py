# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed effect assertions for artifact-producing jobs (RT-5).

A *producer* is any job whose purpose is to emit an artifact — publish an event,
push a tag, open a downstream PR, write a receipt. The recurring silent-failure
disease (``docs/plans/2026-07-12-mechanical-release-trains.md`` §1) is a producer
that "ran successfully" while producing *nothing* and exiting green: the deploy
trigger that prints ``"KAFKA_BOOTSTRAP_SERVERS is not set -- skipping publish"``
then ``sys.exit(0)`` even though a real runtime change merged.

The invariant, stated once and shared by every producer (deploy trigger, publish
step, pin cascade, OCC publisher):

    "Ran successfully" is NOT a completion signal for a producer.
    "Produced N>0, and here it is" is.

Emitting zero — whether because a required precondition is missing or because the
emit itself delivered nothing — must FAIL CLOSED (non-zero, red), never skip
green.

This module is pure: no I/O, no environment reads, no topic literals. Callers
resolve their own preconditions and emit counts, then translate a
:class:`ProducerZeroOutputError` into a non-zero process exit at their boundary.

Ticket: OMN-14467 (RT-5); epic OMN-13674.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "ProducerZeroOutputError",
    "assert_producer_emitted",
    "require_producer_preconditions",
]


class ProducerZeroOutputError(RuntimeError):
    """A producer expected to emit >=1 artifact emitted zero.

    Raised by :func:`require_producer_preconditions` (a precondition for emitting
    is missing, so the producer cannot emit) and by :func:`assert_producer_emitted`
    (the emit completed but delivered nothing). Callers convert this into a
    non-zero process exit — a producer that emits nothing must go RED.
    """


def require_producer_preconditions(
    *,
    artifact: str,
    preconditions: Mapping[str, object],
) -> None:
    """Fail closed when a required precondition for emitting ``artifact`` is absent.

    ``preconditions`` maps a human-readable name (e.g. an env var) to its resolved
    value. A value that is falsy (empty string, ``None``, ``0``) means the
    producer cannot emit its artifact — that is zero output, not a reason to skip
    green.

    Raises:
        ProducerZeroOutputError: if any precondition value is falsy. The message
            names every missing precondition and the artifact that would not be
            produced.
    """
    missing = [name for name, value in preconditions.items() if not value]
    if missing:
        raise ProducerZeroOutputError(
            f"producer for {artifact!r} cannot emit: missing required "
            f"precondition(s) {', '.join(missing)}. A producer that emits nothing "
            f"must fail closed (RT-5), not skip green."
        )


def assert_producer_emitted(
    produced_count: int,
    *,
    artifact: str,
    detail: str = "",
) -> None:
    """Fail closed when a producer emitted fewer than one artifact.

    Call this *after* the emit with the number of artifacts actually delivered
    (messages published, tags pushed, PRs opened, receipts written). ``0`` (or a
    negative count) is the silent-producer failure and must go RED.

    Raises:
        ProducerZeroOutputError: if ``produced_count < 1``.
    """
    if produced_count < 1:
        suffix = f" ({detail})" if detail else ""
        raise ProducerZeroOutputError(
            f"producer for {artifact!r} emitted {produced_count} artifact(s); "
            f"expected at least 1{suffix}. 'Ran successfully' is not completion "
            f"for a producer — producing zero must fail closed (RT-5)."
        )

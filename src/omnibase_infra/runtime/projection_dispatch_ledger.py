# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which projections this process subscribes but deliberately does not dispatch.

OMN-17448. There is a third way for a projection to persist nothing, and until
this module existed no health surface could see it.

The mask
--------
``handler_wiring._make_projection_dispatch_callback`` opens its callback with::

    if is_projection_runner:
        logger.debug("Projection runner skipped by DB-injection auto-wiring: ...")
        return None

That skip is deliberate and correct (OMN-15905): a handler with the standalone
runner shape owns its own consume loop and its own DB pool, so the shared kernel
must not dispatch it in-process. The sanctioned way to run it for real is a
dedicated writer process — a k8s Deployment or a compose service — outside the
kernel.

The defect is what happens when that dedicated writer does not exist. The
kernel still *subscribes* the contract's topics, so the Kafka consumer joins,
takes every message, and commits every offset. The callback returns ``None``
before any handler runs. Net: **consume, ack, write nothing, no DLQ record, no
terminal event, no ERROR log.** Silent by construction.

Both pre-existing liveness masks read green through it, by construction rather
than by accident:

* ``unattached_projections`` — the topic *is* attached. The subscription is
  real; only the dispatch is a no-op. Measured live on the ``.201`` dev lane
  2026-09-01: ``service_runtime_health_monitor`` logged ``status=HEALTHY ...
  projections=13 unattached_projections=0`` while ``node_projection_tenant_
  registry`` had no writer on ANY lane and ``tenant_registry_mirror`` held 0
  rows.
* ``dlq_saturated_projections`` — nothing is routed to a DLQ, because nothing
  raises. ``messages_dlq`` stays 0 and the ratio never reaches 1.0.

What this module claims, and what it does not
---------------------------------------------
It records a **process-local fact**: "this process subscribed contract X and
its dispatch here is a no-op." That fact is true and cheaply knowable at wiring
time — the wiring seam already computes it to decide the branch.

It deliberately does NOT claim "X has no writer anywhere". A kernel process
cannot see a sibling Deployment in another pod or another compose service, and
inventing that claim would report a permanent false outage on every lane where
the standalone writer is correctly deployed. The corpus-level "every subscribing
lane has a deployed writer" assertion is a static gate over the deployment
manifests, not a runtime health dimension (OMN-17448 AC5).

So the health dimension this feeds says what is observable and stops: the
kernel persists nothing for these projections, and their rows depend entirely
on a writer this process cannot see. That is strictly more than the previous
surface, which said ``HEALTHY`` and named nothing.

Keyed by (contract, handler entry), not by contract
--------------------------------------------------
OMN-17562. Both recording branches run once per HANDLER ENTRY, but the original
ledger stored bare contract names. A contract with a standalone-runner entry
AND a live in-process entry therefore landed in the ledger even though it
dispatched and wrote rows on every message.

That is not a rounding error. Live on both ``.201`` compose lanes 2026-09-02
the main runtime profile reported ``projection_count=37 nonwriting=15`` where
only 9 genuinely dispatched nothing: ``node_projection_receipt_gate``,
``projection_baselines``, ``projection_intent_classification``,
``projection_session_outcome``, ``projection_pattern_learning`` and
``projection_routing_decision`` all carry a live sibling entry. Six of fifteen
names on the health detail were false, which is how an operator learns to
ignore a detail.

So both branches are recorded — the skipped one and the live one — and the
contract-level question "does anything dispatch this here?" is answered by
:func:`projections_with_no_live_dispatcher` from the two together. Absence of a
skip row was never evidence of a live dispatcher: a contract this process never
wired at all is equally absent.

What this is NOT used for
-------------------------
The kernel's decision to withhold a Kafka subscription (OMN-17562) is taken
from a typed per-entry field carried on ``PreparedWiring`` and
``ModelContractWiringResult``, NOT from this ledger. The ledger is
process-global and describes every wiring pass that has run in this process;
the subscribe decision is about one contract in one pass. Reading it at the
subscribe seam would couple two facts that can legitimately differ.

Shape
-----
Two module-level sets plus six functions. No class, no lifecycle, no
``Plugin*``. Process-local by design: the ledger describes THIS process's
wiring, so it must not be shared, persisted, or read across a process boundary.

Related Tickets:
    - OMN-17448: this module — the third mask
    - OMN-17562: per-entry keying, and the kernel unsubscribing what it cannot dispatch
    - OMN-15905: why the skip exists, and the standalone-writer fix pattern
    - OMN-16994: the two masks this one sits alongside
    - OMN-16874: ``_is_standalone_projection_runner``, the predicate recorded here
"""

from __future__ import annotations

# (contract_name, handler_name) pairs whose dispatch is a no-op in this process.
_DISPATCH_SKIPPED_ENTRIES: set[tuple[str, str]] = set()

# (contract_name, handler_name) pairs this process really dispatches. Recorded
# so a contract's total absence from the skipped set is distinguishable from a
# contract that has a live sibling entry.
_DISPATCH_LIVE_ENTRIES: set[tuple[str, str]] = set()


def _entry_key(contract_name: str, handler_name: str) -> tuple[str, str] | None:
    """Return the ledger key, or None when either half is unnameable.

    Blank names are refused rather than stored under a placeholder: an entry no
    operator can look up is worse than an omission, and a placeholder would
    additionally collide with every other unnamed entry, silently merging
    distinct handler entries into one row.
    """
    contract = contract_name.strip()
    handler = handler_name.strip()
    if not contract or not handler:
        return None
    return (contract, handler)


def record_dispatch_skipped_projection(contract_name: str, handler_name: str) -> None:
    """Record that this process wires one handler entry with a no-op dispatch.

    Called from the wiring seam at the moment it takes the standalone-runner
    branch (OMN-15905) or the zero-route branch (OMN-17519), so the ledger
    cannot drift from the branch it describes.

    Args:
        contract_name: The contract owning the handler entry.
        handler_name: The handler class whose dispatch is a no-op here. Required
            because the same contract may also own an entry that IS dispatched.
    """
    key = _entry_key(contract_name, handler_name)
    if key is not None:
        _DISPATCH_SKIPPED_ENTRIES.add(key)


def record_live_projection_dispatch(contract_name: str, handler_name: str) -> None:
    """Record that this process really dispatches one projection handler entry.

    The counterpart of :func:`record_dispatch_skipped_projection`, written on
    the branch that builds a live dispatch callback. Without it, "this contract
    has no skipped entry" and "this contract has a live entry" are the same
    observation, and a mixed contract cannot be told from a wholly non-writing
    one.
    """
    key = _entry_key(contract_name, handler_name)
    if key is not None:
        _DISPATCH_LIVE_ENTRIES.add(key)


def dispatch_skipped_entries() -> frozenset[tuple[str, str]]:
    """Return the (contract, handler) entries wired with a no-op dispatch."""
    return frozenset(_DISPATCH_SKIPPED_ENTRIES)


def dispatch_skipped_projections() -> frozenset[str]:
    """Return contracts with at least one no-op handler entry in this process."""
    return frozenset(contract for contract, _handler in _DISPATCH_SKIPPED_ENTRIES)


def projections_with_no_live_dispatcher() -> frozenset[str]:
    """Return contracts this process subscribes and dispatches NOTHING for.

    A contract qualifies only when it has at least one no-op entry and NO live
    entry. This is the honest form of the OMN-17448 count: a contract whose
    runner entry is skipped while its in-process sibling writes rows on every
    message is not a non-writing projection, and naming it as one is a false
    outage report.
    """
    live = {contract for contract, _handler in _DISPATCH_LIVE_ENTRIES}
    return frozenset(dispatch_skipped_projections() - live)


def reset_dispatch_skipped_projections() -> None:
    """Clear the ledger. For tests and for a re-wire within one process."""
    _DISPATCH_SKIPPED_ENTRIES.clear()
    _DISPATCH_LIVE_ENTRIES.clear()


__all__: list[str] = [
    "dispatch_skipped_entries",
    "dispatch_skipped_projections",
    "projections_with_no_live_dispatcher",
    "record_dispatch_skipped_projection",
    "record_live_projection_dispatch",
    "reset_dispatch_skipped_projections",
]

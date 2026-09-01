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

Shape
-----
A module-level set plus three functions. No class, no lifecycle, no
``Plugin*``. Process-local by design: the ledger describes THIS process's
wiring, so it must not be shared, persisted, or read across a process boundary.

Related Tickets:
    - OMN-17448: this module — the third mask
    - OMN-15905: why the skip exists, and the standalone-writer fix pattern
    - OMN-16994: the two masks this one sits alongside
    - OMN-16874: ``_is_standalone_projection_runner``, the predicate recorded here
"""

from __future__ import annotations

_DISPATCH_SKIPPED_PROJECTIONS: set[str] = set()


def record_dispatch_skipped_projection(contract_name: str) -> None:
    """Record that this process wires ``contract_name`` with a no-op dispatch.

    Called from the wiring seam at the moment it takes the standalone-runner
    branch, so the ledger cannot drift from the branch it describes.

    Args:
        contract_name: The contract whose projection dispatch is a no-op here.
            Blank names are ignored rather than stored: an unnamed entry cannot
            be rendered on a health detail an operator can look up, and a
            placeholder would be worse than an omission.
    """
    name = contract_name.strip()
    if name:
        _DISPATCH_SKIPPED_PROJECTIONS.add(name)


def dispatch_skipped_projections() -> frozenset[str]:
    """Return the projections this process subscribes but does not dispatch."""
    return frozenset(_DISPATCH_SKIPPED_PROJECTIONS)


def reset_dispatch_skipped_projections() -> None:
    """Clear the ledger. For tests and for a re-wire within one process."""
    _DISPATCH_SKIPPED_PROJECTIONS.clear()


__all__: list[str] = [
    "dispatch_skipped_projections",
    "record_dispatch_skipped_projection",
    "reset_dispatch_skipped_projections",
]

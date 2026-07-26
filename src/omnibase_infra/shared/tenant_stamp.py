# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical verified-tenant payload stamp (OMN-14367).

Single source of truth for how a verified / config-bound tenant identity is
stamped into an inbound payload before local republish or dispatch. Two
producers write ``payload["tenant_id"]`` and MUST route through this helper so
they cannot diverge on the shape again:

* the runtime auto-wiring ``tenant_scoped_ingress`` stamp
  (``handler_wiring._stamp_tenant_id_from_topic_prefix``), which derives the
  slug from a ``tenant-<slug>.`` wire prefix, and
* the gateway forwarder's ``handler_consume_inbound.consume_inbound``, which
  derives it from the config-bound ``ModelGatewayTenantIdentity``.

Canonical shape: ``payload["tenant_id"] = <DNS-safe slug>``, overwriting any
client-supplied value. There is NO separate ``tenant_slug`` key -- the consumer
``ModelDelegateSkillRequest`` (omnimarket) is ``extra="forbid"`` and documents
``tenant_id`` as the slug, not a UUID. The OMN-14208 cross-boundary seam test
``test_tenant_stamp_seam_omn14208`` pins this shape against the real consumer.

Why the divergence this closes (OMN-14367): the two producers each hand-rolled
the dict and drifted -- the auto-wiring stamp emitted the slug while the gateway
emitted ``str(identity.tenant_id)`` (a UUID) plus an extra ``tenant_slug`` key,
a silent seam mismatch the consumer contract rejects.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["stamp_verified_tenant_slug"]


def stamp_verified_tenant_slug(
    payload: Mapping[str, object], slug: str
) -> dict[str, object]:
    """Return a copy of ``payload`` with ``tenant_id`` set to the verified slug.

    The verified slug always wins: any client-supplied ``tenant_id`` is
    overwritten -- never merged-if-absent, never defaulted. Callers decide
    WHETHER to stamp (e.g. only when a trusted slug is actually present); this
    helper owns only the canonical resulting shape.

    Also strips any client-supplied ``tenant_slug`` key from the input. The
    canonical shape carries no separate ``tenant_slug`` key -- without this,
    a forged or stale ``tenant_slug`` in the raw payload would survive
    untouched (this helper only ever WRITES ``tenant_id``, so a pre-existing
    ``tenant_slug`` key was never being cleared), reaching a downstream
    ``extra="forbid"`` consumer or, worse, a laxer one that silently accepts
    it as a second, unverified tenant signal.
    """
    stamped = {**payload, "tenant_id": slug}
    stamped.pop("tenant_slug", None)
    return stamped

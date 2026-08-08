# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Gateway attach/session control-plane effect node (OMN-15750, G6).

Validates a per-tenant Keycloak client-credentials access token, registers a
tenant-bound attach session, and provides heartbeat re-validation (which is
also the revocation path — see ``services.service_keycloak_token_validator``)
and explicit detach. This is the control plane for the productized edge
ingress recommended in ``docs/design/2026-08-08-customer-cloud-connectivity-design.md``
(candidate A) and ``docs/design/2026-08-08-gateway-node-architecture-lift.md`` (G6).

Byte-forwarding stays in ``node_bus_forwarder_effect`` — this node owns only
attach/session/revocation, per that assessment's §2 ("what should NOT become
a node").
"""

from __future__ import annotations

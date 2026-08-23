# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Edge delegation worker (OMN forwarder, unreleased).

Standalone, outbound-only client that lets a LAN-hosted model (e.g. a local
Qwen server) serve delegation-inference traffic mirrored out of the cloud
control plane. It is NOT an ONEX contract node — it never runs inside
``omninode-runtime`` and is never wired into that runtime's handler
dispatch. It is a separate process an operator runs on the edge machine.

Wiring it targets (all pre-existing, none invented here):

- ``POST /v1/gateway/{attach,heartbeat,detach}`` -- the gateway control
  plane in ``omninode_infra/docker/onex-api/routers/gateway.py``, backed by
  ``omnibase_infra.nodes.node_gateway_attach_effect``.
- The mirrored local bus topics ``node_bus_forwarder_effect`` bridges
  between the edge and the cloud Kafka edge (contract-declared inbound:
  delegation-inference-request.v1 / delegation-request.v1; outbound:
  inference-response.v1 / delegation-completed.v1 / delegation-failed.v1 /
  llm-call-completed.v1).
- A local OpenAI-compatible chat-completions endpoint (the LAN model
  server), reached over outbound-only HTTP.

What is genuinely new here (did not exist anywhere before this module): the
local consumer that reads a mirrored delegation envelope, calls the LAN
model, and republishes a result envelope for the forwarder to mirror back
out. Everything else above already existed and is only *used*, not
reimplemented.

Known gaps this worker cannot itself close (see the mapping report this
package was built against):

- Per-tenant ``ga-*`` Keycloak client provisioning has no self-service path
  yet -- a credential must be manually provisioned before attach will work.
- The deployed ``omninode-runtime-effects`` image must be built from
  omnibase_infra commit ``52d8b601`` or later for the three
  ``gateway-*-request.v1`` Kafka subscriptions this control plane depends on
  to exist; an older image makes every ``/v1/gateway/*`` call time out.

This package is built and unit-tested only. It does not attach to any live
queue, does not flip any enabled flag, and is never invoked against a real
endpoint from this codebase or its tests -- see ``tests/unit/scripts/
edge_delegation_worker`` for the fake-server-backed test suite.
"""

from __future__ import annotations

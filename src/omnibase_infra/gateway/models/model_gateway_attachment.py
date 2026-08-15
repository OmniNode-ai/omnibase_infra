# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelGatewayAttachment -- everything one successful attach hands the client.

The client-side reading of ``ModelGatewayAttachResponse``
(``node_gateway_attach_effect`` 0.3.0) reduced to the three things a client
actually acts on: the session it now holds, how often to prove liveness, and
the renewal cycle it must run to survive its own ceiling. The response's
``session_event`` is deliberately not carried -- it is the node's thin-publish
payload for the bus, not instruction to the caller.

``session`` and ``renewal`` are the node's OWN models, imported rather than
mirrored. The client and the server are the same package here, so there is
exactly one definition of the attach contract's shape and no way for the two
sides to drift: a field added to ``ModelGatewaySession`` is immediately a field
this client parses, and a rename breaks the import rather than silently
producing a client that ignores what it was told.

``renewal`` is REQUIRED here, matching the node and NOT the edge. The edge
(``onex-api``) accepts it as optional purely so the two repos may deploy in
either order; a *client* that accepted its absence would run an unattended
runtime with no renewal policy at all, which is the exact gap OMN-15952 was
filed against. Absence is refused at parse time.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_renewal_directive import (
    ModelGatewayRenewalDirective,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)

__all__ = ["ModelGatewayAttachment"]


class ModelGatewayAttachment(BaseModel):
    """A live gateway session plus the terms for keeping it."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    session: ModelGatewaySession
    heartbeat_interval_seconds: int = Field(gt=0)
    renewal: ModelGatewayRenewalDirective

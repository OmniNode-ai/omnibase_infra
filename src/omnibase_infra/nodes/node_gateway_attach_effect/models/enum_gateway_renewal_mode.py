# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How an unattended runtime keeps working across attach-token expiry.

One member today, and the enum exists precisely so that the one member is
*named on the wire* rather than assumed. OMN-15952's design was revised
three times on this exact point, and the final answer is the
counter-intuitive one:

  ``RE_ATTACH`` -- the session's ``expires_at`` is stamped once, at attach,
  from ``min(token exp, max_session_ttl_seconds)``. Nothing ever moves it.
  A heartbeat proves the session is still alive and still backed by a
  non-revoked credential; it does not, and must not, buy the session more
  time. So a runtime that wants to keep working past ``expires_at`` performs
  a fresh ``client_credentials`` grant against Keycloak and then a fresh
  ``gateway.attach`` -- minting a NEW ``session_id``. Continuity across the
  boundary is the runtime's property (and the correlation trail's), never
  the session record's.

There is deliberately no ``RENEW_IN_PLACE`` member. It is not unimplemented;
it is refused. Extending a live session's ceiling on the strength of a
heartbeat would let a credential that Keycloak has already stopped
authorizing keep a session alive indefinitely, one heartbeat at a time --
which is the exact property the fixed ``expires_at`` exists to deny. If a
future lane believes it needs in-place renewal, that is a contract change
with its own security review, not a new enum member.
"""

from __future__ import annotations

from enum import Enum


class EnumGatewayRenewalMode(str, Enum):
    """The renewal mechanism this node's contract declares to its clients."""

    RE_ATTACH = "RE_ATTACH"


__all__ = ["EnumGatewayRenewalMode"]

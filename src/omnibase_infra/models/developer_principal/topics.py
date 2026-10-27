# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic names the developer principal plan's rules name (OMN-20821).

The one command topic a principal-specific grant addition may write, the lab
issuer's ``SCOPED_WRITE_TOPICS`` (OPERATOR-CONSENT 2026-10-25T23:01:04Z: h105's
``onex scm pr-create``). The parity test holds this set equal to the issuer's.
"""

from __future__ import annotations

SCOPED_WRITE_TOPICS: frozenset[str] = frozenset(
    {"onex.cmd.omnimarket.github-repo-write-requested.v1"}
)

__all__: list[str] = ["SCOPED_WRITE_TOPICS"]

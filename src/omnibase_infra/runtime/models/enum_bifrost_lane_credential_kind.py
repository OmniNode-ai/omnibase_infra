# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How a lane-declared Bifrost backend authenticates (OMN-17099)."""

from __future__ import annotations

from enum import StrEnum


class EnumBifrostLaneCredentialKind(StrEnum):
    """The two credential postures a lane-declared backend may state.

    There is no third, defaulted posture. A backend a lane ADDS must say which
    of these it is, so an unauthenticated lab server and a forgotten
    ``secret_ref`` can never be the same input.
    """

    #: The endpoint takes no credential (an unauthenticated lab server).
    NONE = "none"
    #: The endpoint authenticates with the named secret reference, resolved at
    #: the effect boundary exactly as a base-contract backend's ``secret_ref``.
    SECRET_REF = "secret_ref"


__all__ = ["EnumBifrostLaneCredentialKind"]

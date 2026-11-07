# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed refusals of a lane overlay's media capability declarations (OMN-22756).

One subclass per failure, so a caller and a test can tell the four apart without
reading the message. Each message names the backend (or, for an unknown
capability, the capability asked for) and the lane where it is known.
"""

from __future__ import annotations

from omnibase_infra.errors.error_infra import ProtocolConfigurationError


class BifrostLaneMediaCapabilityError(ProtocolConfigurationError):
    """Base of every media capability refusal on a lane overlay."""


class BifrostLaneMediaCapabilityShapeError(BifrostLaneMediaCapabilityError):
    """A binding's capability and its endpoint path disagree."""


class BifrostLaneMediaCapabilityAmbiguousError(BifrostLaneMediaCapabilityError):
    """Two serving backends on one lane declare the same capability."""


class BifrostLaneMediaCapabilityUnknownError(BifrostLaneMediaCapabilityError):
    """A capability was asked for that is not one of the typed capabilities."""


class BifrostLaneMediaCapabilityNotServingError(BifrostLaneMediaCapabilityError):
    """A capability was asked for and no serving backend on the lane declares it."""


__all__ = [
    "BifrostLaneMediaCapabilityAmbiguousError",
    "BifrostLaneMediaCapabilityError",
    "BifrostLaneMediaCapabilityNotServingError",
    "BifrostLaneMediaCapabilityShapeError",
    "BifrostLaneMediaCapabilityUnknownError",
]

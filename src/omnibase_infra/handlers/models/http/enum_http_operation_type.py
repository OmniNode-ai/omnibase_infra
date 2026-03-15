# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HTTP Operation Type Enum.

Defines the discriminator enum for HTTP operation types, used in the
discriminated union for HTTP handler responses.
"""

from __future__ import annotations

from enum import Enum


class EnumHttpOperationType(str, Enum):
    """HTTP operation type discriminator.

    Each value corresponds to a specific HTTP operation type and its
    associated payload model in the HttpPayload discriminated union.

    Attributes:
        GET: GET request operation (ModelHttpGetPayload)
        POST: POST request operation (ModelHttpPostPayload)
    """

    GET = "get"
    POST = "post"


__all__: list[str] = ["EnumHttpOperationType"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import StrEnum


class EnumProjectTrackerBackend(StrEnum):
    """Which ``ProtocolProjectTracker`` implementation a caller selects (OMN-20595).

    ``LINEAR`` is the default and fails loud without a credential. ``LOCAL_STUB``
    is the JSON-file stub, returned only when a caller names it.
    """

    LINEAR = "linear"
    LOCAL_STUB = "local_stub"


__all__: list[str] = [
    "EnumProjectTrackerBackend",
]

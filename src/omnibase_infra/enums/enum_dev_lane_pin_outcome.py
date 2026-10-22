# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How one dev lane pin read or set ended (OMN-18475)."""

from __future__ import annotations

from enum import StrEnum


class EnumDevLanePinOutcome(StrEnum):
    """The outcome of one read or set of a dev lane pin."""

    READ = "READ"
    """The pin was read; nothing was written."""

    PLANNED = "PLANNED"
    """A set without ``execute``: the change was validated and not written."""

    UNCHANGED = "UNCHANGED"
    """The pin already names the value; nothing was written."""

    WRITTEN = "WRITTEN"
    """The pin was written and its ledger row appended."""

    REFUSED = "REFUSED"
    """Nothing was written; the result's reason names why."""

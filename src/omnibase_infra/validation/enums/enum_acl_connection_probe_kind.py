# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""What one post-apply application-database connection probe asserts."""

from enum import StrEnum, unique


@unique
class EnumAclConnectionProbeKind(StrEnum):
    """Separate the three assertions a connection probe can make."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEGATIVE_PUBLIC = "negative_public"


__all__ = ["EnumAclConnectionProbeKind"]

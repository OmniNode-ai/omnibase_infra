# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Authorization boundary for generated application-database ACL matrices."""

from enum import StrEnum, unique


@unique
class EnumApplicationDatabaseAclAuthorizationScope(StrEnum):
    """Separate deployment-authorized evidence from synthetic proof fixtures."""

    DEPLOYMENT = "deployment"
    SYNTHETIC_PROOF = "synthetic_proof"


__all__ = ["EnumApplicationDatabaseAclAuthorizationScope"]

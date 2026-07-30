# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Evidence source kinds accepted by application-database ACL policy."""

from enum import StrEnum, unique


@unique
class EnumApplicationDatabaseAclPolicySourceKind(StrEnum):
    """Distinguish authoritative topology policy from synthetic proof input."""

    TOPOLOGY_CONTRACT = "topology_contract"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


__all__ = ["EnumApplicationDatabaseAclPolicySourceKind"]

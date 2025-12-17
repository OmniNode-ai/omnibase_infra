# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Enumerations Module.

Provides infrastructure-specific enumerations for transport types,
protocol identification, policy classification, and other infrastructure
concerns.

Exports:
    EnumInfraTransportType: Infrastructure transport type enumeration
    EnumPolicyType: Policy type enumeration for PolicyRegistry plugins
"""

from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.enums.enum_policy_type import EnumPolicyType

__all__ = ["EnumInfraTransportType", "EnumPolicyType"]

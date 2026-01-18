# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Enumerations Module.

Provides enumerations specific to the ONEX runtime scheduler and related components.

Exports:
    EnumConfigRefScheme: Supported configuration reference schemes
    EnumSchedulerStatus: Status of the runtime scheduler lifecycle
"""

from omnibase_infra.runtime.enums.enum_config_ref_scheme import EnumConfigRefScheme
from omnibase_infra.runtime.enums.enum_scheduler_status import EnumSchedulerStatus

__all__: list[str] = [
    "EnumConfigRefScheme",
    "EnumSchedulerStatus",
]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Named operator faults for delegation doctor checks."""

from enum import StrEnum


class EnumDelegationDoctorFault(StrEnum):
    """Delegation faults with a specific operator remediation."""

    NO_IDENTITY = "no_identity"
    NO_KEY = "no_key"
    WRONG_KEY = "wrong_key"
    GATEWAY_DOWN = "gateway_down"
    QUOTA_EXHAUSTED = "quota_exhausted"
    NO_LOCAL_MODEL = "no_local_model"
    LOCAL_MODEL_NOT_SERVING = "local_model_not_serving"
    LOCAL_MODEL_ID_MISMATCH = "local_model_id_mismatch"


__all__ = ["EnumDelegationDoctorFault"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed preflight dispositions for the inert RSD lane."""

from __future__ import annotations

from enum import StrEnum


class EnumRsdLiveDelegationPreflightFailure(StrEnum):
    EXECUTION_DISABLED = "execution_disabled"
    PUBLIC_RSD_REVISION_MISMATCH = "public_rsd_revision_mismatch"
    CAPABILITY_REFERENCE_MISSING = "capability_reference_missing"
    CAPABILITY_HEALTH_UNVERIFIED = "capability_health_unverified"
    RESULT_ATTESTOR_UNVERIFIED = "result_attestor_unverified"
    SEALED_AUTHORITY_UNVERIFIED = "sealed_authority_unverified"
    POSTGRES_ACCEPTANCE_BINDING_MISMATCH = "postgres_acceptance_binding_mismatch"
    OBSERVED_MODEL_ID_MISMATCH = "observed_model_id_mismatch"
    ACTIVATION_PATH_UNIMPLEMENTED = "activation_path_unimplemented"


__all__ = ["EnumRsdLiveDelegationPreflightFailure"]

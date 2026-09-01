# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable first-effect recording scaffolds.

The legacy lifecycle ledger records non-authorizing observations.  The
verified-grant ledger stores a payload-free causal projection after the fixed
RSD verifier has accepted a wire grant; that projection is never issuer proof
or effect permission.  This package ships no publisher.
"""

from omnibase_infra.runtime.first_effect_ledger.adapter_postgres import (
    FirstEffectLedgerConflictError,
    PostgresFirstEffectLedger,
)
from omnibase_infra.runtime.first_effect_ledger.composition import (
    FirstEffectGrantIngressRejectedError,
    FirstEffectPublishCompositionError,
    ProtocolTransactionalFirstEffectOutbox,
    build_rsd_verified_first_effect_grant_ingress,
    require_transactional_first_effect_outbox,
)
from omnibase_infra.runtime.first_effect_ledger.enum_first_effect_ledger_state import (
    EnumFirstEffectLedgerState,
)
from omnibase_infra.runtime.first_effect_ledger.enum_verified_first_effect_grant_state import (
    EnumVerifiedFirstEffectGrantState,
)
from omnibase_infra.runtime.first_effect_ledger.model_expected_output_pin import (
    ModelExpectedOutputPin,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    ModelFirstEffectAuthorizationRequest,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_consumer_claim import (
    ModelFirstEffectConsumerClaim,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_ledger_record import (
    ModelFirstEffectLedgerRecord,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_output_identity import (
    ModelFirstEffectOutputIdentity,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_stage_request import (
    ModelFirstEffectStageRequest,
)
from omnibase_infra.runtime.first_effect_ledger.model_verified_first_effect_grant_record import (
    ModelVerifiedFirstEffectGrantRecord,
)
from omnibase_infra.runtime.first_effect_ledger.protocol import (
    ProtocolFirstEffectLedger,
)
from omnibase_infra.runtime.first_effect_ledger.protocol_signed_first_effect_grant_ingress import (
    ProtocolSignedFirstEffectGrantIngress,
)

__all__ = [
    "EnumFirstEffectLedgerState",
    "FirstEffectLedgerConflictError",
    "FirstEffectGrantIngressRejectedError",
    "FirstEffectPublishCompositionError",
    "build_rsd_verified_first_effect_grant_ingress",
    "ModelFirstEffectAuthorizationRequest",
    "ModelFirstEffectConsumerClaim",
    "ModelFirstEffectOutputIdentity",
    "ModelFirstEffectStageRequest",
    "ModelExpectedOutputPin",
    "ModelFirstEffectLedgerRecord",
    "ModelVerifiedFirstEffectGrantRecord",
    "PostgresFirstEffectLedger",
    "ProtocolFirstEffectLedger",
    "ProtocolSignedFirstEffectGrantIngress",
    "EnumVerifiedFirstEffectGrantState",
    "ProtocolTransactionalFirstEffectOutbox",
    "require_transactional_first_effect_outbox",
]

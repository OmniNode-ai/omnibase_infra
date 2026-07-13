# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Validation ledger models for cross-repo validation event persistence.

Pydantic models for the validation_event_ledger table,
supporting append, query, and replay operations for cross-repository
validation events.

Models:
    - ModelValidationLedgerEntry: Single row in the validation_event_ledger table
    - ModelValidationLedgerQuery: Filter model for validation ledger queries
    - ModelValidationLedgerReplayBatch: Paginated result set from a query
    - ModelValidationLedgerAppendResult: Result of appending a validation event
    - ModelPayloadValidationLedgerAppend: Intent payload for the write-effect
      (OMN-14524)

Example:
    >>> from omnibase_infra.models.validation_ledger import (
    ...     ModelValidationLedgerEntry,
    ...     ModelValidationLedgerQuery,
    ...     ModelValidationLedgerReplayBatch,
    ...     ModelValidationLedgerAppendResult,
    ...     ModelPayloadValidationLedgerAppend,
    ... )
"""

from omnibase_infra.models.validation_ledger.model_payload_validation_ledger_append import (
    ModelPayloadValidationLedgerAppend,
)
from omnibase_infra.models.validation_ledger.model_validation_ledger_append_result import (
    ModelValidationLedgerAppendResult,
)
from omnibase_infra.models.validation_ledger.model_validation_ledger_entry import (
    ModelValidationLedgerEntry,
)
from omnibase_infra.models.validation_ledger.model_validation_ledger_query import (
    ModelValidationLedgerQuery,
)
from omnibase_infra.models.validation_ledger.model_validation_ledger_replay_batch import (
    ModelValidationLedgerReplayBatch,
)

__all__ = [
    "ModelPayloadValidationLedgerAppend",
    "ModelValidationLedgerAppendResult",
    "ModelValidationLedgerEntry",
    "ModelValidationLedgerQuery",
    "ModelValidationLedgerReplayBatch",
]

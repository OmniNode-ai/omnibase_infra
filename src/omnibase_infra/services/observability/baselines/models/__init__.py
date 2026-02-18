# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Models for the baselines observability service.

This package contains all Pydantic models used in the baselines data
pipeline, from raw database row representations through batch computation
results to the Kafka event payload emitted at the end of each run.

Pipeline roles
--------------
``ModelBaselinesComparisonRow``
    Hydrated from the ``baselines_comparisons`` table (daily treatment vs
    control aggregate).  Used as both the DB row type (via
    ``from_attributes=True``) and the API response type for
    ``/api/baselines/comparisons`` and ``/api/baselines/summary``.

``ModelBaselinesTrendRow``
    Hydrated from the ``baselines_trend`` table (per-cohort, per-day time
    series).  Pairs of (treatment, control) rows for the same date form
    a single data point consumed by ``/api/baselines/trend``.

``ModelBaselinesBreakdownRow``
    Hydrated from the ``baselines_breakdown`` table (per-pattern treatment
    vs control breakdown).  Consumed by ``/api/baselines/breakdown``.

``ModelBatchComputeBaselinesResult``
    Returned by ``ServiceBatchComputeBaselines.compute_and_persist()``.
    Carries row counts and the generated ``snapshot_id`` used to stamp
    the Kafka event.

``ModelBaselinesSnapshotEvent``
    The Kafka event payload published to
    ``onex.evt.omnibase-infra.baselines-computed.v1`` after a successful
    batch run.  Uses ``extra="forbid"``; consumers must match the exact
    declared schema.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
    - OMN-2331: Wire /api/baselines/* to real tables -- remove mockOnEmpty
"""

from omnibase_infra.services.observability.baselines.models.model_baselines_breakdown_row import (
    ModelBaselinesBreakdownRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_comparison_row import (
    ModelBaselinesComparisonRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_snapshot_event import (
    ModelBaselinesSnapshotEvent,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_trend_row import (
    ModelBaselinesTrendRow,
)
from omnibase_infra.services.observability.baselines.models.model_batch_compute_baselines_result import (
    ModelBatchComputeBaselinesResult,
)

__all__: list[str] = [
    "ModelBaselinesBreakdownRow",
    "ModelBaselinesComparisonRow",
    "ModelBaselinesTrendRow",
    "ModelBaselinesSnapshotEvent",
    "ModelBatchComputeBaselinesResult",
]

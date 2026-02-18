# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Baselines observability service.

Provides batch computation for A/B treatment/control comparison data
that powers the Baselines & ROI dashboard page (OMN-2305).

Services:
    ServiceBatchComputeBaselines: Batch computation from existing data.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
"""

from omnibase_infra.services.observability.baselines.service_batch_compute_baselines import (
    ServiceBatchComputeBaselines,
)

__all__: list[str] = ["ServiceBatchComputeBaselines"]

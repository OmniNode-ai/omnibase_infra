# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Post-merge consumer chain for event-driven post-merge automation.

Listens to ``onex.evt.github.pr-merged.v1`` events and dispatches a chain
of quality checks:
1. Hostile review on the merged diff
2. Contract sweep (check-drift CLI)
3. Integration check (boundary verification)

Findings are auto-ticketed in Linear.

Related Tickets:
    - OMN-6727: post-merge consumer chain
    - OMN-6726: GitHub merge event producer (upstream)
    - OMN-6725: contract_sweep skill wrapper (dependency)
"""

from __future__ import annotations

from omnibase_infra.services.post_merge.config import ConfigPostMergeConsumer
from omnibase_infra.services.post_merge.consumer import PostMergeConsumer
from omnibase_infra.services.post_merge.enum_check_stage import EnumCheckStage
from omnibase_infra.services.post_merge.enum_finding_severity import (
    EnumFindingSeverity,
)
from omnibase_infra.services.post_merge.model_post_merge_finding import (
    ModelPostMergeFinding,
)
from omnibase_infra.services.post_merge.model_post_merge_result import (
    ModelPostMergeResult,
)

__all__ = [
    "ConfigPostMergeConsumer",
    "EnumCheckStage",
    "EnumFindingSeverity",
    "ModelPostMergeFinding",
    "ModelPostMergeResult",
    "PostMergeConsumer",
]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Adapter for storing and retrieving agent learning records.

Writes to both Postgres (structured metadata) and Qdrant (vector embeddings).
Reads from Qdrant for similarity search, then hydrates from Postgres.
"""

from __future__ import annotations

import math
from datetime import datetime


def compute_freshness_score(
    created_at: datetime,
    now: datetime,
) -> float:
    """Compute freshness decay: 10% loss per week, asymptotic to 0.

    Uses exponential decay: score = e^(-0.015 * days_old)
    This gives ~90% at 1 week, ~60% at 4 weeks, ~35% at 8 weeks.
    """
    delta = now - created_at
    days_old = max(0.0, delta.total_seconds() / 86400)
    return math.exp(-0.015 * days_old)

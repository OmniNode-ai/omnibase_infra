# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for emit daemon integration tests.

This module provides shared fixtures for the emit daemon integration tests.
Fixtures handle Kafka topic management, daemon lifecycle, and test isolation.

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker address (e.g., "localhost:9092")

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP
"""

from __future__ import annotations

import logging

import pytest

# Module-level logger for test diagnostics
logger = logging.getLogger(__name__)

# =============================================================================
# Module-Level Markers
# =============================================================================

pytestmark = [
    pytest.mark.kafka,
]

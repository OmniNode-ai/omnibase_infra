# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session registry service for multi-session coordination.

Materializes session state from Kafka events into Postgres for
cross-session awareness, resume, and coordination queries.

Part of the Multi-Session Coordination Layer (OMN-6850).
"""

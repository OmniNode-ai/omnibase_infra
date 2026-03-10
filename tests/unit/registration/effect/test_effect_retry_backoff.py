# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for effect retry and backoff behavior.

NOTE (OMN-3540): HandlerConsul has been fully removed from omnibase_infra.
The retry/backoff tests that previously exercised HandlerConsul have been
removed along with the handler. If a replacement handler gains retry/backoff
capabilities, add new tests here targeting that handler.
"""

from __future__ import annotations

__all__: list[str] = []

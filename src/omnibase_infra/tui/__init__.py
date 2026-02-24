# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Status TUI — textual-based terminal status view.

Consumes three Kafka topics and renders a 3-panel layout:
  - Workstreams (left):  Linear epic progress from onex.evt.linear.snapshot.v1
  - PR Triage (center):  GitHub PR triage state from onex.evt.github.pr-status.v1
  - Hook Feed (right):   Real-time git hook feed from onex.evt.git.hook.v1

Entry point: ``uv run python -m omnibase_infra.tui``

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

__all__ = []

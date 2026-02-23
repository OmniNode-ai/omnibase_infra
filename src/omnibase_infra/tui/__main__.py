# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Entry point for the ONEX Status TUI.

Usage:
    uv run python -m omnibase_infra.tui

Or via project script (after installing):
    onex-status

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

import logging

from omnibase_infra.tui.app import StatusApp


def run_status_tui() -> None:
    """Launch the ONEX Status TUI."""
    logging.basicConfig(level=logging.WARNING)
    app = StatusApp()
    app.run()


if __name__ == "__main__":
    run_status_tui()

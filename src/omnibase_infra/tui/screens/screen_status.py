# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Status screen — 3-panel layout for the ONEX TUI.

Layout:
    ┌─────────────────┬──────────────────┬──────────────────┐
    │  Workstreams    │   PR Triage      │  Git Hook Feed   │
    │  (left 25%)     │   (center 45%)   │  (right 30%)     │
    └─────────────────┴──────────────────┴──────────────────┘

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header

from omnibase_infra.tui.widgets.widget_hook_feed import WidgetHookFeed
from omnibase_infra.tui.widgets.widget_pr_triage import WidgetPRTriage
from omnibase_infra.tui.widgets.widget_workstreams import WidgetWorkstreams


class ScreenStatus(Screen[None]):
    """Primary status screen with a 3-panel horizontal layout."""

    DEFAULT_CSS = """
    ScreenStatus {
        layout: horizontal;
    }

    WidgetWorkstreams {
        width: 25%;
        border: solid $primary;
        padding: 0 1;
    }

    WidgetPRTriage {
        width: 45%;
        border: solid $accent;
        padding: 0 1;
    }

    WidgetHookFeed {
        width: 30%;
        border: solid $secondary;
        padding: 0 1;
    }

    #workstreams-header,
    #pr-triage-header,
    #hook-feed-header {
        text-align: center;
        text-style: bold;
        padding: 0 0 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield WidgetWorkstreams(id="workstreams")
        yield WidgetPRTriage(id="pr-triage")
        yield WidgetHookFeed(id="hook-feed")
        yield Footer()


__all__ = ["ScreenStatus"]

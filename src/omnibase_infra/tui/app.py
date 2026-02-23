# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""StatusApp — the main textual application for ONEX TUI.

Launches the ScreenStatus screen and starts the Kafka consumer background
task. Handles all three message types dispatched by the consumer and routes
them to the appropriate widgets.

Keybindings:
    [r]  Refresh all panels (re-render from in-memory state)
    [q]  Quit
    [o]  Open selected PR in browser (delegates to WidgetPRTriage)

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

import asyncio
import logging

from textual.app import App, ComposeResult
from textual.binding import Binding

from omnibase_infra.tui.consumers.consumer_status import (
    HookEventReceived,
    PRStatusReceived,
    SnapshotReceived,
    consume_all,
)
from omnibase_infra.tui.screens.screen_status import ScreenStatus
from omnibase_infra.tui.widgets.widget_hook_feed import WidgetHookFeed
from omnibase_infra.tui.widgets.widget_pr_triage import WidgetPRTriage
from omnibase_infra.tui.widgets.widget_workstreams import WidgetWorkstreams

logger = logging.getLogger(__name__)

_APP_TITLE = "ONEX Status"
_APP_SUB_TITLE = "Live · github.pr-status | git.hook | linear.snapshot"


class StatusApp(App[None]):
    """Main ONEX Status TUI application.

    Subscribes to three Kafka topics via a background consumer task.
    Routes incoming messages to the three status widgets.
    """

    TITLE = _APP_TITLE
    SUB_TITLE = _APP_SUB_TITLE

    BINDINGS = [
        Binding("r", "refresh_panels", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("o", "open_pr", "Open PR"),
    ]

    def compose(self) -> ComposeResult:
        yield ScreenStatus()

    async def on_mount(self) -> None:
        """Push the status screen and start the Kafka consumer task."""
        self._consumer_task: asyncio.Task[None] = asyncio.create_task(
            consume_all(self),
            name="onex-tui-kafka-consumer",
        )

    async def on_unmount(self) -> None:
        """Cancel the consumer task on exit."""
        if hasattr(self, "_consumer_task"):
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def on_p_r_status_received(self, message: PRStatusReceived) -> None:
        """Route PR status event to the PR triage widget."""
        widget: WidgetPRTriage = self.query_one("#pr-triage", WidgetPRTriage)
        widget.update_pr(message.payload)

    def on_hook_event_received(self, message: HookEventReceived) -> None:
        """Route git hook event to the hook feed widget."""
        widget: WidgetHookFeed = self.query_one("#hook-feed", WidgetHookFeed)
        widget.add_event(message.payload)

    def on_snapshot_received(self, message: SnapshotReceived) -> None:
        """Route Linear snapshot event to the workstreams widget."""
        widget: WidgetWorkstreams = self.query_one("#workstreams", WidgetWorkstreams)
        widget.update_snapshot(message.payload)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_refresh_panels(self) -> None:
        """[r] Re-render all panels from in-memory state."""
        try:
            ws: WidgetWorkstreams = self.query_one("#workstreams", WidgetWorkstreams)
            ws.refresh_display()
        except Exception:
            pass
        try:
            hf: WidgetHookFeed = self.query_one("#hook-feed", WidgetHookFeed)
            hf.refresh_all()
        except Exception:
            pass
        try:
            pt: WidgetPRTriage = self.query_one("#pr-triage", WidgetPRTriage)
            pt._refresh_table()
        except Exception:
            pass

    def action_open_pr(self) -> None:
        """[o] Open the selected PR in the browser."""
        try:
            pt: WidgetPRTriage = self.query_one("#pr-triage", WidgetPRTriage)
            pt.open_selected_pr()
        except Exception:
            pass


__all__ = ["StatusApp"]

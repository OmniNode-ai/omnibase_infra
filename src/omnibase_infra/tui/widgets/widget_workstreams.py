# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Workstreams widget — Linear epic progress bars from snapshot events.

Displays ``workstreams`` from ModelLinearSnapshotEvent payloads received
on onex.evt.linear.snapshot.v1.

The workstreams field is a list of strings (workstream names) extracted
from the snapshot. Progress bars show active workstream count vs total
in the snapshot.

No imports from application-specific modules (omniclaude, omnidash).
All display is driven from event field values only.

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Label, Static

from omnibase_core.types import JsonType


class WidgetWorkstreams(Static):
    """Left panel: workstream progress bars from Linear snapshots.

    Each snapshot_id replaces the previous snapshot — only the most recent
    snapshot state is shown. Workstreams are rendered as individual labeled
    progress bars.
    """

    def __init__(
        self,
        content: str = "",
        *,
        expand: bool = False,
        shrink: bool = False,
        markup: bool = True,
        name: str | None = None,
        id: str | None = None,  # noqa: A002  # textual API uses 'id'
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            content,
            expand=expand,
            shrink=shrink,
            markup=markup,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self._latest_snapshot: dict[str, JsonType] | None = None

    def compose(self) -> ComposeResult:
        yield Label("[b]Workstreams[/b]", id="workstreams-header")
        yield Static("Waiting for snapshot...", id="workstreams-content")

    def update_snapshot(self, payload: dict[str, JsonType]) -> None:
        """Accept a new Linear snapshot event and update the display."""
        self._latest_snapshot = payload
        self._render_snapshot(payload)

    def _render_snapshot(self, payload: dict[str, JsonType]) -> None:
        workstreams_raw = payload.get("workstreams", [])
        workstreams: list[str] = (
            [str(w) for w in workstreams_raw]
            if isinstance(workstreams_raw, list)
            else []
        )
        snapshot_id_raw = payload.get("snapshot_id", "")
        snapshot_id = str(snapshot_id_raw)[:8] if snapshot_id_raw else "unknown"
        emitted_raw = payload.get("emitted_at", "")
        emitted = str(emitted_raw)[:19].replace("T", " ") if emitted_raw else ""

        # Get snapshot sub-dict for richer data if available
        snapshot_data_raw = payload.get("snapshot", {})
        snapshot_data: dict[str, JsonType] = (
            snapshot_data_raw if isinstance(snapshot_data_raw, dict) else {}
        )

        content_widget: Static = self.query_one("#workstreams-content", Static)

        if not workstreams:
            content_widget.update(
                f"[dim]Snapshot {snapshot_id} at {emitted}[/dim]\n"
                "[dim]No workstreams in snapshot.[/dim]"
            )
            return

        # Build display lines
        lines: list[str] = []
        lines.append(f"[dim]Snapshot {snapshot_id} at {emitted}[/dim]")
        lines.append(f"[bold]{len(workstreams)} workstreams[/bold]")
        lines.append("")

        # Extract epic-level progress if available in snapshot
        # snapshot_data structure is opaque — we render what we have
        epics_raw = snapshot_data.get("epics", [])
        epics: list[dict[str, JsonType]] = (
            [e for e in epics_raw if isinstance(e, dict)]
            if isinstance(epics_raw, list)
            else []
        )

        epic_by_name: dict[str, dict[str, JsonType]] = {}
        for epic in epics:
            name_raw = epic.get("name", epic.get("title", ""))
            if name_raw:
                epic_by_name[str(name_raw)] = epic

        for ws in workstreams:
            epic = epic_by_name.get(ws, {})
            total_raw = epic.get("total_issues", 0)
            done_raw = epic.get("completed_issues", 0)
            total = int(total_raw) if isinstance(total_raw, (int, float)) else 0
            done = int(done_raw) if isinstance(done_raw, (int, float)) else 0

            if total > 0:
                pct = int(100 * done / total)
                bar_filled = int(pct / 5)  # 20-char bar
                bar_empty = 20 - bar_filled
                bar = f"[green]{'█' * bar_filled}[/green][dim]{'░' * bar_empty}[/dim]"
                lines.append(f"[bold]{ws}[/bold]  {bar}  {done}/{total} ({pct}%)")
            else:
                lines.append(f"[bold]{ws}[/bold]  [dim]no issue data[/dim]")

        content_widget.update("\n".join(lines))

    def refresh_display(self) -> None:
        """Re-render from latest snapshot (used after [r] refresh)."""
        if self._latest_snapshot is not None:
            self._render_snapshot(self._latest_snapshot)


__all__ = ["WidgetWorkstreams"]

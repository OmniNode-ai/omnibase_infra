# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PR Triage widget — displays GitHub PRs grouped by triage_state.

Handles all 8 triage states defined in handler_github_api_poll.py:
    draft | stale | ci_failing | changes_requested | ready_to_merge |
    approved_pending_ci | needs_review | blocked

State is driven purely by event field values — no hardcoded OMN- prefix
or application-specific conditionals.

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

import webbrowser
from collections import defaultdict
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from omnibase_core.types import JsonType

# All 8 triage states — ordering defines display priority
_TRIAGE_ORDER: tuple[str, ...] = (
    "ready_to_merge",
    "approved_pending_ci",
    "needs_review",
    "changes_requested",
    "ci_failing",
    "stale",
    "blocked",
    "draft",
)

_TRIAGE_STYLE: dict[str, str] = {
    "ready_to_merge": "bold green",
    "approved_pending_ci": "green",
    "needs_review": "bold yellow",
    "changes_requested": "bold red",
    "ci_failing": "red",
    "stale": "dim",
    "blocked": "bold magenta",
    "draft": "dim italic",
}


def _triage_label(state: str) -> Text:
    style = _TRIAGE_STYLE.get(state, "")
    return Text(state.replace("_", " ").upper(), style=style)


class WidgetPRTriage(Static):
    """Center panel: PRs grouped by triage state.

    Receives PRStatusReceived events from the app and updates the table.
    The app adds PR data via ``update_pr``.
    Keybinding [o] opens the currently selected PR in a browser.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = set()

    # Map: (repo, pr_number) -> latest payload
    _prs: reactive[dict[tuple[str, int], dict[str, JsonType]]] = reactive({})

    def compose(self) -> ComposeResult:
        yield Label("[b]PR Triage[/b]", id="pr-triage-header")
        yield DataTable(id="pr-triage-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        table: DataTable[Text] = self.query_one("#pr-triage-table", DataTable)
        table.add_columns("State", "Repo", "PR#", "Title")

    def update_pr(self, payload: dict[str, JsonType]) -> None:
        """Accept a new or updated PR status event and refresh the table."""
        repo_raw = payload.get("partition_key", "")
        repo = (
            str(repo_raw).rsplit(":", 1)[0] if ":" in str(repo_raw) else str(repo_raw)
        )
        pr_number_raw = payload.get("pr_number", 0)
        pr_number = int(pr_number_raw) if isinstance(pr_number_raw, (int, float)) else 0
        key = (repo, pr_number)
        new_prs = dict(self._prs)
        new_prs[key] = payload
        self._prs = new_prs
        self._refresh_table()

    def _refresh_table(self) -> None:
        table: DataTable[Text] = self.query_one("#pr-triage-table", DataTable)
        table.clear()

        # Group PRs by triage state
        by_state: dict[str, list[dict[str, JsonType]]] = defaultdict(list)
        for pr in self._prs.values():
            state = str(pr.get("triage_state", "needs_review"))
            by_state[state].append(pr)

        for state in _TRIAGE_ORDER:
            prs_in_state = by_state.get(state, [])
            for pr in prs_in_state:
                repo_raw = pr.get("partition_key", "")
                repo = (
                    str(repo_raw).rsplit(":", 1)[0]
                    if ":" in str(repo_raw)
                    else str(repo_raw)
                )
                pr_number = str(pr.get("pr_number", "?"))
                title_raw = pr.get("title", "")
                title = str(title_raw)[:60] if title_raw else "(no title)"
                table.add_row(
                    _triage_label(state),
                    Text(repo),
                    Text(pr_number),
                    Text(title),
                )

        # Also add any states not in _TRIAGE_ORDER (forward-compat)
        for state, prs_in_state in by_state.items():
            if state not in _TRIAGE_ORDER:
                for pr in prs_in_state:
                    repo_raw = pr.get("partition_key", "")
                    repo = (
                        str(repo_raw).rsplit(":", 1)[0]
                        if ":" in str(repo_raw)
                        else str(repo_raw)
                    )
                    pr_number = str(pr.get("pr_number", "?"))
                    title_raw = pr.get("title", "")
                    title = str(title_raw)[:60] if title_raw else "(no title)"
                    table.add_row(
                        _triage_label(state),
                        Text(repo),
                        Text(pr_number),
                        Text(title),
                    )

    def refresh_table(self) -> None:
        """Public alias for _refresh_table — re-render table from in-memory state."""
        self._refresh_table()

    def open_selected_pr(self) -> None:
        """Open the currently selected PR in the browser via [o] keybinding."""
        table: DataTable[Text] = self.query_one("#pr-triage-table", DataTable)
        if table.cursor_row is None:
            return
        # Reconstruct repo/pr_number from the sorted list matching cursor row
        rows: list[tuple[str, int]] = []
        by_state: dict[str, list[dict[str, JsonType]]] = defaultdict(list)
        for pr in self._prs.values():
            state = str(pr.get("triage_state", "needs_review"))
            by_state[state].append(pr)
        for state in _TRIAGE_ORDER:
            for pr in by_state.get(state, []):
                repo_raw = pr.get("partition_key", "")
                repo = (
                    str(repo_raw).rsplit(":", 1)[0]
                    if ":" in str(repo_raw)
                    else str(repo_raw)
                )
                pr_number_raw = pr.get("pr_number", 0)
                pr_number = (
                    int(pr_number_raw) if isinstance(pr_number_raw, (int, float)) else 0
                )
                rows.append((repo, pr_number))
        for state, prs in by_state.items():
            if state not in _TRIAGE_ORDER:
                for pr in prs:
                    repo_raw = pr.get("partition_key", "")
                    repo = (
                        str(repo_raw).rsplit(":", 1)[0]
                        if ":" in str(repo_raw)
                        else str(repo_raw)
                    )
                    pr_number_raw = pr.get("pr_number", 0)
                    pr_number = (
                        int(pr_number_raw)
                        if isinstance(pr_number_raw, (int, float))
                        else 0
                    )
                    rows.append((repo, pr_number))

        cursor_row = table.cursor_row
        if cursor_row is not None and 0 <= cursor_row < len(rows):
            repo, pr_number = rows[cursor_row]
            url = f"https://github.com/{repo}/pull/{pr_number}"
            webbrowser.open(url)


__all__ = ["WidgetPRTriage"]

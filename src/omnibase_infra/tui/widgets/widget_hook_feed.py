# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Hook Feed widget — scrollable real-time git hook event feed.

Receives ModelGitHookEvent payloads (from onex.evt.git.hook.v1) and
displays them in a scrollable log, newest at top.

Fields displayed: hook, repo, branch, author, outcome, emitted_at.
No imports from application-specific modules.

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

from collections import deque

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Label, RichLog, Static

from omnibase_core.types import JsonType

# Max hook events to keep in memory
_MAX_EVENTS = 200

_OUTCOME_STYLE: dict[str, str] = {
    "pass": "bold green",
    "allowed": "bold green",
    "fail": "bold red",
    "blocked": "bold red",
    "warning": "yellow",
}


def _outcome_text(outcome: str) -> Text:
    style = _OUTCOME_STYLE.get(outcome.lower(), "white")
    return Text(outcome.upper(), style=style)


def _format_hook_line(payload: dict[str, JsonType]) -> Text:
    """Format a single git hook event as a Rich Text line."""
    hook = str(payload.get("hook", "unknown"))
    repo = str(payload.get("repo", "?"))
    branch = str(payload.get("branch", "?"))
    author = str(payload.get("author", "?"))
    outcome = str(payload.get("outcome", "?"))
    emitted_at_raw = payload.get("emitted_at", "")
    emitted_at = str(emitted_at_raw)[:19].replace("T", " ") if emitted_at_raw else ""

    line = Text()
    line.append(emitted_at, style="dim")
    line.append("  ")
    line.append(hook.upper(), style="bold cyan")
    line.append(f"  {repo}", style="white")
    line.append(f"  @{branch}", style="blue")
    line.append(f"  {author}", style="magenta")
    line.append("  ")
    line.append_text(_outcome_text(outcome))
    return line


class WidgetHookFeed(Static):
    """Right panel: scrollable real-time git hook event feed.

    New events are prepended (newest first). Capped at _MAX_EVENTS.
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
        self._events: deque[dict[str, JsonType]] = deque(maxlen=_MAX_EVENTS)

    def compose(self) -> ComposeResult:
        yield Label("[b]Git Hook Feed[/b]", id="hook-feed-header")
        yield RichLog(
            id="hook-feed-log", highlight=False, markup=True, max_lines=_MAX_EVENTS
        )

    def add_event(self, payload: dict[str, JsonType]) -> None:
        """Accept a new git hook event and prepend to the feed."""
        self._events.appendleft(payload)
        log: RichLog = self.query_one("#hook-feed-log", RichLog)
        log.write(_format_hook_line(payload))

    def refresh_all(self) -> None:
        """Rebuild the full log from the in-memory deque (used after [r] refresh).

        Iterates in reverse so oldest events are written first and newest appears
        at the bottom of the RichLog, consistent with add_event behavior.
        """
        log: RichLog = self.query_one("#hook-feed-log", RichLog)
        log.clear()
        for payload in reversed(self._events):
            log.write(_format_hook_line(payload))


__all__ = ["WidgetHookFeed"]

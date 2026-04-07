# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Self-contained dashboard sweep runner for the verify phase.

Hits omnidash pages via HTTP and checks that key routes return 200 with
non-empty content. Advisory only — if omnidash is not running, all checks
produce warnings rather than failures.

Related:
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Key omnidash routes and what we expect to find in the response body
# to confirm the page has real data (not just a shell).
_DASHBOARD_ROUTES: tuple[tuple[str, str], ...] = (
    ("/", "Summary"),
    ("/agents", "Agent"),
    ("/events", "Event"),
    ("/intelligence", "Intelligence"),
    ("/drift", "Drift"),
    ("/pipeline", "Pipeline"),
    ("/metrics", "Metric"),
    ("/settings", "Settings"),
)

_DEFAULT_BASE_URL = "http://localhost:3000"
_REQUEST_TIMEOUT = 8.0


@dataclass(frozen=True)
class DashboardPageResult:
    """Result of checking a single dashboard page."""

    route: str
    status_code: int | None
    has_data: bool
    message: str


@dataclass(frozen=True)
class DashboardSweepResult:
    """Aggregate result of the dashboard sweep."""

    reachable: bool
    pages: tuple[DashboardPageResult, ...]
    pages_with_data: int
    pages_no_data: int
    pages_error: int
    summary: str


async def run_dashboard_sweep(
    base_url: str = _DEFAULT_BASE_URL,
) -> DashboardSweepResult:
    """Hit all known omnidash routes and classify each page.

    Args:
        base_url: The omnidash base URL (default: http://localhost:3000).

    Returns:
        DashboardSweepResult with per-page outcomes.
    """
    # First check if omnidash is reachable at all
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            probe = await client.get(f"{base_url}/")
            if probe.status_code >= 500:
                return DashboardSweepResult(
                    reachable=False,
                    pages=(),
                    pages_with_data=0,
                    pages_no_data=0,
                    pages_error=0,
                    summary=f"omnidash returned {probe.status_code} — not healthy",
                )
    except httpx.HTTPError as exc:
        return DashboardSweepResult(
            reachable=False,
            pages=(),
            pages_with_data=0,
            pages_no_data=0,
            pages_error=0,
            summary=f"omnidash unreachable at {base_url}: {exc}",
        )

    # Sweep each route
    results: list[DashboardPageResult] = []
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        for route, data_signal in _DASHBOARD_ROUTES:
            url = f"{base_url}{route}"
            try:
                resp = await client.get(url)
                body = resp.text
                has_data = (
                    resp.status_code == 200
                    and len(body) > 500
                    and data_signal.lower() in body.lower()
                )
                if resp.status_code != 200:
                    msg = f"HTTP {resp.status_code}"
                elif has_data:
                    msg = "OK — data present"
                else:
                    msg = "NO DATA — page rendered but expected content missing"
                results.append(
                    DashboardPageResult(
                        route=route,
                        status_code=resp.status_code,
                        has_data=has_data,
                        message=msg,
                    )
                )
            except httpx.HTTPError as exc:
                results.append(
                    DashboardPageResult(
                        route=route,
                        status_code=None,
                        has_data=False,
                        message=f"Request failed: {exc}",
                    )
                )

    pages_with_data = sum(1 for p in results if p.has_data)
    pages_no_data = sum(1 for p in results if p.status_code == 200 and not p.has_data)
    pages_error = sum(1 for p in results if p.status_code != 200)

    summary_parts = [
        f"{pages_with_data}/{len(results)} pages show data",
    ]
    if pages_no_data:
        summary_parts.append(f"{pages_no_data} empty")
    if pages_error:
        summary_parts.append(f"{pages_error} errors")

    summary = "; ".join(summary_parts)
    logger.info("Dashboard sweep: %s", summary)

    return DashboardSweepResult(
        reachable=True,
        pages=tuple(results),
        pages_with_data=pages_with_data,
        pages_no_data=pages_no_data,
        pages_error=pages_error,
        summary=summary,
    )

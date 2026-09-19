# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18781 — a CI run that selects a service-backed suite must never skip it.

WHY THIS EXISTS
---------------
Two integration suites carried a module-level ``pytest.mark.skipif`` gated on an
opt-in environment variable that **no workflow in this repository set**. They were
collected on every pull request, skipped in full, and contributed to a green Tests
job. The event-bus one is the bus's own integration proof, and it had never
executed in CI.

A silent skip and a suite that is deliberately not selected look identical in a
junit summary, and only one of them is honest. This helper draws the line:

``run``
    the dependency is provisioned and the opt-in is set — execute.

``fail``
    the run is CI and the opt-in is absent. Something SELECTED this suite in CI
    without provisioning its dependency, which is the exact condition that used
    to be a silent skip. It is now a red failure naming the variable and the
    workflow that owns it.

``skip``
    outside CI only, so a developer running the whole tree on a laptop with no
    broker is not forced to stand one up.

The PR test splits deselect these suites by marker (``not kafka``,
``not qdrant``), so the CI arm above fires only when a job selects the suite and
forgets the env — never on the ordinary PR path.
"""

from __future__ import annotations

import os

import pytest


def _is_ci() -> bool:
    """True when running under a CI provider.

    GitHub Actions sets ``CI=true`` on every runner, hosted and self-hosted
    alike. Any truthy spelling counts: the point is to refuse a silent skip
    wherever a machine, rather than a person, chose the selection.
    """
    return os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}


def require_service_env(
    *,
    opt_in: str,
    endpoint: str,
    workflow: str,
    service: str,
) -> None:
    """Fail in CI, skip locally, when a provisioned service's env is absent.

    Args:
        opt_in: the variable that opts this suite in (e.g. ``KAFKA_INTEGRATION_TESTS``).
            It must be exactly ``"1"``; a set-but-empty value is treated as absent.
        endpoint: the variable carrying the service address. ``tests/conftest.py``
            supplies a localhost default for several of these, so the endpoint
            alone is never sufficient evidence that a service exists — the opt-in
            is what a provisioning job asserts.
        workflow: the workflow file that provisions the service and sets ``opt_in``.
            Named in the failure text so the reader is not left guessing where the
            variable is supposed to come from.
        service: a human name for the dependency, used in the messages.

    Raises:
        Failed: in CI when ``opt_in`` is not ``"1"``.
        Skipped: outside CI when ``opt_in`` is not ``"1"``.
    """
    if os.environ.get(opt_in, "").strip() == "1":
        return

    if _is_ci():
        pytest.fail(
            f"{opt_in} is not set to '1', so this {service} suite would have "
            f"skipped silently in CI. It is deselected from the PR test splits by "
            f"marker and is executed by {workflow}, which provisions {service} and "
            f"exports {opt_in} and {endpoint}. A CI job that selects this suite "
            f"without provisioning {service} is a false green, not a skip — see "
            f"OMN-18781.",
            pytrace=False,
        )

    pytest.skip(
        f"{service} integration suite not opted in locally. "
        f"Set {opt_in}=1 and {endpoint} to run it against a real {service}; "
        f"{workflow} does this in CI."
    )

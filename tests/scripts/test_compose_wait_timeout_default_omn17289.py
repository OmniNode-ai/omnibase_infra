# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The compose wait deadline must not be shorter than the startup budget the
compose file itself declares (OMN-17289).

Real failure this pins: ``RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS`` defaulted to
300s while ``docker/docker-compose.infra.yml`` declares ``start_period: 1800s``
for ``omninode-runtime`` and ``runtime-effects``. A deploy whose services were
still legitimately inside their own declared start_period was therefore killed
at the deadline and reported as a failure -- healthy-but-slow was
indistinguishable from the OMN-15718 permanent-hang case the deadline exists to
bound. The kill then landed in the EXIT trap OMN-17287 had to guard against
``rm -rf``-ing a deploy dir with live bind-mounts.

The assertion is deliberately RELATIVE, not a hardcoded 1800: it re-derives the
budget from the compose file on every run, so raising a service's start_period
without raising this deadline fails here instead of in a deploy on .201.

``docker-compose.infra.yml`` uses YAML merge keys with custom tags
(``!!merge <<: *runtime-base``) that ``yaml.safe_load`` rejects, so the
start_period values are read with a line regex rather than a YAML parse. This
reads the same declarations the operator reads; it does not need the resolved
service graph.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "compose_wait_timeout.sh"
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"
INFRA_COMPOSE = REPO_ROOT / "docker" / "docker-compose.infra.yml"

_START_PERIOD_RE = re.compile(r"^\s*start_period:\s*(\d+)s\s*$", re.MULTILINE)


def _declared_start_periods() -> list[int]:
    """Every ``start_period: <n>s`` declared in the infra compose file."""
    text = INFRA_COMPOSE.read_text(encoding="utf-8")
    return [int(m) for m in _START_PERIOD_RE.findall(text)]


def _sourced_default() -> int:
    """The default the shell library actually assigns, by sourcing it."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{LIB_SCRIPT}"; printf "%s" '
            '"${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}"',
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert result.returncode == 0, result.stderr
    return int(result.stdout.strip())


@pytest.mark.unit
def test_infra_compose_declares_start_periods() -> None:
    """Guard the guard: a regex that silently matched nothing would make the
    real assertion below vacuously true."""
    periods = _declared_start_periods()
    assert periods, f"parsed no start_period values from {INFRA_COMPOSE}"
    assert max(periods) >= 1800, (
        "expected the infra compose file to still declare a long runtime "
        f"start_period; got max={max(periods)}s"
    )


@pytest.mark.unit
def test_default_deadline_covers_declared_start_period() -> None:
    """The deadline must be >= the largest start_period it has to outlast."""
    default_seconds = _sourced_default()
    worst_start_period = max(_declared_start_periods())

    assert default_seconds >= worst_start_period, (
        f"RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS default is {default_seconds}s but "
        f"{INFRA_COMPOSE.name} declares a start_period of {worst_start_period}s. "
        "A deploy would be killed while its services are still inside the "
        "startup window the compose file itself grants them (OMN-17289)."
    )


@pytest.mark.unit
def test_default_deadline_is_still_finite_and_bounded() -> None:
    """OMN-15718 was an UNBOUNDED hang. Raising the default must not drift into
    'effectively unbounded' -- keep it inside the autoheal window (2400s) that
    the compose file uses to clear the same startup budget."""
    default_seconds = _sourced_default()
    assert 0 < default_seconds <= 2400, (
        f"expected a finite deadline within the 2400s autoheal window; "
        f"got {default_seconds}s"
    )


@pytest.mark.unit
def test_default_remains_operator_overridable() -> None:
    """The raised default must stay a default, not become a fixed constant."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{LIB_SCRIPT}"; printf "%s" '
            '"${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}"',
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS": "37",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "37"


@pytest.mark.unit
def test_failed_deploy_fails_the_refresh() -> None:
    """A non-zero deploy-runtime.sh must end the refresh, not fall through.

    Static guard on the source: the health-gate answers "is something healthy?",
    never "is the build I just asked for the one running?", so continuing past a
    failed deploy could report a green refresh over the PREVIOUS image. The
    stability lane is where the ``stability-proven`` premise of every live prod
    grant is resolved from (OMN-15243), so that false green is load-bearing.
    """
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")

    match = re.search(
        r'if \[\[ "\$\{DEPLOY_EXIT\}" -ne 0 \]\]; then(.*?)\nfi\n',
        text,
        re.DOTALL,
    )
    assert match is not None, (
        "could not locate the DEPLOY_EXIT failure block in "
        f"{REFRESH_SCRIPT.name} -- if it was restructured, update this guard "
        "rather than deleting it"
    )
    block = match.group(1)
    # Comments in this block deliberately quote the old fall-through wording to
    # explain why it was removed. Only executable lines are the behaviour.
    code = "\n".join(
        line for line in block.splitlines() if not line.lstrip().startswith("#")
    )

    assert 'exit "${DEPLOY_EXIT}"' in code, (
        "the DEPLOY_EXIT failure block must propagate the deploy's exit code "
        "and stop the refresh (OMN-17289)"
    )
    assert "proceeding to health-gate anyway" not in code, (
        "the failed deploy must not fall through to the health-gate: a green "
        "verdict there would describe the previously-deployed image, not the "
        "build this refresh was asked to install (OMN-17289)"
    )

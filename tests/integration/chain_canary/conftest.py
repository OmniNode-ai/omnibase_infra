# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for the chain-canary integration proofs.

``pg16`` (OMN-18872): ``test_application_migration_ledger_omn15413.py`` owns the
PostgreSQL 16 harness. It is re-exported here rather than imported into the test
module for the reason that file's own siblings record: imported into a module, the
name shadows every ``pg16`` test parameter in it and ruff rejects the redefinition.

``live_deploy_agent_chain`` (OMN-24674): the read side of the deploy agent's job
reading chain. The agent writes one ledger reading row per terminal dev or stability
job through ``node_deploy_job_reading_effect``'s append command; this fixture reads
the agent's own HTTP surface and the ledger rows that came from the agent's host, so a
test can bind a row to the job that produced it. It reads and never writes: the
OMN-19513 guard refuses a test's write to the ledger of record, and this fixture does
not try one.

The live tests read these variables, exported by the runner (OMN-24695):

* ``CHAIN_CANARY_DEPLOY_AGENT_URL`` -- the agent's HTTP surface (``/health``, ``/job/<id>``).
* ``CHAIN_CANARY_LEDGER_FILE`` -- the live ledger file; its ``archive/`` rolls beside it are read too.
* ``CHAIN_CANARY_AGENT_LEDGER_HOST`` -- the host whose reading rows are read; the ledger host
  stamps ``via=bus:<host>`` on a row it took from that host. The agent URL and this host name
  the same agent.
* ``CHAIN_CANARY_WINDOW_SECONDS`` -- how far back the every-row assertions look (default 6 h).
* ``CHAIN_CANARY_LEDGER_IS_CURRENT`` -- ``1`` where the ledger file is the ledger of record.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml
from deploy_agent.events import EnumRuntimeLane
from pydantic import JsonValue

from omnibase_infra.handlers.handler_deploy_job_reading import HandlerDeployJobReading
from omnibase_infra.models.model_deploy_job_reading_request import (
    ModelDeployJobReadingRequest,
)
from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    pg16,
)

__all__ = [
    "ONE_TICK_SECONDS",
    "SETTLE_CEILING_SECONDS",
    "LiveDeployAgentChain",
    "chain_from_environment",
    "handler_failing_step",
    "live_deploy_agent_chain",
    "pg16",
    "reading_problems",
]

#: Read once at collection, from a variable of this module's own: the suite-wide fixture
#: points the shell's ledger variable at a scratch file before every test, so a fixture
#: that read that one at test time would read an empty ledger and prove nothing. The runner
#: sets this one to the ledger copy it wants read.
_LEDGER_FILE = os.environ.get("CHAIN_CANARY_LEDGER_FILE", "")
_AGENT_URL = os.environ.get("CHAIN_CANARY_DEPLOY_AGENT_URL", "")
#: Set to ``1`` where ``CHAIN_CANARY_LEDGER_FILE`` is the ledger of record itself. A lane's copy is a
#: snapshot, so a job that settled after the snapshot has no row in it whatever the agent did.
_LEDGER_IS_CURRENT = os.environ.get("CHAIN_CANARY_LEDGER_IS_CURRENT", "") == "1"
#: The host the agent runs on; the ledger host stamps ``via=bus:<host>`` on a row it took over
#: the bus from that host. A variable, because more than one host runs this agent.
_AGENT_LEDGER_HOST = os.environ.get("CHAIN_CANARY_AGENT_LEDGER_HOST", "")
_DEFAULT_WINDOW_SECONDS = 6 * 3600.0
_WINDOW_VARIABLE = "CHAIN_CANARY_WINDOW_SECONDS"

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_READING_SETTINGS = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["settings"]
#: The lanes the reading contract covers; its own excluded lanes are never read.
_COVERED_LANES = (EnumRuntimeLane.DEV.value, EnumRuntimeLane.STABILITY_TEST.value)
#: The actor the reading node's contract settings (``row_actor``) put on every row.
READING_ACTOR = "deploy-agent:job-reading"
#: One tick of the chain: the agent appends the row inside the job's own terminal
#: publish, so the row is due well inside this bound after the job completes.
ONE_TICK_SECONDS = 120.0
#: The longest a dev job may stay settling: the lab overlay settle and the repair build run after
#: ``completed_at`` is stamped and before the reading row is published (OMN-24716), so a correct
#: dev row is published up to this long after its job completed. The wait on ``/health`` gives up
#: at the same ceiling.
SETTLE_CEILING_SECONDS = 1800.0
_POLL_SECONDS = 5.0
_HTTP_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class LiveDeployAgentChain:
    """The live deploy agent's HTTP surface and the reading rows its host wrote."""

    ledger_path: Path | None
    agent_url: str
    ledger_is_current: bool
    agent_host: str = _AGENT_LEDGER_HOST
    window_seconds: float = _DEFAULT_WINDOW_SECONDS

    def health(self) -> dict[str, object]:
        """The agent's ``/health`` body; fails the test when the agent cannot be read."""
        assert self.agent_url, (
            "CHAIN_CANARY_DEPLOY_AGENT_URL names no deploy agent; this chain test reads the "
            "live agent and fails closed without one"
        )
        response = httpx.get(f"{self.agent_url}/health", timeout=_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
        assert isinstance(body, dict)
        return body

    def job(self, job_id: str) -> dict[str, object]:
        """The agent's own record of one job (``/job/<id>``)."""
        response = httpx.get(
            f"{self.agent_url}/job/{job_id}", timeout=_HTTP_TIMEOUT_SECONDS
        )
        assert response.status_code == 200, (
            f"the agent holds no record of job {job_id}: HTTP {response.status_code}"
        )
        body = response.json()
        assert isinstance(body, dict)
        return body

    def ledger_files(self) -> list[Path]:
        """The ledger's archive rolls, oldest first, then the live file.

        A roll moves rows out of the live file, so a reader of the live file alone loses a
        reading the moment the ledger rolls (OMN-24695).
        """
        assert self.ledger_path is not None and self.ledger_path.is_file(), (
            "CHAIN_CANARY_LEDGER_FILE names no readable ledger copy; this chain test reads the ledger "
            "and fails closed without one"
        )
        rolls = sorted(
            self.ledger_path.parent.glob(f"archive/{self.ledger_path.stem}_*-split.md")
        )
        return [*rolls, self.ledger_path]

    def rows(self) -> list[dict[str, str]]:
        """Every reading row the agent's host appended, oldest first, as its cells.

        Read from the archive rolls and the live file. The row's leading stamp is the
        ledger's own and rides as ``stamp``.
        """
        assert self.agent_host, (
            "CHAIN_CANARY_AGENT_LEDGER_HOST names no host; this chain test reads one agent's "
            "rows and fails closed without it"
        )
        found: list[dict[str, str]] = []
        for path in self.ledger_files():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "name=deploy-agent-job" not in line:
                    continue
                parts = [part.strip() for part in line.split(" | ")]
                cells = dict(part.split("=", 1) for part in parts[2:] if "=" in part)
                if (
                    cells.get("src") == "reading"
                    and cells.get("actor") == READING_ACTOR
                    and cells.get("via") == f"bus:{self.agent_host}"
                ):
                    found.append({"stamp": parts[0], **cells})
        return sorted(found, key=lambda row: row["stamp"])

    def window_rows(self, *, now: datetime | None = None) -> list[dict[str, str]]:
        """The rows whose job completed inside the window that ends now."""
        start = (now or datetime.now(UTC)) - timedelta(seconds=self.window_seconds)
        return [row for row in self.rows() if _moment(row["completed_at"]) >= start]

    def wait_while_settling(
        self,
        job_id: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        ceiling: timedelta = timedelta(seconds=SETTLE_CEILING_SECONDS),
    ) -> datetime:
        """Wait while the agent reports ``job_id`` settling; the moment it stopped.

        Where the lab overlay settle runs, ``completed_at`` is stamped before the settle and the
        reading row is written after it, so the one-tick bound runs from the end of the settle.
        """
        give_up = now() + ceiling
        while True:
            last = self.health().get("last_result")
            settling = (
                isinstance(last, dict)
                and str(last.get("correlation_id")) == job_id
                and last.get("settling") is True
            )
            if not settling:
                return now()
            assert now() < give_up, (
                f"job {job_id} is still settling {ceiling.total_seconds():.0f}s after the wait began"
            )
            sleep(_POLL_SECONDS)

    def wait_for_row(self, job_id: str, *, deadline: datetime) -> dict[str, str] | None:
        """The job's reading row, polling the ledger until ``deadline``; ``None`` if none came."""
        while True:
            for row in self.rows():
                if row.get("job") == job_id:
                    return row
            if datetime.now(UTC) >= deadline:
                return None
            time.sleep(_POLL_SECONDS)


def _moment(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def handler_failing_step(
    settings: Mapping[str, JsonValue], phase_results: Mapping[str, str]
) -> str:
    """The failing step the reading handler itself names for a failed job with these phases.

    The handler's own rule, not a copy of it: the first phase that failed, else the first that
    did not succeed (a lane-lock or host-slot job settles every phase skipped and reads
    ``preflight``), else ``unknown``.
    """
    handler = HandlerDeployJobReading(settings=settings, environ={})
    request = ModelDeployJobReadingRequest(
        correlation_id=uuid.uuid4(),
        runtime_lane="dev",
        scope="runtime",
        job_outcome="failed",
        phase_results=dict(phase_results),
        published_at=datetime.now(UTC),
    )
    return handler.failing_step(request)


def reading_problems(
    rows: list[dict[str, str]],
    *,
    bound: float = ONE_TICK_SECONDS,
    dev_bound: float = SETTLE_CEILING_SECONDS + ONE_TICK_SECONDS,
) -> list[str]:
    """What is wrong with each row, one line per defect; empty when every row is correct.

    Checks every row it is given, the shape the handler writes (its step vocabulary included:
    ``unknown`` and ``not-started-*`` are rows the handler emits) and that the row was
    published inside a bound of the job's completion: ``dev_bound`` for a dev-lane row, whose
    job may settle (overlay apply, repair build) between ``completed_at`` and the publish, and
    ``bound`` for every other lane, which never settles.
    """
    order = _READING_SETTINGS["phase_order"]
    problems: list[str] = []
    for row in rows:
        job = row.get("job", "?")
        failed = row.get("verdict") == "NOT_WORKING"
        if row.get("verdict") not in ("WORKING", "NOT_WORKING"):
            problems.append(f"job {job}: verdict {row.get('verdict')!r}")
        if row.get("key") != job:
            problems.append(f"job {job}: key {row.get('key')!r} is not the job id")
        if failed:
            step = row.get("failing_step", "")
            known = (
                step in order or step == "unknown" or step.startswith("not-started-")
            )
            if not known:
                problems.append(f"job {job}: failing_step {step!r} is no handler step")
            if row.get("reason") != f"job-failed:{step}":
                problems.append(
                    f"job {job}: reason {row.get('reason')!r} for step {step!r}"
                )
        elif row.get("reason") != "job-success" or "failing_step" in row:
            problems.append(f"job {job}: a WORKING row names a failure")
        if row.get("deploy_lane") not in _COVERED_LANES:
            problems.append(f"job {job}: deploy_lane {row.get('deploy_lane')!r}")
        if (
            row.get("piece") != _READING_SETTINGS["piece"]
            or row.get("name") != (_READING_SETTINGS["piece_name"])
        ):
            problems.append(f"job {job}: piece or name differs from the contract's")
        expected = uuid.uuid5(uuid.NAMESPACE_URL, f"onex:deploy-job-reading:{job}")
        if row.get("req") != str(expected):
            problems.append(f"job {job}: req is not the handler's derived id")
        lag = (
            _moment(row["published_at"]) - _moment(row["completed_at"])
        ).total_seconds()
        limit = (
            dev_bound if row.get("deploy_lane") == EnumRuntimeLane.DEV.value else bound
        )
        if not 0 <= lag <= limit:
            problems.append(
                f"job {job}: published {lag:.0f}s after completion, past the {limit:.0f}s bound"
            )
    return problems


def chain_from_environment(environ: Mapping[str, str]) -> LiveDeployAgentChain:
    """The live chain's read side from ``environ``; the window is parsed here, at test time.

    A window that is not a positive number fails the test that asked for the chain, never the
    collection of every chain-canary test.
    """
    raw = environ.get(_WINDOW_VARIABLE, "")
    try:
        window = float(raw) if raw else _DEFAULT_WINDOW_SECONDS
    except ValueError:
        window = 0.0
    assert window > 0, (
        f"{_WINDOW_VARIABLE} must be a positive number of seconds, got {raw!r}"
    )
    return LiveDeployAgentChain(
        ledger_path=Path(_LEDGER_FILE) if _LEDGER_FILE else None,
        agent_url=_AGENT_URL.rstrip("/"),
        ledger_is_current=_LEDGER_IS_CURRENT,
        window_seconds=window,
    )


@pytest.fixture
def live_deploy_agent_chain() -> LiveDeployAgentChain:
    """The live chain's read side, from the variables as the shell exported them."""
    return chain_from_environment(os.environ)

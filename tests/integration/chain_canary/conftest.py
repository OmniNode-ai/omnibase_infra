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
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    pg16,
)

__all__ = ["LiveDeployAgentChain", "live_deploy_agent_chain", "pg16"]

#: Read once at collection, from a variable of this module's own: the suite-wide fixture
#: points the shell's ledger variable at a scratch file before every test, so a fixture
#: that read that one at test time would read an empty ledger and prove nothing. The runner
#: sets this one to the ledger copy it wants read.
_LEDGER_FILE = os.environ.get("CHAIN_CANARY_LEDGER_FILE", "")
_AGENT_URL = os.environ.get("CHAIN_CANARY_DEPLOY_AGENT_URL", "")
#: Set to ``1`` where ``CHAIN_CANARY_LEDGER_FILE`` is the ledger of record itself. A lane's copy is a
#: snapshot, so a job that settled after the snapshot has no row in it whatever the agent did.
_LEDGER_IS_CURRENT = os.environ.get("CHAIN_CANARY_LEDGER_IS_CURRENT", "") == "1"

#: The lab host the dev deploy agent runs on; the ledger host stamps ``via=bus:<host>``
#: on a row it took over the bus from that host.
AGENT_LEDGER_HOST = "h201"
#: The actor the reading node's contract settings (``row_actor``) put on every row.
READING_ACTOR = "deploy-agent:job-reading"
#: One tick of the chain: the agent appends the row inside the job's own terminal
#: publish, so the row is due well inside this bound after the job completes.
ONE_TICK_SECONDS = 120.0
_POLL_SECONDS = 5.0
_HTTP_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class LiveDeployAgentChain:
    """The live deploy agent's HTTP surface and the reading rows its host wrote."""

    ledger_path: Path | None
    agent_url: str
    ledger_is_current: bool

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

    def rows(self) -> list[dict[str, str]]:
        """Every reading row the agent's host appended, oldest first, as its cells.

        The row's leading stamp is the ledger's own and rides as ``stamp``.
        """
        assert self.ledger_path is not None and self.ledger_path.is_file(), (
            "CHAIN_CANARY_LEDGER_FILE names no readable ledger copy; this chain test reads the ledger "
            "and fails closed without one"
        )
        found: list[dict[str, str]] = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if "name=deploy-agent-job" not in line:
                continue
            parts = [part.strip() for part in line.split(" | ")]
            cells = dict(part.split("=", 1) for part in parts[2:] if "=" in part)
            if (
                cells.get("src") == "reading"
                and cells.get("actor") == READING_ACTOR
                and cells.get("via") == f"bus:{AGENT_LEDGER_HOST}"
            ):
                found.append({"stamp": parts[0], **cells})
        return found

    def wait_for_row(self, job_id: str, *, deadline: datetime) -> dict[str, str] | None:
        """The job's reading row, polling the ledger until ``deadline``; ``None`` if none came."""
        while True:
            for row in self.rows():
                if row.get("job") == job_id:
                    return row
            if datetime.now(UTC) >= deadline:
                return None
            time.sleep(_POLL_SECONDS)


@pytest.fixture
def live_deploy_agent_chain() -> LiveDeployAgentChain:
    """The live chain's read side, from the variables as the shell exported them."""
    return LiveDeployAgentChain(
        ledger_path=Path(_LEDGER_FILE) if _LEDGER_FILE else None,
        agent_url=_AGENT_URL.rstrip("/"),
        ledger_is_current=_LEDGER_IS_CURRENT,
    )

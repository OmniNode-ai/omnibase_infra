# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared fixtures for the OMN-19233 staging-delivery lane-selection tests.

A fake GitHub REST surface keyed exactly as the real one is: artifacts by exact
name, artifact zips by id, workflow runs by workflow file, jobs by run attempt.
Every read the gate and the re-run selector make goes through
``scripts.ci.lab_pass_receipt._gh_api``, so this is the one seam they are
tested through.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    ModelLabPassCheck,
    ModelLabPassReceipt,
    artifact_name,
    build_receipt,
)

REPO = "OmniNode-ai/omnibase_infra"

#: The two measured shas the plan's PS-5 replays name (section 0b, PS-3).
SHA_4ACA = "4aca83d9373e823c55792253f9e943b1b19f0294"
SHA_3E4A = "3e4aaded3081654053177df79db91f897b880c3a"
#: df6ffe02 was the dev head whose PS-1 subject was 3e4aaded.
SHA_DF6F = "df6ffe0261e554c4442fbee7c34713993ac50c9d"

GATE_JOB = "Require a passing lab-pass receipt for this sha"


def ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)


def receipt(
    sha: str,
    lane: EnumLabLane = EnumLabLane.COMPOSE_DEV,
    *,
    outcome: str = "ok",
) -> ModelLabPassReceipt:
    """A receipt whose one probe check is ``ok``, ``fail`` or ``indeterminate``."""
    checks: list[ModelLabPassCheck] = [
        ModelLabPassCheck(name="ready_main", ok=True, evidence="GET /ready -> 200")
    ]
    if outcome == "fail":
        checks.append(
            ModelLabPassCheck(
                name="health_dimensions",
                ok=False,
                evidence="GET /health -> 200, unhealthy: ['consumer_coverage']",
            )
        )
    elif outcome == "indeterminate":
        checks.append(
            ModelLabPassCheck.indeterminate_check(
                name="deployed_revision",
                evidence="INDETERMINATE: queued behind 4 deploy commands",
            )
        )
    return build_receipt(
        sha=sha,
        lane=lane,
        started_at=datetime(2026, 9, 22, 19, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 22, 19, 47, 4, tzinfo=UTC),
        checks=checks,
        agent_command_id=None,
    )


def zip_of(member: str, body: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(member, body)
    return buf.getvalue()


@dataclass
class FakeSurface:
    """A mutable fake of the REST surface. Tests add receipts mid-scenario."""

    receipts: list[ModelLabPassReceipt] = field(default_factory=list)
    #: artifact name -> (member name, JSON body)
    extra_artifacts: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: workflow file -> list of run dicts as the REST API returns them
    runs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: (run_id, attempt) -> list of job dicts
    jobs: dict[tuple[int, int], list[dict[str, Any]]] = field(default_factory=dict)
    reads: list[str] = field(default_factory=list)
    fail_paths: Sequence[str] = ()

    def add(self, *items: ModelLabPassReceipt) -> None:
        self.receipts.extend(items)

    def _artifacts(self) -> dict[str, tuple[int, str, str]]:
        table: dict[str, tuple[int, str, str]] = {}
        for i, r in enumerate(self.receipts):
            table[artifact_name(r.lane, r.sha)] = (
                1000 + i,
                "receipt.json",
                r.to_json(),
            )
        for j, (name, (member, body)) in enumerate(
            sorted(self.extra_artifacts.items())
        ):
            table[name] = (5000 + j, member, body)
        return table

    def __call__(self, path: str) -> bytes:
        self.reads.append(path)
        for bad in self.fail_paths:
            if bad in path:
                msg = f"`gh api {path}` exited 1: HTTP 502"
                raise RuntimeError(msg)
        table = self._artifacts()
        if "/actions/artifacts?name=" in path:
            name = path.split("name=")[1].split("&")[0]
            if name not in table:
                return json.dumps({"artifacts": []}).encode()
            artifact_id = table[name][0]
            return json.dumps(
                {
                    "artifacts": [
                        {
                            "id": artifact_id,
                            "name": name,
                            "expired": False,
                            "created_at": "2026-09-22T19:47:06Z",
                        }
                    ]
                }
            ).encode()
        if "/actions/artifacts/" in path and path.endswith("/zip"):
            artifact_id = int(path.split("/artifacts/")[1].split("/")[0])
            for aid, member, body in table.values():
                if aid == artifact_id:
                    return zip_of(member, body)
            msg = f"no artifact {artifact_id}"
            raise RuntimeError(msg)
        if "/actions/workflows/" in path and "/runs" in path:
            workflow = path.split("/actions/workflows/")[1].split("/")[0]
            return json.dumps({"workflow_runs": self.runs.get(workflow, [])}).encode()
        if "/attempts/" in path and "/jobs" in path:
            run_id = int(path.split("/actions/runs/")[1].split("/")[0])
            attempt = int(path.split("/attempts/")[1].split("/")[0])
            return json.dumps({"jobs": self.jobs.get((run_id, attempt), [])}).encode()
        msg = f"unexpected read {path}"
        raise AssertionError(msg)

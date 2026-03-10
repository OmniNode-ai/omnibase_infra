# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""compare-environments.py — local Docker vs k8s environment parity checker.

Usage:
    uv run python scripts/compare-environments.py [--mode check|fix] [--checks CHECKS]
    uv run python scripts/compare-environments.py --all-checks
    uv run python scripts/compare-environments.py --json

Checks (default: credential,ecr,infisical):
    credential   CRITICAL  Service secret POSTGRES_USER/PASSWORD vs onex-runtime-credentials
    ecr          CRITICAL  Deployment image tags still exist in ECR
    infisical    CRITICAL  InfisicalSecret paths exist in Infisical project
    schema       WARNING   DB migration_history latest id matches local vs cloud
    services     WARNING   Deployments present in local Docker vs cloud k8s
    flags        WARNING   Feature flag env vars consistent
    kafka        WARNING   Kafka topic sets match on both buses
    packages     WARNING   omnibase-core/spi/infra package versions match
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelParityFinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    check_id: str
    severity: Literal["CRITICAL", "WARNING", "INFO"]
    title: str
    detail: str
    local_value: str | None = None
    cloud_value: str | None = None
    auto_fixable: bool
    fix_hint: str


class ModelParitySummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    critical_count: int
    warning_count: int
    info_count: int
    checks_skipped: list[str]


class ModelParityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str
    generated_at: str
    mode: str
    checks_run: list[str]
    findings: list[ModelParityFinding]
    summary: ModelParitySummary


# ---------------------------------------------------------------------------
# transport — SsmRunner, SsmResult
# ---------------------------------------------------------------------------


@dataclass
class SsmResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False
    skip_reason: str = ""


class SsmRunner:
    def __init__(self, instance_id: str, region: str, timeout: int = 90) -> None:
        self.instance_id = instance_id
        self.region = region
        self.timeout = timeout

    def run(self, command: str) -> SsmResult:
        if not shutil.which("aws"):
            return SsmResult(
                skipped=True, skip_reason="aws CLI not found — install awscli"
            )
        try:
            send = subprocess.run(
                [
                    "aws",
                    "ssm",
                    "send-command",
                    "--instance-ids",
                    self.instance_id,
                    "--region",
                    self.region,
                    "--document-name",
                    "AWS-RunShellScript",
                    "--parameters",
                    f'commands=["{command}"]',
                    "--output",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:
            return SsmResult(skipped=True, skip_reason=f"send-command failed: {exc}")
        if send.returncode != 0:
            if (
                "ExpiredTokenException" in send.stderr
                or "ExpiredTokenException" in send.stdout
            ):
                return SsmResult(
                    skipped=True, skip_reason="SSO session expired — run: aws sso login"
                )
            return SsmResult(
                skipped=True, skip_reason=f"send-command error: {send.stderr[:200]}"
            )
        try:
            command_id = json.loads(send.stdout)["Command"]["CommandId"]
        except Exception as exc:
            return SsmResult(
                skipped=True, skip_reason=f"could not parse CommandId: {exc}"
            )
        for _ in range(self.timeout // 2):
            time.sleep(2)
            try:
                poll = subprocess.run(
                    [
                        "aws",
                        "ssm",
                        "get-command-invocation",
                        "--command-id",
                        command_id,
                        "--instance-id",
                        self.instance_id,
                        "--region",
                        self.region,
                        "--output",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if poll.returncode != 0:
                    continue
                inv = json.loads(poll.stdout)
                if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
                    return SsmResult(
                        returncode=0 if inv["Status"] == "Success" else 1,
                        stdout=inv.get("StandardOutputContent", ""),
                        stderr=inv.get("StandardErrorContent", ""),
                    )
            except Exception:
                continue
        return SsmResult(
            skipped=True, skip_reason=f"instance unreachable after {self.timeout}s"
        )


# ---------------------------------------------------------------------------
# checks registry
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    "credential",
    "ecr",
    "infisical",
    "schema",
    "services",
    "flags",
    "kafka",
    "packages",
]
DEFAULT_CHECKS = ["credential", "ecr", "infisical"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local Docker vs k8s parity checker")
    p.add_argument("--mode", choices=["check", "fix"], default="check")
    p.add_argument("--checks", default=",".join(DEFAULT_CHECKS))
    p.add_argument("--all-checks", action="store_true")
    p.add_argument("--namespace", default="onex-dev")
    p.add_argument("--instance-id", default="i-0e596e8b557e27785")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--json", dest="json_output", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=int, default=90)
    return p


def main() -> None:
    args = build_parser().parse_args()
    checks = (
        ALL_CHECKS if args.all_checks else [c.strip() for c in args.checks.split(",")]
    )
    report = ModelParityReport(
        run_id=str(uuid.uuid4())[:8],
        generated_at=datetime.now(tz=UTC).isoformat(),
        mode=args.mode,
        checks_run=checks,
        findings=[],
        summary=ModelParitySummary(
            critical_count=0, warning_count=0, info_count=0, checks_skipped=checks
        ),
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

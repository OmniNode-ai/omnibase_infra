#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Check delegation verdicts for runtime PRs and enforce shadow rollout age."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from typing import TextIO

import yaml
from pydantic import BaseModel, ConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import runtime_change_classifier
from scripts.ci import lab_pass_receipt

FIX_FORWARD_LABEL = "delegation-fix-forward"
CHECK_JOB_NAME = "Delegation Health Check (shadow)"


class ModelVerdictSource(BaseModel):
    """A recurring workflow measurement read by the shared verdict reader."""

    model_config = ConfigDict(extra="forbid")

    name: str
    repo: str
    workflow: str
    branch: str
    max_age_hours: float
    events: tuple[str, ...]
    dispatch_title_contains: str = ""


class ModelRepoRollout(BaseModel):
    """The start of shadow observation and required-check state for a repo."""

    model_config = ConfigDict(extra="forbid")

    shadow_started_at: date
    required: bool


class ModelHealthConfig(BaseModel):
    """Configured verdict sources and mutable per-repository rollout state."""

    model_config = ConfigDict(extra="forbid")

    shadow_days_required: int
    sources: tuple[ModelVerdictSource, ...]
    repos: dict[str, ModelRepoRollout]


class ModelFixForwardReading(BaseModel):
    """Valid fix-forward tickets and verbatim malformed fix-forward labels."""

    model_config = ConfigDict(extra="forbid")

    tickets: tuple[str, ...]
    refused: tuple[str, ...]


def load_config(path: Path) -> ModelHealthConfig:
    """Read YAML safely and validate its configuration contract."""
    with path.open(encoding="utf-8") as stream:
        return ModelHealthConfig.model_validate(yaml.safe_load(stream))


def parse_fix_forward_labels(labels: Sequence[str]) -> ModelFixForwardReading:
    """Accept only exact, ticket-qualified fix-forward labels."""
    tickets: list[str] = []
    refused: list[str] = []
    prefix = f"{FIX_FORWARD_LABEL}:"
    for label in labels:
        if label == FIX_FORWARD_LABEL:
            refused.append(label)
        elif label.startswith(prefix):
            ticket = label[len(prefix) :]
            if re.fullmatch(r"OMN-\d+", ticket):
                tickets.append(ticket)
            else:
                refused.append(label)
    return ModelFixForwardReading(tickets=tuple(tickets), refused=tuple(refused))


def evaluate_delegation_health(
    sources: Sequence[ModelVerdictSource],
    *,
    runtime_affecting: bool,
    labels: Sequence[str],
    out: TextIO,
    now: datetime,
    record: dict[str, object],
) -> int:
    """Collect every red verdict, admitting only ticketed fix-forward repairs."""
    red_runs: list[dict[str, str]] = []
    record["red_runs"] = red_runs
    record["admitted_by"] = []
    if not runtime_affecting:
        print("Delegation health: not runtime-affecting; no verdicts read.", file=out)
        return 0

    for source in sources:
        detail = StringIO()
        run_id = "unreadable"
        try:
            code = lab_pass_receipt.evaluate_workflow_verdict(
                source.repo,
                source.workflow,
                source.branch,
                source.max_age_hours,
                source.events,
                detail,
                now=now,
                dispatch_title_contains=source.dispatch_title_contains,
            )
        except Exception as exc:  # noqa: BLE001 - unreadable sources fail closed
            code = 1
            print(f"unreadable: {exc}", file=detail)
        else:
            match = re.search(r"newest run\s*:\s*(\d+)", detail.getvalue())
            if match:
                run_id = match.group(1)
        state = "red" if code else "green"
        print(f"{source.name}: {state}, run {run_id}", file=out)
        print(detail.getvalue().rstrip(), file=out)
        if code:
            red_runs.append({"source": source.name, "run_id": run_id})

    if not red_runs:
        print("Delegation health: all sources green.", file=out)
        return 0

    reds = ", ".join(f"{run['source']} run {run['run_id']}" for run in red_runs)
    reading = parse_fix_forward_labels(labels)
    if reading.refused:
        print(
            "::error::Fix-forward requires an OMN-<digits> ticket; refused labels: "
            f"{', '.join(reading.refused)}. Red verdicts: {reds}",
            file=out,
        )
        return 1
    if reading.tickets:
        record["admitted_by"] = list(reading.tickets)
        print(
            f"Delegation health: admitted by fix-forward ticket(s) "
            f"{', '.join(reading.tickets)} despite red verdicts: {reds}",
            file=out,
        )
        return 0
    print(f"::error::Delegation health red verdicts: {reds}", file=out)
    return 1


def validate_rollout(cfg: ModelHealthConfig, *, now: datetime) -> list[str]:
    """Refuse required checks before the configured shadow period has elapsed."""
    errors: list[str] = []
    for repo, rollout in cfg.repos.items():
        days = (now.date() - rollout.shadow_started_at).days
        if rollout.required and days < cfg.shadow_days_required:
            errors.append(
                f"{repo}: only {days} shadow days elapsed; "
                f"{cfg.shadow_days_required} days required before making the check required."
            )
    return errors


def _comma_separated(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """Classify the PR, validate rollout, and persist the check's decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "config/delegation_health_check.yaml",
    )
    parser.add_argument(
        "--repo-key", choices=("omnibase_infra", "omnimarket"), required=True
    )
    parser.add_argument("--changed-files", default="")
    parser.add_argument("--labels", default="")
    parser.add_argument("--runtime-validator", type=Path, required=True)
    parser.add_argument("--record-out", type=Path, required=True)
    args = parser.parse_args(argv)
    now = datetime.now(UTC)
    record: dict[str, object] = {
        "repo": args.repo_key,
        "checked_at": now.isoformat(),
        "red_runs": [],
        "admitted_by": [],
    }
    code = 1
    errors: list[str] = []
    try:
        cfg = load_config(args.config)
        errors = validate_rollout(cfg, now=now)
        if not errors:
            labels = _comma_separated(args.labels)
            classifier = runtime_change_classifier.load_runtime_path_classifier(
                args.runtime_validator
            )
            runtime_paths = runtime_change_classifier.classify_runtime_paths(
                _comma_separated(args.changed_files),
                classifier,
                source_repo=args.repo_key,
            )
            runtime_affecting = runtime_change_classifier.is_runtime_affecting(
                runtime_paths, labels
            )
            record["runtime_affecting"] = runtime_affecting
            code = evaluate_delegation_health(
                cfg.sources,
                runtime_affecting=runtime_affecting,
                labels=labels,
                out=sys.stdout,
                now=now,
                record=record,
            )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(str(exc))
    for error in errors:
        print(f"::error::{error}")
    record["errors"] = errors
    record["exit_code"] = code
    args.record_out.parent.mkdir(parents=True, exist_ok=True)
    args.record_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            print(f"\n### {CHECK_JOB_NAME}\n", file=summary)
            print(
                f"Result: {'PASS' if code == 0 else 'FAIL'} ({args.repo_key}).",
                file=summary,
            )
            print(f"\nRed runs: {json.dumps(record['red_runs'])}", file=summary)
            print(f"\nAdmitted by: {json.dumps(record['admitted_by'])}", file=summary)
            for error in errors:
                print(f"\n- {error}", file=summary)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

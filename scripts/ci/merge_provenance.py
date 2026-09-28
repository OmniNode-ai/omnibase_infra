# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI boundary for merge provenance (OMN-19927, plan task S2).

Runs ``node_merge_provenance_observe_effect`` and then
``node_merge_provenance_compute`` for one pushed commit and writes the result.
This module is the I/O boundary only: it binds the effect's GitHub reader,
calls the two handlers, and writes ``verdict`` and ``force_full_suite`` to the
step's ``$GITHUB_OUTPUT`` plus the full result as JSON. Every decision is the
compute handler's.

WHY NOT ``onex run-node``. That command dispatches a packaged node to a remote
runtime over Kafka, and a CI job has no bus. The same handlers execute either
way; the precedent and its measurement are ``scripts/ci/runner_route_decision.py``
(OMN-18412), which calls ``node_ci_runner_route_compute``'s handler the same way.

FAIL CLOSED. A missing token or any failed read is an observation with
``read_ok=False``, which the compute node grades UNDECIDABLE, which forces the
full suite. If this module cannot run at all it writes nothing, and the
workflow's selection step treats anything other than an explicit
``force_full_suite=false`` as forced.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omnibase_infra.nodes.node_merge_provenance_compute.handlers.handler_merge_provenance import (
    HandlerMergeProvenance,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.model_merge_provenance_request import (
    ModelMergeProvenanceRequest,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.model_merge_provenance_result import (
    ModelMergeProvenanceResult,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.handlers.handler_merge_group_run_read_github import (
    HandlerMergeGroupRunReadGithub,
    MergeGroupReadError,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.handlers.handler_merge_provenance_observe import (
    HandlerMergeProvenanceObserve,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_provenance_observe_request import (
    ModelMergeProvenanceObserveRequest,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_workflow_job_fact import (
    ModelWorkflowJobFact,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_workflow_run_fact import (
    ModelWorkflowRunFact,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.protocols.protocol_merge_group_run_reader import (
    ProtocolMergeGroupRunReader,
)


class _UnavailableReader:
    """A reader that fails every call, used when no token was supplied."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def list_merge_group_runs(
        self, repository: str, head_sha: str
    ) -> list[ModelWorkflowRunFact]:
        raise MergeGroupReadError(self._reason)

    def list_latest_attempt_jobs(
        self, repository: str, run_id: int
    ) -> list[ModelWorkflowJobFact]:
        raise MergeGroupReadError(self._reason)


def evaluate(
    repository: str,
    sha: str,
    reader: ProtocolMergeGroupRunReader,
    workflow_path: str = ".github/workflows/ci.yml",
    summary_job: str = "CI Summary",
) -> ModelMergeProvenanceResult:
    """Observe through ``reader``, then grade. No I/O besides the reader."""
    observation = asyncio.run(
        HandlerMergeProvenanceObserve(reader=reader).handle(
            ModelMergeProvenanceObserveRequest(
                repository=repository,
                sha=sha,
                workflow_path=workflow_path,
                summary_job=summary_job,
            )
        )
    )
    return HandlerMergeProvenance().handle(
        ModelMergeProvenanceRequest(
            repository=repository, sha=sha, observation=observation
        )
    )


def evaluate_and_write(
    repository: str,
    sha: str,
    reader: ProtocolMergeGroupRunReader,
    github_output: Path | None,
    result_json: Path | None,
) -> ModelMergeProvenanceResult:
    result = evaluate(repository, sha, reader)
    if result_json is not None:
        result_json.write_text(result.model_dump_json(indent=2) + "\n")
    lines = [
        f"verdict={result.verdict.value}",
        f"reason={result.reason.value}",
        f"force_full_suite={'true' if result.forces_full_suite else 'false'}",
    ]
    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", required=True, help="owner/name")
    parser.add_argument("--sha", required=True, help="full commit sha")
    parser.add_argument("--github-output", type=Path, default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    args = parser.parse_args(argv)

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    reader: ProtocolMergeGroupRunReader = (
        HandlerMergeGroupRunReadGithub(token=token)
        if token
        else _UnavailableReader("no GH_TOKEN or GITHUB_TOKEN in the environment")
    )
    evaluate_and_write(
        repository=args.repository,
        sha=args.sha,
        reader=reader,
        github_output=args.github_output,
        result_json=args.result_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

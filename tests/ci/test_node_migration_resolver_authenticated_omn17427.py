# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: every step that runs the node-migration source resolver is authenticated.

``scripts/resolve_node_migration_source_ref.py`` reads the paired omnimarket pull
request (and, since omnibase_infra#4323, a compare of the declared SHA against its
live head) through ``scripts/ci/check_pin_reachability._api_get``. That helper
attaches ``GH_TOKEN``/``GITHUB_TOKEN`` only when the environment carries one.
Neither workflow step passed a token, so every call was anonymous, capped at 60
requests an hour per runner IP, which hosted runners share.

On 2026-09-30 the merge-group run 36663905835 for omnibase_infra#4304 failed
``Application Database Domain Enforcement (OMN-15361)`` with ``could not prove
node-migration source PR is open (fail-closed): omnimarket#3079; HTTP 403: API
rate limit exceeded``, and the queue ejected the PR. The resolver fails closed,
which is correct, so the fix is to give it the token it already knows how to use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESOLVER = "scripts/resolve_node_migration_source_ref.py"
_WORKFLOWS = (
    _REPO_ROOT / ".github/workflows/ci.yml",
    _REPO_ROOT / ".github/workflows/node-migration-sync.yml",
)


def _resolver_steps() -> list[tuple[str, str, dict[str, Any]]]:
    found: list[tuple[str, str, dict[str, Any]]] = []
    for workflow in _WORKFLOWS:
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        for job_name, job in (document.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if _RESOLVER in str(step.get("run", "")):
                    found.append((workflow.name, job_name, step))
    return found


def test_the_resolver_runs_in_both_known_workflows() -> None:
    workflows = {name for name, _, _ in _resolver_steps()}
    assert workflows == {"ci.yml", "node-migration-sync.yml"}


@pytest.mark.parametrize(
    ("workflow", "job", "step"),
    _resolver_steps(),
    ids=lambda value: value if isinstance(value, str) else "step",
)
def test_every_resolver_step_passes_a_github_token(
    workflow: str, job: str, step: dict[str, Any]
) -> None:
    env = step.get("env") or {}
    token = str(env.get("GH_TOKEN", ""))
    assert "github.token" in token, (
        f"{workflow} job {job!r} runs {_RESOLVER} without GH_TOKEN, so its GitHub "
        "API reads are anonymous and fail closed on the shared 60/hour limit"
    )

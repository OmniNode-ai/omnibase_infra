# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17135 defect 1, executor half: the requested SHA pins omnibase_infra, and
the siblings get a resolvable ref of their own.

``ModelRedeployStartCommandWire.validate_git_ref`` constrains the CI-published
``git_ref`` to a 7-64 character lowercase hex commit SHA -- of omnibase_infra,
the repo whose merge fired the trigger. ``_compose_build`` passed that SHA
straight into ``stage_workspace.sh`` as ``DEPLOY_REF``, where RT-1 tried to
check omnibase_core / omnibase_compat / omnimarket out at it. No such commit
exists in any of them, so every CI-triggered rebuild failed in the CORE phase
seconds after acceptance (job ``a5b200d5``, 2026-09-09T21:02Z).

The executor now names both halves: the requested ref stays the infra pin, and
the declared tracking head (``DEPLOY_AGENT_TRACKING_REF`` -> ``origin/<branch>``)
is exported as the sibling fallback. It also reads back the per-repo refs RT-1
resolved, so the terminal event states which sibling commit was actually
vendored instead of leaving it to be inferred from the image.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildCompleted,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobState
from deploy_agent.publisher import build_completion_payload

pytestmark = [pytest.mark.unit, pytest.mark.promotion_guard]

# A real-shaped omnibase_infra merge SHA: exactly what the trigger publishes.
INFRA_SHA = "cc8974408d6c543311a40b8beb414787a80fef2e"


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _ok() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _install_recording_stage_script(repo: Path, *, resolved: dict[str, str]) -> Path:
    """Stand-in stage_workspace.sh: records its env and emits an expected-refs
    manifest at DEPLOY_SOURCE_REFS_OUT, exactly as the real RT-1 checkout does."""
    script_dir = repo / "scripts" / "runtime_build"
    script_dir.mkdir(parents=True, exist_ok=True)
    record = repo / "staging-env.json"
    payload = json.dumps(resolved)
    script = script_dir / "stage_workspace.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "python3 - <<'PY'\n"
        "import json, os\n"
        "keys = ('DEPLOY_REF', 'DEPLOY_SIBLING_FALLBACK_REF', 'OMNI_HOME',\n"
        "        'DEPLOY_SOURCE_REFS_OUT', 'ALLOW_UNPINNED_DEPLOY_SOURCE')\n"
        f"json.dump({{k: os.environ.get(k) for k in keys}}, open({str(record)!r}, 'w'))\n"
        f"resolved = json.loads({payload!r})\n"
        "out = os.environ.get('DEPLOY_SOURCE_REFS_OUT')\n"
        "if out:\n"
        "    os.makedirs(os.path.dirname(out), exist_ok=True)\n"
        "    json.dump({'ref_pinned': True, 'repos': {\n"
        "        repo: {'ref': 'origin/dev', 'expected_sha': sha,\n"
        "               'fallback_from': os.environ.get('DEPLOY_REF', '')}\n"
        "        for repo, sha in resolved.items()}}, open(out, 'w'))\n"
        "PY\n"
    )
    script.chmod(0o755)
    return record


# ---------------------------------------------------------------------------
# (a) an infra SHA never becomes a sibling ref
# ---------------------------------------------------------------------------
def test_sha_form_ref_pins_infra_and_gives_siblings_their_own_tracking_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    record = _install_recording_stage_script(repo, resolved={})
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(repo))
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "omni_home"))
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.delenv("ALLOW_UNPINNED_DEPLOY_SOURCE", raising=False)
    monkeypatch.setattr(executor_mod, "_run", lambda cmd, timeout, **kwargs: _ok())

    DeployExecutor()._compose_build(
        Scope.RUNTIME,
        INFRA_SHA[:7],
        _noop_phase_update,
        build_source=BuildSource.WORKSPACE,
        runtime_lane=EnumRuntimeLane.DEV,
        git_ref=INFRA_SHA,
    )

    staged = json.loads(record.read_text())
    # The infra pin is unchanged: the requested SHA is still what the infra
    # clone is asserted against.
    assert staged["DEPLOY_REF"] == INFRA_SHA
    # ...and the siblings are handed a ref that exists in THEIR history.
    assert staged["DEPLOY_SIBLING_FALLBACK_REF"] == "origin/dev"
    # The OMN-17291 guard is satisfied by supplying pins, never opted out of.
    assert staged["ALLOW_UNPINNED_DEPLOY_SOURCE"] is None


# ---------------------------------------------------------------------------
# (c) a symbolic git_ref is unchanged
# ---------------------------------------------------------------------------
def test_symbolic_ref_still_reaches_staging_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    record = _install_recording_stage_script(repo, resolved={})
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(repo))
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "omni_home"))
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setattr(executor_mod, "_run", lambda cmd, timeout, **kwargs: _ok())

    DeployExecutor()._compose_build(
        Scope.RUNTIME,
        "abc1234",
        _noop_phase_update,
        build_source=BuildSource.WORKSPACE,
        runtime_lane=EnumRuntimeLane.DEV,
        git_ref="origin/dev",
    )

    staged = json.loads(record.read_text())
    assert staged["DEPLOY_REF"] == "origin/dev"


# ---------------------------------------------------------------------------
# (d) the per-repo resolved SHAs are read back and carried
# ---------------------------------------------------------------------------
def test_compose_build_records_the_per_repo_resolved_shas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = {
        "omnibase_core": "1111111111111111111111111111111111111111",
        "omnibase_compat": "2222222222222222222222222222222222222222",
        "omnimarket": "3333333333333333333333333333333333333333",
    }
    repo = tmp_path / "repo"
    repo.mkdir()
    _install_recording_stage_script(repo, resolved=resolved)
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(repo))
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "omni_home"))
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setattr(executor_mod, "_run", lambda cmd, timeout, **kwargs: _ok())

    executor = DeployExecutor()
    executor._compose_build(
        Scope.RUNTIME,
        INFRA_SHA[:7],
        _noop_phase_update,
        build_source=BuildSource.WORKSPACE,
        runtime_lane=EnumRuntimeLane.DEV,
        git_ref=INFRA_SHA,
    )

    assert executor.sibling_source_refs == resolved


def test_completion_event_carries_the_sibling_refs() -> None:
    resolved = {"omnibase_core": "1111111111111111111111111111111111111111"}
    now = datetime.now(UTC)
    job = JobState(
        correlation_id=uuid4(),
        command={
            "git_ref": INFRA_SHA,
            "scope": "runtime",
            "runtime_lane": EnumRuntimeLane.DEV.value,
        },
        accepted_at=now,
        completed_at=now,
        status="success",
        phase_results={Phase.RUNTIME: PhaseStatus.SUCCESS},
    )

    payload = build_completion_payload(
        job,
        INFRA_SHA,
        [],
        services_restarted=["omninode-runtime"],
        sibling_refs=resolved,
    )

    assert payload["sibling_refs"] == resolved
    # The requested ref stays the infra pin; the sibling SHAs are EVIDENCE
    # beside it, never the key (rule 24's lab-pass receipt is keyed by the
    # infra sha and must stay that way).
    assert payload["requested_git_ref"] == INFRA_SHA
    # The field is declared on the model the payload is validated through, so a
    # consumer sees a typed map and not a free-form dict smuggled past it.
    assert "sibling_refs" in ModelRebuildCompleted.model_fields
    # A release-mode / prod-digest deploy vendors no sibling trees and says so
    # with an empty map rather than omitting the field.
    assert build_completion_payload(job, INFRA_SHA, [])["sibling_refs"] == {}

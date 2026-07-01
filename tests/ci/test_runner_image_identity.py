# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coverage for the versioned runner image contract (OMN-12567).

The runner image version is a *binding*, not a human label. The binding must
fold image digest inputs, the dependency manifest, the Python version, the uv
version, and the shared CI env (canary) version into one reproducible identity.
Every CI job must emit that identity in its startup evidence, and the image
version must bump on release while mutating jobs opt out explicitly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
IDENTITY_SCRIPT = SCRIPTS_CI / "runner_image_identity.py"
LOCK_FILE = REPO_ROOT / "docker" / "runners" / "runner-image.lock.json"
RUNNER_DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
EMIT_ACTION = REPO_ROOT / ".github" / "actions" / "emit-runner-identity" / "action.yml"
SETUP_ACTION = REPO_ROOT / ".github" / "actions" / "setup-python-uv" / "action.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
CANARY_DOC = REPO_ROOT / "docs" / "ci" / "versioned-ci-env-canary.md"


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _load_identity_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "runner_image_identity", IDENTITY_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # ci_env_digest lives next to the identity module; ensure it resolves.
    sys.path.insert(0, str(SCRIPTS_CI))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(SCRIPTS_CI) in sys.path:
            sys.path.remove(str(SCRIPTS_CI))
    return module


# --------------------------------------------------------------------------- #
# Binding identity                                                            #
# --------------------------------------------------------------------------- #


def test_lock_file_exists_and_is_well_formed() -> None:
    assert LOCK_FILE.exists(), "runner-image.lock.json must exist"
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))

    # Version identity is a binding, not a label: every component is present.
    assert isinstance(data["image_version"], int)
    assert data["image_version"] >= 1
    assert data["base_image_digest"].startswith("sha256:")
    assert data["python_version"]
    assert data["uv_version"]
    assert data["runner_version"]
    assert isinstance(data["shared_env_install_args"], str)
    # The bound identity digest folds every component above.
    assert isinstance(data["identity_digest"], str)
    assert len(data["identity_digest"]) >= 24


def test_identity_digest_binds_all_required_components() -> None:
    module = _load_identity_module()
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))

    base = module.compute_identity(REPO_ROOT, data)

    # Flipping any single binding component changes the digest. A label model
    # would leave the digest stable; a binding model must not.
    for field, mutated in (
        ("base_image_digest", "sha256:" + "0" * 64),
        ("python_version", "3.13"),
        ("uv_version", "9.9.9"),
        ("runner_version", "0.0.0"),
        ("shared_env_install_args", "--frozen"),
        ("image_version", data["image_version"] + 1),
    ):
        drifted = dict(data)
        drifted[field] = mutated
        assert module.compute_identity(REPO_ROOT, drifted) != base, (
            f"identity digest must change when {field} changes"
        )


def test_recorded_identity_matches_recomputed_identity() -> None:
    """The committed lock digest must equal the recomputed binding (verify)."""
    module = _load_identity_module()
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    recomputed = module.compute_identity(REPO_ROOT, data)
    assert data["identity_digest"] == recomputed, (
        "lock identity_digest is stale; regenerate with "
        "scripts/ci/runner_image_identity.py --mode generate"
    )
    # The verify entrypoint must agree with the recomputed value.
    assert module.verify_lock(REPO_ROOT, LOCK_FILE) == 0


def test_verify_fails_loudly_on_drift(tmp_path: Path) -> None:
    module = _load_identity_module()
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    data["identity_digest"] = "deadbeef" * 3
    drifted = tmp_path / "runner-image.lock.json"
    drifted.write_text(json.dumps(data), encoding="utf-8")
    assert module.verify_lock(REPO_ROOT, drifted) != 0


def test_emit_produces_machine_readable_startup_evidence() -> None:
    module = _load_identity_module()
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    evidence = module.format_startup_evidence(data)
    parsed = json.loads(evidence)
    assert parsed["runner_image_version"] == data["image_version"]
    assert parsed["runner_image_identity"] == data["identity_digest"]
    assert parsed["python_version"] == data["python_version"]
    assert parsed["uv_version"] == data["uv_version"]


def test_identity_uses_shared_env_digest_as_a_binding_component() -> None:
    """The shared-env (canary) version must participate in the binding."""
    module = _load_identity_module()
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    # The lock must record the shared-env digest produced by ci_env_digest so
    # an env rebuild forces a new runner identity.
    assert "shared_env_digest" in data
    recomputed_env = module.compute_shared_env_digest(REPO_ROOT, data)
    assert data["shared_env_digest"] == recomputed_env


# --------------------------------------------------------------------------- #
# Image contract: prebuilt env baked into the image                          #
# --------------------------------------------------------------------------- #


def test_dockerfile_bakes_prebuilt_env_and_lock() -> None:
    source = RUNNER_DOCKERFILE.read_text(encoding="utf-8")
    # The lock file is copied into the image as the bound identity contract.
    assert "runner-image.lock.json" in source
    # The image stamps the bound identity digest as an arg/label, not just a
    # human version string.
    assert "OMNI_RUNNER_IMAGE_IDENTITY" in source
    # The repo-specific prebuilt env is part of the image contract.
    assert "OMNI_CI_ENV_ROOT" in source or "ci-envs" in source
    assert "uv sync" in source


# --------------------------------------------------------------------------- #
# Startup-evidence emission wired into every CI job                          #
# --------------------------------------------------------------------------- #


def test_emit_runner_identity_action_exists_and_writes_step_summary() -> None:
    action = _load_yaml(EMIT_ACTION)
    steps = action["runs"]["steps"]
    run_blob = "\n".join(step.get("run", "") for step in steps)
    assert "runner_image_identity.py" in run_blob
    assert "GITHUB_STEP_SUMMARY" in run_blob
    # The emit must also export the identity into the job env for downstream
    # debugging.
    assert "GITHUB_ENV" in run_blob or "GITHUB_OUTPUT" in run_blob


def test_setup_action_emits_runner_identity_on_every_setup() -> None:
    """Routing identity emission through setup-python-uv covers every job that
    prepares Python — which is every dependency-consuming CI job."""
    action = _load_yaml(SETUP_ACTION)
    steps = action["runs"]["steps"]
    emit_steps = [
        step
        for step in steps
        if str(step.get("uses", "")).endswith("/emit-runner-identity")
        or step.get("name") == "Emit runner image identity"
    ]
    assert emit_steps, "setup-python-uv must emit the runner image identity"


def test_every_ci_python_job_emits_runner_identity() -> None:
    """Every CI job that sets up Python must surface the runner identity in its
    startup evidence (otherwise image-drift debugging is guesswork)."""
    workflow = _load_yaml(CI_WORKFLOW)
    for job_name, job in workflow["jobs"].items():
        steps = job.get("steps", [])
        uses_setup = any(
            str(step.get("uses", "")).endswith("/setup-python-uv")
            or step.get("uses") == "./.github/actions/setup-python-uv"
            for step in steps
        )
        if not uses_setup:
            continue
        emits = (
            any(
                str(step.get("uses", "")).endswith("/emit-runner-identity")
                or step.get("name") == "Emit runner image identity"
                for step in steps
            )
            or uses_setup
        )  # setup-python-uv emits internally
        assert emits, f"job {job_name} must emit runner image identity"


# --------------------------------------------------------------------------- #
# Release bump + mutating-job opt-out                                         #
# --------------------------------------------------------------------------- #


def test_release_workflow_verifies_or_bumps_runner_image_identity() -> None:
    workflow = _load_yaml(RELEASE_WORKFLOW)
    blob = json.dumps(workflow)
    assert "runner_image_identity.py" in blob, (
        "release must verify/bump the bound runner image identity"
    )


def test_canary_doc_enumerates_mutating_jobs_and_zero_sync_happy_path() -> None:
    doc = CANARY_DOC.read_text(encoding="utf-8")
    assert "versioned **runner image contract**" in doc
    # Mutating jobs are enumerated and must opt out explicitly.
    assert "compliance" in doc
    assert 'shared-env-enabled: "false"' in doc
    assert "zero `uv sync`" in doc or "zero uv sync" in doc

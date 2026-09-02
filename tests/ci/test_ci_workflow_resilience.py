# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for CI resilience fixes."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCKER_BUILD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-build.yml"
REJECT_SKIP_CALLER_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "call-reject-skip.yml"
)
FRESH_DEPLOY_FITNESS_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "fresh-deploy-fitness.yml"
)
RUNTIME_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.runtime"
ENV_PARITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "env-parity.yml"
ARTIFACT_RECONCILIATION_WEBHOOK_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "artifact-reconciliation-webhook.yml"
)
PR_MERGED_EVENT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-merged-event.yml"
RUNTIME_REBUILD_TRIGGER_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"
)
PROD_PROMOTION_LINEAGE_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "prod-promotion-lineage.yml"
)
OMNI_STANDARDS_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "omni-standards-compliance.yml"
)
SECURITY_SCAN_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security-scan.yml"
CHECK_HANDSHAKE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "check-handshake.yml"
CODEQL_CONFIG = REPO_ROOT / ".github" / "codeql" / "codeql-config.yml"
SETUP_PYTHON_UV_ACTION = (
    REPO_ROOT / ".github" / "actions" / "setup-python-uv" / "action.yml"
)
CHECKOUT_V7_SHA = "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0"
CODEQL_V4_SHA = "dc73d59c2d7bd4f8194098a91219eeee6d8a1719"
OMNICLAUDE_REJECT_SKIP_NO_CHECKOUT_SHA = "b441ff9d979e248ac20c51a00c135a3ce273cef2"


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _runs_on_expression(job: dict[str, Any]) -> str:
    runs_on = job.get("runs-on")
    if runs_on is None:
        return ""
    if isinstance(runs_on, str):
        return runs_on
    return str(runs_on)


def test_migration_freeze_checkout_is_bounded_for_merge_group() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["migration-freeze"]
    steps = job["steps"]

    assert job["timeout-minutes"] == 15

    checkout_step = next(step for step in steps if step.get("name") == "Checkout code")
    assert checkout_step["uses"] == f"actions/checkout@{CHECKOUT_V7_SHA}"
    assert checkout_step["timeout-minutes"] == 10
    assert checkout_step["with"]["fetch-depth"] == 2
    assert checkout_step["with"]["fetch-tags"] is False

    freeze_step = next(
        step for step in steps if step.get("name") == "Check migration freeze"
    )
    assert freeze_step["run"] == "./scripts/check_migration_freeze.sh --ci"


def test_prod_promotion_lineage_guard_uses_uncached_direct_setup() -> None:
    workflow = _load_yaml(PROD_PROMOTION_LINEAGE_WORKFLOW)
    setup_step = next(
        step
        for step in workflow["jobs"]["guard-tests"]["steps"]
        if step.get("name") == "Setup Python and uv"
    )

    assert setup_step["uses"] == "./.github/actions/setup-python-uv"
    assert setup_step["with"]["cache-enabled"] == "false"
    assert setup_step["with"]["shared-env-enabled"] == "false"


def test_migration_integration_resolves_reachable_postgres_host() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["migration-integration"]

    ports = job["services"]["postgres"]["ports"]
    assert ports == ["5432/tcp"]

    steps = job["steps"]
    resolve_step = next(
        step for step in steps if step.get("name") == "Resolve Postgres service host"
    )
    assert resolve_step["id"] == "postgres_host"
    assert (
        resolve_step["env"]["POSTGRES_PORT"]
        == "${{ job.services.postgres.ports['5432'] }}"
    )
    assert "socket.create_connection" in resolve_step["run"]
    assert "/proc/net/route" in resolve_step["run"]

    apply_step = next(
        step for step in steps if step.get("name") == "Apply all migrations"
    )
    assert (
        apply_step["env"]["OMNIBASE_INFRA_DB_URL"]
        == "postgresql://postgres:test_password@${{ steps.postgres_host.outputs.host }}:${{ job.services.postgres.ports['5432'] }}/omnibase_infra"
    )

    assert_step = next(
        step for step in steps if step.get("name") == "Assert manifest tables exist"
    )
    assert (
        assert_step["env"]["POSTGRES_HOST"] == "${{ steps.postgres_host.outputs.host }}"
    )
    assert (
        assert_step["env"]["POSTGRES_PORT"]
        == "${{ job.services.postgres.ports['5432'] }}"
    )
    assert 'host=os.environ["POSTGRES_HOST"]' in assert_step["run"]
    assert 'port=int(os.environ["POSTGRES_PORT"])' in assert_step["run"]

    client_step_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Verify Python Postgres client"
    )
    assert_step_index = steps.index(assert_step)
    assert client_step_index < assert_step_index
    client_step = steps[client_step_index]
    assert "import asyncpg" in client_step["run"]
    assert "apt-get" not in client_step["run"]
    assert "sudo" not in client_step["run"]

    assert "import asyncpg" in assert_step["run"]
    assert "asyncpg.connect" in assert_step["run"]
    assert "EXPECTED_TABLES" in assert_step["run"]
    assert "psql" not in assert_step["run"]


def test_migration_conflict_action_is_blocking() -> None:
    """OMN-11163 graduated migration-conflict-check from advisory to blocking.

    The validate-boundaries step must run with warn-only disabled and without
    continue-on-error at either the job or step level, so an upstream conflict
    fails the gate instead of merely emitting a warning.
    """
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["migration-conflict-check"]

    assert "continue-on-error" not in job

    validate_step = next(
        step
        for step in job["steps"]
        if step.get("uses")
        == "OmniNode-ai/onex_change_control/.github/actions/validate-boundaries@main"
    )
    assert "continue-on-error" not in validate_step
    assert validate_step["with"]["warn-only"] == "false"
    assert (
        validate_step["with"]["repos"]
        == "omniclaude,omnidash,omniintelligence,omnibase_core,omnimemory"
    )
    assert "omnibase_infra" not in validate_step["with"]["repos"].split(",")
    # OMN-16373: CROSS_REPO_PAT retired in favor of a minted
    # onexbot-occ-writer App installation token.
    #
    # OMN-16414: the mint step itself now carries continue-on-error (fork PRs
    # get no org secrets, so minting fails there) and this env now falls back
    # to github.token -- safe because every repo in `repos` above is PUBLIC.
    # This does not weaken "blocking": the fallback only changes which token
    # authenticates the read; warn-only stays "false" and this step keeps no
    # continue-on-error of its own, so a real conflict still fails the job.
    assert (
        validate_step["env"]["OMNI_REPO_CLONE_TOKEN"]
        == "${{ steps.app-token.outputs.token || github.token }}"
    )

    report_steps = [
        step
        for step in job["steps"]
        if step.get("name") == "Report non-blocking boundary validator startup failure"
    ]
    assert report_steps == []


def test_docker_integration_build_timeout_matches_workflow_budget() -> None:
    workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    job = workflow["jobs"]["docker-integration-tests"]
    step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Run Docker integration tests"
    )

    assert step["env"]["OMNI_DOCKER_BUILD_TIMEOUT_SECONDS"] == "1200"
    assert '--timeout="${OMNI_DOCKER_BUILD_TIMEOUT_SECONDS}"' in step["run"]


def test_docker_integration_tests_do_not_run_on_pull_requests() -> None:
    workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    job = workflow["jobs"]["docker-integration-tests"]

    assert "github.event_name != 'pull_request'" in job["if"]
    assert "github.event.inputs.run_full_tests != 'false'" in job["if"]
    assert job["continue-on-error"] is True


def test_short_dependency_gates_have_checkout_budget() -> None:
    docker_workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    freshness_workflow = _load_yaml(FRESH_DEPLOY_FITNESS_WORKFLOW)

    docker_pin_job = docker_workflow["jobs"]["dockerfile-pin-check"]
    sibling_lock_job = freshness_workflow["jobs"]["sibling-lock-pins"]

    assert docker_pin_job["timeout-minutes"] >= 15
    assert sibling_lock_job["timeout-minutes"] >= 20


def test_reject_skip_token_gate_uses_no_checkout_reusable() -> None:
    workflow = _load_yaml(REJECT_SKIP_CALLER_WORKFLOW)
    job = workflow["jobs"]["call-reject-skip-token"]

    assert (
        job["uses"]
        == "OmniNode-ai/omniclaude/.github/workflows/reject-deploy-gate-skip.yml"
        f"@{OMNICLAUDE_REJECT_SKIP_NO_CHECKOUT_SHA}"
    )


def test_runtime_boot_smoke_is_not_run_on_pull_requests() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["runtime-boot-smoke"]
    summary = workflow["jobs"]["ci-summary"]

    assert "github.event_name != 'pull_request'" in job["if"]
    assert "needs.tests-gate.result == 'success'" in job["if"]
    # ci-summary is a NO-`needs` fail-closed poller (OMN-14127); regardless of
    # whether it declares `needs`, runtime-boot-smoke must never be a dependency
    # of it (a PR-skipped advisory job must not wedge the required summary gate).
    assert "runtime-boot-smoke" not in summary.get("needs", [])


def test_compose_required_env_gate_has_checkout_budget() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["compose-required-env-coverage"]

    assert job["timeout-minutes"] >= 20


def test_docker_integration_installs_compose_plugin_before_tests() -> None:
    workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    assert workflow["env"]["DOCKER_COMPOSE_VERSION"] == "v2.40.3"

    job = workflow["jobs"]["docker-integration-tests"]
    steps = job["steps"]
    compose_step_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Install Docker Compose plugin"
    )
    test_step_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Run Docker integration tests"
    )

    compose_step = steps[compose_step_index]
    assert compose_step_index < test_step_index
    assert "docker compose version" in compose_step["run"]
    assert "DOCKER_COMPOSE_VERSION" in compose_step["run"]
    assert "docker-compose-linux-x86_64" in compose_step["run"]


def test_short_gates_can_disable_uv_cache_cleanup() -> None:
    action = _load_yaml(SETUP_PYTHON_UV_ACTION)
    assert action["inputs"]["cache-enabled"]["default"] == "true"
    assert action["inputs"]["shared-env-enabled"]["default"] == "auto"
    assert (
        action["inputs"]["shared-env-root"]["default"]
        == "/home/runner/.cache/omni/ci-envs"
    )
    assert (
        action["inputs"]["shared-env-install-args"]["default"]
        == "--frozen --all-extras --all-groups --no-install-project"
    )

    shared_mode_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Resolve shared CI env mode"
    )
    assert shared_mode_step["id"] == "shared_env_mode"
    assert "OMNI_CI_SHARED_ENV_ENABLED" in shared_mode_step["run"]

    shared_env_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Prepare shared CI env"
    )
    assert (
        shared_env_step["if"]
        == "steps.shared_env_mode.outputs.enabled == 'true' && inputs.skip-install != 'true'"
    )
    assert (
        shared_env_step["run"].strip()
        == '"${GITHUB_ACTION_PATH}/../../../scripts/ci/ensure_ci_env.sh"'
    )
    assert shared_env_step["env"]["OMNI_CI_ENV_ROOT"] == "${{ inputs.shared-env-root }}"

    cache_step = next(
        step for step in action["runs"]["steps"] if step.get("name") == "Load cached uv"
    )
    assert (
        cache_step["if"]
        == "steps.shared_env_mode.outputs.enabled != 'true' && inputs.cache-enabled != 'false'"
    )

    install_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Install dependencies"
    )
    assert (
        install_step["if"]
        == "steps.shared_env_mode.outputs.enabled != 'true' && inputs.skip-install != 'true'"
    )

    ci_workflow = _load_yaml(CI_WORKFLOW)
    assert ci_workflow["env"]["OMNI_CI_ENV_ROOT"] == "/home/runner/.cache/omni/ci-envs"
    assert "OMNI_CI_SHARED_ENV_ENABLED" in ci_workflow["env"]
    assert (
        "head.repo.full_name != github.repository"
        in ci_workflow["env"]["OMNI_CI_SHARED_ENV_ENABLED"]
    )
    standards_workflow = _load_yaml(OMNI_STANDARDS_WORKFLOW)
    assert (
        standards_workflow["env"]["OMNI_CI_ENV_ROOT"]
        == "/home/runner/.cache/omni/ci-envs"
    )
    assert "OMNI_CI_SHARED_ENV_ENABLED" in standards_workflow["env"]

    for job_name, job in ci_workflow["jobs"].items():
        setup_steps = [
            step
            for step in job.get("steps", [])
            if str(step.get("uses", "")).endswith("/.github/actions/setup-python-uv")
            or step.get("uses") == "./.github/actions/setup-python-uv"
        ]
        if not setup_steps:
            continue

        assert len(setup_steps) == 1
        setup_step = setup_steps[0]
        assert setup_step["with"]["cache-enabled"] == "false"

    env_parity_workflow = _load_yaml(ENV_PARITY_WORKFLOW)
    setup_step = next(
        step
        for step in env_parity_workflow["jobs"]["env-parity"]["steps"]
        if step.get("uses") == "./omnibase_infra/.github/actions/setup-python-uv"
    )
    assert setup_step["with"]["cache-enabled"] == "false"

    sibling_workflow = _load_yaml(
        REPO_ROOT / ".github" / "workflows" / "check-sibling-compat.yml"
    )
    sibling_steps = sibling_workflow["jobs"]["sibling-compat"]["steps"]
    assert all("setup-python-uv" not in step.get("uses", "") for step in sibling_steps)
    run_lines = [
        line.strip()
        for step in sibling_steps
        for line in step.get("run", "").splitlines()
    ]
    assert not any(re.search(r"\buv\s+sync\b", line) for line in run_lines)
    assert not any(re.search(r"\buv\s+pip\s+install\b", line) for line in run_lines)
    assert any("OMN-12563" in step.get("run", "") for step in sibling_steps)

    docker_workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    docker_setup_step = next(
        step
        for step in docker_workflow["jobs"]["docker-integration-tests"]["steps"]
        if step.get("uses") == "./.github/actions/setup-python-uv"
    )
    assert docker_setup_step["with"]["cache-enabled"] == "false"
    assert docker_setup_step["with"]["cache-version"] == "docker"
    assert docker_setup_step["with"]["cache-key-prefix"] == "uv-docker"
    assert docker_setup_step["with"]["shared-env-enabled"] == "true"
    assert docker_setup_step["with"]["github-token"] == "${{ secrets.GITHUB_TOKEN }}"


def test_shared_ci_env_scripts_are_digest_keyed_and_read_only() -> None:
    digest_script = REPO_ROOT / "scripts" / "ci" / "ci_env_digest.py"
    ensure_script = REPO_ROOT / "scripts" / "ci" / "ensure_ci_env.sh"

    digest_source = digest_script.read_text(encoding="utf-8")
    assert "pyproject.toml" in digest_source
    assert "uv.lock" in digest_source
    assert "python_version" in digest_source
    assert "uv_version" in digest_source
    assert "install_args" in digest_source

    ensure_source = ensure_script.read_text(encoding="utf-8")
    assert "/home/runner/.cache/omni/ci-envs" in ensure_source
    assert "flock 9" in ensure_source
    assert 'mkdir "${lock_path}"' in ensure_source
    assert 'UV_PROJECT_ENVIRONMENT="${venv_dir}"' in ensure_source
    assert 'cat > "${manifest_path}"' in ensure_source
    assert 'workspace_venv="${repo_root}/.venv"' in ensure_source
    assert 'wrapper_parent="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"' in ensure_source
    assert 'wrapper_dir="${wrapper_parent%/}/omni-ci-bin-${digest}"' in ensure_source
    assert (
        'metadata_root="${wrapper_parent%/}/omni-ci-metadata-${digest}"'
        in ensure_source
    )
    assert 'ln -sfn "${venv_dir}" "${workspace_venv}"' in ensure_source
    assert "OMNI_CI_SHARED_UV_RUN_DIRECT=1" in ensure_source
    assert 'if [[ "\\${OMNI_CI_SHARED_UV_RUN_DIRECT:-0}" == "1"' in ensure_source
    assert 'exec "\\${workspace_venv}/bin/\\${cmd}" "\\$@"' in ensure_source
    assert 'echo "UV_PROJECT_ENVIRONMENT=${workspace_venv}"' in ensure_source
    assert 'echo "PATH=${wrapper_dir}:${workspace_venv}/bin:${PATH}"' in ensure_source
    assert 'echo "PYTHONPATH=${metadata_root}:${repo_root}/src' in ensure_source
    assert "write_project_metadata" in ensure_source
    assert "entry_points.txt" in ensure_source
    assert 'project.get("entry-points", {})' in ensure_source
    assert "uv sync" in ensure_source
    assert 'sync_attempts="${OMNI_CI_ENV_SYNC_ATTEMPTS:-5}"' in ensure_source
    assert (
        'retry_delay_seconds="${OMNI_CI_ENV_SYNC_RETRY_DELAY_SECONDS:-10}"'
        in ensure_source
    )
    assert 'until uv sync "${install_argv[@]}"; do' in ensure_source
    assert "shared CI env uv sync attempt" in ensure_source
    assert "shared CI env uv sync failed after" in ensure_source
    assert "chmod -R a-w" in ensure_source
    assert "UV_NO_SYNC=1" in ensure_source


def test_ci_jobs_that_mutate_python_env_disable_shared_env() -> None:
    ci_workflow = _load_yaml(CI_WORKFLOW)

    compliance_setup = next(
        step
        for step in ci_workflow["jobs"]["compliance"]["steps"]
        if step.get("uses") == "./.github/actions/setup-python-uv"
    )
    assert compliance_setup["with"].get("shared-env-enabled") != "false"
    assert not any(
        step.get("name") == "Install dependencies"
        for step in ci_workflow["jobs"]["compliance"]["steps"]
    )

    for job_name in ("schema-handshake", "kafka-boundary-compat"):
        setup_step = next(
            step
            for step in ci_workflow["jobs"][job_name]["steps"]
            if step.get("uses") == "./omnibase_infra/.github/actions/setup-python-uv"
        )
        assert setup_step["with"]["shared-env-enabled"] == "false"


def test_contract_compliance_uv_sync_is_bounded_and_retried() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["contract-compliance"]

    assert job["timeout-minutes"] == 20
    steps = job["steps"]
    checkout_occ = next(
        step for step in steps if step.get("name") == "Checkout onex_change_control"
    )
    # OMN-16373: CROSS_REPO_PAT retired in favor of a minted
    # onexbot-occ-writer App installation token.
    assert (
        checkout_occ["with"]["token"]
        == "${{ steps.app-token.outputs.token || github.token }}"
    )

    setup_uv = next(
        step for step in steps if step.get("uses") == "astral-sh/setup-uv@v7"
    )
    assert setup_uv["with"]["enable-cache"] is False
    assert "cache-dependency-glob" not in setup_uv["with"]

    install_step = next(
        step for step in steps if step.get("name") == "Install onex_change_control"
    )
    run_script = install_step["run"]
    assert 'export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"' in run_script
    assert "max_attempts=3" in run_script
    assert "until uv sync --no-cache --all-extras" in run_script
    assert "uv sync onex_change_control failed after" in run_script


def test_merge_group_and_docker_workflows_have_runner_pool_overrides() -> None:
    ci_workflow = _load_yaml(CI_WORKFLOW)
    for job_name, job in ci_workflow["jobs"].items():
        expression = _runs_on_expression(job)
        if "self-hosted" not in expression:
            continue

        assert "OMNI_PUBLIC_PR_RUNS_ON_JSON" in expression, job_name
        assert "OMNI_REQUIRED_CI_RUNS_ON_JSON" in expression, job_name
        assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" in expression, job_name
        assert "github.event_name == 'merge_group'" in expression, job_name

    docker_workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    for job_name, job in docker_workflow["jobs"].items():
        expression = _runs_on_expression(job)
        if "self-hosted" not in expression:
            continue

        assert "OMNI_PUBLIC_PR_RUNS_ON_JSON" in expression, job_name
        assert "OMNI_DOCKER_CI_RUNS_ON_JSON" in expression, job_name
        assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" in expression, job_name


def test_cross_repo_ci_jobs_use_retrying_uv_install() -> None:
    ci_workflow = _load_yaml(CI_WORKFLOW)
    for job_name in ("topic-drift-check", "schema-handshake"):
        steps = ci_workflow["jobs"][job_name]["steps"]
        setup_step = next(
            step
            for step in steps
            if step.get("uses") == "./omnibase_infra/.github/actions/setup-python-uv"
        )

        assert setup_step["with"]["cache-enabled"] == "false"
        assert setup_step["with"]["working-directory"] == "omnibase_infra"
        assert setup_step["with"].get("skip-install") != "true"
        assert not any(step.get("run") == "uv sync --no-cache" for step in steps)


def test_heavy_cross_repo_boundary_installs_retry_and_have_timeout_budget() -> None:
    ci_workflow = _load_yaml(CI_WORKFLOW)
    for job_name in ("schema-handshake", "kafka-boundary-compat"):
        job = ci_workflow["jobs"][job_name]
        assert job["timeout-minutes"] >= 45

        install_step = next(
            step
            for step in job["steps"]
            if str(step.get("name", "")).startswith("Install sibling repos as editable")
        )
        checkout_names = {str(step.get("name", "")) for step in job["steps"]}
        assert "Checkout omnimarket (sibling)" in checkout_names

        run_script = install_step["run"]
        assert "max_attempts=5" in run_script
        assert "until uv pip install --no-deps" in run_script
        assert "-e ../omnibase_compat" in run_script
        assert "-e ../omnimarket" in run_script
        assert "-e ../omnimemory" in run_script
        assert "-e ../omniintelligence" in run_script
        assert "sibling deps attempt" in run_script
        assert "sibling deps failed after" in run_script


def test_heavy_cross_repo_jobs_use_cpu_torch_for_sibling_install() -> None:
    ci_workflow = _load_yaml(CI_WORKFLOW)
    for job_name, label in (
        ("schema-handshake", "schema-handshake"),
        ("kafka-boundary-compat", "kafka-boundary"),
    ):
        job = ci_workflow["jobs"][job_name]
        steps = job["steps"]

        assert job["timeout-minutes"] >= 45

        torch_step = next(
            step
            for step in steps
            if step.get("name") == "Preinstall CPU-only torch for sibling deps"
        )
        torch_script = torch_step["run"]
        assert "https://download.pytorch.org/whl/cpu" in torch_script
        assert f"{label} torch CPU wheel attempt" in torch_script
        assert f"{label} torch CPU wheel failed after" in torch_script


def test_topic_enum_drift_has_install_retry_budget() -> None:
    """OMN-12432: topic enum drift must survive one uv git fetch retry."""
    ci_workflow = _load_yaml(CI_WORKFLOW)
    job = ci_workflow["jobs"]["topic-enum-drift"]

    assert job["timeout-minutes"] >= 15
    setup_step = next(
        step
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/setup-python-uv"
    )
    assert setup_step["with"]["cache-enabled"] == "false"
    assert setup_step["with"].get("skip-install") != "true"


def test_onex_validators_have_retry_timeout_budget() -> None:
    ci_workflow = _load_yaml(CI_WORKFLOW)
    job = ci_workflow["jobs"]["onex-validation"]

    assert job["timeout-minutes"] >= 20


def test_architecture_handshake_has_checkout_retry_timeout_budget() -> None:
    workflow = _load_yaml(CHECK_HANDSHAKE_WORKFLOW)
    job = workflow["jobs"]["check-handshake"]

    assert job["timeout-minutes"] >= 10


def test_retrying_uv_action_defaults_to_cpu_torch_backend() -> None:
    action_text = SETUP_PYTHON_UV_ACTION.read_text(encoding="utf-8")
    assert 'export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"' in action_text
    assert "UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-<unset>}" in action_text


def test_docker_integration_uses_retrying_uv_setup_action() -> None:
    docker_workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    steps = docker_workflow["jobs"]["docker-integration-tests"]["steps"]

    setup_step = next(
        step
        for step in steps
        if step.get("uses") == "./.github/actions/setup-python-uv"
    )
    assert setup_step["name"] == "Setup Python and uv"
    assert setup_step["with"]["cache-enabled"] == "false"
    assert setup_step["with"]["github-token"] == "${{ secrets.GITHUB_TOKEN }}"
    assert not any(
        step.get("run")
        == "uv sync --reinstall-package omnibase-core --reinstall-package omnibase-spi"
        for step in steps
    )


def test_runtime_plugin_dependency_install_retries_package_index_flakes() -> None:
    dockerfile = RUNTIME_DOCKERFILE.read_text(encoding="utf-8")

    assert "UV_HTTP_TIMEOUT=600" in dockerfile
    assert "UV_RETRY_ATTEMPTS=8" in dockerfile
    assert "cat > /usr/local/bin/uv-with-retry" in dockerfile
    assert "uv-with-retry pip install \\" in dockerfile
    assert "uv $* attempt ${attempt}/${max_attempts} failed" in dockerfile


def test_runtime_dockerfile_retries_builder_uv_sync_transport_flakes() -> None:
    dockerfile = RUNTIME_DOCKERFILE.read_text(encoding="utf-8")

    assert "git config --global http.version HTTP/1.1" in dockerfile
    assert "UV_HTTP_TIMEOUT=600" in dockerfile
    assert "UV_RETRY_ATTEMPTS=8" in dockerfile
    assert "uv-with-retry sync --no-dev --no-install-project" in dockerfile
    assert "uv-with-retry sync --no-dev" in dockerfile
    assert "uv $* attempt ${attempt}/${max_attempts} failed" in dockerfile


def test_runtime_dockerfile_retries_torch_cpu_index_transport_flakes() -> None:
    dockerfile = RUNTIME_DOCKERFILE.read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert (
        "uv-with-retry pip install torch --index-url https://download.pytorch.org/whl/cpu"
        in dockerfile
    )
    assert (
        'uv-with-retry pip install --no-deps "torch>=2.6.0,<3.0.0" --index-url https://download.pytorch.org/whl/cpu'
        in dockerfile
    )
    assert "UV_RETRY_ATTEMPTS=8" in dockerfile


def test_omni_standards_jobs_use_retrying_uv_install() -> None:
    workflow = _load_yaml(OMNI_STANDARDS_WORKFLOW)

    assert workflow["env"]["UV_HTTP_TIMEOUT"] == "600"

    for job_name in ("type-safety", "type-union-check"):
        steps = workflow["jobs"][job_name]["steps"]
        setup_step = next(
            step
            for step in steps
            if step.get("uses") == "./.github/actions/setup-python-uv"
        )

        assert setup_step["with"]["python-version"] == "${{ env.PYTHON_VERSION }}"
        assert setup_step["with"]["uv-version"] == "${{ env.UV_VERSION }}"
        assert setup_step["with"]["cache-enabled"] == "false"
        assert setup_step["with"]["install-args"] == "--all-extras"
        assert setup_step["with"]["sync-attempts"] == "3"
        assert setup_step["with"]["sync-retry-delay-seconds"] == "10"
        assert not any(
            step.get("run") == "uv sync --no-cache --all-extras" for step in steps
        )


def test_setup_python_uv_retries_uv_sync_and_logs_transport_settings() -> None:
    action = _load_yaml(SETUP_PYTHON_UV_ACTION)

    assert action["inputs"]["sync-attempts"]["default"] == "5"
    assert action["inputs"]["sync-retry-delay-seconds"]["default"] == "10"

    setup_step = next(
        step for step in action["runs"]["steps"] if step.get("name") == "Set up Python"
    )
    assert setup_step["uses"] == "actions/setup-python@v6"

    install_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Install dependencies"
    )
    setup_step = next(
        step for step in action["runs"]["steps"] if step.get("name") == "Set up Python"
    )
    assert setup_step["uses"] == "actions/setup-python@v6"

    assert install_step["env"]["UV_SYNC_ATTEMPTS"] == "${{ inputs.sync-attempts }}"
    assert (
        install_step["env"]["UV_SYNC_RETRY_DELAY_SECONDS"]
        == "${{ inputs.sync-retry-delay-seconds }}"
    )

    run_script = install_step["run"]
    assert 'export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"' in run_script
    assert (
        'export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"' in run_script
    )
    assert 'export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"' in run_script
    assert 'export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"' in run_script
    assert "git config --global http.version HTTP/1.1" in run_script
    assert "git config --global http.lowSpeedLimit 0" in run_script
    assert "git config --global http.lowSpeedTime 999999" in run_script
    assert "sync_cmd=(uv sync)" in run_script
    assert "sync_cmd+=(--no-cache)" in run_script
    assert 'until "${sync_cmd[@]}"; do' in run_script
    assert (
        'echo "::warning::uv sync attempt ${attempt}/${sync_attempts} failed'
        in run_script
    )
    assert 'echo "::error::uv sync failed after ${attempt} attempt(s)"' in run_script
    assert 'echo "UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-<unset>}"' in run_script
    assert (
        'echo "UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-<unset>}"'
        in run_script
    )


def test_setup_python_uv_authenticates_git_fetches() -> None:
    """OMN-12432: uv's git+https dependency fetches must be authenticated.

    Anonymous github.com fetches from the self-hosted runners hit the 60/hr
    anonymous rate limit and fail with "Empty reply from server" when many
    parallel --no-cache uv syncs run from one egress IP. The action configures
    a process-scoped insteadOf rewrite (via GIT_CONFIG_* env vars, never a
    persisted gitconfig) using a github token so uv's internal `git fetch`
    authenticates and gets the 5000/hr limit.
    """
    action = _load_yaml(SETUP_PYTHON_UV_ACTION)

    token_input = action["inputs"]["github-token"]
    assert token_input["default"] == "${{ github.token }}"

    install_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Install dependencies"
    )
    assert install_step["env"]["GIT_FETCH_TOKEN"] == "${{ inputs.github-token }}"

    run_script = install_step["run"]
    assert 'if [ -n "${GIT_FETCH_TOKEN}" ]; then' in run_script
    assert "export GIT_CONFIG_COUNT=1" in run_script
    assert (
        'export GIT_CONFIG_KEY_0="url.https://x-access-token:${GIT_FETCH_TOKEN}@github.com/.insteadOf"'
        in run_script
    )
    assert 'export GIT_CONFIG_VALUE_0="https://github.com/"' in run_script
    # Token must never be written to a persistent global gitconfig on the runner.
    assert "git config --global url." not in run_script


def test_omni_standards_uv_jobs_use_authenticated_composite_action() -> None:
    """OMN-12432: the uv-sync jobs that block #1781/#1782 must authenticate.

    type-safety and type-union-check previously inlined an unauthenticated
    `uv sync --no-cache --all-extras`. They now route through setup-python-uv
    with an explicit token so the git fetches are authenticated and retried.
    """
    workflow = _load_yaml(OMNI_STANDARDS_WORKFLOW)

    for job_name in ("type-safety", "type-union-check"):
        steps = workflow["jobs"][job_name]["steps"]
        setup_step = next(
            step
            for step in steps
            if step.get("uses") == "./.github/actions/setup-python-uv"
        )
        assert setup_step["with"]["install-args"] == "--all-extras"
        assert setup_step["with"]["cache-enabled"] == "false"
        # OMN-16373: CROSS_REPO_PAT retired in favor of a minted
        # onexbot-occ-writer App installation token.
        assert (
            setup_step["with"]["github-token"]
            == "${{ steps.app-token.outputs.token || github.token }}"
        )
        # No raw unauthenticated uv sync left behind.
        assert not any(
            step.get("run") == "uv sync --no-cache --all-extras" for step in steps
        )

    # The pinned onex_change_control git+https install must also authenticate.
    occ_steps = workflow["jobs"]["handler-contract-compliance"]["steps"]
    install_step = next(
        step
        for step in occ_steps
        if step.get("name") == "Install onex_change_control (pinned)"
    )
    # OMN-16373: CROSS_REPO_PAT retired in favor of a minted
    # onexbot-occ-writer App installation token.
    assert (
        install_step["env"]["GIT_FETCH_TOKEN"]
        == "${{ steps.app-token.outputs.token || github.token }}"
    )
    assert "export GIT_CONFIG_COUNT=1" in install_step["run"]
    assert 'export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"' in install_step["run"]
    assert "max_attempts=3" in install_step["run"]
    assert "until uv pip install" in install_step["run"]
    assert "uv pip install onex_change_control failed after" in install_step["run"]


def test_webhook_workflows_use_ci_python_environment() -> None:
    """Webhook producer jobs must not resolve Python deps outside CI env setup."""
    workflow_paths = (
        ARTIFACT_RECONCILIATION_WEBHOOK_WORKFLOW,
        PR_MERGED_EVENT_WORKFLOW,
        RUNTIME_REBUILD_TRIGGER_WORKFLOW,
    )

    for workflow_path in workflow_paths:
        workflow = _load_yaml(workflow_path)
        for job_name, job in workflow["jobs"].items():
            if "uses" in job:
                assert job["uses"].endswith(
                    "occ-preflight.yml@789d175d78a7a802f4f0f4aa2af7083bdfd312c2"
                )
                continue

            steps = job["steps"]
            setup_steps = [
                step
                for step in steps
                if step.get("uses") == "./.github/actions/setup-python-uv"
            ]
            assert setup_steps, (
                f"{workflow_path.name}:{job_name} must use setup-python-uv"
            )
            assert all(
                step["with"]["install-args"] == "--frozen" for step in setup_steps
            )
            assert all(step["with"]["cache-enabled"] == "false" for step in setup_steps)
            assert all(
                step["with"]["shared-env-enabled"] == "true" for step in setup_steps
            )

            run_scripts = [
                step.get("run", "")
                for step in steps
                if isinstance(step.get("run"), str)
            ]
            assert not any(
                re.search(r"(^|\n)\s*(?:python -m )?pip install\b", script)
                for script in run_scripts
            ), f"{workflow_path.name} must not run pip install directly"
            assert any("uv run python scripts/" in script for script in run_scripts)


def test_codeql_uses_repo_config_that_ignores_github_metadata() -> None:
    """OMN-12432: CodeQL must not upload malformed .github directory results."""
    workflow = _load_yaml(SECURITY_SCAN_WORKFLOW)
    config = _load_yaml(CODEQL_CONFIG)

    checkout_step = next(
        step
        for step in workflow["jobs"]["codeql"]["steps"]
        if step.get("name") == "Checkout repository"
    )
    assert checkout_step["uses"] == f"actions/checkout@{CHECKOUT_V7_SHA}"
    assert checkout_step["with"]["persist-credentials"] is False

    init_step = next(
        step
        for step in workflow["jobs"]["codeql"]["steps"]
        if step.get("name") == "Initialize CodeQL"
    )
    assert init_step["uses"] == f"github/codeql-action/init@{CODEQL_V4_SHA}"
    assert init_step["with"]["languages"] == "python"
    assert init_step["with"]["queries"] == "security-and-quality"
    assert init_step["with"]["config-file"] == "./.github/codeql/codeql-config.yml"

    autobuild_step = next(
        step
        for step in workflow["jobs"]["codeql"]["steps"]
        if step.get("name") == "Autobuild"
    )
    assert autobuild_step["uses"] == f"github/codeql-action/autobuild@{CODEQL_V4_SHA}"

    analyze_step = next(
        step
        for step in workflow["jobs"]["codeql"]["steps"]
        if step.get("name") == "Perform CodeQL Analysis"
    )
    assert analyze_step["uses"] == f"github/codeql-action/analyze@{CODEQL_V4_SHA}"
    assert analyze_step["with"]["category"] == "/language:python"
    assert analyze_step["with"]["upload"] == "never"
    assert analyze_step["with"]["wait-for-processing"] is False

    assert config["paths"] == ["src", "scripts", "tests"]
    assert ".github/**" in config["paths-ignore"]


# OMN-15815's concurrency-group non-collision regression lived here: a
# CodeRabbit `issue_comment` run sharing a group key with the `pull_request`
# run of the same PR cancelled the real gate run, and the CodeRabbit-actored
# replacement skipped its own job, leaving the required context stuck
# cancelled with no successor. Both the test and its renderer were deleted in
# OMN-16933 along with cr-thread-gate-caller.yml, the only workflow whose
# group template had that shape. If a future workflow keys concurrency on a
# `||` fallback chain with no event discriminator, restore the property test
# with it — the failure mode is a property of the template, not of CodeRabbit.


_TIMEOUT_METHOD_RE = re.compile(r"--timeout-method=(\S+)")


def _has_pytest_token(command_text: str) -> bool:
    """True when a non-comment line of the command text invokes pytest."""
    return any(
        token == "pytest" or token.endswith("/pytest")
        for line in command_text.splitlines()
        if not line.lstrip().startswith("#")
        for token in line.split()
    )


def _all_workflow_run_commands() -> list[tuple[str, str, str]]:
    """Every ``(workflow file, job, run script)`` triple in .github/workflows."""
    workflows_dir = CI_WORKFLOW.parent
    commands: list[tuple[str, str, str]] = []
    for path in sorted(workflows_dir.glob("*.yml")) + sorted(
        workflows_dir.glob("*.yaml")
    ):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            steps = job.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if isinstance(step, dict) and "run" in step:
                    commands.append((path.name, str(job_name), str(step["run"])))
    return commands


def test_no_workflow_pytest_invocation_uses_thread_timeout_method() -> None:
    """OMN-16348: every ``--timeout-method`` in any workflow must be ``signal``.

    OMN-15977 (omnibase_core) banned pytest-timeout's ``thread`` method: its
    watcher thread fires only when the GIL is released, so a CPU-bound
    pure-Python runaway holds the GIL continuously and the declared
    ``--timeout`` ceiling silently never fires (the config behind the
    2026-08-12 46/53-minute pre-push runaways that needed manual SIGKILL).
    The only remaining backstop is the job's ``timeout-minutes``, which
    cancels the whole shard with no attributable test. The original guards
    were per-file, which is exactly why additional surfaces stayed invisible
    — so this assertion is per-invocation-surface: it scans every run step of
    every workflow file, and none may pass ``--timeout-method=`` with any
    value other than ``signal``.
    """
    commands = _all_workflow_run_commands()

    # Positive control: the scanner must actually be seeing ci.yml's pytest
    # steps — an empty scan would vacuously pass while enforcing nothing.
    assert any(
        source == CI_WORKFLOW.name and _has_pytest_token(run)
        for source, _, run in commands
    )

    violations = [
        f"{source}::{job}: {line.strip()}"
        for source, job, run in commands
        for line in run.splitlines()
        for method in _TIMEOUT_METHOD_RE.findall(line)
        if method != "signal"
    ]
    assert violations == [], (
        "workflow passes a non-signal --timeout-method (banned by OMN-15977; "
        "a CLI flag overrides any addopts signal default): "
        f"{violations}"
    )


def test_no_precommit_or_script_surface_uses_thread_timeout_method() -> None:
    """OMN-16348: the thread-method ban covers non-workflow command surfaces.

    The workflow scan above cannot see the two other places this repo builds
    pytest command lines: ``.pre-commit-config.yaml`` hook entries and the
    committed operational scripts under ``scripts/`` (e.g.
    ``publish_test_failure_baseline.py`` assembles a subprocess pytest command).
    Each carried ``--timeout-method=thread`` until OMN-16348, which is exactly
    the per-file-guard blind spot that let ci.yml keep the banned method after
    OMN-15977 — so this scan is enumerated per command surface, not per file
    that happened to be broken once.
    """
    banned = "--timeout-method=thread"

    precommit = yaml.safe_load(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    entries = [
        str(hook.get("entry", ""))
        for repo in precommit.get("repos", [])
        if isinstance(repo, dict)
        for hook in repo.get("hooks", [])
        if isinstance(hook, dict)
    ]
    # Positive control: the config's pytest-running hooks must be visible to
    # the scan — an empty entry list would vacuously pass.
    assert any(_has_pytest_token(entry) for entry in entries)
    entry_violations = [
        entry
        for entry in entries
        for method in _TIMEOUT_METHOD_RE.findall(entry)
        if method != "signal"
    ]
    assert entry_violations == [], (
        "pre-commit hook entry passes a non-signal --timeout-method "
        "(OMN-15977: the thread watcher never fires under a GIL-bound "
        f"runaway): {entry_violations}"
    )

    script_files = [
        path
        for pattern in ("*.py", "*.sh")
        for path in sorted((REPO_ROOT / "scripts").rglob(pattern))
        if path.is_file()
    ]
    # Positive control: the surface OMN-16348 actually fixed must be scanned.
    assert REPO_ROOT / "scripts" / "publish_test_failure_baseline.py" in script_files
    # Scripts are arbitrary source text (quoted flags, list literals), so the
    # scan is a literal ban on the banned flag rather than value extraction.
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in script_files
        if banned in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert violations == [], (
        "committed script builds a pytest command with the banned "
        f"--timeout-method=thread (OMN-15977/OMN-16348): {violations}"
    )


# ---------------------------------------------------------------------------
# OMN-16555: shared-env mode must survive an ephemeral (GitHub-hosted) runner
# ---------------------------------------------------------------------------
#
# In shared-env mode the action skips `actions/setup-python` and
# `astral-sh/setup-uv` entirely and calls `scripts/ci/ensure_ci_env.sh`, which
# assumes a pre-provisioned persistent runner filesystem: a warm host-local
# cache root and `uv` already on PATH. Both hold on the long-lived self-hosted
# `omnibase-ci` fleet; neither holds on a fresh GitHub-hosted VM, where
# `ensure_ci_env.sh:40` (`real_uv="$(command -v uv)"`) aborts under `set -e`
# with no diagnostic. That killed 2/10 rows of the OMN-16511 Stage-0a canary
# and blocks the OMN-16682 hosted-runner migration.

EPHEMERAL_TOOLCHAIN_STEP_NAME = "Provision ephemeral runner toolchain"
EPHEMERAL_TOOLCHAIN_STEP_ID = "ephemeral_toolchain"


def _setup_python_uv_steps() -> list[dict[str, Any]]:
    action = _load_yaml(SETUP_PYTHON_UV_ACTION)
    steps = action["runs"]["steps"]
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


def _step_index(name: str) -> int:
    return next(
        index
        for index, step in enumerate(_setup_python_uv_steps())
        if step.get("name") == name
    )


def _sandbox_bin(tmp_path: Path, *, with_uv: bool) -> Path:
    """Return a PATH directory holding only what the step legitimately needs.

    The step must work on a bare VM, so the sandbox deliberately excludes
    everything except `mkdir` (plus an optional `uv` stub). Restricting PATH is
    what makes "uv is absent" a real condition rather than a mocked one.
    """
    bin_dir = tmp_path / "sandbox-bin"
    bin_dir.mkdir()
    mkdir_bin = shutil.which("mkdir")
    assert mkdir_bin is not None, "mkdir must exist to build the PATH sandbox"
    (bin_dir / "mkdir").symlink_to(mkdir_bin)
    if with_uv:
        uv_stub = bin_dir / "uv"
        uv_stub.write_text(
            "#!/bin/sh\nexit 0\n",
            encoding="utf-8",
        )
        uv_stub.chmod(0o755)
    return bin_dir


def _run_ephemeral_toolchain_step(
    tmp_path: Path,
    *,
    with_uv: bool,
    runner_environment: str | None,
    shared_env_root: Path | str,
) -> tuple[int, str, dict[str, str]]:
    """Execute the step's real `run:` body and return (rc, output, step outputs)."""
    step = next(
        candidate
        for candidate in _setup_python_uv_steps()
        if candidate.get("name") == EPHEMERAL_TOOLCHAIN_STEP_NAME
    )
    run_text = step["run"]
    assert "${{" not in run_text, (
        "the step body must be free of GitHub expression interpolation so it is "
        "directly executable and testable; pass inputs through `env:` instead"
    )

    bin_dir = _sandbox_bin(tmp_path, with_uv=with_uv)
    github_output = tmp_path / "github_output"
    github_output.touch()

    env = {
        "PATH": str(bin_dir),
        "GITHUB_OUTPUT": str(github_output),
        "SHARED_ENV_ROOT": str(shared_env_root),
    }
    if runner_environment is not None:
        env["RUNNER_ENVIRONMENT"] = runner_environment

    bash_bin = shutil.which("bash")
    assert bash_bin is not None
    completed = subprocess.run(
        [bash_bin, "-c", run_text],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    outputs: dict[str, str] = {}
    for line in github_output.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            outputs[key] = value
    return completed.returncode, completed.stdout + completed.stderr, outputs


def test_shared_env_mode_provisions_python_and_uv_on_ephemeral_runners() -> None:
    """OMN-16555: the shared-env branch must provision its own toolchain."""
    steps = _setup_python_uv_steps()

    detect_step = next(
        step for step in steps if step.get("name") == EPHEMERAL_TOOLCHAIN_STEP_NAME
    )
    assert detect_step["id"] == EPHEMERAL_TOOLCHAIN_STEP_ID
    assert detect_step["if"] == "steps.shared_env_mode.outputs.enabled == 'true'"
    assert detect_step["env"]["SHARED_ENV_ROOT"] == "${{ inputs.shared-env-root }}"

    provision_guard = f"steps.{EPHEMERAL_TOOLCHAIN_STEP_ID}.outputs.provision == 'true'"

    python_step = next(
        step
        for step in steps
        if step.get("uses") == "actions/setup-python@v6"
        and step.get("if") == provision_guard
    )
    assert python_step["with"]["python-version"] == "${{ inputs.python-version }}"

    uv_step = next(
        step
        for step in steps
        if step.get("uses") == "astral-sh/setup-uv@v7"
        and step.get("if") == provision_guard
    )
    # The ephemeral install must use the SAME pins as the non-shared branch, so
    # a hosted runner cannot silently resolve a different uv than the fleet.
    assert uv_step["with"]["version"] == "${{ inputs.uv-version }}"

    # Ordering: detect -> provision python -> provision uv -> run ensure_ci_env.sh.
    detect_index = _step_index(EPHEMERAL_TOOLCHAIN_STEP_NAME)
    prepare_index = _step_index("Prepare shared CI env")
    python_index = steps.index(python_step)
    uv_index = steps.index(uv_step)
    assert detect_index < python_index < prepare_index
    assert detect_index < uv_index < prepare_index


def test_ephemeral_toolchain_step_keeps_the_fleet_warm_path(tmp_path: Path) -> None:
    """uv already on PATH => no provisioning, on either runner class.

    This is the no-regression assertion for DoD item 3: a self-hosted job must
    keep hitting the warm shared env, never a cold re-provision.
    """
    root = tmp_path / "ci-envs"
    root.mkdir()
    rc, output, outputs = _run_ephemeral_toolchain_step(
        tmp_path,
        with_uv=True,
        runner_environment="self-hosted",
        shared_env_root=root,
    )
    assert rc == 0, output
    assert outputs["provision"] == "false"
    assert "::error::" not in output


def test_ephemeral_toolchain_step_provisions_on_fresh_hosted_vm(
    tmp_path: Path,
) -> None:
    """Fresh GitHub-hosted VM: no uv, no cache root => provision both."""
    root = tmp_path / "fresh" / "ci-envs"
    assert not root.exists()
    rc, output, outputs = _run_ephemeral_toolchain_step(
        tmp_path,
        with_uv=False,
        runner_environment="github-hosted",
        shared_env_root=root,
    )
    assert rc == 0, output
    assert outputs["provision"] == "true"
    assert root.is_dir(), "the absent shared-env cache root must be created"
    assert "::error::" not in output


def test_ephemeral_toolchain_step_aborts_on_corrupt_self_hosted_runner(
    tmp_path: Path,
) -> None:
    """Missing uv on a self-hosted runner is corrupt state, not a fresh VM.

    Fail-fast is preserved: the fleet image bakes uv in, so self-healing here
    would mask a real runner misprovisioning instead of surfacing it.
    """
    root = tmp_path / "ci-envs"
    root.mkdir()
    rc, output, outputs = _run_ephemeral_toolchain_step(
        tmp_path,
        with_uv=False,
        runner_environment="self-hosted",
        shared_env_root=root,
    )
    assert rc != 0
    assert "::error::" in output
    assert "provision" not in outputs


def test_ephemeral_toolchain_step_fails_closed_on_unknown_runner_class(
    tmp_path: Path,
) -> None:
    """An unset RUNNER_ENVIRONMENT is indeterminate and must not provision."""
    root = tmp_path / "ci-envs"
    root.mkdir()
    rc, output, outputs = _run_ephemeral_toolchain_step(
        tmp_path,
        with_uv=False,
        runner_environment=None,
        shared_env_root=root,
    )
    assert rc != 0
    assert "::error::" in output
    assert "provision" not in outputs


def test_ephemeral_toolchain_step_aborts_when_cache_root_is_uncreatable(
    tmp_path: Path,
) -> None:
    """A cache root that cannot be created is a config error, not a fresh VM."""
    rc, output, outputs = _run_ephemeral_toolchain_step(
        tmp_path,
        with_uv=False,
        runner_environment="github-hosted",
        shared_env_root="/dev/null/omni-ci-envs",
    )
    assert rc != 0
    assert "::error::" in output
    assert outputs.get("provision") != "true"


SHARED_ENV_PARITY_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "shared-env-runner-parity.yml"
)
RUNNER_ROUTING_POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"


def test_shared_env_parity_gate_covers_both_runner_classes() -> None:
    """OMN-16555: the shared-env branch must be exercised on hosted AND fleet.

    The defect was invisible for as long as it was because shared-env mode was
    only ever run on the self-hosted fleet. A gate that runs on one class only
    would leave that hole open.
    """
    workflow = _load_yaml(SHARED_ENV_PARITY_WORKFLOW)
    jobs = workflow["jobs"]

    # BOTH halves pin literal runner labels rather than the var-driven
    # OMNI_RUNNER_SELECTOR_V1 expression. A parity gate whose runner classes are
    # decided by repo variables is not a parity gate: OMN-16682 retargeted
    # OMNI_TRUSTED_CI_RUNS_ON_JSON to ["ubuntu-latest"] on 2026-08-26, which
    # would have collapsed both halves onto hosted while still reading green.
    assert jobs["hosted"]["runs-on"] == "ubuntu-latest"
    assert jobs["self-hosted-fleet"]["runs-on"] == ["self-hosted", "omnibase-ci"]

    # Pinning fleet labels is only safe because fork PRs never reach that job;
    # untrusted code must never be routed onto a self-hosted runner.
    fleet_if = jobs["self-hosted-fleet"]["if"]
    assert "head.repo.full_name == github.repository" in fleet_if

    # Both jobs must assert their own runner class before trusting the result:
    # a green conclusion is not evidence of where the job ran.
    for job_name, expected_class in (
        ("hosted", "github-hosted"),
        ("self-hosted-fleet", "self-hosted"),
    ):
        assertion_steps = [
            step
            for step in jobs[job_name]["steps"]
            if "RUNNER_ENVIRONMENT" in step.get("run", "")
            and f'!= "{expected_class}"' in step.get("run", "")
        ]
        assert assertion_steps, (
            f"{job_name} must fail closed when RUNNER_ENVIRONMENT is not "
            f"{expected_class!r} (OMN-16555 evidence rule)"
        )

    # Both jobs must actually enter the shared-env branch, and both must prove
    # the resulting environment is usable rather than merely exiting zero.
    for job_name in ("hosted", "self-hosted-fleet"):
        setup_step = next(
            step
            for step in jobs[job_name]["steps"]
            if step.get("uses") == "./.github/actions/setup-python-uv"
        )
        assert setup_step["with"]["shared-env-enabled"] == "true"
        verify_steps = [
            step
            for step in jobs[job_name]["steps"]
            if "OMNI_CI_ENV_DIR" in step.get("run", "")
        ]
        assert verify_steps, (
            f"{job_name} must verify the shared env, not just install it"
        )
        verify_source = "\n".join(step["run"] for step in verify_steps)
        # Importing a real synced dependency is what separates "the setup step
        # exited 0" from "the environment is actually installed and usable".
        assert "import sys, pydantic" in verify_source
        assert "manifest.json" in verify_source
        assert "-L .venv" in verify_source

    # The gate has to fire on every input that can reintroduce the defect.
    # PyYAML resolves the bare `on:` key to the boolean True (YAML 1.1).
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    paths = triggers["pull_request"]["paths"]
    for required in (
        ".github/actions/setup-python-uv/action.yml",
        "scripts/ci/ensure_ci_env.sh",
        ".github/workflows/shared-env-runner-parity.yml",
    ):
        assert required in paths


def test_shared_env_parity_hosted_job_is_routing_allowlisted() -> None:
    """A literal hosted runner is only legal with a recorded justification."""
    policy = _load_yaml(RUNNER_ROUTING_POLICY)
    entry = next(
        item
        for item in policy["hosted_runner_allowlist"]
        if item["path"] == ".github/workflows/shared-env-runner-parity.yml"
    )
    assert "OMN-16555" in entry["reason"]


def test_shared_env_parity_jobs_carry_positive_controls() -> None:
    """Each job must prove its runner class really has the shape it claims.

    Without these, the hosted job could silently stop exercising the ephemeral
    bootstrap (if hosted images ever ship uv) and keep reporting green while
    proving nothing — the exact vacuous-proof failure OMN-16511 warned about.
    """
    jobs = _load_yaml(SHARED_ENV_PARITY_WORKFLOW)["jobs"]

    hosted_controls = "\n".join(step.get("run", "") for step in jobs["hosted"]["steps"])
    assert "uv is unexpectedly pre-installed" in hosted_controls
    assert "already exists on a fresh hosted VM" in hosted_controls

    fleet_controls = "\n".join(
        step.get("run", "") for step in jobs["self-hosted-fleet"]["steps"]
    )
    # The fleet's warm path depends on uv coming from the runner image; if it
    # ever does not, that is corrupt runner state and must surface as such.
    assert "uv is missing from a self-hosted runner" in fleet_controls

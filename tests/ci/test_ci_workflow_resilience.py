# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for CI resilience fixes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCKER_BUILD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-build.yml"
RUNTIME_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.runtime"
ENV_PARITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "env-parity.yml"
ARTIFACT_RECONCILIATION_WEBHOOK_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "artifact-reconciliation-webhook.yml"
)
PR_MERGED_EVENT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-merged-event.yml"
RUNTIME_REBUILD_TRIGGER_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"
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
CHECKOUT_V6_SHA = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
CODEQL_V4_SHA = "dc73d59c2d7bd4f8194098a91219eeee6d8a1719"


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_migration_freeze_uses_shallow_checkout_for_merge_group() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["migration-freeze"]
    steps = job["steps"]

    checkout_step = next(
        step for step in steps if step.get("uses") == "actions/checkout@v6"
    )
    assert checkout_step["with"]["fetch-depth"] == 2

    freeze_step = next(
        step for step in steps if step.get("name") == "Check migration freeze"
    )
    assert freeze_step["run"] == "./scripts/check_migration_freeze.sh --ci"


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
    assert (
        validate_step["env"]["OMNI_REPO_CLONE_TOKEN"] == "${{ secrets.CROSS_REPO_PAT }}"
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
    assert (
        checkout_occ["with"]["token"] == "${{ secrets.CROSS_REPO_PAT || github.token }}"
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
        run_script = install_step["run"]
        assert "max_attempts=5" in run_script
        assert (
            "until uv pip install --overrides /tmp/sibling-overrides.txt" in run_script
        )
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


def test_kafka_boundary_sibling_install_retries_transient_download_failures() -> None:
    ci_workflow = _load_yaml(CI_WORKFLOW)
    steps = ci_workflow["jobs"]["kafka-boundary-compat"]["steps"]
    install_step = next(
        step
        for step in steps
        if step.get("name") == "Install sibling repos as editable (boundary test deps)"
    )

    run_script = install_step["run"]
    assert "max_attempts=5" in run_script
    assert "until uv pip install --overrides /tmp/sibling-overrides.txt" in run_script
    assert (
        'echo "::warning::uv pip install sibling deps attempt ${attempt}/${max_attempts} failed'
        in run_script
    )
    assert (
        'echo "::error::uv pip install sibling deps failed after ${attempt} attempt(s)"'
        in run_script
    )


def test_architecture_handshake_has_checkout_retry_timeout_budget() -> None:
    workflow = _load_yaml(CHECK_HANDSHAKE_WORKFLOW)
    job = workflow["jobs"]["check-handshake"]

    assert job["timeout-minutes"] >= 10


def test_onex_validators_have_retry_timeout_budget() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["onex-validation"]

    assert job["timeout-minutes"] >= 20


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


def test_setup_python_uv_retries_uv_sync_and_logs_transport_settings() -> None:
    action = _load_yaml(SETUP_PYTHON_UV_ACTION)

    assert action["inputs"]["sync-attempts"]["default"] == "5"
    assert action["inputs"]["sync-retry-delay-seconds"]["default"] == "10"

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
        assert (
            setup_step["with"]["github-token"]
            == "${{ secrets.CROSS_REPO_PAT || github.token }}"
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
    assert (
        install_step["env"]["GIT_FETCH_TOKEN"]
        == "${{ secrets.CROSS_REPO_PAT || github.token }}"
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
        for job in workflow["jobs"].values():
            steps = job["steps"]
            setup_steps = [
                step
                for step in steps
                if step.get("uses") == "./.github/actions/setup-python-uv"
            ]
            assert setup_steps, f"{workflow_path.name} must use setup-python-uv"
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
    assert checkout_step["uses"] == f"actions/checkout@{CHECKOUT_V6_SHA}"
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

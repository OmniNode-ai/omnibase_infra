# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21004: the caller owns the unchanged allowlist and uses core's guard.

The recorded paths come from the guard's omnibase_infra allowlist at
ebe30bc3589c9f803e8920b941ddaef07b17ed90. Behavioural probes use the canary's isolated 0.47.41 validator environment
without changing the application's lock.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
CALLER = WORKFLOWS / "imperative-contract-guard.yml"
ALLOWLIST = ROOT / "imperative-contract-allowlist.yaml"
CORE_SHA_PREFIX = "031148215"
CORE_USES = "OmniNode-ai/omnibase_core/.github/workflows/imperative-contract-guard.yml"
RECORDED_PATHS = [
    "omnibase_infra/nodes/node_architecture_validator/handlers/handler_architecture_validation.py",
    "omnibase_infra/nodes/node_baseline_comparison_compute/handlers/handler_baseline_comparison.py",
    "omnibase_infra/nodes/node_build_loop_write_effect/handlers/handler_build_loop_append.py",
    "omnibase_infra/nodes/node_checkpoint_effect/handlers/util_checkpoint_sorting.py",
    "omnibase_infra/nodes/node_contract_loader_effect/handlers/handler_contract_scan.py",
    "omnibase_infra/nodes/node_decision_store_query_compute/handlers/handler_query_decisions.py",
    "omnibase_infra/nodes/node_event_bus_wiring_effect/handlers/handler_event_bus_wiring.py",
    "omnibase_infra/nodes/node_impact_analyzer_compute/handlers/constants.py",
    "omnibase_infra/nodes/node_ledger_write_effect/handlers/handler_ledger_append.py",
    "omnibase_infra/nodes/node_ledger_write_effect/handlers/handler_ledger_query.py",
    "omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/config/bifrost_shadow.py",
    "omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/config/model_shadow_decision_log.py",
    "omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/config_loader_bifrost.py",
    "omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/handler_bifrost_gateway.py",
    "omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/model_bifrost_config.py",
    "omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/model_bifrost_request.py",
    "omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/model_bifrost_response.py",
    "omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/model_bifrost_routing_rule.py",
    "omnibase_infra/nodes/node_pattern_lifecycle_effect/handlers/handler_lifecycle_update.py",
    "omnibase_infra/nodes/node_runtime_orchestrator/handlers/handler_runtime_lifecycle.py",
    "omnibase_infra/nodes/node_savings_estimation_compute/handlers/handler_savings_estimation.py",
    "omnibase_infra/nodes/node_session_state_effect/handlers/handler_run_context_read.py",
    "omnibase_infra/nodes/node_session_state_effect/handlers/handler_run_context_write.py",
    "omnibase_infra/nodes/node_session_state_effect/handlers/handler_session_index_read.py",
    "omnibase_infra/nodes/node_session_state_effect/handlers/handler_session_index_write.py",
    "omnibase_infra/nodes/node_session_state_effect/handlers/handler_stale_run_gc.py",
    "omnibase_infra/nodes/node_validation_executor/handlers/handler_run_checks.py",
    "src/omnibase_infra/adapters/github/adapter_github_client.py",
    "src/omnibase_infra/backends/backend_probe.py",
    "src/omnibase_infra/cli/artifact_reconcile.py",
    "src/omnibase_infra/cli/cli_env.py",
    "src/omnibase_infra/cli/cli_kafka.py",
    "src/omnibase_infra/cli/commands.py",
    "src/omnibase_infra/cli/infra_test/run_suite.py",
    "src/omnibase_infra/cli/infra_test/verify.py",
    "src/omnibase_infra/cli/service_demo_reset.py",
    "src/omnibase_infra/dlq/plugin_dlq.py",
    "src/omnibase_infra/dlq/service_dlq_tracking.py",
    "src/omnibase_infra/enums/generated/enum_artifact_topic.py",
    "src/omnibase_infra/enums/generated/enum_omnibase_infra_topic.py",
    "src/omnibase_infra/enums/generated/enum_omnimarket_topic.py",
    "src/omnibase_infra/enums/generated/enum_platform_topic.py",
    "src/omnibase_infra/event_bus/event_bus_kafka.py",
    "src/omnibase_infra/event_bus/kafka_transport.py",
    "src/omnibase_infra/event_bus/topic_constants.py",
    "src/omnibase_infra/event_bus/topic_violation_alerter.py",
    "src/omnibase_infra/handlers/handler_db.py",
    "src/omnibase_infra/handlers/handler_gmail_api.py",
    "src/omnibase_infra/handlers/handler_linear_db_error_reporter.py",
    "src/omnibase_infra/handlers/handler_slack_webhook.py",
    "src/omnibase_infra/handlers/registration_storage/handler_registration_storage_postgres.py",
    "src/omnibase_infra/idempotency/store_postgres.py",
    "src/omnibase_infra/mixins/mixin_llm_http_transport.py",
    "src/omnibase_infra/models/discovery/model_introspection_config.py",
    "src/omnibase_infra/models/projection/model_topic_projection.py",
    "src/omnibase_infra/models/validation_ledger/model_validation_ledger_entry.py",
    "src/omnibase_infra/nodes/node_chain_canary_effect/deploy_agent_window.py",
    "src/omnibase_infra/nodes/node_dlq_replay_effect/engine_dlq_replay.py",
    "src/omnibase_infra/nodes/node_registration_orchestrator/plugin.py",
    "src/omnibase_infra/probes/capability_probe.py",
    "src/omnibase_infra/projectors/snapshot_publisher_registration.py",
    "src/omnibase_infra/runtime/auto_wiring/handler_wiring.py",
    "src/omnibase_infra/runtime/emit_daemon/topics.py",
    "src/omnibase_infra/runtime/providers/provider_http_client.py",
    "src/omnibase_infra/runtime/providers/provider_kafka_producer.py",
    "src/omnibase_infra/runtime/providers/provider_postgres_pool.py",
    "src/omnibase_infra/runtime/request_response_wiring.py",
    "src/omnibase_infra/runtime/runtime_scheduler.py",
    "src/omnibase_infra/runtime/service_kernel.py",
    "src/omnibase_infra/runtime/service_pattern_b_broker.py",
    "src/omnibase_infra/runtime/util_schema_fingerprint.py",
    "src/omnibase_infra/services/service_llm_endpoint_health.py",
    "src/omnibase_infra/services/session/consumer.py",
    "src/omnibase_infra/services/session/store.py",
    "src/omnibase_infra/topics/platform_topic_suffixes.py",
    "src/omnibase_infra/tui/consumers/consumer_status.py",
]
LIVE_HANDLER = (
    "src/omnibase_infra/nodes/node_auth_gate_compute/handlers/handler_auth_gate.py"
)


def _mapping(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _core(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--no-project", "--with", "omnibase-core==0.47.41", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        cwd=ROOT,
    )


def _caller_sha() -> str:
    jobs = _mapping(CALLER)["jobs"]
    assert isinstance(jobs, dict)
    uses = jobs["imperative-contract-guard"]["uses"]
    assert isinstance(uses, str)
    repository, sha = uses.split("@")
    assert repository == CORE_USES
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert sha.startswith(CORE_SHA_PREFIX)
    return sha


def test_caller_uses_core_and_keeps_the_job_and_context() -> None:
    workflow = _mapping(CALLER)
    assert workflow["name"] == "Imperative Contract Guard"
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"imperative-contract-guard"}
    job = jobs["imperative-contract-guard"]
    assert job["uses"] == f"{CORE_USES}@{_caller_sha()}"
    assert job["with"] == {"allowlist-path": ALLOWLIST.name}
    assert workflow["permissions"] == {"contents": "read"}
    for path in WORKFLOWS.iterdir():
        if path.suffix in {".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8")
            assert not re.search(
                r"onex_change_control/\.github/workflows/imperative-contract-guard\.yml",
                text,
            ), path


def test_canonical_shape_hook_and_ci_use_the_exemption_commit() -> None:
    repos = _mapping(ROOT / ".pre-commit-config.yaml")["repos"]
    assert isinstance(repos, list)
    (hook_repo,) = [
        repo
        for repo in repos
        if any(hook["id"] == "canonical-file-shape" for hook in repo["hooks"])
    ]
    assert hook_repo["repo"] == "https://github.com/OmniNode-ai/omnibase_core"
    assert hook_repo["rev"] == _caller_sha()
    (boundary_repo,) = [
        repo
        for repo in repos
        if any(hook["id"] == "check-node-boundary-imports" for hook in repo["hooks"])
    ]
    assert boundary_repo["repo"] == hook_repo["repo"]
    assert boundary_repo["rev"] == _caller_sha()
    jobs = _mapping(WORKFLOWS / "ci.yml")["jobs"]
    assert isinstance(jobs, dict)
    assert jobs["canonical-file-shape"]["env"]["CANONICAL_FILE_SHAPE_CORE"].endswith(
        f"@{_caller_sha()}"
    )
    assert jobs["node-boundary-imports"]["env"]["NODE_BOUNDARY_CORE"].endswith(
        f"@{_caller_sha()}"
    )


def test_own_allowlist_matches_the_recorded_paths() -> None:
    entries = _mapping(ALLOWLIST)["allowlisted_handlers"]
    assert isinstance(entries, list)
    assert [entry["path"] for entry in entries] == RECORDED_PATHS
    # yamlfmt removes the optional document marker; all other bytes stay historical.
    historical_bytes = b"---\n" + ALLOWLIST.read_bytes().removeprefix(b"---\n")
    assert hashlib.sha256(historical_bytes).hexdigest() == (
        "31c48377eebce9b29efda0b661e50365d2cd4edc8407194bf57ebca53766057d"
    )


def test_core_ratchet_accepts_unchanged_and_shrinking_but_refuses_an_extra_path() -> (
    None
):
    text = ALLOWLIST.read_text(encoding="utf-8")
    entries = _mapping(ALLOWLIST)["allowlisted_handlers"]
    assert isinstance(entries, list)
    extra = "src/omnibase_infra/nodes/node_planted_effect/handlers/handler_planted.py"
    request = {
        "base": text,
        "head": yaml.safe_dump({"allowlisted_handlers": [*entries, {"path": extra}]}),
        "shrunk": yaml.safe_dump({"allowlisted_handlers": entries[:-1]}),
    }
    result = _core(
        "python",
        "-c",
        "import json, sys; "
        "from omnibase_core.handlers.handler_imperative_allowlist_ratchet import "
        "HandlerImperativeAllowlistRatchet; "
        "from omnibase_core.models.nodes.imperative_contract_guard."
        "model_allowlist_ratchet_input import ModelAllowlistRatchetInput; "
        "p=json.load(sys.stdin); h=HandlerImperativeAllowlistRatchet(); "
        "reports=[h.handle(ModelAllowlistRatchetInput(base_text=p['base'], "
        "head_text=p[k])) for k in ('base', 'shrunk', 'head')]; "
        "print(json.dumps([dict(r.model_dump(), admitted=r.admitted) for r in reports]))",
        stdin=json.dumps(request),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    unchanged, shrunk, planted = json.loads(result.stdout)
    assert unchanged == {
        "added": [],
        "removed": [],
        "bootstrap": False,
        "admitted": True,
    }
    assert shrunk["admitted"] is True
    assert shrunk["removed"] == [RECORDED_PATHS[-1]]
    assert planted["admitted"] is False
    assert planted["bootstrap"] is False
    assert planted["added"] == [extra]


@pytest.mark.live_contact("tests/ci/fixtures/imperative_contract_guard_omn21004.json")
def test_core_guard_accepts_clean_tree_and_refuses_a_planted_live_handler(
    tmp_path: Path,
    recorded_response: dict[str, object],
) -> None:
    # Use the caller's input; the merge-base caller cannot provide it.
    jobs = _mapping(CALLER)["jobs"]
    assert isinstance(jobs, dict)
    assert jobs["imperative-contract-guard"]["with"]["allowlist-path"] == ALLOWLIST.name
    root = tmp_path / "omnibase_infra"
    shutil.copytree(
        ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copyfile(ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copyfile(ALLOWLIST, root / ALLOWLIST.name)
    args = (
        "check-imperative-contracts",
        "--repo-root",
        str(root),
        "--allowlist-path",
        ALLOWLIST.name,
        "--scan-freestanding",
        "--json",
    )
    recorded = recorded_response["response"]
    assert isinstance(recorded, dict)
    assert [
        recorded[key]["exit"]
        for key in ("core-clean", "occ-clean", "core-planted", "occ-planted")
    ] == [0, 0, 1, 1]
    clean = _core(*args)
    assert clean.returncode == recorded["core-clean"]["exit"]
    assert clean.returncode == 0, clean.stdout + clean.stderr
    handler = root / LIVE_HANDLER
    original = handler.read_text(encoding="utf-8")
    # The scanner reads this source; it does not execute a network request.
    handler.write_text(
        original + '\nimport httpx\nhttpx.get("http://example.invalid/planted")\n',
        encoding="utf-8",
    )
    planted = _core(*args)
    assert planted.returncode == 1, planted.stdout + planted.stderr
    report = json.loads(planted.stdout)[0]
    assert planted.returncode == recorded["core-planted"]["exit"]
    assert (
        report["new_violation_count"] == recorded["core-planted"]["new_violation_count"]
    )
    assert report["new_violation_count"] > 0
    (violation,) = [
        violation
        for violation in report["new_violations"]
        if violation["handler_path"].endswith("handlers/handler_auth_gate.py")
    ]
    assert violation["reachability"] == "live"
    assert "HTTP" in violation["undeclared_transports"]
    assert "undeclared_transport" in violation["violations"]

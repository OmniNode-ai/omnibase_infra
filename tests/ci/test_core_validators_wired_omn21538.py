# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omnibase_core validators are consumed by hook id from one core pin (OMN-21538).

AC1: every wired hook id resolves to the omnibase_core repository URL in
``.pre-commit-config.yaml`` and runs in ``ci.yml``. AC3: no baseline carries a
file that a canon-audit finding names. AC5: the tag has one home, the
omnibase_core block of ``.pre-commit-config.yaml``; the workflow reads it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CORE_URL = "https://github.com/OmniNode-ai/omnibase_core"
CORE_JOB = "core-validators"
SHAPE_JOB = "canonical-file-shape"
SHAPE_HOOK = "canonical-file-shape"
DEV_MERGE_TAG = re.compile(r"dev-\d{14}-[0-9a-f]{12}")

# Every core validator this repository consumes by hook id for OMN-21538.
WIRED_HOOK_IDS = (
    "check-no-raw-sqlite3",
    "check-local-paths-compute",
    "check-hardcoded-topic-compute",
    "check-hardcoded-model-config-compute",
    "check-no-io-outside-effects",
    "handler-di-gate",
)

# Files that findings F44 to F47 (omnibase_infra#5426 and #5425) name. A baseline
# for a wired hook never lists one: the file is fixed or the hook fails.
FINDING_FILES = (
    "src/omnibase_infra/event_bus/event_bus_kafka.py",
    "src/omnibase_infra/event_bus/kafka_transport.py",
    "src/omnibase_infra/handlers/handler_kafka_topic_demand_source.py",
    "src/omnibase_infra/runtime/auto_wiring/handler_wiring.py",
    "src/omnibase_infra/runtime/service_kernel.py",
    "src/omnibase_infra/runtime/models/model_runtime_config.py",
    "src/omnibase_infra/runtime/models/model_boot_subscriptions_config.py",
    "scripts/runtime_build/prepr_pool_prove.sh",
    "scripts/runtime_build/prepr_runtime_pool.py",
    "scripts/runtime_build/prepr_verify_rendered_slot.py",
    "docker/docker-compose.prepr.yml",
)


def _repo_blocks() -> list[dict[str, Any]]:
    loaded = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text("utf-8"))
    blocks: list[dict[str, Any]] = loaded["repos"]
    return blocks


def _workflow_jobs() -> dict[str, dict[str, Any]]:
    loaded = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
    jobs: dict[str, dict[str, Any]] = loaded["jobs"]
    return jobs


def _job_text(job: str) -> str:
    return yaml.safe_dump(_workflow_jobs()[job])


def _core_hooks() -> list[dict[str, Any]]:
    hooks: list[dict[str, Any]] = []
    for block in _repo_blocks():
        if block["repo"] == CORE_URL:
            hooks.extend(block["hooks"])
    return hooks


@pytest.mark.parametrize("hook_id", WIRED_HOOK_IDS)
def test_hook_id_resolves_to_the_core_repository(hook_id: str) -> None:
    owners = [
        block["repo"]
        for block in _repo_blocks()
        if any(hook["id"] == hook_id for hook in block["hooks"])
    ]
    assert owners == [CORE_URL], f"{hook_id} is declared under {owners}"


@pytest.mark.parametrize("hook_id", WIRED_HOOK_IDS)
def test_hook_id_is_not_defined_locally(hook_id: str) -> None:
    for block in _repo_blocks():
        if block["repo"] == "local":
            assert hook_id not in [hook["id"] for hook in block["hooks"]]


@pytest.mark.parametrize("hook_id", WIRED_HOOK_IDS)
def test_ci_job_runs_the_hook_id(hook_id: str) -> None:
    assert hook_id in _job_text(CORE_JOB)


def test_ci_job_is_unconditional_and_hosted() -> None:
    job = _workflow_jobs()[CORE_JOB]
    assert "needs" not in job
    assert "if" not in job
    assert job["runs-on"] == "ubuntu-latest"
    assert job["name"] == "Core Validators (OMN-21538)"


def test_ci_job_has_a_positive_control() -> None:
    text = _job_text(CORE_JOB)
    assert "POSITIVE CONTROL FAILED" in text
    assert "pre-commit@" in text


def test_the_tag_has_one_home() -> None:
    """The workflow holds no dev-merge tag for these jobs; the config block does."""
    for job in (CORE_JOB, SHAPE_JOB):
        assert not DEV_MERGE_TAG.search(_job_text(job)), job
    assert ".pre-commit-config.yaml" in _job_text(SHAPE_JOB)
    (block,) = [
        block
        for block in _repo_blocks()
        if any(hook["id"] == SHAPE_HOOK for hook in block["hooks"])
    ]
    block_ids = {hook["id"] for hook in block["hooks"]}
    assert set(WIRED_HOOK_IDS) <= block_ids
    assert DEV_MERGE_TAG.fullmatch(block["rev"])


def _baseline_paths() -> list[Path]:
    """Every baseline file a wired hook reads, from the hook args in the config."""
    paths = [ROOT / "config/hardcoded_model_config_baseline.yaml"]
    for hook in _core_hooks():
        if hook["id"] not in WIRED_HOOK_IDS:
            continue
        args: list[str] = list(hook.get("args", []))
        if "--baseline" in args:
            paths.append(ROOT / args[args.index("--baseline") + 1])
    return [path for path in paths if path.is_file()]


def test_baselines_name_no_finding_file_and_carry_no_suppression_marker() -> None:
    baselines = _baseline_paths()
    assert baselines, "no baseline of a wired hook is on disk"
    for baseline in baselines:
        text = baseline.read_text("utf-8")
        assert [name for name in FINDING_FILES if name in text] == [], baseline.name
        assert not re.search(r"#\s*(noqa|nosec|di-ok|io-ok|local-path-ok)\b", text), (
            baseline.name
        )

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

# Hooks the CI positive control plants a defect for. check-no-io-outside-effects
# and handler-di-gate judge a module by its place in the package (node, handler),
# which one planted file at the root does not reproduce.
PLANTED_CONTROL_HOOK_IDS = (
    "check-no-raw-sqlite3",
    "check-local-paths-compute",
    "check-hardcoded-topic-compute",
    "check-hardcoded-model-config-compute",
)

HANDLER_DI_BASELINE = ".onex_ratchets/handler_di_baseline.txt"
MODEL_CONFIG_BASELINE = "config/hardcoded_model_config_baseline.yaml"
SHA1 = re.compile(r"[0-9a-f]{40}")
SUPPRESSION_MARKER = re.compile(
    r"#\s*(noqa|nosec|di-ok|io-ok|local-path-ok|onex-allow-topic-literal)\b"
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


def _step_script(name_prefix: str) -> list[str]:
    """Executable lines of the named step's ``run`` script, comments removed."""
    steps = _workflow_jobs()[CORE_JOB]["steps"]
    (step,) = [s for s in steps if str(s.get("name", "")).startswith(name_prefix)]
    return [
        line.strip()
        for line in str(step["run"]).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _joined(lines: list[str]) -> str:
    return " ".join(line.rstrip("\\").strip() for line in lines)


def _loop_words(lines: list[str], variable: str) -> list[str]:
    """The words a ``for <variable> in ...; do`` header iterates over."""
    match = re.search(rf"for {variable} in (.*?); do", _joined(lines))
    assert match, f"no for-{variable} loop in the step script"
    return match.group(1).split()


def test_ci_job_executes_exactly_the_wired_hook_ids() -> None:
    """The run loop iterates the wired ids (comments and strings elsewhere do not count)."""
    lines = _step_script("Run the core validators")
    assert sorted(_loop_words(lines, "hook")) == sorted(WIRED_HOOK_IDS)
    invocation = [line for line in lines if "pre-commit" in line]
    assert len(invocation) == 1
    assert 'run "$hook"' in invocation[0]
    assert "--from-ref" in invocation[0]
    assert "--to-ref HEAD" in invocation[0]


@pytest.mark.parametrize("step", ["Positive control", "Run the core validators"])
def test_ci_step_fails_on_the_first_failing_command(step: str) -> None:
    """Each step opens with ``set -euo pipefail``: a failing hook in the loop stops it.

    The run loop has no ``|| true``, no ``set +e`` and no ``continue-on-error``, so
    the step's exit status is the first failing hook's.
    """
    lines = _step_script(step)
    assert lines[0] == "set -euo pipefail", lines[0]
    text = _joined(lines)
    assert "|| true" not in text
    assert "|| :" not in text
    assert "set +e" not in text
    assert "continue-on-error" not in yaml.safe_dump(_workflow_jobs()[CORE_JOB])


def test_ci_job_runs_the_hooks_of_the_core_block() -> None:
    """The job runs pre-commit against the repository config, so the hook ids and
    the core rev it runs are the ones the omnibase_core block declares."""
    (invocation,) = [
        line for line in _step_script("Run the core validators") if "pre-commit" in line
    ]
    assert "--config" not in invocation
    assert " -c " not in invocation
    assert not DEV_MERGE_TAG.search(invocation)
    core_ids = {hook["id"] for hook in _core_hooks()}
    assert set(WIRED_HOOK_IDS) <= core_ids


def test_ci_run_step_derives_the_base_before_judging() -> None:
    text = _joined(_step_script("Run the core validators"))
    assert 'from_ref="$(git merge-base "$base" HEAD)"' in text
    assert "github.event.pull_request.base.sha" in yaml.safe_dump(
        _workflow_jobs()[CORE_JOB]
    )


def test_ci_job_is_unconditional_and_hosted() -> None:
    job = _workflow_jobs()[CORE_JOB]
    assert "needs" not in job
    assert "if" not in job
    assert job["runs-on"] == "ubuntu-latest"
    assert job["name"] == "Core Validators (OMN-21538)"


def test_ci_job_has_a_positive_control_for_each_planted_hook() -> None:
    """Each control pair names a wired hook, plants a file, and fails closed on acceptance."""
    lines = _step_script("Positive control")
    pairs = [word.split(":") for word in _loop_words(lines, "pair")]
    assert sorted(hook for hook, _ in pairs) == sorted(PLANTED_CONTROL_HOOK_IDS)
    text = _joined(lines)
    for hook, planted in pairs:
        assert hook in WIRED_HOOK_IDS
        assert f"> {planted}" in text, f"{planted} is never written"
    # An accepting hook must stop the job: the if-branch exits non-zero.
    accepted = re.search(
        r'if uvx pre-commit@4\.5\.1 run "\$hook" --files "\$planted"; '
        r"then (.*?) fi",
        text,
    )
    assert accepted, "the control does not branch on the hook accepting the plant"
    assert "POSITIVE CONTROL FAILED" in accepted.group(1)
    assert "exit 1" in accepted.group(1)
    assert lines[0] == "set -euo pipefail"
    assert "set +e" not in text


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


def _handler_di_entries() -> dict[str, int]:
    """``path count`` lines of the handler-DI baseline, one per file."""
    entries: dict[str, int] = {}
    for line in (ROOT / HANDLER_DI_BASELINE).read_text("utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path, count = line.split()
        assert path not in entries, f"{path} is listed twice"
        assert int(count) > 0, line
        entries[path] = int(count)
    return entries


def _model_config_entries() -> list[dict[str, Any]]:
    loaded = yaml.safe_load((ROOT / MODEL_CONFIG_BASELINE).read_text("utf-8"))
    assert loaded["schema_version"] == 1
    entries: list[dict[str, Any]] = loaded["entries"]
    return entries


def test_handler_di_gate_reads_its_baseline() -> None:
    (hook,) = [hook for hook in _core_hooks() if hook["id"] == "handler-di-gate"]
    assert hook["args"] == ["--baseline", HANDLER_DI_BASELINE]


def test_baselines_are_per_file() -> None:
    """Each entry names one repository file; none is a glob, directory or blanket."""
    di_paths = set(_handler_di_entries())
    model_entries = _model_config_entries()
    for entry in model_entries:
        assert set(entry) == {"path", "family", "content_sha1"}, entry
        assert SHA1.fullmatch(entry["content_sha1"]), entry
    model_paths = {entry["path"] for entry in model_entries}
    assert di_paths and model_paths
    for path in di_paths | model_paths:
        assert "*" not in path and not path.endswith("/"), path
        assert (ROOT / path).is_file(), f"{path} is not a file in the repository"


def test_baselines_name_no_finding_file_and_carry_no_suppression_marker() -> None:
    listed = set(_handler_di_entries()) | {
        entry["path"] for entry in _model_config_entries()
    }
    assert sorted(listed & set(FINDING_FILES)) == []
    for name in (HANDLER_DI_BASELINE, MODEL_CONFIG_BASELINE):
        text = (ROOT / name).read_text("utf-8")
        # Substring check as well: it also catches a path written another way.
        assert [f for f in FINDING_FILES if f in text] == [], name
        assert not SUPPRESSION_MARKER.search(text), name


def test_model_config_baseline_names_its_zero_date() -> None:
    """The non-empty model-config baseline names the date it reaches zero."""
    header = (ROOT / MODEL_CONFIG_BASELINE).read_text("utf-8").split("entries:")[0]
    assert re.search(r"Target zero: \d{4}-\d{2}-\d{2}\b", header), header


LOCAL_HOME_PATH = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")


def test_lab_proof_plan_tests_name_no_home_directory() -> None:
    """The lab proof plan tests hold no home-directory literal.

    check-local-paths-compute judges every file a change touches, so a path of
    this shape in a file any change edits turns the Core Validators job red on
    each pull request whose range includes that edit.
    """
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted(
            (ROOT / "tests/unit/nodes/node_lab_proof_plan_compute").glob("*.py")
        )
        for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1)
        if LOCAL_HOME_PATH.search(line + "/")
    ]
    assert offenders == []

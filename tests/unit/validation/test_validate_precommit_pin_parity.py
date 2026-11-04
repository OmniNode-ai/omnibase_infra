# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pin-parity read-through is structural, not a substring match (OMN-22360).

A workflow with no literal core pin passes the parity check only when one job
reads the hook block's ``rev`` from ``.pre-commit-config.yaml``, exports it,
and a later step of that job runs the validator with that exact variable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "validate_precommit_pin_parity",
    ROOT / "scripts/validation/validate_precommit_pin_parity.py",
)
assert _SPEC is not None and _SPEC.loader is not None
parity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = parity
_SPEC.loader.exec_module(parity)

HOOK = "check-canonical-inference"
VALIDATOR = "validator_canonical_inference"

READ_STEP = """\
set -eu
spec="$(python -c '
import yaml
config = yaml.safe_load(open(".pre-commit-config.yaml"))
blocks = [r for r in config["repos"] if any(h["id"] == "check-canonical-inference" for h in r["hooks"])]
print("omnibase-core @ git+" + blocks[0]["repo"] + "@" + blocks[0]["rev"])
')"
echo "CORE_SPEC=${spec}" >> "$GITHUB_ENV"
"""
GATE_STEP = (
    'uv run --with "$CORE_SPEC" python -m '
    "omnibase_core.validation.validator_canonical_inference --all\n"
)


def _workflow(*runs: str) -> object:
    return {"jobs": {"gate": {"steps": [{"run": run} for run in runs]}}}


def test_read_then_consume_passes() -> None:
    assert parity._reads_hook_rev(_workflow(READ_STEP, GATE_STEP), HOOK, VALIDATOR)


def test_live_canonical_inference_workflow_reads_the_hook_rev() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/canonical-inference-gate.yml").read_text(
            encoding="utf-8"
        )
    )
    assert parity._reads_hook_rev(workflow, HOOK, VALIDATOR)


def test_yaml_comment_naming_file_and_hook_is_not_a_read() -> None:
    text = (
        '# reads .pre-commit-config.yaml where h["id"] == "check-canonical-inference"\n'
        "jobs:\n  gate:\n    steps:\n      - run: |\n"
        '          uv run --with "$CORE_SPEC" python -m '
        "omnibase_core.validation.validator_canonical_inference\n"
    )
    assert not parity._reads_hook_rev(yaml.safe_load(text), HOOK, VALIDATOR)


def test_shell_comment_read_is_not_a_read() -> None:
    commented = "\n".join(f"# {line}" for line in READ_STEP.splitlines())
    assert not parity._reads_hook_rev(_workflow(commented, GATE_STEP), HOOK, VALIDATOR)


def test_exported_variable_never_consumed_fails() -> None:
    other = GATE_STEP.replace("$CORE_SPEC", "$OTHER_SPEC")
    assert not parity._reads_hook_rev(_workflow(READ_STEP, other), HOOK, VALIDATOR)


def test_consumer_that_does_not_run_the_validator_fails() -> None:
    other = GATE_STEP.replace(VALIDATOR, "validator_url_authority")
    assert not parity._reads_hook_rev(_workflow(READ_STEP, other), HOOK, VALIDATOR)


def test_read_of_another_hook_fails() -> None:
    read = READ_STEP.replace(HOOK, "check-url-authority")
    assert not parity._reads_hook_rev(_workflow(read, GATE_STEP), HOOK, VALIDATOR)


def test_consume_before_read_fails() -> None:
    assert not parity._reads_hook_rev(_workflow(GATE_STEP, READ_STEP), HOOK, VALIDATOR)


def test_read_and_consume_in_different_jobs_fails() -> None:
    workflow = {
        "jobs": {
            "read": {"steps": [{"run": READ_STEP}]},
            "gate": {"steps": [{"run": GATE_STEP}]},
        }
    }
    assert not parity._reads_hook_rev(workflow, HOOK, VALIDATOR)

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16958 incident replay -- bare-interpreter pre-commit hooks.

THE INCIDENT. On 2026-09-15 an OMN-18172 commit in ``omnibase_infra`` on macOS
was refused by ``workflow-input-default-type-parity`` and
``workflow-uv-available`` with ``Executable `python` not found``: both hooks
were ``language: system`` with an ``entry:`` of bare ``python``, and macOS ships
no bare ``python`` outside an activated venv. The same config carried seven
more hooks whose ``entry:`` was a bare ``python3``, which resolves whatever
interpreter the committer's PATH happens to expose instead of the project
environment.

THE REGRESSION CLASS IS ``false_green``: nothing in this repo read hook
interpreters at all, so every commit gate was green while the config refused
commits on a stock Mac. The guard is omnimarket's OMN-17219
``validate_precommit_interpreter.py``, ported unchanged.

THE ARTIFACT is the byte-for-byte ``.pre-commit-config.yaml`` blob at
``omnibase_infra@9114ea9cc`` -- the ``origin/dev`` head the OMN-16958 fix
branched from -- taken with ``git show``, not retyped. The replay points the
real guard's ``CONFIG_PATH`` at those bytes and requires it to refuse, naming
exactly the nine ``language: system`` entries. The accept control is the live
config (``test_live_config_has_no_bare_interpreter_hooks`` in
``tests/ci/test_precommit_interpreter_gate.py``), so a guard stuck closed fails.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "validation" / "validate_precommit_interpreter.py"
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "omn16958" / "pre-commit-config.yaml.captured"
)
EXPECTED_SHA256 = "3049619aa68af6d5e660234111a7e25ab38d400fcdaebd25f48435c5c06e3a9b"

BARE_PYTHON_SYSTEM_HOOKS = {
    "workflow-input-default-type-parity",
    "workflow-uv-available",
}
BARE_PYTHON3_HOOKS = {
    "exposed-identifier-gate",
    "onex-check-migration-append-only",
    "onex-check-migration-rls-policy-atomicity",
    "onex-check-topology-grant-delivery",
    "onex-check-duplication-sweep",
    "check-reconciler-movement-proof",
    "check-reconciler-privilege",
}


def _load_gate() -> Any:
    spec = importlib.util.spec_from_file_location("precommit_interpreter_gate", GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_captured_config_is_unmodified() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_the_real_guard_rejects_the_config_that_refused_the_commit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate()
    original = gate.CONFIG_PATH
    gate.CONFIG_PATH = FIXTURE
    try:
        rc = gate.main()
    finally:
        gate.CONFIG_PATH = original
    err = capsys.readouterr().err

    assert rc == 1, err
    flagged = dict(re.findall(r"^  - ([\w-]+): entry invokes bare `(\w+)`", err, re.M))
    assert flagged == {
        **dict.fromkeys(BARE_PYTHON_SYSTEM_HOOKS, "python"),
        **dict.fromkeys(BARE_PYTHON3_HOOKS, "python3"),
    }, err

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the onex CLI registry-vocabulary guard (OMN-19407).

The artifact is ``src/omnibase_infra/cli/cli_delegate.py`` exactly as it stood
on ``dev`` at 1888c0a83 on 2026-09-24, captured with ``git show`` from the
object, not retyped. It carries the three hand-written vocabularies the
operator ruled out that day: ``TASK_TYPE_CHOICES`` (11 classes against a
task-class contract that declared 15, so ``documentation``,
``validator_generation`` and ``escalation`` were unreachable by name),
``DELEGATE_SOURCE_CHOICES`` and a literal ``--criteria-mode`` choice. Each was
"pinned" by a test holding another copy, and nothing refused a new one.

The discriminator is load-bearing: a guard that refused every CLI module would
replay this perfectly and block every change to the CLI, so the same guard is
run over the real package on this branch and must accept it.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _ROOT / "tests/fixtures/omn19407/cli_delegate.py.captured"
_SHA256 = "ed879dd330394da11c06afafbadf8b5be28d333db49d2ae00eafe2bb7134184a"
_GUARD = _ROOT / "scripts/validation/check_cli_registry_vocabulary.py"


def _guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_cli_registry_vocabulary", _GUARD
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fixture_is_the_captured_bytes() -> None:
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _SHA256


def test_the_real_guard_refuses_the_cli_that_kept_its_own_lists() -> None:
    findings = _guard().check_file(_FIXTURE)
    joined = "\n".join(findings)
    assert "TASK_TYPE_CHOICES" in joined
    assert "DELEGATE_SOURCE_CHOICES" in joined
    assert "extend-task-class" in joined
    assert len(findings) == 3, findings


def test_the_same_guard_accepts_the_cli_that_reads_the_registry() -> None:
    assert _guard().main([str(_ROOT / "src/omnibase_infra/cli")]) == 0

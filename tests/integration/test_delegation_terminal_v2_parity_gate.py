# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the delegation v2 terminal topics (OMN-15622).

The unit suite (``tests/unit/topics/test_delegation_terminal_v2_provisioning.py``)
asserts the constants, the ``ModelTopicSpec`` entries and the allowlist entries in
isolation. This module exercises the surfaces that only exist when the pieces are
put together:

* the real ``scripts/check_contract_topic_parity.py`` run end to end as a
  subprocess, over the real ``src/omnibase_infra/nodes`` tree and the real
  provisioned-suffix registry, asserting a MEASURED zero rather than an empty
  result;
* the seeded-RED direction — with one v2 allowlist entry removed in memory, the
  same scanner must FAIL and name the exact wire topic, so the zero above is
  known to be load-bearing for these three suffixes specifically;
* the cross-package seam — omnibase_core's canonical topic-suffix validator
  accepting the suffixes this repo declares.

No broker is required: the parity scanner is a static cross-repo reader, and the
core validator is pure.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

from omnibase_core.validation.validator_topic_suffix import validate_topic_suffix
from omnibase_infra.topics import (
    ALL_PROVISIONED_SUFFIXES,
    SUFFIX_DELEGATION_COMPLETED_V2,
    SUFFIX_DELEGATION_FAILED_ROUTED_V2,
    SUFFIX_DELEGATION_FAILED_UNROUTED_V2,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = REPO_ROOT / "scripts" / "check_contract_topic_parity.py"

V2_TERMINAL_SUFFIXES: tuple[str, ...] = (
    SUFFIX_DELEGATION_COMPLETED_V2,
    SUFFIX_DELEGATION_FAILED_ROUTED_V2,
    SUFFIX_DELEGATION_FAILED_UNROUTED_V2,
)


def _load_parity_module() -> ModuleType:
    """Import the parity script from its real path.

    Loaded from the real location so the script's own
    ``Path(__file__).parent.parent`` repo-root resolution still points at this
    checkout. A fresh module object per call keeps one test's in-memory edit out
    of the next test.
    """
    spec = importlib.util.spec_from_file_location(
        "_omn15622_parity_integration", PARITY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_parity_gate_passes_end_to_end_with_zero_python_only_gaps() -> None:
    """The real script, run as CI runs it, reports zero python-only gaps.

    stderr is captured and surfaced on failure rather than discarded: a scanner
    that errors out would otherwise return no gap rows and read exactly like a
    clean bill of health.
    """
    completed = subprocess.run(
        [sys.executable, str(PARITY_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined
    assert "python-only gaps:   0" in completed.stdout, combined


@pytest.mark.integration
@pytest.mark.parametrize("suffix", V2_TERMINAL_SUFFIXES)
def test_removing_one_v2_allowlist_entry_fails_the_gate_and_names_it(
    suffix: str,
) -> None:
    """Seeded RED: each v2 entry is individually load-bearing.

    The edit is in memory, on a freshly loaded copy of the module — the
    repository file is never written. Without this direction, the passing run
    above would not distinguish "these three suffixes are correctly allowlisted"
    from "the scanner never looked at them".
    """
    module = _load_parity_module()
    module._LEGACY_ALLOWLIST = {
        k: v for k, v in module._LEGACY_ALLOWLIST.items() if k != suffix
    }
    module._ALLOWLISTED_SUFFIXES = frozenset(module._LEGACY_ALLOWLIST.keys())

    captured = io.StringIO()
    with redirect_stdout(captured):
        exit_code = module.run_parity_check()
    output = captured.getvalue()

    assert exit_code != 0, output
    assert suffix in output, output


@pytest.mark.integration
def test_unpatched_in_process_run_passes() -> None:
    """Positive control for the harness above.

    Proves the in-process runner reaches a PASS on an unmodified allowlist, so a
    failure in the seeded-RED tests is attributable to the removed entry and not
    to the way the module is loaded.
    """
    module = _load_parity_module()
    captured = io.StringIO()
    with redirect_stdout(captured):
        exit_code = module.run_parity_check()
    assert exit_code == 0, captured.getvalue()


@pytest.mark.integration
@pytest.mark.parametrize("suffix", V2_TERMINAL_SUFFIXES)
def test_v2_suffix_is_valid_under_the_core_topic_validator(suffix: str) -> None:
    """Cross-package seam: omnibase_core accepts what omnibase_infra declares."""
    result = validate_topic_suffix(suffix)
    assert result.is_valid, result.error


@pytest.mark.integration
def test_core_validator_rejects_a_malformed_sibling() -> None:
    """Negative control: the validator is not accepting everything handed to it."""
    result = validate_topic_suffix("onex.evt.omnibase-infra.delegation-failed-routed")
    assert not result.is_valid


@pytest.mark.integration
@pytest.mark.parametrize("suffix", V2_TERMINAL_SUFFIXES)
def test_v2_suffix_reaches_the_assembled_provisioning_registry(suffix: str) -> None:
    """The composed registry the provisioner consumes carries all three."""
    assert suffix in ALL_PROVISIONED_SUFFIXES

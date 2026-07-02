# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard tests for the node-skill-package install script (OMN-13829).

These are hermetic: they never hit the network or mutate a venv. They validate
the committed script's invariants (immutable pin, --no-deps composition of the
market provider layer, an --execute gate, and portability) and exercise the
dry-run path, which prints the plan without installing anything.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "install-node-skill-package.sh"

# Immutable dev-branch rev pinned by the script (see script header for why a dev
# rev + --no-deps is the canonical composition of the market provider layer).
# omnimarket@dev HEAD as of 2026-07-02, carrying OMN-13836 (clean uv overrides).
_EXPECTED_REV = "bc516ef5da67a348947fbb0e3c88dc964b2cd541"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_script_exists_and_executable() -> None:
    assert _SCRIPT.is_file(), f"missing install script: {_SCRIPT}"
    mode = _SCRIPT.stat().st_mode
    assert mode & 0o111, "install script must be executable"


def test_script_has_spdx_header() -> None:
    head = _script_text().splitlines()[:4]
    joined = "\n".join(head)
    assert "SPDX-License-Identifier: MIT" in joined
    assert "SPDX-FileCopyrightText" in joined


def test_pins_immutable_full_sha_not_mutable_tag() -> None:
    text = _script_text()
    assert _EXPECTED_REV in text, "script must pin the vetted omnimarket rev"
    # The default rev must be a full 40-hex SHA, never a branch name or short tag
    # (mutable refs defeat reproducibility — CLAUDE.md rule: prefer SHAs).
    m = re.search(r'OMNIMARKET_REF="\$\{OMNIMARKET_REF:-([^}"]+)\}"', text)
    assert m is not None, "OMNIMARKET_REF default not found in expected form"
    default_ref = m.group(1)
    assert re.fullmatch(r"[0-9a-f]{40}", default_ref), (
        f"default OMNIMARKET_REF must be a full 40-hex SHA, got: {default_ref!r}"
    )


def test_no_deps_used_for_market_provider_layer() -> None:
    text = _script_text()
    # omnimarket sits ABOVE the infra layer; it must be composed --no-deps so its
    # metadata never re-resolves (or downgrades) the infra layer beneath it.
    assert "--no-deps" in text
    assert "omnibase-compat==0.5.5" in text
    assert "omninode-memory==0.15.0" in text
    assert "OmniNode-ai/omnimarket.git" in text


def test_verifies_merge_sweep_and_session_nodes() -> None:
    text = _script_text()
    # DoD: after install, the nodes behind `onex skill merge_sweep` and
    # `onex skill session` must resolve. Step 3 asserts both entry points.
    assert "node_pr_lifecycle_orchestrator" in text
    assert "node_session_orchestrator" in text


def test_framed_as_canonical_not_interim() -> None:
    # Operator-confirmed direction (OMN-13829): this co-install is the CANONICAL
    # mechanism, not a stopgap. Guard against interim/retire framing regressing.
    lowered = _script_text().lower()
    assert "canonical co-install" in lowered
    for banned in ("retire this script", "interim", "workaround", "stopgap"):
        assert banned not in lowered, f"interim-framing token present: {banned!r}"


def test_has_execute_gate() -> None:
    text = _script_text()
    assert "--execute" in text, "script must gate mutation behind --execute"
    assert "DRY RUN" in text, "script must default to a dry-run plan"


def test_no_hardcoded_absolute_machine_paths() -> None:
    # CLAUDE.md rule #6: no /Users/ or /Volumes/ absolute paths in source.
    text = _script_text()
    for token in ("/Users/", "/Volumes/"):
        assert token not in text, f"hardcoded machine path {token!r} present"


def test_dry_run_prints_plan_and_does_not_require_execute() -> None:
    # Dry run against a non-existent python still prints nothing that installs;
    # it fails fast on interpreter resolution rather than touching any venv.
    result = subprocess.run(
        ["bash", str(_SCRIPT), "/nonexistent/python"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "not executable" in (result.stdout + result.stderr)


def test_dry_run_with_current_interpreter_prints_plan() -> None:
    # Using the running interpreter (guaranteed executable) exercises the plan
    # print path; without --execute it must not install anything.
    result = subprocess.run(
        ["bash", str(_SCRIPT), sys.executable],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "node-skill-package install plan" in result.stdout
    assert "DRY RUN" in result.stdout
    assert _EXPECTED_REV in result.stdout

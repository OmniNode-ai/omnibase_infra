# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard tests for the node-skill-package install script (OMN-13829, OMN-14060).

These are hermetic: they never hit the network or mutate a venv. Every exec
test pins ``OMNIMARKET_REF`` explicitly so the dynamic-resolution path (which
does a live ``git ls-remote``, OMN-14060) is never exercised here — that path
was verified manually against the live repo (see the OMN-14060 PR body) rather
than baked into this suite, to keep it network-independent. The tests below
validate the committed script's invariants (dynamic-by-default ref resolution
with a pinning override, --no-deps composition of the market provider layer,
an --execute gate, and portability) and exercise the dry-run path, which
prints the plan without installing anything.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "install-node-skill-package.sh"

# A fixed, syntactically-valid 40-hex SHA used to pin OMNIMARKET_REF in every
# exec test below, so none of them trigger the live `git ls-remote` resolution
# path (OMN-14060) — keeps this suite hermetic regardless of network state.
_PINNED_TEST_REF = "0123456789abcdef0123456789abcdef01234567"


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


def test_resolves_ref_dynamically_from_live_dev_head() -> None:
    # OMN-14060: a hand-edited SHA literal goes stale the moment omnimarket@dev
    # advances past it (the OMN-13829 recurrence mechanism). The default path
    # must resolve from the live `dev` branch, never a baked-in literal.
    text = _script_text()
    assert "git ls-remote" in text
    assert '"$OMNIMARKET_GIT" dev' in text or 'OMNIMARKET_GIT}" dev' in text


def test_omnimarket_ref_override_still_takes_precedence() -> None:
    # An operator-set OMNIMARKET_REF (pinned/offline use) must win outright —
    # the dynamic resolution branch must never run when it's set.
    text = _script_text()
    assert 'if [[ -n "${OMNIMARKET_REF:-}" ]]; then' in text


def test_has_offline_fallback_to_local_canonical_clone() -> None:
    # When `git ls-remote` is unreachable (offline) and no explicit override is
    # given, fall back to the already-checked-out local clone at
    # $OMNI_HOME/omnimarket rather than hard-failing outright.
    text = _script_text()
    assert "OMNI_HOME" in text
    assert "omnimarket/.git" in text
    assert "rev-parse HEAD" in text


def test_fails_fast_when_ref_cannot_be_resolved() -> None:
    # No silent fallback to a stale default (CLAUDE.md rule #8) — if neither
    # live resolution nor the local-clone fallback succeeds, exit non-zero
    # with an actionable message.
    text = _script_text()
    assert "could not resolve an omnimarket ref" in text
    assert "exit 1" in text


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
    # print path; without --execute it must not install anything. OMNIMARKET_REF
    # is pinned so this never triggers the live `git ls-remote` resolution path
    # (OMN-14060) — keeps the test hermetic regardless of network state.
    result = subprocess.run(
        ["bash", str(_SCRIPT), sys.executable],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={**os.environ, "OMNIMARKET_REF": _PINNED_TEST_REF},
    )
    assert result.returncode == 0, result.stderr
    assert "node-skill-package install plan" in result.stdout
    assert "DRY RUN" in result.stdout
    assert _PINNED_TEST_REF in result.stdout
    assert "OMNIMARKET_REF override (pinned/offline use)" in result.stdout


def test_dry_run_without_override_does_not_touch_network_before_python_check() -> None:
    # Reordered (OMN-14060): interpreter resolution runs BEFORE ref resolution,
    # so a bad interpreter path fails fast without ever doing a `git ls-remote`.
    # No OMNIMARKET_REF is set here on purpose -- this proves the ordering.
    env = {k: v for k, v in os.environ.items() if k != "OMNIMARKET_REF"}
    result = subprocess.run(
        ["bash", str(_SCRIPT), "/nonexistent/python"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=env,
    )
    assert result.returncode != 0
    assert "not executable" in (result.stdout + result.stderr)
    assert "Resolving omnimarket ref" not in (result.stdout + result.stderr)

# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""deploy-runtime.sh step 3b must resolve omnibase_core runtime contracts from the
OMNI_HOME sibling clone's real source-tree path, not only from a pip-installed
omnibase_core (OMN-15122).

Live failure (run https://github.com/OmniNode-ai/omnibase_infra/actions/runs/30166226948,
2026-07-25T16:44Z): every release-train-lab deploy job aborted in sync_files() at
step 3b with "Could not locate omnibase_core runtime contracts." Root cause,
reproduced directly on the deploy runner (not inferred):

  * ``importlib.util.find_spec('omnibase_core')`` returned ``None`` -- the
    runner's python3 has no omnibase_core installed at all.
  * The previous editable-install fallback assumed a site-packages-shaped
    layout (``<pkg_dir>/../../contracts/runtime_data``) that does not match
    the real omnibase_core source-tree layout
    (``<repo>/src/omnibase_core/contracts/runtime_data``), so even an
    ``ls`` of the runner's staged ``omnibase_core`` checkout came up empty.

The fix (``resolve_core_contracts_dir()``) resolves the contracts directory
from the OMNI_HOME sibling clone/checkout filesystem path FIRST -- the deploy
source of truth, matching the pinned sibling clone the rest of the workspace
build vendors from -- and falls back to python import resolution second, for
hosts where omnibase_core happens to be pip/editable-installed. Neither
resolving is a hard, named failure that prints every path probed; there is no
silent default.

These tests extract and execute the pure ``resolve_core_contracts_dir()``
bash function in isolation against a real (fake) filesystem -- proving RED
against the exact "directory exists but wrong shape" failure mode this ticket
diagnosed, not merely asserting the source text mentions the right strings.

A stubbed ``python3`` is always placed first on PATH so the secondary (import)
probe is deterministic across hosts: without it, a machine where
``omnibase_core`` happens to be pip/editable-installed into the active venv
(e.g. ``omnibase_infra``'s own venv on a dev box, since omnibase_core is a
runtime dependency) would make the secondary probe succeed unpredictably,
masking whether the *primary* (OMNI_HOME filesystem) probe actually did the
work these tests exist to prove.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"

_EXPECTED_RUNTIME_YAML_NAMES = (
    "contract_loader_effect.yaml",
    "contract_registry_reducer.yaml",
    "event_bus_wiring_effect.yaml",
    "node_graph_reducer.yaml",
    "runtime_orchestrator.yaml",
)

# A fake python3 that always behaves as if omnibase_core is NOT importable
# (exit 0, empty stdout) -- deterministic stand-in for the OMN-15122 runner
# state, regardless of whatever the host running this test actually has
# installed.
_STUB_PYTHON3 = """\
#!/usr/bin/env bash
# Deterministic stub: always answers as if omnibase_core is not importable.
exit 0
"""


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    """Return the source text of a single top-level bash function ``name()``.

    Mirrors the extraction helper in test_deploy_runtime_lane_overlay.py: the
    function under test is pure (no top-level script deps beyond ``python3``
    on PATH), so it can be executed in isolation for a real behavioral
    assertion without sourcing (and running the ``main`` of) the whole script.
    """
    text = _script_text()
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runtime.sh"
    )
    return match.group(0)


def _run_resolver(
    omni_home: str | None, *, stub_bin_dir: Path
) -> subprocess.CompletedProcess[str]:
    """Execute the extracted resolver in a harness with a deterministic PATH."""
    stub_python3 = stub_bin_dir / "python3"
    stub_python3.write_text(_STUB_PYTHON3, encoding="utf-8")
    stub_python3.chmod(0o755)

    env_line = (
        f'export OMNI_HOME="{omni_home}"'
        if omni_home is not None
        else "unset OMNI_HOME || true"
    )

    harness = "\n".join(
        [
            "set -uo pipefail",  # no -e: we need the resolver's own exit code, not an early abort
            f'export PATH="{stub_bin_dir}:$PATH"',
            env_line,
            _extract_function("resolve_core_contracts_dir"),
            "declare -a probed=()",
            "resolved=''",
            "resolve_core_contracts_dir probed resolved",
            "rc=$?",
            'printf "RESOLVED=%s\\n" "$resolved"',
            'printf "PROBED_COUNT=%s\\n" "${#probed[@]}"',
            'for p in "${probed[@]:-}"; do printf "PROBED: %s\\n" "$p"; done',
            "exit $rc",
        ]
    )
    return subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_defines_resolve_core_contracts_dir_function() -> None:
    text = _script_text()
    assert re.search(r"^resolve_core_contracts_dir\s*\(\)", text, re.MULTILINE), (
        "deploy-runtime.sh must define resolve_core_contracts_dir()"
    )


@pytest.mark.unit
def test_step_3b_calls_resolver_not_inline_python() -> None:
    """Step 3b must delegate to the shared resolver, not re-inline find_spec logic."""
    lines = [
        line
        for line in _script_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    text = "\n".join(lines)
    assert (
        "resolve_core_contracts_dir core_contracts_probed core_contracts_dir" in text
    ), (
        "sync_files()'s step 3b must call resolve_core_contracts_dir(), passing "
        "both the probed-paths array and the resolved-dir output variable by name"
    )


@pytest.mark.unit
def test_resolves_from_omni_home_sibling_clone_source_tree(tmp_path: Path) -> None:
    """Primary path: the real src/omnibase_core/contracts/runtime_data layout.

    This is the exact shape of the OMN-15122 live failure reproduced: the
    deploy runner has an OMNI_HOME-rooted omnibase_core checkout, and the
    contracts live under src/omnibase_core/contracts/runtime_data relative to
    that checkout's root -- not under a bare omnibase_core/contracts/runtime_data
    (the shape the old editable-install fallback assumed and which never
    existed on the runner, per the ticket's `ls .../omnibase_core/contracts/
    runtime_data/*.yaml -> No such file or directory` readback).
    """
    fake_omni_home = tmp_path / "omni_home"
    runtime_data = (
        fake_omni_home
        / "omnibase_core"
        / "src"
        / "omnibase_core"
        / "contracts"
        / "runtime_data"
    )
    runtime_data.mkdir(parents=True)
    for name in _EXPECTED_RUNTIME_YAML_NAMES:
        (runtime_data / name).write_text("contract: {}\n", encoding="utf-8")

    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    result = _run_resolver(str(fake_omni_home), stub_bin_dir=stub_bin)

    assert result.returncode == 0, result.stderr
    assert f"RESOLVED={runtime_data}\n" in result.stdout, result.stdout
    assert "PROBED_COUNT=1" in result.stdout, result.stdout
    assert str(runtime_data) in result.stdout


@pytest.mark.unit
def test_wrong_shape_directory_is_red_not_silently_accepted(tmp_path: Path) -> None:
    """RED against 'the directory exists but at the wrong (old-assumed) path'.

    Reproduces the ticket's exact failed probe: an OMNI_HOME sibling clone
    exists, but only the (incorrect) bare `omnibase_core/contracts/runtime_data`
    shape is present -- not the real `omnibase_core/src/omnibase_core/contracts/
    runtime_data` shape. The resolver must NOT accept this and must fail closed
    (the stubbed python3 guarantees the secondary path cannot rescue it either).
    """
    fake_omni_home = tmp_path / "omni_home"
    wrong_shape_dir = fake_omni_home / "omnibase_core" / "contracts" / "runtime_data"
    wrong_shape_dir.mkdir(parents=True)
    for name in _EXPECTED_RUNTIME_YAML_NAMES:
        (wrong_shape_dir / name).write_text("contract: {}\n", encoding="utf-8")

    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    result = _run_resolver(str(fake_omni_home), stub_bin_dir=stub_bin)

    assert result.returncode == 1, (
        f"expected fail-closed exit 1, got rc={result.returncode} stdout={result.stdout!r}"
    )
    assert "RESOLVED=\n" in result.stdout


@pytest.mark.unit
def test_omni_home_unset_fails_closed_with_named_probe(tmp_path: Path) -> None:
    """No OMNI_HOME and no importable omnibase_core -> fail closed, not silent."""
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    result = _run_resolver(None, stub_bin_dir=stub_bin)

    assert result.returncode == 1, (
        f"expected fail-closed exit 1, got rc={result.returncode} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "RESOLVED=\n" in result.stdout
    assert "OMNI_HOME unset" in result.stdout


@pytest.mark.unit
def test_neither_probe_resolves_reports_both_paths(tmp_path: Path) -> None:
    """OMNI_HOME set but missing the sibling clone -> both probes named on failure."""
    fake_omni_home = tmp_path / "omni_home_without_core"
    fake_omni_home.mkdir()
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()

    result = _run_resolver(str(fake_omni_home), stub_bin_dir=stub_bin)

    assert result.returncode == 1, (
        f"expected fail-closed exit 1, got rc={result.returncode} stdout={result.stdout!r}"
    )
    assert "PROBED_COUNT=2" in result.stdout, result.stdout
    assert (
        str(fake_omni_home / "omnibase_core" / "src" / "omnibase_core") in result.stdout
    )


@pytest.mark.unit
def test_step_3b_error_path_names_every_probed_path() -> None:
    """On total failure, step 3b must log every path probed, not a fixed hint only."""
    text = "\n".join(
        line
        for line in _script_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "Probed the following paths:" in text
    assert 'for probed_path in "${core_contracts_probed[@]}"' in text

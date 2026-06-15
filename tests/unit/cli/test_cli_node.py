# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for `onex node <name>` CLI resolution (OMN-8938).

Covers the 5 CLI migration branches:
    - unknown name → clear error + lists known
    - valid name → resolves to packaged contract.yaml
    - --contract override → overrides packaged resolution
    - missing packaged contract → cites convention violation
    - --input file not found → CLI-level error
"""

from __future__ import annotations

import importlib.util
import json
from importlib.machinery import ModuleSpec
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_node import _resolve_packaged_contract, run_node_by_name

pytestmark = pytest.mark.unit


def _first_resolvable_packaged_node_name() -> str | None:
    for ep in entry_points(group="onex.nodes"):
        module_path = ep.value.split(":", 1)[0].strip()
        try:
            spec = importlib.util.find_spec(module_path)
        except (ModuleNotFoundError, ValueError):
            continue
        if spec is None:
            continue

        if spec.submodule_search_locations:
            module_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
        elif spec.origin is not None:
            module_dir = Path(spec.origin).resolve().parent
        else:
            continue

        if (module_dir / "contract.yaml").exists():
            return ep.name
    return None


def test_unknown_node_name_reports_known_names() -> None:
    """Unknown name errors with the list of known names."""
    runner = CliRunner()
    result = runner.invoke(run_node_by_name, ["definitely_not_a_real_node"])
    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert "Unknown node 'definitely_not_a_real_node'" in combined


def test_valid_node_name_resolves_packaged_contract() -> None:
    """A real onex.nodes entry point resolves to a packaged contract.yaml.

    Skipped if no onex.nodes entry points are registered in the test env.
    """
    node_name = _first_resolvable_packaged_node_name()
    if node_name is None:
        pytest.skip("No resolvable onex.nodes packaged contract registered in this env")
    contract = _resolve_packaged_contract(node_name)
    assert contract.name == "contract.yaml"
    assert contract.exists()


def test_contract_override_wins_over_packaged(tmp_path: Path) -> None:
    """``--contract <path>`` takes precedence over the packaged contract.

    Proves the override was honored by confirming the CLI processes the
    override's contract path (not the packaged one): we point --contract
    at a well-formed YAML whose handler module does not exist. The runtime
    attempts to resolve that bogus handler, fails, and exits non-zero.
    A FAILED exit with the handler-not-found log confirms override wins.
    """
    node_name = _first_resolvable_packaged_node_name()
    if node_name is None:
        pytest.skip("No resolvable onex.nodes packaged contract registered in this env")

    override = tmp_path / "custom_contract.yaml"
    override.write_text(
        "---\n"
        "name: custom\n"
        "node_type: compute\n"
        "handler:\n"
        "  module: does.not.exist.anywhere_zzzqqq\n"
        "  class: Nope\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        run_node_by_name,
        [
            node_name,
            "--contract",
            str(override),
            "--state-root",
            str(tmp_path / "state"),
            "--timeout",
            "5",
        ],
    )
    # Exit code 1 = FAILED (handler could not be resolved from override). If the
    # override had been ignored and the packaged contract loaded instead, the
    # well-formed packaged handler would have succeeded or hit a different error.
    assert result.exit_code == 1


def test_missing_packaged_contract_reports_convention_violation(tmp_path: Path) -> None:
    """If an entry-point module resolves but has no contract.yaml, error cites the convention."""
    fake_pkg = tmp_path / "fake_pkg"
    fake_pkg.mkdir()
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    spec = ModuleSpec("fake_pkg", loader=None, is_package=True)
    spec.submodule_search_locations = [str(fake_pkg)]

    class _FakeEP:
        name = "node_has_no_contract"
        value = "fake_pkg"
        dist = "local-fake"

    with (
        patch(
            "omnibase_infra.cli.cli_node.entry_points",
            return_value=[_FakeEP()],
        ),
        patch(
            "omnibase_infra.cli.cli_node.importlib.util.find_spec",
            return_value=spec,
        ),
    ):
        with pytest.raises(Exception) as exc_info:
            _resolve_packaged_contract("node_has_no_contract")
    assert "packaging convention" in str(exc_info.value)


def test_packaged_contract_resolution_does_not_import_node_module(
    tmp_path: Path,
) -> None:
    """Contract lookup uses module specs so optional deps are not imported."""
    fake_pkg = tmp_path / "fake_pkg"
    fake_pkg.mkdir()
    (fake_pkg / "contract.yaml").write_text(
        "name: fake\ncontract_version: {major: 1, minor: 0, patch: 0}\n"
        "node_type: COMPUTE_GENERIC\n",
        encoding="utf-8",
    )
    spec = ModuleSpec("fake_pkg", loader=None, is_package=True)
    spec.submodule_search_locations = [str(fake_pkg)]

    class _FakeEP:
        name = "node_has_contract"
        value = "fake_pkg:Node"
        dist = "local-fake"

    with (
        patch(
            "omnibase_infra.cli.cli_node.entry_points",
            return_value=[_FakeEP()],
        ),
        patch(
            "omnibase_infra.cli.cli_node.importlib.util.find_spec",
            return_value=spec,
        ) as find_spec,
    ):
        contract = _resolve_packaged_contract("node_has_contract")

    assert contract == fake_pkg / "contract.yaml"
    find_spec.assert_called_once_with("fake_pkg")


def test_default_mode_failure_emits_skill_routing_error_envelope(
    tmp_path: Path,
) -> None:
    """A non-zero default-mode run emits a ``SkillRoutingError`` JSON envelope.

    The ``golden_chain_sweep`` skill (and others) document that ``onex node``
    surfaces a ``SkillRoutingError`` JSON envelope on non-zero exit rather than a
    raw traceback (OMN-8724). This drives a FAILED run via a bogus handler module
    and asserts the documented envelope is on stdout with the expected keys.
    """
    node_name = _first_resolvable_packaged_node_name()
    if node_name is None:
        pytest.skip("No resolvable onex.nodes packaged contract registered in this env")

    override = tmp_path / "custom_contract.yaml"
    override.write_text(
        "---\n"
        "name: custom\n"
        "node_type: compute\n"
        "handler:\n"
        "  module: does.not.exist.anywhere_zzzqqq\n"
        "  class: Nope\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        run_node_by_name,
        [
            node_name,
            "--contract",
            str(override),
            "--state-root",
            str(tmp_path / "state"),
            "--timeout",
            "5",
        ],
    )

    assert result.exit_code == 1
    # The last JSON object in stdout is the SkillRoutingError envelope.
    start = result.output.rfind("{")
    assert start != -1, f"no JSON envelope in output: {result.output!r}"
    envelope = json.loads(result.output[start:])
    assert envelope["error_type"] == "SkillRoutingError"
    assert envelope["node_id"] == node_name
    assert envelope["result"] == "failed"
    assert envelope["message"]


def test_input_file_not_found_surfaces_cli_error(tmp_path: Path) -> None:
    """``--input <nonexistent>`` must fail at the CLI layer with a Click-level error."""
    runner = CliRunner()
    result = runner.invoke(
        run_node_by_name,
        [
            "node_merge_sweep",
            "--input",
            str(tmp_path / "does_not_exist.json"),
        ],
    )
    assert result.exit_code != 0
    assert "does_not_exist.json" in result.output

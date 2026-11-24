# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A merged ticket's relative-cwd checks run in the product clone (OMN-26650).

Recorded failure (``evidence-autoclose-sweep.yml`` run 44672842883, job
136077602971, head e952143d3, 2026-11-20T04:12:23Z; fixture): 26 behaviour
checks of 8 omnibase_infra tickets were recorded FAILED with
``ModuleNotFoundError: No module named 'scripts'``, ``exit_code=4`` and no test
collected, each reading ``ImportError while loading conftest
'<RUNNER_TEMP>/contract-roots/omnibase_infra/conftest.py'``.

``contract-roots/<repo>`` is the sparse, contracts-only clone the sweep reads
contracts from. In cone mode it keeps the repository's root files (so
``conftest.py`` is there) and nothing else (so ``scripts/`` is not). The
verifier anchors a relative check ``cwd`` at the repository owning the contract
file, so every contract saying ``cwd: .`` ran pytest in that tree, where the root
``conftest.py`` imports ``scripts.hooks...``. The fix hands the verifier the
materialised product clone beside ``OMNI_HOME`` as its ``CONTRACT_REPO_DIR``.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from omnibase_infra.handlers import done_write_receipt_guard as guard
from omnibase_infra.handlers.done_write_receipt_guard import (
    materialised_product_clone,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "sweep_scripts_import_failures_run_44672842883.json"
)

_ROOT_CONFTEST = textwrap.dedent(
    """\
    from scripts.hooks.pytest_full_suite_host_guard import enforce
    """
)

# Stands in for the verifier's check phase: it anchors where the real
# `EvidenceCollector._resolve_contract_repo_dir` does (an explicit
# CONTRACT_REPO_DIR first, else the parent of the contract's `contracts/`
# directory) and loads that tree's root conftest the way pytest does.
_VERIFIER = textwrap.dedent(
    """\
    import json, os, pathlib, runpy, sys

    contract = pathlib.Path(sys.argv[1])
    anchor = os.environ.get("CONTRACT_REPO_DIR") or str(contract.parent.parent)
    sys.path.insert(0, anchor)
    try:
        runpy.run_path(str(pathlib.Path(anchor) / "conftest.py"))
        error = ""
    except ModuleNotFoundError as exc:
        error = f"ModuleNotFoundError: {exc}"
    print(json.dumps({"anchor": anchor, "error": error}))
    """
)


def _recorded() -> list[dict[str, str]]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["failed_checks"]


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The hosted runner's two trees: sparse contract root, materialised clone."""
    sparse = tmp_path / "temp" / "contract-roots" / "omnibase_infra"
    (sparse / "contracts").mkdir(parents=True)
    (sparse / "conftest.py").write_text(_ROOT_CONFTEST)
    (sparse / "contracts" / "OMN-26106.yaml").write_text("ticket_id: OMN-26106\n")

    home = tmp_path / "work"
    full = home / "omnibase_infra"
    hooks = full / "scripts" / "hooks"
    hooks.mkdir(parents=True)
    (full / "scripts" / "__init__.py").write_text("")
    (hooks / "__init__.py").write_text("")
    (hooks / "pytest_full_suite_host_guard.py").write_text(
        "def enforce(config, target):\n    return None\n"
    )
    (full / "conftest.py").write_text(_ROOT_CONFTEST)
    return sparse, home, full


def test_scripts_import_recorded_failures_ran_in_the_contract_reading_clone() -> None:
    rows = _recorded()
    assert len(rows) == 26
    assert len({row["ticket_id"] for row in rows}) == 8
    for row in rows:
        assert row["status"] == "failed"
        assert "No module named 'scripts'" in row["message_excerpt"], row
    in_sparse_tree = [
        r for r in rows if "<RUNNER_TEMP>/contract-roots/omnib" in r["message_excerpt"]
    ]
    assert len(in_sparse_tree) == 25
    assert not any(
        "work/omnibase_infra/omnibase_infra" in r["message_excerpt"] for r in rows
    )


def test_scripts_import_materialised_clone_resolution(tmp_path: Path) -> None:
    sparse, home, full = _layout(tmp_path)

    assert materialised_product_clone(str(sparse), str(home)) == str(full.resolve())
    # A contract root that already is the product clone needs no override.
    assert materialised_product_clone(str(full), str(home)) == ""
    # No registry root, or no materialised clone of that repository: nothing to point at.
    assert materialised_product_clone(str(sparse), "") == ""
    assert materialised_product_clone(str(sparse), str(tmp_path / "elsewhere")) == ""
    assert materialised_product_clone("", str(home)) == ""


async def test_scripts_import_sweep_verifier_loads_the_product_clone_conftest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real runner and a real subprocess, the recorded layout."""
    sparse, home, full = _layout(tmp_path)

    def argv(
        ticket_id: str, *, contract_path: str = "", state_root: str = ""
    ) -> list[str]:
        return [sys.executable, "-c", _VERIFIER, contract_path]

    monkeypatch.setattr(guard, "dod_verify_argv", argv)
    monkeypatch.setenv("OMNI_HOME", str(home))
    handler = HandlerEvidenceAutocloseSweep(autoclose_disabled=False)

    result, code, error = await handler._run_dod_verify_command_real(
        "OMN-26106", str(sparse), 30
    )

    assert (code, error) == (0, "")
    assert result == {"anchor": str(full.resolve()), "error": ""}


async def test_scripts_import_without_a_materialised_clone_keeps_the_contract_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: with no product clone to anchor at, the verifier is
    unchanged and still meets the recorded error, so the test above is the fix
    working and not a verifier that cannot fail."""
    sparse, _home, _full = _layout(tmp_path)

    def argv(
        ticket_id: str, *, contract_path: str = "", state_root: str = ""
    ) -> list[str]:
        return [sys.executable, "-c", _VERIFIER, contract_path]

    monkeypatch.setattr(guard, "dod_verify_argv", argv)
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "no-such-home"))
    handler = HandlerEvidenceAutocloseSweep(autoclose_disabled=False)

    result, code, error = await handler._run_dod_verify_command_real(
        "OMN-26106", str(sparse), 30
    )

    assert (code, error) == (0, "")
    assert result == {
        "anchor": str(sparse),
        "error": "ModuleNotFoundError: No module named 'scripts'",
    }

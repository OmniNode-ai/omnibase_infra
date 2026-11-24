# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A check of a merged ticket runs pytest in the product clone (OMN-26650).

Integrated, with real git and a real pytest process: the sweep reads a ticket's
contract from a sparse, contracts-only clone (``git clone --sparse`` then
``git sparse-checkout set contracts``), which keeps the repository's root
``conftest.py`` and drops ``scripts/``. Run in that tree, pytest stops with
``ImportError while loading conftest ... ModuleNotFoundError: No module named
'scripts'`` and exit code 4, the failure recorded for 26 checks of 8 tickets in
``evidence-autoclose-sweep.yml`` run 44672842883. Handed the full clone beside
``OMNI_HOME`` as ``CONTRACT_REPO_DIR`` by the sweep handler's real runner, the
same pytest collects and passes.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from omnibase_infra.handlers import done_write_receipt_guard as guard
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)

pytestmark = pytest.mark.integration

_REPO = "product_repo"

# What the verifier's `uv run pytest` check does, reduced to its anchoring: take
# CONTRACT_REPO_DIR when given, else the repository that owns the contract file,
# and run pytest there.
_VERIFIER = textwrap.dedent(
    """\
    import json, os, pathlib, subprocess, sys

    contract = pathlib.Path(sys.argv[1])
    anchor = os.environ.get("CONTRACT_REPO_DIR") or str(contract.parent.parent)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=anchor, capture_output=True, text=True, check=False,
    )
    print(json.dumps({
        "anchor": anchor,
        "exit": proc.returncode,
        "scripts_import_error": "No module named 'scripts'" in proc.stdout + proc.stderr,
    }))
    """
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "HOME": str(cwd),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )


def _build_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "remote" / _REPO
    (remote / "scripts" / "hooks").mkdir(parents=True)
    (remote / "contracts").mkdir()
    (remote / "tests").mkdir()
    (remote / "scripts/__init__.py").write_text("")
    (remote / "scripts" / "hooks" / "__init__.py").write_text("")
    (remote / "scripts" / "hooks" / "guard.py").write_text(
        "def enforce(config, target):\n    return None\n"
    )
    (remote / "conftest.py").write_text(
        "from scripts.hooks.guard import enforce\n\n\n"
        "def pytest_configure(config):\n    enforce(config, 'tests')\n"
    )
    (remote / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\n'
    )
    (remote / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (remote / "contracts" / "OMN-26106.yaml").write_text("ticket_id: OMN-26106\n")
    _git(remote, "init", "-q", "-b", "dev")
    _git(remote, "add", "-A")
    _git(remote, "commit", "-q", "-m", "seed")
    return remote


def _sparse_clone(remote: Path, dest: Path) -> None:
    _git(dest.parent, "clone", "-q", "--sparse", f"file://{remote}", str(dest))
    _git(dest, "sparse-checkout", "set", "contracts")


async def test_scripts_import_sparse_contract_clone_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _build_remote(tmp_path)
    sparse = tmp_path / "contract-roots" / _REPO
    sparse.parent.mkdir()
    _sparse_clone(remote, sparse)
    registry = tmp_path / "registry"
    registry.mkdir()
    _git(registry, "clone", "-q", f"file://{remote}", str(registry / _REPO))

    # The layout the recorded run had: contracts present, root conftest.py
    # present, scripts/ and tests/ absent.
    assert (sparse / "contracts" / "OMN-26106.yaml").is_file()
    assert (sparse / "conftest.py").is_file()
    assert not (sparse / "scripts").exists()

    def argv(
        ticket_id: str, *, contract_path: str = "", state_root: str = ""
    ) -> list[str]:
        return [sys.executable, "-c", _VERIFIER, contract_path]

    monkeypatch.setattr(guard, "dod_verify_argv", argv)

    # Recorded failure: no product clone to anchor at, the check runs in the sparse tree.
    handler = HandlerEvidenceAutocloseSweep(
        autoclose_disabled=False, registry_root=str(tmp_path / "no-registry")
    )
    broken, code, error = await handler._run_dod_verify_command_real(
        "OMN-26106", str(sparse), 120
    )
    assert (code, error) == (0, "")
    assert broken == {
        "anchor": str(sparse),
        "exit": 4,
        "scripts_import_error": True,
    }

    # The fix: the sweep's real runner hands the verifier the full clone.
    handler = HandlerEvidenceAutocloseSweep(
        autoclose_disabled=False, registry_root=str(registry)
    )
    fixed, code, error = await handler._run_dod_verify_command_real(
        "OMN-26106", str(sparse), 120
    )
    assert (code, error) == (0, "")
    assert fixed == {
        "anchor": str((registry / _REPO).resolve()),
        "exit": 0,
        "scripts_import_error": False,
    }

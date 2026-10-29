# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canary for the canonical-file-shape node entry and its burn-down baseline (OMN-21523).

The repository runs omnibase_core's canonical-file-shape hook at the dev-merge
tag pinned in ``.pre-commit-config.yaml``. Each case builds a throwaway git
repository, commits a base, commits a change and runs that same module over
``--base``..``--head``:

* a new handler no ``contract.yaml`` binds is refused as ``handler_outside_node``;
* one more line in the committed burn-down baseline is refused as
  ``baseline-growth``;
* a change that touches neither (a docs file) is accepted, so a refusal above
  is the ratchet's and not a broken invocation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
TICKET = "OMN-21523"
BURNDOWN = ".onex_ratchets/canonical_file_shape_burndown_baseline.txt"
BASELINE = ".onex_ratchets/canonical_file_shape_baseline.txt"
RUNTIME = (
    "omnibase_core.nodes.node_canonical_file_shape_check_compute."
    "runtime_canonical_file_shape_check"
)
HANDLER = "def handle() -> None:\n    return None\n"


def _core_spec() -> str:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    (rev,) = [
        entry["rev"]
        for entry in config["repos"]
        if any(hook["id"] == "canonical-file-shape" for hook in entry["hooks"])
    ]
    return f"omnibase-core @ git+https://github.com/OmniNode-ai/omnibase_core@{rev}"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=canary", "-c", "user.email=canary@example.com", *args],
        cwd=repo,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _burndown(*paths: str) -> str:
    header = (
        f"# Burn-down ticket: {TICKET}\n"
        "# canonical-file-shape burn-down baseline. Shrink-only once installed.\n"
    )
    return header + "".join(f"{path}\n" for path in paths)


def _ratchet(repo: Path, base: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            _core_spec(),
            "python",
            "-m",
            RUNTIME,
            "--rules",
            "file-shape",
            "--base",
            base,
            "--head",
            "HEAD",
        ],
        cwd=repo,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, BASELINE, "# canonical-file-shape baseline. Shrink-only.\n")
    return tmp_path


def test_new_unbound_handler_is_refused_as_handler_outside_node(repo: Path) -> None:
    _write(repo, "docs/readme.md", "base\n")
    base = _commit(repo, f"{TICKET}: base")

    _write(repo, "docs/other.md", "docs only\n")
    clean = _commit(repo, f"{TICKET}: docs only")
    accepted = _ratchet(repo, base)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr

    _write(repo, "src/zz_pkg/handlers/handler_zz_canary.py", HANDLER)
    _commit(repo, f"{TICKET}: plant an unbound handler")
    refused = _ratchet(repo, clean)
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert (
        "src/zz_pkg/handlers/handler_zz_canary.py: [handler_outside_node]"
        in refused.stderr
    )


def test_one_more_burndown_line_is_refused_as_baseline_growth(repo: Path) -> None:
    # Every handler in the base is listed in the burn-down baseline, and the
    # added handler is listed in the same commit that adds it, so the only thing
    # that can refuse the change is the growth itself.
    listed = [
        "src/zz_pkg/handlers/handler_a.py",
        "src/zz_pkg/handlers/handler_b.py",
    ]
    added = "src/zz_pkg/handlers/handler_c.py"
    for path in listed:
        _write(repo, path, HANDLER)
    _write(repo, BURNDOWN, _burndown(*listed))
    base = _commit(repo, f"{TICKET}: base with a burn-down baseline")

    _write(repo, "docs/readme.md", "unrelated\n")
    _commit(repo, f"{TICKET}: unrelated change")
    control = _ratchet(repo, base)
    assert control.returncode == 0, control.stdout + control.stderr

    _write(repo, added, HANDLER)
    _write(repo, BURNDOWN, _burndown(*listed, added))
    _commit(repo, f"{TICKET}: grow the burn-down baseline")
    refused = _ratchet(repo, base)
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert f"{BURNDOWN}: [baseline-growth]" in refused.stderr
    assert added in refused.stderr
    assert "[handler_outside_node]" not in refused.stderr, refused.stderr


def test_committed_burndown_baseline_names_the_ticket_and_no_existing_entry() -> None:
    burndown = (ROOT / BURNDOWN).read_text(encoding="utf-8").splitlines()
    assert burndown[0] == f"# Burn-down ticket: {TICKET}"
    entries = {line for line in burndown if line and not line.startswith("#")}
    baseline = {
        line
        for line in (ROOT / BASELINE).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert entries
    assert not entries & baseline
    missing = sorted(path for path in entries if not (ROOT / path).is_file())
    assert not missing, f"burn-down entries naming no file: {missing}"


def test_whole_tree_passes_at_head_with_the_committed_baselines() -> None:
    done = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            _core_spec(),
            "python",
            "-m",
            RUNTIME,
            "--rules",
            "file-shape",
            "--head",
            "HEAD",
        ],
        cwd=ROOT,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert done.returncode == 0, done.stdout + done.stderr

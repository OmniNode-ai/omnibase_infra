# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A fake ``onex`` and a pass-through ``strace`` for the probe state-root tests (OMN-20505).

The fake CLI resolves its state root the way ``onex delegate`` does since
omnibase_infra 0.38.64 (OMN-19232): an absolute ``ONEX_STATE_DIR``, then
``$HOME/.onex_state``, never the working directory. A probe that looks for the
run directory anywhere else finds nothing, which is the C13/C29 red of
2026-10-04.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

_FAKE_ONEX = """\
import json, os, pathlib, sys
home = pathlib.Path(os.environ["HOME"])
state = os.environ.get("ONEX_STATE_DIR")
root = pathlib.Path(state) if state and os.path.isabs(state) else home / ".onex_state"
verb = sys.argv[1]
if verb == "local":
    print("{}")
    sys.exit(0)
if verb == "secret":
    if sys.argv[2] == "set":
        (home / ".fake-secret").write_text(sys.stdin.read())
        sys.exit(0)
    print("registered:")
    if (home / ".fake-secret").exists():
        print("  llm.openrouter.api_key")
    sys.exit(0)
if verb == "delegate":
    configured = (
        (home / ".omninode" / "delegation" / "bifrost_overrides.yaml").exists()
        or (home / ".fake-secret").exists()
    )
    run_id = "run-configured" if configured else "run-unconfigured"
    run = root / "runs" / run_id
    run.mkdir(parents=True)
    doc = {"run_id": run_id, "correlation_id": "corr-1"}
    (run / "result.txt").write_text("the answer")
    (run / "receipt.json").write_text(json.dumps({**doc, "status": "success"}))
    (run / "run.json").write_text(json.dumps(doc))
    print(json.dumps(doc))
    sys.exit(0 if configured else 1)
sys.exit(2)
"""

_FAKE_STRACE = """\
import os, sys
argv = sys.argv[sys.argv.index("--") + 1:]
os.execv(argv[0], argv)
"""


def _script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def install_fakes(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Return (customer_home, customer_bin, workdir, strace) for one fake session."""
    customer_home = tmp_path / "home"
    customer_bin = customer_home / ".local" / "bin"
    customer_bin.mkdir(parents=True)
    _script(customer_bin / "onex", _FAKE_ONEX)
    strace = _script(tmp_path / "strace", _FAKE_STRACE)
    return customer_home, customer_bin, tmp_path / "work", strace

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lock layer behind the provider's floor is applied BEFORE the co-install (OMN-20154).

MEASURED on .202 (``/data/omninode/omni_home``), every reconcile tick from
22:12Z to 23:17Z on 2026-09-30: omnibase_infra's lock pinned omnibase-core
0.47.27 and omnimarket dev declared ``omnibase-core>=0.47.27``, while the
dispatch venv carried 0.47.25. The reconciler ran the provider co-install
FIRST (the OMN-16262 ordering), the co-install's OMN-18752 floor readback read
``omnibase-core (floor from omnimarket) installed 0.47.25 != expected
>=0.47.27`` and refused, and the lock pass that would have installed 0.47.27
never ran. Every tick repeated that, so the floor was never proven and the
onex wrapper's below-floor self-heal kept firing reconcile-host.

The fix applies the lock layer before the co-install whenever the lock check
says it is behind, and keeps the mandatory lock pass AFTER the co-install, so
the OMN-16262 guarantee (the lock has the last word over pins the co-install
moved) is unchanged.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.integration.scripts.test_dispatch_gate_venv_split_omn17819 import (
    _SCRIPT,
    _SHA_LEN,
    _git,
    _make_clone,
    _make_fake_venv,
    _make_uv_shim,
    _read_lines,
)

pytestmark = pytest.mark.integration


def _make_ordering_install_shim(path: Path) -> Path:
    path.write_text(
        '#!/usr/bin/env bash\nprintf "INSTALL %s\\n" "$*" >> "$UV_SHIM_LOG"\nexit 0\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run(tmp_path: Path, *, check_exit: int) -> list[str]:
    omni_home = tmp_path / "omni_home"
    _make_clone(omni_home, "omnimarket")
    infra = omni_home / "omnibase_infra"
    (infra / "scripts").mkdir(parents=True)
    (infra / "uv.lock").write_text("lock-v1\n", encoding="utf-8")
    (omni_home / "omniclaude").mkdir(parents=True)
    (omni_home / "omniclaude" / "uv.lock").write_text("c\n", encoding="utf-8")
    _make_fake_venv(infra / ".venv", None)
    dispatch_venv = omni_home / ".onex-dispatch-venv"
    _make_fake_venv(dispatch_venv, "0" * _SHA_LEN)
    bin_dir = omni_home / "shimbin"
    log = omni_home / "order.log"
    install = infra / "scripts" / "install-node-skill-package.sh"
    _make_uv_shim(bin_dir, check_exit=check_exit)
    _make_ordering_install_shim(install)
    env = {
        **os.environ,
        "OMNI_HOME": str(omni_home),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "UV_SHIM_LOG": str(log),
        "ONEX_RECONCILE_INSTALL_SCRIPT": str(install),
        "CLAUDE_PLUGIN_DATA": str(omni_home / "no-such-plugin-data"),
    }
    result = subprocess.run(
        ["bash", str(_SCRIPT)], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert _git("rev-parse", "HEAD", cwd=omni_home / "omnimarket")
    return [
        line
        for line in _read_lines(log)
        if line.startswith("INSTALL ")
        or (
            f"env={dispatch_venv}" in line
            and " sync " in f" {line} "
            and "--check" not in line
        )
    ]


def test_a_behind_lock_layer_is_applied_before_the_provider_co_install(
    tmp_path: Path,
) -> None:
    order = _run(tmp_path, check_exit=1)

    install_at = next(i for i, line in enumerate(order) if line.startswith("INSTALL"))
    syncs_before = [line for line in order[:install_at] if "sync" in line]
    syncs_after = [line for line in order[install_at + 1 :] if "sync" in line]
    assert syncs_before, f"no lock pass before the co-install: {order!r}"
    # OMN-16262: the lock still has the last word after the co-install.
    assert syncs_after, f"no lock pass after the co-install: {order!r}"
    assert all("--frozen" in line and "--inexact" in line for line in syncs_before)


def test_an_in_sync_lock_layer_adds_no_extra_pass_before_the_co_install(
    tmp_path: Path,
) -> None:
    """Positive control: a lock check that passes keeps the one post-install pass."""
    order = _run(tmp_path, check_exit=0)

    install_at = next(i for i, line in enumerate(order) if line.startswith("INSTALL"))
    assert not [line for line in order[:install_at] if "sync" in line], order
    assert [line for line in order[install_at + 1 :] if "sync" in line], order

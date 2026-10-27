# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for runtime fresh-volume bootstrap."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]


def test_entrypoint_repairs_runtime_volume_paths_before_the_boot_preflight() -> None:
    """The volume repair and the privilege drop must precede every render.

    OMN-17372 collapsed the schema stamp and both renders into one warm
    interpreter (``omnibase_infra.runtime.entrypoint_preflight``), so the
    ordering guard now anchors on that single launch rather than on the two
    module names it replaced. The invariant is unchanged: nothing writes to
    /app/data before it is owned by the runtime user and privileges are
    dropped.
    """
    entrypoint = (DOCKER_DIR / "entrypoint-runtime.sh").read_text()

    bootstrap_pos = entrypoint.index("Fresh Volume Bootstrap")
    data_dir_pos = entrypoint.index("/app/data/delegation")
    chown_pos = entrypoint.index("chown -R omniinfra:omniinfra")
    drop_pos = entrypoint.index('exec gosu omniinfra "$0" "$@"')
    preflight_pos = entrypoint.index(
        'exec python -m omnibase_infra.runtime.entrypoint_preflight "$@"'
    )

    assert bootstrap_pos < data_dir_pos < chown_pos < drop_pos < preflight_pos
    assert (
        "install -d -o omniinfra -g omniinfra /app/data /app/data/delegation /app/logs /app/tmp"
        in entrypoint
    )
    assert "chown -R omniinfra:omniinfra /app/data /app/logs /app/tmp" in entrypoint


def test_entrypoint_hands_the_secret_store_volume_to_the_runtime_user() -> None:
    """The runtime-owned secret store is a named volume under omniinfra's home.

    A fresh volume is root-owned and the kernel runs as omniinfra after the
    privilege drop. Measured on the lab (OMN-17099, 2026-09-30): with the store
    left root-owned the kernel could not read a tenant's registered key, the
    tenant route failed, and the delegation fell through to a local rung.
    """
    entrypoint = (DOCKER_DIR / "entrypoint-runtime.sh").read_text()

    repair_pos = entrypoint.index(
        "chown -R omniinfra:omniinfra /home/omniinfra/.omninode"
    )
    drop_pos = entrypoint.index('exec gosu omniinfra "$0" "$@"')
    assert repair_pos < drop_pos
    assert (
        "install -d -o omniinfra -g omniinfra /home/omniinfra/.omninode "
        "/home/omniinfra/.omninode/delegation"
    ) in entrypoint


def test_runtime_image_installs_gosu_for_privilege_drop() -> None:
    dockerfile = (DOCKER_DIR / "Dockerfile.runtime").read_text()

    assert "gosu \\" in dockerfile
    assert "USER root" in dockerfile
    assert "dropping to the non-root runtime user" in dockerfile


FOLDED_ROOT = "/app/effects-data/.onex_state_stability_test"


def _run_root_bootstrap(
    tmp_path: Path, folded_root: str | None
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the real root bootstrap with filesystem commands recorded, never executed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    calls.touch()
    for name in ("id", "install", "chown", "gosu"):
        script = bin_dir / name
        script.write_text(
            "#!/bin/sh\nprintf '0\\n'\n"
            if name == "id"
            else f'#!/bin/sh\nprintf \'{name} %s\\n\' "$*" >> "$CALLS"\n'
        )
        script.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "CALLS": str(calls)}
    if folded_root is not None:
        env["ONEX_FOLDED_EFFECTS_STATE_ROOT"] = folded_root
    result = subprocess.run(
        ["sh", str(DOCKER_DIR / "entrypoint-runtime.sh"), "onex-runtime"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, calls.read_text().splitlines()


@pytest.mark.parametrize("folded", [False, True])
def test_effects_state_root_is_owned_before_privilege_drop_only_when_folded(
    tmp_path: Path, folded: bool
) -> None:
    result, recorded = _run_root_bootstrap(tmp_path, FOLDED_ROOT if folded else None)
    assert result.returncode == 0, result.stderr
    install = f"install -d -o omniinfra -g omniinfra /app/effects-data {FOLDED_ROOT}"
    repair = f"chown -hR omniinfra:omniinfra {FOLDED_ROOT}"
    assert (repair in recorded) is folded
    assert (install in recorded) is folded
    # The archives beside the state root are never walked.
    assert not any(
        call.startswith("chown") and call.endswith(" /app/effects-data")
        for call in recorded
    )
    if folded:
        gosu = next(
            index for index, call in enumerate(recorded) if call.startswith("gosu ")
        )
        assert recorded.index(install) < recorded.index(repair) < gosu


@pytest.mark.parametrize(
    "folded_root",
    [
        "/app/effects-data",
        "/app/effects-data/",
        "/app/effects-data/.",
        "/app/effects-data/state/",
        "/app/effects-data/../data",
        "/app/effects-data/a/b",
        "/etc",
    ],
)
def test_folded_state_root_outside_the_effects_volume_refuses_boot(
    tmp_path: Path, folded_root: str
) -> None:
    result, recorded = _run_root_bootstrap(tmp_path, folded_root)
    assert result.returncode == 2
    assert "FOLDED_EFFECTS_STATE_REFUSED" in result.stderr
    assert not any(call.startswith("gosu ") for call in recorded)
    assert not any("effects-data" in call or "/etc" in call for call in recorded)


@pytest.mark.parametrize(
    "linked_path",
    ["effects-data", "effects-data/state", "effects-data/state/delegation"],
)
def test_folded_symlink_refuses_before_any_root_filesystem_operation(
    tmp_path: Path, linked_path: str
) -> None:
    app = tmp_path / "app"
    outside = tmp_path / "outside"
    outside.mkdir()
    claim = outside / "claim"
    claim.write_text("preserve")
    link = app / linked_path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("id", "install", "chown", "gosu"):
        script = bin_dir / name
        script.write_text(
            "#!/bin/sh\nprintf '0\\n'\n"
            if name == "id"
            else "#!/bin/sh\necho 'unexpected root filesystem operation' >&2\nexit 77\n"
        )
        script.chmod(0o755)
    # Run the real shell with only its absolute /app paths relocated into this
    # isolated tree; the guard and command ordering are unchanged.
    entrypoint = (DOCKER_DIR / "entrypoint-runtime.sh").read_text()
    result = subprocess.run(
        ["sh", "-c", entrypoint.replace("/app/", f"{app}/")],
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "ONEX_FOLDED_EFFECTS_STATE_ROOT": str(app / "effects-data/state"),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2, result.stderr
    assert "must not be a symlink" in result.stderr
    assert "unexpected root filesystem operation" not in result.stderr
    assert link.is_symlink()
    assert claim.read_text() == "preserve"
    assert list(outside.iterdir()) == [claim]

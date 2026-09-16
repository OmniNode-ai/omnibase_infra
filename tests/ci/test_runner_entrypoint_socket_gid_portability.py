# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The socket-GID fix must work on a host whose socket GID is already taken.

MEASURED 2026-09-16 on the `.101` host, not inferred. Docker Desktop for Mac
presents the bind-mounted `/var/run/docker.sock` inside the container with GID
**0**. The entrypoint's fix was `groupmod -g <socket gid> docker`, which on that
host means "renumber the docker group to 0" -- and 0 is already `root`.
`groupmod` refuses with `GID '0' already exists`, the entrypoint dies under
`set -e`, and the container enters a restart loop that never registers a runner.

The failure is invisible in the obvious place: the container is `Restarting`,
not `Exited`, and the log line immediately above the failure reads
"Adjusting container docker group GID to 0" -- which looks like progress.

The fix is to grant membership rather than renumber. When the socket's GID
already belongs to a group, add `runner` to THAT group; only renumber `docker`
when the GID is free. Renumbering was never the goal -- socket access was.

These are shell-level tests run against the real script with a fake
``groupmod``/``usermod``/``stat`` on PATH, so they exercise the actual control
flow rather than a transcription of it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker" / "runners" / "entrypoint.sh"

HARNESS = r"""
set -uo pipefail
export PATH="$FAKEBIN:$PATH"
# Source only the GID-fix function, not the whole entrypoint: everything after
# it requires a live GitHub registration.
sed -n '/^_fix_docker_socket_gid()/,/^}/p' "$ENTRYPOINT_PATH" > "$FAKEBIN/fn.sh"
# shellcheck disable=SC1090
. "$FAKEBIN/fn.sh"
set -e
_fix_docker_socket_gid
"""


def _run(
    tmp_path: Path, socket_gid: str, groups: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run the GID fix with a fabricated socket GID and group database."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    # A real socket, so the function's `-S` guard passes deterministically
    # rather than depending on whether the machine running the tests happens to
    # have a Docker socket of its own. Bound from INSIDE the directory: macOS
    # caps an AF_UNIX path at 104 bytes and pytest's tmp_path is longer.
    import socket as _socket

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        sock = _socket.socket(_socket.AF_UNIX)
        sock.bind("docker.sock")
        sock.close()
    finally:
        os.chdir(cwd)

    # `getent group docker` and `getent group -> gid` come from this table.
    group_lines = "\n".join(f"{name}:x:{gid}:" for name, gid in groups.items())
    (fakebin / "getent").write_text(
        "#!/bin/bash\n"
        f"TABLE='{group_lines}'\n"
        'if [[ "$1" != group ]]; then exit 2; fi\n'
        'if [[ -z "${2:-}" ]]; then echo "$TABLE"; exit 0; fi\n'
        "while IFS=: read -r n _ g _; do\n"
        '  [[ -z "$n" ]] && continue\n'
        '  if [[ "$n" == "$2" || "$g" == "$2" ]]; then echo "$n:x:$g:"; exit 0; fi\n'
        'done <<< "$TABLE"\n'
        "exit 2\n"
    )
    (fakebin / "stat").write_text(f"#!/bin/bash\necho {socket_gid}\n")
    # groupmod refuses a GID that already exists, exactly as the real one does.
    (fakebin / "groupmod").write_text(
        "#!/bin/bash\n"
        f"TAKEN='{' '.join(groups.values())}'\n"
        'want="$2"\n'
        "for g in $TAKEN; do\n"
        '  if [[ "$g" == "$want" ]]; then\n'
        "    echo \"groupmod: GID '$want' already exists\" >&2; exit 4\n"
        "  fi\n"
        "done\n"
        'echo "GROUPMOD $*"\n'
    )
    (fakebin / "usermod").write_text('#!/bin/bash\necho "USERMOD $*"\n')
    for f in fakebin.iterdir():
        f.chmod(0o755)

    env = dict(os.environ)
    env.update(FAKEBIN=str(fakebin), ENTRYPOINT_PATH=str(ENTRYPOINT))
    env["DOCKER_SOCKET_PATH"] = str(tmp_path / "docker.sock")
    script = HARNESS
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )


def test_socket_gid_already_taken_grants_membership_instead_of_renumbering(
    tmp_path: Path,
) -> None:
    """GID 0 is `root`, and `groupmod -g 0 docker` cannot succeed.

    This is the live `.101` case. The correct outcome is that `runner` joins the
    group that already owns the socket GID, and the function returns 0 so the
    entrypoint proceeds to registration.
    """
    result = _run(
        tmp_path,
        socket_gid="0",
        groups={"root": "0", "runner": "1001", "docker": "1002"},
    )
    assert result.returncode == 0, (
        "the socket-GID fix must not fail when the socket's GID already belongs "
        "to another group -- that is a restart loop with no runner registered.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "USERMOD" in result.stdout, (
        "the fix must add `runner` to the group that owns the socket GID; "
        f"got:\n{result.stdout}"
    )


def test_socket_gid_free_still_renumbers_the_docker_group(tmp_path: Path) -> None:
    """The primary host's behaviour is unchanged.

    `.201`'s socket GID (984) belongs to no container group, so the original
    renumber is still both available and correct. A fix that switched every
    host to membership-only would silently change the 60-runner fleet.
    """
    result = _run(
        tmp_path,
        socket_gid="984",
        groups={"root": "0", "runner": "1001", "docker": "1002"},
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "GROUPMOD" in result.stdout, (
        f"a free socket GID must still renumber the docker group; got:\n{result.stdout}"
    )


def test_matching_gid_is_a_no_op(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        socket_gid="1002",
        groups={"root": "0", "runner": "1001", "docker": "1002"},
    )
    assert result.returncode == 0
    assert "GROUPMOD" not in result.stdout
    assert "USERMOD" not in result.stdout

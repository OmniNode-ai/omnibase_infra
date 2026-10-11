# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shipped defaults name no lab host (OMN-20931).

Three defaults used to fall back to a lab address when a setting was absent,
so a fork or a customer that set nothing was pointed at our lab:

* the Redpanda catalog manifest advertised the lab's LAN address;
* ``scripts/push_lanes/deploy-push-lanes.sh`` deployed to a lab host when run
  with no argument;
* ``scripts/hooks/prepush_smart_tests.sh`` carried two lab SSH targets.

The Redpanda default is now the neutral ``localhost`` (a deployment sets
``REDPANDA_ADVERTISE_HOST`` in its env file), the deploy script refuses with a
usage line when no target is passed, and the hook carries no SSH target.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
REDPANDA = REPO_ROOT / "docker" / "catalog" / "services" / "redpanda.yaml"
DEPLOY_PUSH_LANES = REPO_ROOT / "scripts" / "push_lanes" / "deploy-push-lanes.sh"
PREPUSH = REPO_ROOT / "scripts" / "hooks" / "prepush_smart_tests.sh"

_ADVERTISE_DEFAULT = re.compile(r"\$\{REDPANDA_ADVERTISE_HOST:-([^}]*)\}")


@pytest.mark.unit
def test_redpanda_advertises_the_neutral_default() -> None:
    command = yaml.safe_load(REDPANDA.read_text(encoding="utf-8"))["command"]
    defaults = _ADVERTISE_DEFAULT.findall(" ".join(command))
    # Two listeners advertise the host: Kafka and the HTTP proxy.
    assert defaults == ["localhost", "localhost"]


@pytest.mark.unit
def test_deploy_push_lanes_refuses_without_a_target(tmp_path: Path) -> None:
    # ssh and scp are stubs that fail at once, so a script that still falls
    # back to a host never reaches it: it fails fast here instead of hanging
    # on a connect or copying files to a real machine.
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name in ("ssh", "scp"):
        stub = stubs / name
        stub.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
        stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(DEPLOY_PUSH_LANES)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=tmp_path,
        env={"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode != 0
    assert "usage: deploy-push-lanes.sh <ssh-target>" in result.stderr
    # It refused before reaching any host.
    assert "[deploy-push-lanes] target:" not in result.stdout


@pytest.mark.unit
def test_prepush_hook_carries_no_ssh_target() -> None:
    text = PREPUSH.read_text(encoding="utf-8")
    assert re.search(r"^PREPUSH_20[01]_SSH_TARGET=", text, re.MULTILINE) is None

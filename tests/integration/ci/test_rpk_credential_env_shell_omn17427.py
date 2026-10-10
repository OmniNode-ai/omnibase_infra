# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the broker password reaches a real ``rpk`` process only through its environment.

``ConsumerFlowLane.rpk`` builds a ``sh -c`` script that runs inside the broker container.
This test runs that exact script under a real ``sh`` with a stand-in ``rpk`` executable that
records its argv and environment, and asserts the SASL password is in the environment and
in no argv entry.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)

pytestmark = pytest.mark.integration

_PASSWORD = "s3cret-sasl-password-omn17427"
_FAKE_RPK = """#!/bin/sh
python3 - "$@" <<'PY'
import json, os, sys
json.dump({"argv": sys.argv[1:], "env": dict(os.environ)}, open(os.environ["RPK_CAPTURE"], "w"))
PY
"""


def test_password_is_in_rpk_environment_and_not_in_its_argv(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "rpk"
    fake.write_text(_FAKE_RPK, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    capture = tmp_path / "capture.json"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RPK_CAPTURE": str(capture),
        "DEV_KAFKA_SASL_USERNAME": "probe-user",
        "DEV_KAFKA_SASL_PASSWORD": _PASSWORD,
    }

    def runner(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        # Drop the `docker exec -i <container>` prefix: run the in-container shell here.
        inner = argv[4:]
        return subprocess.run(
            inner, capture_output=True, text=True, check=False, env=env
        )

    lane = ConsumerFlowLane(
        docker="fake-docker", base_url="http://projection.test", runner=runner
    )
    lane.rpk("topic", "describe", "-p", "some.topic")

    seen = json.loads(capture.read_text(encoding="utf-8"))
    assert seen["argv"][:4] == ["topic", "describe", "-p", "some.topic"]
    assert not any(_PASSWORD in arg for arg in seen["argv"])
    assert seen["env"]["RPK_PASS"] == _PASSWORD
    assert seen["env"]["RPK_USER"] == "probe-user"
    assert seen["env"]["RPK_SASL_MECHANISM"] == "SCRAM-SHA-256"

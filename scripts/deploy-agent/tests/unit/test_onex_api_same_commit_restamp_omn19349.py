# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A re-stamped onex-api image of an unchanged commit is not a pin advance (OMN-19349).

THE DEFECT
----------
The lab-overlay applier builds ``onex-lab/omnicloud-core:<sha8>-<stamp>`` on every
deploy-agent job. The repoint script compared the pin with the newest tag as a
whole string, so a newer stamp of the SAME commit read as an advance, and
``deliver_onex_api_pin`` recreated ``onex-api`` on every dev-lane job. Observed
2026-09-23 at 14:52Z on the .201 dev lane: ``1b128e1d-...T141329Z`` was repointed
to ``1b128e1d-...T144642Z`` and the container recreated. Each recreate takes
:8090 down for seconds, and a chain canary that submitted inside one of those
windows failed with ``ingress_unreachable``.

HOW THESE TESTS DRIVE IT
------------------------
``deliver_onex_api_pin`` runs for real, and so does the repoint script it
invokes. Only the two external binaries the script calls are replaced, by
executables this file writes, and the compose recreate is recorded instead of
run. A recreate therefore happens in these tests exactly when the shipped code
would issue one on the lab host.
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane

pytestmark = pytest.mark.unit

#: The script in THIS tree. The executor's constant names the lab host's clone.
_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runtime_build"
    / "repoint_dev_lane_onex_api.py"
)
_IMAGE = "onex-lab/omnicloud-core"
_SHA = "1b128e1d" + "4" * 32
_PINNED = f"{_IMAGE}:1b128e1d-20260923T141329Z"
_RESTAMPED = f"{_IMAGE}:1b128e1d-20260923T144642Z"
_NEXT_SHA = "9c0ffee1" + "5" * 32
_NEXT = f"{_IMAGE}:9c0ffee1-20260923T150000Z"


def _write_exec(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_docker(tmp_path: Path, *, tags: list[str]) -> Path:
    """Lists ``tags``; every listed tag is resident and unlabelled."""
    fixture = tmp_path / "docker-fixture.json"
    fixture.write_text(json.dumps({"tags": tags}), encoding="utf-8")
    return _write_exec(
        tmp_path / "fake-docker",
        f"""#!/usr/bin/env python3
import json, sys
fx = json.load(open({str(fixture)!r}))
argv = sys.argv[1:]
if argv[0] == "images":
    print("\\n".join(fx["tags"]))
    sys.exit(0)
if argv[0] == "image" and argv[1] == "inspect" and argv[2] in fx["tags"]:
    print(json.dumps([{{"Id": "sha256:" + argv[2][-6:], "Config": {{"Env": []}}}}]))
    sys.exit(0)
sys.stderr.write("Error: No such image\\n")
sys.exit(1)
""",
    )


def _fake_git(tmp_path: Path) -> Path:
    """Resolves both lineages' sha8s and reports the newer one as origin/dev."""
    known = {_SHA[:8]: _SHA, _NEXT_SHA[:8]: _NEXT_SHA}
    fixture = tmp_path / "git-fixture.json"
    fixture.write_text(json.dumps({"known": known}), encoding="utf-8")
    return _write_exec(
        tmp_path / "fake-git",
        f"""#!/usr/bin/env python3
import json, sys
fx = json.load(open({str(fixture)!r}))
rev = sys.argv[-1]
if rev.startswith("origin/dev"):
    print({_NEXT_SHA!r})
    sys.exit(0)
short = rev.split("^")[0]
if short in fx["known"]:
    print(fx["known"][short])
    sys.exit(0)
sys.exit(128)
""",
    )


class _Lab:
    """The seams around ``deliver_onex_api_pin``, and what went through them."""

    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tags: list[str]
    ) -> None:
        self.env_file = tmp_path / "operator.env"
        self.env_file.write_text(f"A=1\nONEX_API_IMAGE={_PINNED}\nB=2\n", "utf-8")
        self.clone = tmp_path / "omninode_clone"
        (self.clone / ".git").mkdir(parents=True)
        docker = _fake_docker(tmp_path, tags=tags)
        git = _fake_git(tmp_path)
        self.compose_calls: list[list[str]] = []
        monkeypatch.setattr(executor_mod, "REPOINT_ONEX_API_SCRIPT", _SCRIPT)
        real_script = str(_SCRIPT)

        def run(
            cmd: list[str], timeout: int, **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if len(cmd) > 1 and cmd[1] == real_script:
                argv = [
                    sys.executable,
                    *cmd[1:],
                    "--docker",
                    str(docker),
                    "--git",
                    str(git),
                ]
                return subprocess.run(
                    argv, capture_output=True, text=True, check=False, timeout=timeout
                )
            self.compose_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(self.env_file))
        monkeypatch.setattr(executor_mod, "_run", run)
        monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})

    def deliver(self, sha: str) -> dict[str, Any]:
        executor = executor_mod.DeployExecutor.__new__(executor_mod.DeployExecutor)
        return executor.deliver_onex_api_pin(
            sha=sha, omninode_clone=self.clone, lane=EnumRuntimeLane.DEV
        )

    def pin(self) -> str:
        for line in self.env_file.read_text("utf-8").splitlines():
            if line.startswith("ONEX_API_IMAGE="):
                return line.partition("=")[2]
        raise AssertionError("the pin line is gone")


def test_a_newer_stamp_of_the_pinned_commit_does_not_recreate_onex_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: the measured 14:52Z repoint, which recreated onex-api for nothing."""
    lab = _Lab(tmp_path, monkeypatch, tags=[_PINNED, _RESTAMPED])

    record = lab.deliver(_SHA)

    assert record["tag_advanced"] is False, record
    assert record["recreated"] is False
    assert record["result"] == "UNCHANGED"
    assert "1b128e1d" in record["reason"]
    assert lab.compose_calls == []
    assert lab.pin() == _PINNED


def test_a_descendant_commit_still_advances_and_recreates_only_onex_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2: a real advance is still delivered, as one service and nothing else."""
    lab = _Lab(tmp_path, monkeypatch, tags=[_PINNED, _RESTAMPED, _NEXT])

    record = lab.deliver(_NEXT_SHA)

    assert record["tag_advanced"] is True, record
    assert record["recreated"] is True
    assert record["pin_before"] == _PINNED
    assert record["pin_after"] == _NEXT
    assert lab.pin() == _NEXT
    assert len(lab.compose_calls) == 1
    recreate = lab.compose_calls[0]
    assert recreate[-1] == executor_mod.ONEX_API_SERVICE
    assert "--no-deps" in recreate
    assert "--force-recreate" in recreate


def test_a_collected_pin_is_repaired_by_the_newer_stamp_of_its_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pin naming an image GC removed renders a service that cannot start."""
    lab = _Lab(tmp_path, monkeypatch, tags=[_RESTAMPED])

    record = lab.deliver(_SHA)

    assert record["tag_advanced"] is True, record
    assert lab.pin() == _RESTAMPED
    assert len(lab.compose_calls) == 1

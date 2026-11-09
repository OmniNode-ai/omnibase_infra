# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A candidate onex-api image that cannot import its modules is not delivered (OMN-23039).

THE DEFECT
----------
``deliver_onex_api_pin`` advanced ``ONEX_API_IMAGE`` and recreated ``onex-api`` on
the strength of the image's provenance. On 2026-11-09T03:58Z it delivered
the onex-api image tagged ``145fd4d6-20261109T032509Z``, whose omnimarket no longer
carried ``omnimarket.routing.byok_provider_backends`` (omnimarket#4810 moved it).
``onex-api`` crash-looped 246 times on ``ModuleNotFoundError``, every other
compose service waits on it being healthy, and every later job failed.

HOW THESE TESTS DRIVE IT
------------------------
``deliver_onex_api_pin`` runs for real, and so does the repoint script it
invokes and the import-check script it ships. Only the binaries are replaced:
``docker images`` and ``docker image inspect`` by the fake the repoint script
takes, and ``docker run`` by a function that runs the shipped check script with
this interpreter inside a fixture ``/app`` whose omnimarket is a fixture package.
The recreate is recorded instead of run.

THE FAILING CONTROL
-------------------
``test_a_candidate_that_imports_a_moved_module_is_refused_and_the_pin_restored``
is the 03:58Z delivery. Take the gate out of ``deliver_onex_api_pin`` and it
fails: the pin stays advanced and onex-api is recreated.
"""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import (
    EnumOnexApiDeliveryResult,
    EnumRuntimeLane,
    ModelOnexApiDelivery,
)

pytestmark = pytest.mark.unit

# The deploy agent loads the handler from its own checkout (the standalone agent
# venv carries a released omnibase_infra that predates it), and so do these tests.
_HANDLER = executor_mod.load_dev_boot_guard("handler_onex_api_image_import_check")
HandlerOnexApiImageImportCheck = _HANDLER.HandlerOnexApiImageImportCheck
ModelOnexApiImageImportRequest = _HANDLER.ModelOnexApiImageImportRequest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runtime_build"
    / "repoint_dev_lane_onex_api.py"
)
_REPOINT = importlib.util.spec_from_file_location("repoint_onex_api", _SCRIPT)
assert _REPOINT is not None and _REPOINT.loader is not None
_REPOINT_MODULE = importlib.util.module_from_spec(_REPOINT)
sys.modules[_REPOINT.name] = _REPOINT_MODULE
_REPOINT.loader.exec_module(_REPOINT_MODULE)
_IMAGE = _REPOINT_MODULE.API_IMAGE_NAME
_PREV_SHA = "1b128e1d" + "4" * 32
_PINNED = f"{_IMAGE}:1b128e1d-20261102T173353Z"
_NEXT_SHA = "145fd4d6" + "5" * 32
_NEXT = f"{_IMAGE}:145fd4d6-20261109T032509Z"
_OLD_IMPORT = "omnimarket.routing.byok_provider_backends"
_NEW_IMPORT = "omnimarket.handlers.routing.byok_provider_backends"


def _write_exec(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_docker(tmp_path: Path, *, tags: list[str]) -> Path:
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
    known = {_PREV_SHA[:8]: _PREV_SHA, _NEXT_SHA[:8]: _NEXT_SHA}
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


def _build_image_tree(root: Path, *, router_imports: str) -> tuple[Path, Path]:
    """A fixture ``/app`` and the site-packages its interpreter would see.

    The omnimarket here carries ONLY the post-#4810 layout, which is what the
    145fd4d6 image carried. A test file under ``tests/`` imports the old path on
    purpose: the check must not read it.
    """
    site = root / "site"
    for pkg in (
        "omnimarket",
        "omnimarket/handlers",
        "omnimarket/handlers/routing",
        "omnimarket/projection",
    ):
        (site / pkg).mkdir(parents=True, exist_ok=True)
        (site / pkg / "__init__.py").write_text("", encoding="utf-8")
    (site / "omnimarket/handlers/routing/byok_provider_backends.py").write_text(
        "X = 1\n", encoding="utf-8"
    )
    (site / "omnimarket/projection/credential_publisher.py").write_text(
        "Y = 1\n", encoding="utf-8"
    )
    app = root / "app"
    (app / "routers").mkdir(parents=True)
    (app / "tests").mkdir()
    (app / "routers" / "inference_credentials.py").write_text(
        "import os\n"
        "from omnimarket.projection.credential_publisher import Y\n"
        f"{router_imports}\n",
        encoding="utf-8",
    )
    (app / "broken.py").write_text("def (:\n", encoding="utf-8")
    (app / "tests" / "test_old.py").write_text(
        f"from {_OLD_IMPORT} import X\n", encoding="utf-8"
    )
    return app, site


class _Lab:
    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        router_imports: str,
        docker_run: str = "image",
    ) -> None:
        self.env_file = tmp_path / "operator.env"
        self.env_file.write_text(f"A=1\nONEX_API_IMAGE={_PINNED}\nB=2\n", "utf-8")
        self.clone = tmp_path / "omninode_clone"
        (self.clone / ".git").mkdir(parents=True)
        docker = _fake_docker(tmp_path, tags=[_PINNED, _NEXT])
        git = _fake_git(tmp_path)
        app, site = _build_image_tree(tmp_path, router_imports=router_imports)
        self.compose_calls: list[list[str]] = []
        self.docker_run_calls: list[list[str]] = []
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
            if cmd[:2] == ["docker", "run"]:
                self.docker_run_calls.append(cmd)
                if docker_run == "unavailable":
                    return subprocess.CompletedProcess(
                        cmd, 125, stdout="", stderr="docker: daemon not reachable"
                    )
                if docker_run == "timeout":
                    raise subprocess.TimeoutExpired(cmd, timeout)
                code = cmd[cmd.index("-c") + 1]
                return subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=app,
                    env={"PYTHONPATH": str(site), "PATH": ""},
                    timeout=timeout,
                )
            self.compose_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(self.env_file))
        monkeypatch.setattr(executor_mod, "_run", run)
        monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})

    def deliver(self) -> dict[str, Any]:
        executor = executor_mod.DeployExecutor.__new__(executor_mod.DeployExecutor)
        return executor.deliver_onex_api_pin(
            sha=_NEXT_SHA, omninode_clone=self.clone, lane=EnumRuntimeLane.DEV
        )

    def pin(self) -> str:
        for line in self.env_file.read_text("utf-8").splitlines():
            if line.startswith("ONEX_API_IMAGE="):
                return line.partition("=")[2]
        raise AssertionError("the pin line is gone")


def test_a_candidate_that_imports_a_moved_module_is_refused_and_the_pin_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 03:58Z delivery: the router still imports the pre-#4810 path."""
    lab = _Lab(tmp_path, monkeypatch, router_imports=f"from {_OLD_IMPORT} import X")

    record = lab.deliver()

    assert record["result"] == "REFUSED", record
    assert record["reason"].startswith("IMPORT_CHECK_FAILED_ROLLED_BACK: ")
    assert record["tag_advanced"] is False
    assert record["recreated"] is False
    assert record["rejected_candidate"] == _NEXT
    assert record["pin_after"] == _PINNED
    assert _OLD_IMPORT in record["reason"]
    assert "routers/inference_credentials.py" in record["reason"]
    assert lab.pin() == _PINNED, "the advanced pin must be put back"
    assert lab.compose_calls == [], "a rejected candidate must never be recreated"
    # The rollback changed that one line and no other.
    assert lab.env_file.read_text("utf-8") == f"A=1\nONEX_API_IMAGE={_PINNED}\nB=2\n"


def test_the_refusal_is_a_failure_on_the_typed_record_and_survives_a_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loud and typed: a REFUSED failure, not an UNRECOGNISED verdict."""
    lab = _Lab(tmp_path, monkeypatch, router_imports=f"from {_OLD_IMPORT} import X")

    delivery = ModelOnexApiDelivery.from_record(lab.deliver(), requested_sha="145fd4d6")

    assert delivery.result is EnumOnexApiDeliveryResult.REFUSED
    assert delivery.is_failure is True
    assert delivery.pin_after == _PINNED
    assert ModelOnexApiDelivery.model_validate(delivery.model_dump()) == delivery


def test_a_candidate_that_imports_the_new_path_is_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: the check passes a good image and the recreate happens."""
    lab = _Lab(tmp_path, monkeypatch, router_imports=f"from {_NEW_IMPORT} import X")

    record = lab.deliver()

    assert record["result"] == "WRITTEN", record
    assert record["tag_advanced"] is True
    assert record["recreated"] is True
    assert record["pin_after"] == _NEXT
    assert lab.pin() == _NEXT
    assert len(lab.docker_run_calls) == 1
    assert (
        lab.docker_run_calls[0][lab.docker_run_calls[0].index("--entrypoint") + 1]
        == "python"
    )
    assert "--network" in lab.docker_run_calls[0]
    assert len(lab.compose_calls) == 1
    assert lab.compose_calls[0][-1] == executor_mod.ONEX_API_SERVICE


def test_a_guarded_or_function_level_import_is_not_a_crash_and_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only module-level imports crash the container before it serves."""
    lab = _Lab(
        tmp_path,
        monkeypatch,
        router_imports=(
            "try:\n"
            f"    from {_OLD_IMPORT} import X\n"
            "except ImportError:\n"
            "    X = None\n"
            "def late():\n"
            f"    from {_OLD_IMPORT} import X\n"
            "    return X\n"
        ),
    )

    record = lab.deliver()

    assert record["result"] == "WRITTEN", record
    assert lab.pin() == _NEXT


@pytest.mark.parametrize("failure", ["unavailable", "timeout"])
def test_a_check_that_could_not_run_keeps_the_previous_pin_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Fails closed, and worded as 'could not run', not as a missing module."""
    lab = _Lab(
        tmp_path,
        monkeypatch,
        router_imports=f"from {_NEW_IMPORT} import X",
        docker_run=failure,
    )

    record = lab.deliver()

    assert record["result"] == "REFUSED", record
    assert "could not run" in record["reason"]
    assert lab.pin() == _PINNED
    assert lab.compose_calls == []


def test_a_rollback_that_cannot_find_one_pin_line_says_so_and_does_not_guess(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "operator.env"
    env_file.write_text("A=1\n", encoding="utf-8")
    handler = HandlerOnexApiImageImportCheck(
        run=lambda argv, timeout: subprocess.CompletedProcess(
            argv, 125, stdout="", stderr="no docker"
        )
    )

    verdict = handler.handle(
        ModelOnexApiImageImportRequest(
            image=_NEXT, env_file=env_file, previous_pin=_PINNED
        )
    )

    assert verdict.importable is False
    assert verdict.rolled_back is False
    assert "0 ONEX_API_IMAGE= lines" in verdict.reason
    assert verdict.pin_after == _NEXT
    assert env_file.read_text("utf-8") == "A=1\n"

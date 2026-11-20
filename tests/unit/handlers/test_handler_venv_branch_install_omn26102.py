# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The host branch install of one omnimarket pull request head (OMN-26102).

No declared installer put one omnimarket pull request head into a host's dispatch
venv and took it out again. These tests use real ``uv`` and a real venv; the
packages are wheels built here byte by byte, so nothing needs a network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.handlers import handler_venv_branch_install as module
from omnibase_infra.handlers.handler_venv_branch_install import (
    HandlerVenvBranchInstall,
    head_from_watcher_facts,
    read_installed_revision,
    read_installed_source,
)
from omnibase_infra.handlers.venv_atomic_install import atomic_venv_install as engine
from omnibase_infra.models.model_venv_branch_install_request import (
    ModelVenvBranchInstallRequest,
)

pytestmark = pytest.mark.unit

OLD_SHA = "279f2d8799faf887f8bc0490a1209571b5ac9193"
HEAD_SHA = "9142221309c2ce47e4e78d3645c9f1d250850806"
UNITS = ("onex-floor-fill-tick.service",)


def _wheel(out: Path, name: str, version: str, revision: str) -> Path:
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": f"VERSION = {version!r}\n",
        f"{name}/REVISION": revision,
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest())
        record.append(f"{path},sha256={digest.rstrip(b'=').decode()},{len(text)}")
    record.append(f"{dist}/RECORD,,")
    files[f"{dist}/RECORD"] = "\n".join(record) + "\n"
    wheel = out / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return wheel


def _uv(*args: str) -> None:
    subprocess.run(["uv", *args], check=True, capture_output=True, text=True)


def _read_fixture_revision(venv_or_python: str | Path) -> str:
    path = Path(venv_or_python)
    venv = path.parent.parent if path.name == "python" else path
    for revision in venv.glob("lib/python*/site-packages/omnimarket/REVISION"):
        return revision.read_text(encoding="utf-8")
    return ""


def _listing(venv: Path) -> dict[str, str]:
    """Every file of the venv with its digest, bytecode caches left out."""
    out: dict[str, str] = {}
    for path in sorted(venv.rglob("*")):
        if "__pycache__" in path.parts or path.is_dir():
            continue
        key = str(path.relative_to(venv))
        if path.is_symlink():
            out[key] = f"link:{path.readlink()}"
        else:
            out[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


class _Host:
    def __init__(self, root: Path, *, revision: str = OLD_SHA) -> None:
        self.root = root
        wheels = root / "wheels"
        wheels.mkdir(parents=True)
        self.venv = root / "dispatch-venv"
        _uv("venv", "--relocatable", "--python", os.sys.executable, str(self.venv))
        self.python = str(self.venv / "bin" / "python")
        self.state = root / "state"
        self.state.mkdir()
        self.old = _wheel(wheels, "omnimarket", "1.0", revision)
        self.branch = _wheel(wheels, "omnimarket", "1.1", HEAD_SHA)
        self.core = _wheel(wheels, "core_pkg", "3.0", "core")
        self.spi = _wheel(wheels, "spi_pkg", "4.0", "spi")
        _uv(
            "pip",
            "install",
            "--python",
            self.python,
            "--no-deps",
            str(self.old),
            str(self.core),
            str(self.spi),
        )

    def handler(self) -> HandlerVenvBranchInstall:
        return HandlerVenvBranchInstall(
            ModelONEXContainer(),
            requirement_for=lambda _source, _sha: str(self.branch),
            source_reader=lambda _python: "https://example.invalid/omnimarket.git",
            revision_reader=_read_fixture_revision,
        )

    def install(self, **overrides: object) -> object:
        fields: dict[str, object] = {
            "command": "install",
            "host": "h051",
            "venv": self.venv,
            "state_dir": self.state,
            "pr_number": 6393,
            "head_sha": HEAD_SHA,
            "units": UNITS,
        }
        fields.update(overrides)
        return self.handler().handle(ModelVenvBranchInstallRequest(**fields))

    def revert(self) -> object:
        return self.handler().handle(
            ModelVenvBranchInstallRequest(
                command="revert", host="h051", venv=self.venv, state_dir=self.state
            )
        )


@pytest.fixture
def host(tmp_path: Path) -> _Host:
    return _Host(tmp_path)


def test_install_moves_only_omnimarket_and_records_the_installed_revision(
    host: _Host,
) -> None:
    before = _listing(host.venv)
    result = host.install()
    assert result.status == "installed", result.output
    assert result.installed == f"installed=h051:{HEAD_SHA}"
    after = _listing(host.venv)
    assert _read_fixture_revision(host.venv) == HEAD_SHA
    moved = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert moved, "nothing moved"
    assert all("omnimarket" in key for key in moved), sorted(moved)
    bystanders = [k for k in before if "core_pkg" in k or "spi_pkg" in k]
    assert bystanders
    assert all(before[k] == after[k] for k in bystanders)


def test_the_receipt_names_the_installed_head_and_the_revision_it_replaced(
    host: _Host,
) -> None:
    result = host.install()
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["host"] == "h051"
    assert receipt["repository"] == "omnimarket"
    assert receipt["pr_number"] == 6393
    assert receipt["head_sha"] == HEAD_SHA
    assert receipt["replaced_revision"] == OLD_SHA
    assert receipt["units"] == list(UNITS)
    assert receipt["state"] == "active"
    assert receipt["installed"] == f"h051:{HEAD_SHA}"
    assert "restart" not in json.dumps(receipt).lower()


def test_revert_returns_the_venv_to_the_recorded_revision(host: _Host) -> None:
    before = _listing(host.venv)
    installed = host.install()
    assert installed.status == "installed", installed.output
    result = host.revert()
    assert result.status == "reverted", result.output
    assert result.replaced_revision == OLD_SHA
    assert _read_fixture_revision(host.venv) == OLD_SHA
    assert _listing(host.venv) == before
    receipt = json.loads(Path(installed.receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "reverted"
    assert receipt["reverted_at"]


def test_a_failure_after_the_first_exchange_leaves_the_file_listing_identical(
    host: _Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _listing(host.venv)
    real_put = engine._put
    calls: list[str] = []

    def failing_put(*args: object, **kwargs: object) -> None:
        calls.append("put")
        if len(calls) == 2:
            raise OSError("injected failure after the first exchange")
        real_put(*args, **kwargs)

    monkeypatch.setattr(engine, "_put", failing_put)
    result = host.install()
    assert len(calls) == 2
    assert result.status == "failed"
    assert _listing(host.venv) == before
    assert not list((host.state / "receipts").glob("*.json"))


def test_a_host_with_no_recorded_rollback_target_is_refused(tmp_path: Path) -> None:
    host = _Host(tmp_path, revision="")
    before = _listing(host.venv)
    result = host.install()
    assert result.status == "refused"
    assert "rollback" in result.output
    assert _listing(host.venv) == before
    assert not (host.state / "receipts").exists()


def test_a_second_install_is_refused_until_the_first_is_reverted(host: _Host) -> None:
    assert host.install().status == "installed"
    second = host.install(head_sha="a" * 40)
    assert second.status == "refused"
    assert "revert" in second.output


def test_revert_without_an_active_receipt_is_refused(host: _Host) -> None:
    assert host.revert().status == "refused"


def test_no_git_or_restart_call_is_made_against_a_clone_or_the_runtime(
    host: _Host, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clone = tmp_path / "registry" / "omnimarket"
    clone.mkdir(parents=True)
    monkeypatch.setenv("OMNI_HOME", str(clone.parent))
    recorded: list[tuple[str, str]] = []
    real_run = subprocess.run

    def recording_run(argv: object, *args: object, **kwargs: object) -> object:
        recorded.append((" ".join(map(str, argv)), str(kwargs.get("cwd", ""))))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    assert host.install().status == "installed"
    assert host.revert().status == "reverted"
    assert recorded
    for command, cwd in recorded:
        words = command.split()
        assert words[0] != "git", command
        assert "systemctl" not in words and "docker" not in words, command
        assert not any(w in ("restart", "reload-or-restart") for w in words), command
        assert str(clone.parent) not in command and str(clone.parent) not in cwd


def test_a_request_without_units_or_a_full_sha_is_refused_at_the_model() -> None:
    base: dict[str, object] = {
        "command": "install",
        "host": "h051",
        "venv": Path("/v"),
        "state_dir": Path("/s"),
        "pr_number": 6393,
        "head_sha": HEAD_SHA,
        "units": UNITS,
    }
    for bad in ({"units": ()}, {"head_sha": "9142221"}, {"pr_number": 0}):
        with pytest.raises(ValueError):
            ModelVenvBranchInstallRequest(**{**base, **bad})


def test_the_installed_revision_is_the_commit_id_of_the_direct_url_record(
    host: _Host,
) -> None:
    dist = next(host.venv.glob("lib/python*/site-packages/omnimarket-1.0.dist-info"))
    assert read_installed_revision(host.python) == ""
    (dist / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "git+https://example.invalid/omnimarket.git",
                "vcs_info": {"vcs": "git", "commit_id": OLD_SHA},
            }
        ),
        encoding="utf-8",
    )
    assert read_installed_revision(host.python) == OLD_SHA
    assert (
        read_installed_source(host.python) == "https://example.invalid/omnimarket.git"
    )


def test_the_head_comes_from_the_watcher_state_and_only_for_an_open_pull_request() -> (
    None
):
    facts = {
        "facts": {
            "repo": "omnimarket",
            "number": 6393,
            "state": "OPEN",
            "head_sha": HEAD_SHA,
        }
    }
    assert head_from_watcher_facts(facts, repo="omnimarket", number=6393) == HEAD_SHA
    for wrong in (
        {"facts": {**facts["facts"], "state": "MERGED"}},
        {"facts": {**facts["facts"], "repo": "omnibase_infra"}},
        {"facts": {**facts["facts"], "number": 1}},
        {"facts": {**facts["facts"], "head_sha": "abc"}},
    ):
        with pytest.raises(ValueError):
            head_from_watcher_facts(wrong, repo="omnimarket", number=6393)


def test_the_module_names_no_restart_or_git_write() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    for word in ("systemctl", "docker", "launchctl"):
        assert word not in source, word
    assert not re.search(r"git\s+(fetch|pull|checkout|reset|clone|push)", source)
    assert not re.search(r"""["']git["']""", source)


def test_the_command_line_reads_the_head_from_the_watcher_record_only(
    tmp_path: Path,
) -> None:
    from omnibase_infra.nodes.node_venv_branch_install import (
        runtime_venv_branch_install as cli,
    )

    merged = tmp_path / "merged.json"
    merged.write_text(
        json.dumps(
            {
                "facts": {
                    "repo": "omnimarket",
                    "number": 6393,
                    "state": "MERGED",
                    "head_sha": HEAD_SHA,
                }
            }
        ),
        encoding="utf-8",
    )
    common = [
        "install",
        "--host",
        "h051",
        "--venv",
        str(tmp_path / "v"),
        "--state-dir",
        str(tmp_path / "s"),
        "--pr",
        "6393",
        "--unit",
        UNITS[0],
    ]
    assert cli.main(common) == 2
    assert cli.main([*common, "--pr-state", str(merged)]) == 2
    assert not (tmp_path / "s").exists()


def test_the_node_contract_routes_the_handler_and_the_models() -> None:
    import importlib

    import yaml

    contract = yaml.safe_load(
        (
            Path(module.__file__).parents[1]
            / "nodes/node_venv_branch_install/contract.yaml"
        ).read_text(encoding="utf-8")
    )
    entry = contract["handler_routing"]["handlers"][0]
    handler = importlib.import_module(entry["handler"]["module"])
    assert getattr(handler, entry["handler"]["name"]) is HandlerVenvBranchInstall
    for model in (contract["input_model"], contract["output_model"]):
        assert hasattr(importlib.import_module(model["module"]), model["name"])

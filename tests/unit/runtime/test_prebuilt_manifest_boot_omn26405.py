# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The image's prebuilt manifest passes its boot stamp check in the image it was built in (OMN-26405).

Measured on the dev image (revision 8b76765e2fbb): the build RUN passed
``PREBUILT_ACTIVE_RUNTIME_PACKAGES=`` (empty), so the stamp recorded
``active_packages=''``; the runtime compose gives the container
``ONEX_ACTIVE_RUNTIME_PACKAGES`` from ``docker/runtime-policy.env``, so boot read
``'omnibase_infra,omnimarket'``. Every boot of that image read the file as stale and took
the announce-and-discover path.

These tests read the instruction order of ``docker/Dockerfile.runtime`` and run the manifest
build RUN exactly as it is written there (its environment prefix, its build arguments' defaults,
its flags), then boot under the environment the compose file gives the runtime. Nothing here
is a shell: the one RUN is parsed, never executed.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.handlers import handler_package_announcer as announcer
from omnibase_infra.handlers.handler_prebuilt_manifest import run_boot_catalog
from omnibase_infra.nodes.node_boot_catalog_effect import runtime_prebuilt_manifest
from omnibase_infra.runtime.auto_wiring import discovery
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_cache_clear

_REPO = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO / "docker" / "Dockerfile.runtime"
_POLICY_ENV = _REPO / "docker" / "runtime-policy.env"
_EP = "omnibase_infra.handlers.handler_package_announcer.entry_points"
_EP_STAMP = "omnibase_infra.handlers.handler_prebuilt_manifest.entry_points"
_MANIFEST_MODULE = (
    "omnibase_infra.nodes.node_boot_catalog_effect.runtime_prebuilt_manifest"
)
_ACTIVE = "ONEX_ACTIVE_RUNTIME_PACKAGES"
# Instructions that cannot change an installed distribution, a contract file or a switch
# the stamp reads; every other instruction placed after the manifest RUN would.
_INERT_AFTER_BUILD = frozenset(
    {"HEALTHCHECK", "ENTRYPOINT", "CMD", "LABEL", "EXPOSE", "STOPSIGNAL", "ARG"}
)


def _instructions() -> list[str]:
    """Logical Dockerfile instructions in order: continuations joined, comments dropped."""
    out: list[str] = []
    current = ""
    for raw in _DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        out.append((current + line).strip())
        current = ""
    return out


def _manifest_build_index(instructions: list[str]) -> int:
    hits = [
        i for i, ins in enumerate(instructions) if f"{_MANIFEST_MODULE} build" in ins
    ]
    assert len(hits) == 1, f"expected one manifest build RUN, found {len(hits)}"
    return hits[0]


def _arg_defaults(instructions: list[str]) -> dict[str, str]:
    """Every ``ARG name=default`` in the file; the shell of a RUN sees each as an environment variable."""
    out: dict[str, str] = {}
    for ins in instructions:
        match = re.fullmatch(r"ARG\s+(\w+)(?:=(.*))?", ins)
        if match:
            raw = match.group(2) or ""
            out[match.group(1)] = shlex.split(raw)[0] if raw else ""
    return out


def _run_the_manifest_instruction(tmp_path: Path) -> tuple[str, list[str]]:
    """Run the manifest RUN's shell text as written, with ``python`` replaced by a recorder.

    The bind mount is mapped to the repository file it names and each ``ARG`` default is
    the environment the shell sees. The shell expands whatever the RUN's environment prefix
    says; the recorder writes down the ``ONEX_ACTIVE_RUNTIME_PACKAGES`` it was handed and the
    arguments after ``build``, which is exactly what the image build would have given the module.
    """
    instructions = _instructions()
    text = instructions[_manifest_build_index(instructions)]
    mounts: dict[str, Path] = {}

    def take_mount(match: re.Match[str]) -> str:
        fields = dict(f.split("=", 1) for f in match.group(1).split(","))
        if fields.get("type") == "bind":
            mounts[fields["target"]] = _REPO / fields["source"]
        return ""

    shell = re.sub(r"--mount=(\S+)\s*", take_mount, text.removeprefix("RUN ")).strip()
    for target, source in mounts.items():
        shell = shell.replace(target, str(source))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorded = tmp_path / "recorded.json"
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"json.dump({{'active': os.environ.get('{_ACTIVE}', ''), 'argv': sys.argv[1:]}}, "
        f"open({str(recorded)!r}, 'w'))\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    env = {**_arg_defaults(instructions), "PATH": f"{bin_dir}:/usr/bin:/bin"}
    subprocess.run(["/bin/sh", "-c", shell], env=env, check=True)
    seen = json.loads(recorded.read_text(encoding="utf-8"))
    argv: list[str] = seen["argv"]
    assert argv[:3] == ["-m", _MANIFEST_MODULE, "build"], argv
    return str(seen["active"]), argv[2:]


def _policy_env() -> dict[str, str]:
    """The ``--env-file docker/runtime-policy.env`` every compose service is given."""
    out: dict[str, str] = {}
    for line in _POLICY_ENV.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key] = value.strip().strip("'\"")
    return out


def _contract(name: str) -> str:
    return textwrap.dedent(f"""\
        name: "{name}"
        node_type: "EFFECT_GENERIC"
        contract_version:
          major: 1
          minor: 0
          patch: 0
        node_version: "1.0.0"
        description: "fixture"
        runtime_profiles: [main]
        event_bus:
          subscribe_topics:
            - "onex.evt.platform.{name.replace("_", "-")}-in.v1"
          publish_topics:
            - "onex.evt.platform.{name.replace("_", "-")}-out.v1"
        handler_routing:
          routing_strategy: operation_match
          handlers:
            - operation: run
              handler:
                name: HandlerFixture
                module: omnibase_infra.fixture
    """)


def _entry_point(name: str, dist: str, version: str, path: Path) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.dist = MagicMock()
    ep.dist.name = dist
    ep.dist.version = version
    ep.load.return_value = type("FixtureNode", (), {"contract_path": path})
    return ep


@pytest.fixture(autouse=True)
def _fresh_memo() -> Iterator[None]:
    discover_contracts_cache_clear()
    yield
    discover_contracts_cache_clear()


@pytest.fixture
def entry_points(tmp_path: Path) -> list[MagicMock]:
    """Two installed distributions named as the runtime policy's active package set names them."""
    out: list[MagicMock] = []
    for node, dist in (
        ("node_alpha_effect", "omnibase_infra"),
        ("node_beta_effect", "omnimarket"),
    ):
        directory = tmp_path / node
        directory.mkdir()
        path = directory / "contract.yaml"
        path.write_text(_contract(node), encoding="utf-8")
        out.append(_entry_point(node, dist, "1.0.0", path))
    return out


def _image_after_build(
    tmp_path: Path, entry_points: list[MagicMock], monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Apply the Dockerfile's layers in order: the fixture tree is installed, then the manifest RUN runs."""
    active, argv = _run_the_manifest_instruction(tmp_path)
    path = tmp_path / "image" / "prebuilt_node_manifest.json"
    monkeypatch.setenv(_ACTIVE, active)
    with (
        patch(_EP, return_value=entry_points),
        patch(_EP_STAMP, return_value=entry_points),
    ):
        assert runtime_prebuilt_manifest.main([*argv, "--path", str(path)]) == 0
    discover_contracts_cache_clear()
    return path


def _runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The container environment compose gives the main runtime (policy file plus the profile)."""
    monkeypatch.setenv(_ACTIVE, _policy_env()[_ACTIVE])
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    monkeypatch.delenv("ONEX_GATEWAY_CLOUD_MIRRORING_ENABLED", raising=False)


@pytest.mark.unit
def test_prebuilt_manifest_run_is_the_last_instruction_that_changes_a_stamp_input() -> (
    None
):
    """Measured: the build order is not the cause. No instruction after the manifest RUN changes an input."""
    instructions = _instructions()
    index = _manifest_build_index(instructions)

    after = instructions[index + 1 :]
    assert after, "the manifest RUN is followed by the healthcheck and entrypoint"
    assert {ins.split()[0] for ins in after} <= _INERT_AFTER_BUILD, after


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prebuilt_manifest_boot_uses_manifest_under_the_runtime_policy_environment(
    tmp_path: Path,
    entry_points: list[MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _image_after_build(tmp_path, entry_points, monkeypatch)
    _runtime_env(monkeypatch)
    announce = MagicMock(side_effect=AssertionError("boot fell back to announce"))
    parses = MagicMock(wraps=discovery.read_contract_yaml)
    with (
        patch(_EP, return_value=entry_points),
        patch(_EP_STAMP, return_value=entry_points),
        patch.object(discovery, "read_contract_yaml", parses),
        patch.object(announcer, "read_contract_yaml", parses),
        caplog.at_level(logging.WARNING),
    ):
        boot = await run_boot_catalog(workload="main", announce=announce, path=path)

    stale = [r.getMessage() for r in caplog.records if "not used" in r.getMessage()]
    assert not stale, stale
    assert boot.manifest is not None
    assert boot.manifest.total_discovered == 2
    announce.assert_not_called()
    parses.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prebuilt_manifest_boot_uses_manifest_names_the_stamp_field_when_it_falls_back(
    tmp_path: Path,
    entry_points: list[MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Positive control: a boot under a different active package set reads the same file stale, by field name."""
    path = _image_after_build(tmp_path, entry_points, monkeypatch)
    _runtime_env(monkeypatch)
    monkeypatch.setenv(_ACTIVE, "omnibase_infra")
    with (
        patch(_EP, return_value=entry_points),
        patch(_EP_STAMP, return_value=entry_points),
        caplog.at_level(logging.WARNING),
    ):
        boot = await run_boot_catalog(
            workload="main", announce=announcer.run_boot_step0, path=path
        )

    assert boot.manifest is None
    assert any("active_packages:" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["version", "contract", "cloud_mirroring"])
async def test_prebuilt_manifest_boot_uses_manifest_negative_input_changed_after_the_build_reads_stale(
    tmp_path: Path,
    entry_points: list[MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    change: str,
) -> None:
    """A stamp input that moves after the manifest step still reads stale and falls back."""
    path = _image_after_build(tmp_path, entry_points, monkeypatch)
    _runtime_env(monkeypatch)
    expected = {
        "version": "distribution_version:omnimarket:node_beta_effect:1.0.0->1.0.1",
        "contract": "contract_hash:",
        "cloud_mirroring": "gateway_cloud_mirroring",
    }[change]
    if change == "version":
        entry_points[1].dist.version = "1.0.1"
    elif change == "contract":
        contract = entry_points[0].load.return_value.contract_path
        contract.write_text(contract.read_text(encoding="utf-8") + "# edited\n")
    else:
        monkeypatch.setenv("ONEX_GATEWAY_CLOUD_MIRRORING_ENABLED", "true")
    with (
        patch(_EP, return_value=entry_points),
        patch(_EP_STAMP, return_value=entry_points),
        caplog.at_level(logging.WARNING),
    ):
        boot = await run_boot_catalog(
            workload="main", announce=announcer.run_boot_step0, path=path
        )

    assert boot.manifest is None
    assert len(boot.announced.catalog.entries) == len(entry_points)
    assert any(expected in r.getMessage() for r in caplog.records)

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the in-image required-config-path assertion (OMN-15676).

``runner_fleet.yaml`` was tracked in the repo, valid, and referenced by config
code -- and absent from every built runtime image, because no Dockerfile
``COPY`` shipped it. The deployed runtime raised
``FileNotFoundError: /app/config/runner_fleet.yaml`` during auto-wiring. Every
repo-level check passed the whole time; the third occurrence of that class
(after the grants fixture and ``routing_tiers.yaml``) is what this mechanism
exists to be the last of.

Two properties are held here:

* the assertion is **fail-closed** -- a probe that cannot run, a path with no
  verdict, or an emptied registry must all be failures, never green;
* the Dockerfile actually ships every path the typed registry declares, so the
  registry cannot drift into decoration.

The Dockerfile check is deliberately *supplementary*: it inspects the working
tree, which is exactly the surface that was green during all three incidents.
The load-bearing gate is ``assert_image_config_paths.py`` running inside the
built image, wired ahead of the push step in both runtime image builds.
"""

from __future__ import annotations

import importlib.util
import re
import stat
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ci" / "assert_image_config_paths.py"
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "assert_image_config_paths", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script()


def _write_fake_docker(tmp_path: Path, body: str) -> str:
    """Write a stub `docker` executable and return its path."""
    fake = tmp_path / "fake-docker"
    fake.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(fake)


def test_all_paths_present_passes(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every declared path reported PRESENT -> exit 0."""
    paths = script.REQUIRED_IMAGE_CONFIG_PATHS
    echoes = "\n".join(f"echo 'PRESENT {entry.image_path}'" for entry in paths)
    docker = _write_fake_docker(tmp_path, f"{echoes}\nexit 0")

    rc = script.main(["--image", "img:test", "--docker-bin", docker])

    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_missing_path_fails_and_names_it(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A single missing path fails the assertion and is named in the output.

    This is the runner_fleet.yaml scenario reproduced against the mechanism.
    """
    entries = script.REQUIRED_IMAGE_CONFIG_PATHS
    target = entries[0].image_path
    lines = [
        f"echo 'MISSING {entry.image_path}'"
        if entry.image_path == target
        else f"echo 'PRESENT {entry.image_path}'"
        for entry in entries
    ]
    docker = _write_fake_docker(tmp_path, "\n".join(lines) + "\nexit 0")

    rc = script.main(["--image", "img:test", "--docker-bin", docker])

    assert rc == 1
    captured = capsys.readouterr()
    assert target in captured.err
    assert "COPY" in captured.err


def test_probe_container_failure_fails_closed(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A probe that cannot run proves nothing and must NOT report green."""
    docker = _write_fake_docker(tmp_path, "echo 'no such image' >&2\nexit 125")

    rc = script.main(["--image", "img:test", "--docker-bin", docker])

    assert rc == 1
    assert "failing closed" in capsys.readouterr().err


def test_missing_docker_binary_fails_closed(script: ModuleType, tmp_path: Path) -> None:
    """An absent container CLI is a failure, not a skip."""
    rc = script.main(
        ["--image", "img:test", "--docker-bin", str(tmp_path / "not-a-real-binary")]
    )
    assert rc == 1


def test_silent_probe_yields_no_verdict_failure(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 0 with no verdict lines must not be read as 'all present'."""
    docker = _write_fake_docker(tmp_path, "exit 0")

    rc = script.main(["--image", "img:test", "--docker-bin", docker])

    assert rc == 1
    assert "no verdict" in capsys.readouterr().err


def test_empty_registry_fails_closed(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry emptied by a bad refactor must fail, not vacuously pass."""
    monkeypatch.setattr(script, "REQUIRED_IMAGE_CONFIG_PATHS", ())
    docker = _write_fake_docker(tmp_path, "exit 0")

    assert script.main(["--image", "img:test", "--docker-bin", docker]) == 1


def test_registry_rejects_unsafe_paths(script: ModuleType, tmp_path: Path) -> None:
    """Relative or shell-metacharacter paths are rejected before any probe runs."""
    template = script.REQUIRED_IMAGE_CONFIG_PATHS[0]
    bad = (
        template.model_copy(update={"image_path": "config/relative.yaml"}),
        template.model_copy(update={"image_path": "/app/config/$(whoami).yaml"}),
    )
    problems = script._validate_paths(bad)
    assert any("absolute" in p for p in problems)
    assert any("quoting" in p for p in problems)


def test_registry_is_not_empty(script: ModuleType) -> None:
    """The live registry must actually declare something."""
    assert len(script.REQUIRED_IMAGE_CONFIG_PATHS) >= 2


def test_runner_fleet_config_is_registered(script: ModuleType) -> None:
    """The OMN-15676 defect path is covered by the registry."""
    paths = {entry.image_path for entry in script.REQUIRED_IMAGE_CONFIG_PATHS}
    assert "/app/config/runner_fleet.yaml" in paths


def test_dockerfile_ships_every_registered_path(script: ModuleType) -> None:
    """Supplementary static check: every registry entry has a COPY destination.

    Not a substitute for the in-image assertion -- the working tree was green
    during all three incidents. This only catches the drift where an entry is
    added to the registry with no matching COPY, before the (slower) image
    build reports it.
    """
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    # Join Dockerfile line continuations so multi-line COPYs are one token run.
    flattened = re.sub(r"\\\s*\n\s*", " ", dockerfile)
    copy_destinations: list[str] = []
    for line in flattened.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        tokens = [t for t in stripped.split()[1:] if not t.startswith("--")]
        if tokens:
            copy_destinations.append(tokens[-1])

    missing = [
        entry.image_path
        for entry in script.REQUIRED_IMAGE_CONFIG_PATHS
        if not any(
            dest == entry.image_path
            or (dest.endswith("/") and entry.image_path.startswith(dest))
            for dest in copy_destinations
        )
    ]
    assert not missing, (
        "registry entries with no COPY destination in docker/Dockerfile.runtime: "
        f"{missing}"
    )


def test_registered_source_files_exist_in_build_context() -> None:
    """The repo-side source of the runner-fleet COPY is present and tracked."""
    assert (_REPO_ROOT / "config" / "runner_fleet.yaml").is_file()


def test_replay_real_prefix_image_probe_is_rejected(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """OMN-15676 replay: the real pre-fix image's probe bytes must be REJECTED.

    The fixture is the verbatim stdout of this guard's own probe, run inside
    ghcr.io/omninode-ai/omnibase-infra-runtime@sha256:403b6092 (the image family
    that was live on onex-dev), captured on omni-201-ts. Not a reconstruction:
    the same two lines the probe emitted, MISSING for runner_fleet.yaml and
    PRESENT for routing_tiers.yaml.

    The PRESENT line is load-bearing as a control. A broken implementation that
    reported every path missing would satisfy an all-missing capture and look
    correct; it cannot satisfy this one, because the guard must simultaneously
    accept the path that really is in the image and reject the one that is not.
    """
    fixture = (
        _REPO_ROOT
        / "tests"
        / "fixtures"
        / "omn15676"
        / "runtime-image-config-probe-403b6092.txt.captured"
    )
    captured_bytes = fixture.read_text(encoding="utf-8")

    # Sanity-bind the replay to the registry: the capture predates any registry
    # entry added later, so a case that grew a third path would silently stop
    # being replayed by these bytes.
    registry_paths = {entry.image_path for entry in script.REQUIRED_IMAGE_CONFIG_PATHS}
    captured_paths = {
        line.split(" ", 1)[1] for line in captured_bytes.splitlines() if " " in line
    }
    assert registry_paths == captured_paths, (
        "the captured probe no longer covers the registry; re-capture against a "
        "current image rather than editing the fixture"
    )

    replay = tmp_path / "captured-stdout"
    replay.write_text(captured_bytes, encoding="utf-8")
    docker = _write_fake_docker(tmp_path, f"cat {replay}")

    rc = script.main(["--image", "replay:403b6092", "--docker-bin", docker])

    assert rc == 1, "the guard accepted the image that boot-crashed onex-dev"
    out = capsys.readouterr()
    assert "/app/config/runner_fleet.yaml" in out.err
    # The control: the path that IS in the image must not be reported missing.
    assert "/app/config/delegation/routing_tiers.yaml" not in out.err
    assert "OK       /app/config/delegation/routing_tiers.yaml" in out.out

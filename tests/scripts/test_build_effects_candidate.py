# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Candidate source/content gates with isolated Git fixtures and stubbed builds."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/runtime_build"
BASE = "sha256:" + "1" * 64
CANDIDATE = "sha256:" + "2" * 64


@pytest.mark.unit
def test_offline_builder_backend_is_declared_in_frozen_dev_environment() -> None:
    """The no-isolation builder must not depend on an ambient Hatch install."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert config["build-system"]["build-backend"] == "hatchling.build"
    dev_dependencies = config["dependency-groups"]["dev"]
    assert any(dependency.startswith("hatchling") for dependency in dev_dependencies)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=True
    ).stdout.strip()


@pytest.fixture(scope="module")
def source_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("candidate-sources")
    pins = {}
    for name in ("omnibase_core", "omnibase_compat", "omnimarket", "omnibase_infra"):
        clone = root / name
        package = clone / "src" / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"PACKAGE = {name!r}\n")
        (clone / "pyproject.toml").write_text(
            f"[project]\nname = '{name.replace('_', '-')}'\nversion = '1.0.0'\n"
        )
        _git(clone, "init", "-q", "-b", "dev")
        _git(clone, "config", "user.name", "Candidate fixture")
        _git(clone, "config", "user.email", "candidate@example.test")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", "immutable source fixture")
        if name == "omnibase_infra":
            (clone / "entrypoint.sh").write_text("changed startup material\n")
            _git(clone, "add", "entrypoint.sh")
            _git(clone, "commit", "-q", "-m", "startup change without package changes")
        pins[name] = _git(clone, "rev-parse", "HEAD")
    return {
        "schema_version": "1",
        "ticket_id": "OMN-17991",
        "reason": "Build only the approved Market candidate",
        "base_image_id": BASE,
        "source_root": str(root),
        "source_pins": {
            name: sha for name, sha in pins.items() if name != "omnibase_infra"
        },
        "infra_source_root": str(root / "omnibase_infra"),
        "infra_source_sha": pins["omnibase_infra"],
    }


@pytest.fixture
def builder(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = importlib.import_module("build_effects_candidate")
    original_version = module.importlib.metadata.version
    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        lambda name: "1.27.0" if name == "hatchling" else original_version(name),
    )
    return module


@pytest.fixture
def plan(builder: Any, source_data: dict[str, Any]) -> Any:
    return builder.ModelEffectsCandidatePlan.model_validate_json(
        json.dumps(source_data)
    )


class BuildBoundary:
    """Only external Docker/build operations are replaced; sources are real."""

    def __init__(self, builder: Any, plan: Any, mode: str = "ok") -> None:
        self.calls: list[list[str]] = []
        self.mode = mode
        self.plan = plan
        self.digests = builder.validate_sources(plan)
        self.aliases = {BASE: BASE}
        self.built = False
        self.base = {
            "Id": BASE,
            "Config": {"User": "root", "Env": ["STABLE=yes"]},
            "RootFS": {"Layers": ["sha256:" + "a" * 64]},
        }

    def __call__(self, command: list[str], stdin: str | None, timeout: int) -> str:
        self.calls.append(command)
        assert timeout <= 900
        if command[:3] == ["docker", "image", "inspect"]:
            reference = command[-1]
            found = self.aliases[reference]
            if found == BASE:
                image = copy.deepcopy(self.base)
                if self.mode == "base-alias-drift" and self.built and reference != BASE:
                    image["Id"] = CANDIDATE
                return json.dumps([image])
            image = copy.deepcopy(self.base)
            image["Id"] = CANDIDATE
            image["RootFS"]["Layers"].append("sha256:" + "b" * 64)
            image["Config"]["Labels"] = {
                "com.omninode.build_source": "scoped-derived",
                "com.omninode.promotion_class": "stability-candidate",
                "com.omninode.non_main_lineage": "true",
                "com.omninode.allowed_lane": "dev",
                "com.omninode.base_image_id": BASE,
            }
            if self.mode == "rootfs":
                image["RootFS"]["Layers"] = ["wrong-parent"]
            if self.mode == "config":
                image["Config"]["Env"] = ["CHANGED=yes"]
            if self.mode == "labels":
                image["Config"]["Labels"]["com.omninode.non_main_lineage"] = "false"
            return json.dumps([image])
        if command[:2] == ["docker", "run"]:
            assert (
                "--read-only" in command
                and command[command.index("--network") + 1] == "none"
            )
            assert stdin and "package_digests" in stdin
            is_candidate = CANDIDATE in command
            result = {
                "package_digests": dict(self.digests),
                "source_pins": dict(self.plan.source_pins) if is_candidate else {},
                "base_image_id": BASE if is_candidate else None,
                "infra_source_ref": self.plan.infra_source_sha,
                "dependency_fingerprint": "d" * 64,
                "shared_runtime_fingerprint": "e" * 64,
                "required_topics": ["onex.evt.omnimarket.closeout.v1"],
            }
            if not is_candidate:
                result["package_digests"]["omnimarket"] = "0" * 64
                if self.mode == "base-source":
                    result["package_digests"]["omnibase_core"] = "f" * 64
                if self.mode == "infra-lineage":
                    result["infra_source_ref"] = _git(
                        self.plan.infra_source_root, "rev-parse", "HEAD^"
                    )
            elif self.mode == "source":
                result["package_digests"]["omnimarket"] = "f" * 64
            elif self.mode == "dependencies":
                result["dependency_fingerprint"] = "f" * 64
            elif self.mode == "startup":
                result["shared_runtime_fingerprint"] = "f" * 64
            return json.dumps(result)
        if command[:2] == ["uv", "build"]:
            assert {
                "--offline",
                "--no-build-isolation",
                "--no-sources",
                "--no-python-downloads",
            } <= set(command)
            payload = Path(command[command.index("--out-dir") + 1])
            source = Path(command[-1]) / "src/omnimarket"
            with zipfile.ZipFile(
                payload / "omnimarket-1.0.0-py3-none-any.whl", "w"
            ) as archive:
                for path in source.rglob("*"):
                    if path.is_file():
                        archive.writestr(
                            "omnimarket/" + str(path.relative_to(source)),
                            path.read_bytes(),
                        )
                archive.writestr(
                    "omnimarket-1.0.0.dist-info/METADATA",
                    "Name: omnimarket\nVersion: 1.0.0\n",
                )
            return "offline wheel built"
        if command[:3] == ["docker", "image", "ls"]:
            return "existing" if self.mode == "tag-exists" else ""
        if command[:3] == ["docker", "image", "tag"]:
            assert command[-2] == BASE
            assert command[-1].startswith("onex-effects-candidate-base:omn-17991-")
            assert command[-1] not in self.aliases
            self.aliases[command[-1]] = BASE
            return ""
        if command[:2] == ["docker", "build"]:
            if self.mode == "unfenced":
                from scoped_effects_deploy import CommandProcessUnfencedError

                raise CommandProcessUnfencedError(
                    "build process group termination is unproved"
                )
            assert {"--pull=false", "--network=none"} <= set(command)
            tag = command[command.index("--tag") + 1]
            assert tag.startswith("onex-effects-candidate:omn-17991-")
            context = Path(command[-1])
            recipe = (context / "Dockerfile").read_text()
            assert recipe.startswith("FROM onex-effects-candidate-base:")
            assert "--network=none" in recipe and "--mount=type=bind" in recipe
            assert "Dockerfile.runtime" not in recipe
            manifest = json.loads((context / "payload/manifest.json").read_text())
            assert len(manifest["proofs"]) == 4
            assert all(row["status"] == "pending" for row in manifest["proofs"])
            Path(command[command.index("--iidfile") + 1]).write_text(CANDIDATE)
            self.aliases[tag] = CANDIDATE
            self.built = True
            return "candidate built"
        raise AssertionError(f"unexpected external command: {command}")


@pytest.mark.unit
def test_preview_only_reads_sources_and_base(
    builder: Any, plan: Any, tmp_path: Path
) -> None:
    boundary = BuildBoundary(builder, plan)
    output = tmp_path / "candidate"
    result = builder.build_candidate(plan, output, runner=boundary)
    assert result["status"] == "PREVIEW"
    assert not output.exists()
    assert boundary.calls == [["docker", "image", "inspect", BASE]]


@pytest.mark.unit
def test_candidate_preserves_baseline_and_retains_artifacts(
    builder: Any, plan: Any, tmp_path: Path
) -> None:
    boundary = BuildBoundary(builder, plan)
    output = tmp_path / "candidate"
    result = builder.build_candidate(plan, output, execute=True, runner=boundary)
    assert result["status"] == "BUILT"
    assert result["candidate_image_id"] == CANDIDATE
    assert result["base_image_id"] == BASE
    assert len(result["artifact_actions"]) == 2
    assert (
        json.loads((output / "candidate-receipt.json").read_text())["status"] == "BUILT"
    )
    trace = "\n".join(" ".join(command) for command in boundary.calls)
    assert not any(
        forbidden in trace
        for forbidden in (":latest", "compose", "push", "prune", "restart")
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode",
    [
        "base-source",
        "infra-lineage",
        "source",
        "dependencies",
        "startup",
        "rootfs",
        "config",
        "labels",
        "base-alias-drift",
        "tag-exists",
        "unfenced",
    ],
)
def test_incompatible_artifacts_fail_closed(
    builder: Any, plan: Any, tmp_path: Path, mode: str
) -> None:
    boundary = BuildBoundary(builder, plan, mode)
    output = tmp_path / "candidate"
    with pytest.raises(RuntimeError):
        builder.build_candidate(plan, output, execute=True, runner=boundary)
    if output.exists():
        assert (
            json.loads((output / "candidate-receipt.json").read_text())["status"]
            == "REFUSED"
        )
    assert not any(
        "compose" in command or "push" in command for command in boundary.calls
    )
    if mode == "tag-exists":
        assert not any(
            command[:3] == ["docker", "image", "tag"] for command in boundary.calls
        )


@pytest.mark.unit
def test_source_mismatch_refuses_before_docker(
    builder: Any, plan: Any, tmp_path: Path
) -> None:
    invalid = plan.model_copy(
        update={"source_pins": {**plan.source_pins, "omnimarket": "f" * 40}}
    )
    calls: list[list[str]] = []

    def no_external(command: list[str], _stdin: str | None, _timeout: int) -> str:
        calls.append(command)
        raise AssertionError("invalid source must fail before Docker")

    with pytest.raises(Exception, match="cannot resolve"):
        builder.build_candidate(
            invalid, tmp_path / "candidate", execute=True, runner=no_external
        )
    assert calls == []


@pytest.mark.unit
def test_existing_output_is_not_overwritten(
    builder: Any, plan: Any, tmp_path: Path
) -> None:
    boundary = BuildBoundary(builder, plan)
    output = tmp_path / "candidate"
    output.mkdir()
    sentinel = output / "operator-work"
    sentinel.write_text("preserve me")
    with pytest.raises(ValueError, match="already exists"):
        builder.build_candidate(plan, output, execute=True, runner=boundary)
    assert sentinel.read_text() == "preserve me"
    assert not boundary.built


@pytest.mark.unit
@pytest.mark.parametrize(
    "path", ["../outside", "another_package/payload.py", "unowned.pth"]
)
def test_wheel_cannot_write_foreign_paths(
    builder: Any, tmp_path: Path, path: str
) -> None:
    wheel = tmp_path / "invalid.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(path, "foreign payload")
    with pytest.raises(ValueError):
        builder.validate_wheel(wheel, tmp_path / "source", tmp_path / "proof")
    assert not (tmp_path / "outside").exists()


@pytest.mark.unit
def test_wheel_content_parity_is_not_a_source_label(
    builder: Any, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("EXPECTED=True\n")
    wheel = tmp_path / "stale.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("omnimarket/__init__.py", "STALE=True\n")
        archive.writestr("omnimarket-1.0.dist-info/METADATA", "Name: omnimarket\n")
    with pytest.raises(ValueError, match="differs from pinned"):
        builder.validate_wheel(wheel, source, tmp_path / "proof")


@pytest.mark.unit
def test_builder_uses_the_process_group_fenced_runner(builder: Any) -> None:
    from scoped_effects_deploy import _run_command

    assert builder.build_candidate.__kwdefaults__["runner"] is _run_command

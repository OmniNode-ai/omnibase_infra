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

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

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
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=True,
        env=scrub_git_location_env(),
    ).stdout.strip()


_MARKET_FORCE_INCLUDE = (
    "[tool.hatch.build.targets.wheel.force-include]\n"
    '"config/ci_bus_lanes.yaml" = "omnimarket/config/ci_bus_lanes.yaml"\n'
)


def _write_force_included(archive: zipfile.ZipFile, repo: Path) -> None:
    """Honour the wheel force-include table the way hatch does."""
    config = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    wheel_target = config.get("tool", {}).get("hatch", {}).get("build", {})
    table = wheel_target.get("targets", {}).get("wheel", {}).get("force-include", {})
    for source_rel, destination in table.items():
        archive.writestr(destination, (repo / source_rel).read_bytes())


@pytest.fixture(scope="module")
def source_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("candidate-sources")
    pins = {}
    for name in ("omnibase_core", "omnibase_compat", "omnimarket", "omnibase_infra"):
        clone = root / name
        package = clone / "src" / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"PACKAGE = {name!r}\n")
        pyproject = f"[project]\nname = '{name.replace('_', '-')}'\nversion = '1.0.0'\n"
        if name == "omnimarket":
            # The real Market wheel force-includes root files into its package.
            (clone / "config").mkdir()
            (clone / "config/ci_bus_lanes.yaml").write_text("lanes: []\n")
            pyproject += _MARKET_FORCE_INCLUDE
        (clone / "pyproject.toml").write_text(pyproject)
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
        self.probe_timeouts: list[int] = []
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
        assert timeout <= 1800
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
            self.probe_timeouts.append(timeout)
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
                _write_force_included(archive, Path(command[-1]))
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
    (source / "src/omnimarket").mkdir(parents=True)
    (source / "src/omnimarket/__init__.py").write_text("EXPECTED=True\n")
    wheel = tmp_path / "stale.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("omnimarket/__init__.py", "STALE=True\n")
        archive.writestr("omnimarket-1.0.dist-info/METADATA", "Name: omnimarket\n")
    with pytest.raises(ValueError, match="differs from pinned"):
        builder.validate_wheel(wheel, source, tmp_path / "proof")


def _force_include_market(root: Path, *, declare_source: bool = True) -> Path:
    (root / "src/omnimarket").mkdir(parents=True)
    (root / "src/omnimarket/__init__.py").write_text("EXPECTED=True\n")
    if declare_source:
        (root / "config").mkdir()
        (root / "config/x.yaml").write_text("pinned: true\n")
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'omnimarket'\nversion = '1.0.0'\n"
        "[tool.hatch.build.targets.wheel.force-include]\n"
        '"config/x.yaml" = "omnimarket/config/x.yaml"\n'
    )
    return root


def _market_wheel(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("omnimarket/__init__.py", "EXPECTED=True\n")
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("omnimarket-1.0.dist-info/METADATA", "Name: omnimarket\n")
    return path


@pytest.mark.unit
def test_wheel_force_included_file_matches_pinned_source(
    builder: Any, tmp_path: Path
) -> None:
    """OMN-17991: a declared force-include with the same bytes is not drift."""
    source = _force_include_market(tmp_path / "source")
    wheel = _market_wheel(
        tmp_path / "market.whl", {"omnimarket/config/x.yaml": "pinned: true\n"}
    )
    assert (
        builder.validate_wheel(wheel, source, tmp_path / "proof")
        == "omnimarket-1.0.dist-info"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("files", "declare_source", "match"),
    [
        ({"omnimarket/config/x.yaml": "pinned: false\n"}, True, "differs from"),
        (
            {
                "omnimarket/config/x.yaml": "pinned: true\n",
                "omnimarket/undeclared.yaml": "extra\n",
            },
            True,
            "differs from",
        ),
        ({"omnimarket/config/x.yaml": "pinned: true\n"}, False, "force-include"),
    ],
    ids=["changed-bytes", "undeclared-extra", "declared-source-missing"],
)
def test_wheel_force_include_drift_still_refuses(
    builder: Any,
    tmp_path: Path,
    files: dict[str, str],
    declare_source: bool,
    match: str,
) -> None:
    source = _force_include_market(tmp_path / "source", declare_source=declare_source)
    wheel = _market_wheel(tmp_path / "market.whl", files)
    with pytest.raises(ValueError, match=match):
        builder.validate_wheel(wheel, source, tmp_path / "proof")


@pytest.mark.unit
def test_builder_uses_the_process_group_fenced_runner(builder: Any) -> None:
    from scoped_effects_deploy import _run_command

    assert builder.build_candidate.__kwdefaults__["runner"] is _run_command


def test_offline_image_probe_needs_no_writable_filesystem(builder: Any) -> None:
    # Same live defect as the executor's probe: `uv run` cannot create its
    # cache directory in a --read-only container (live proof, .105, OMN-17991).
    seen: list[list[str]] = []

    def runner(args: list[str], stdin: str | None, timeout: int) -> str:
        seen.append(args)
        assert timeout == 720
        assert stdin and "namespace['installed_package_root']" in stdin
        return json.dumps({"package_digests": {}})

    builder._probe(runner, "sha256:" + "1" * 64, timeout=720)
    (args,) = seen
    assert "--read-only" in args
    assert args[args.index("--entrypoint") + 1] == "/app/.venv/bin/python"
    assert args[args.index("sha256:" + "1" * 64) + 1 :] == ["-B", "-"]


def test_source_digest_includes_wheel_force_included_files(
    builder: Any, source_data: dict[str, Any], tmp_path: Path
) -> None:
    # omnibase_core force-includes architecture-handshakes/gitignore-baseline.yaml
    # into its wheel (OMN-18033), so the installed tree carries a file the
    # src/ tree never does. The workspace provenance already resolves that;
    # a src/-only digest refused every real base (live proof, .105, OMN-17991).
    from compute_workspace_provenance import _digest_files, _tracked_files

    root = tmp_path / "sources"
    pins = {}
    for name in ("omnibase_core", "omnibase_compat", "omnimarket", "omnibase_infra"):
        clone = root / name
        package = clone / "src" / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"PACKAGE = {name!r}\n")
        pyproject = f"[project]\nname = '{name.replace('_', '-')}'\nversion = '1.0.0'\n"
        if name == "omnibase_core":
            (clone / "architecture-handshakes").mkdir()
            (clone / "architecture-handshakes" / "gitignore-baseline.yaml").write_text(
                "baseline: []\n"
            )
            pyproject += (
                "[tool.hatch.build.targets.wheel.force-include]\n"
                '"architecture-handshakes/gitignore-baseline.yaml" = '
                '"omnibase_core/data/gitignore-baseline.yaml"\n'
            )
        (clone / "pyproject.toml").write_text(pyproject)
        _git(clone, "init", "-q", "-b", "dev")
        _git(clone, "config", "user.name", "Candidate fixture")
        _git(clone, "config", "user.email", "candidate@example.test")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", "immutable source fixture")
        pins[name] = _git(clone, "rev-parse", "HEAD")
    plan = builder.ModelEffectsCandidatePlan.model_validate_json(
        json.dumps(
            {
                **source_data,
                "source_root": str(root),
                "source_pins": {k: v for k, v in pins.items() if k != "omnibase_infra"},
                "infra_source_root": str(root / "omnibase_infra"),
                "infra_source_sha": pins["omnibase_infra"],
            }
        )
    )
    digests = builder.validate_sources(plan)
    core = root / "omnibase_core" / "src" / "omnibase_core"
    installed = {
        **_tracked_files(core),
        "data/gitignore-baseline.yaml": b"baseline: []\n",
    }
    assert digests["omnibase_core"] == _digest_files(installed)
    compat = root / "omnibase_compat" / "src" / "omnibase_compat"
    assert digests["omnibase_compat"] == _digest_files(_tracked_files(compat))


def test_in_image_proof_resolves_the_editable_project_tree(builder: Any) -> None:
    # The in-image proof loop covers omnibase_infra, which the runtime image
    # installs editable from /app/src; locate_file() names a site-packages path
    # that does not exist, so every real candidate failed its own proof (live
    # proof, .105, OMN-17991). It must share the probe's package resolution.
    assert "installed_package_root(proof['repo'])" in builder._INSTALLER
    assert "locate_file(proof['repo'])" not in builder._INSTALLER
    assert "scoped_image_probe.py" in builder.PROOF_TOOLS


@pytest.mark.unit
def test_loaded_host_scales_both_probe_bounds(
    builder: Any, plan: Any, tmp_path: Path
) -> None:
    # A fixed 120 s probe bound timed out on a loaded 32-core lab host at load1 48 to 63
    # (497.9 s measured, OMN-17991).
    boundary = BuildBoundary(builder, plan)
    output = tmp_path / "candidate"
    result = builder.build_candidate(
        plan,
        output,
        execute=True,
        runner=boundary,
        loadavg=lambda: (48.0, 0.0, 0.0),
        cpu_count=lambda: 32,
    )
    assert result["status"] == "BUILT"
    assert boundary.probe_timeouts == [720, 720]
    receipt = json.loads((output / "candidate-receipt.json").read_text())
    assert receipt["probe_timeout"]["seconds"] == 720
    assert receipt["probe_timeout"]["source"] == "host-load"
    preview = builder.build_candidate(
        plan,
        tmp_path / "preview",
        runner=BuildBoundary(builder, plan),
        loadavg=lambda: (48.0, 0.0, 0.0),
        cpu_count=lambda: 32,
    )
    assert preview["probe_timeout"]["seconds"] == 720


@pytest.mark.unit
def test_declared_probe_bound_wins(builder: Any, plan: Any, tmp_path: Path) -> None:
    declared = plan.model_copy(update={"probe_timeout_seconds": 900})
    boundary = BuildBoundary(builder, declared)
    builder.build_candidate(
        declared,
        tmp_path / "candidate",
        execute=True,
        runner=boundary,
        loadavg=lambda: (48.0, 0.0, 0.0),
        cpu_count=lambda: 32,
    )
    assert boundary.probe_timeouts == [900, 900]

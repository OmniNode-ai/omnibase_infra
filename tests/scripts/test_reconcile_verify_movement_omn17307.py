# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ``scripts/reconcile_verify_movement.py`` (OMN-17307).

The invariant under test is stated once, negatively, because that is the shape
the four motivating incidents took:

    a reconcile step is judged by reading the surface back, never by the exit
    status of the command that was supposed to move it.

`.201`, 2026-08-31 (OMN-17291) is the proof that the distinction is not
academic: ``omnibase_core`` carried ``core.bare=true`` on a clone with a full
working tree, so ``git fetch`` exited 0 *forever* while ``git checkout`` exited
128. Any loop reading the fetch's status saw progress. Any loop reading HEAD saw
the truth immediately.

Everything here is hermetic and offline -- real ``git init`` in ``tmp_path`` for
the clone observations, hand-built ``*.dist-info`` directories for the venv
observations. Nothing imports the package under test through the project venv:
the module is loaded from source by path, because it must run on a host where
that venv is exactly what is broken.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "reconcile_verify_movement.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_verify_movement", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because ``@dataclass`` resolves annotations through
    # ``sys.modules[cls.__module__]``; an unregistered module makes every
    # dataclass in the file raise at import time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vm() -> ModuleType:
    return _load()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_dist(
    site_packages: Path, name: str, version: str, commit: str | None = None
) -> Path:
    """Build a ``*.dist-info`` directory the way an installer would.

    The directory name is the only place a version is recorded that can be read
    without starting the interpreter, which is the whole point: the verifier has
    to work on a venv whose ``python`` will not run.
    """
    dist_info = site_packages / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )
    if commit is not None:
        (dist_info / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": "https://github.com/OmniNode-ai/omnimarket.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": commit,
                        "requested_revision": commit,
                    },
                }
            ),
            encoding="utf-8",
        )
    return dist_info


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _init_clone(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--quiet", str(repo)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--quiet", "-m", "init")
    return _git(repo, "rev-parse", "HEAD")


# --------------------------------------------------------------------------- #
# The verdict table -- the core of AC1/AC2/AC3
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("before", "after", "target", "expected", "ok"),
    [
        ("aaa", "bbb", "bbb", "MOVED", True),
        ("bbb", "bbb", "bbb", "ALREADY_AT_TARGET", True),
        # The OMN-17291 shape: the repair "succeeded" and the surface is still
        # where it started. before == after != target.
        ("aaa", "aaa", "bbb", "DID_NOT_MOVE", False),
        # Moved, but not to the target. Worse than not moving, and it must not
        # be mistaken for success just because something changed.
        ("aaa", "ccc", "bbb", "DID_NOT_MOVE", False),
        # Unreadable surface. `UNKNOWN` is never a pass (CLAUDE.md rule 12's
        # fail-closed posture, applied to host state).
        ("aaa", None, "bbb", "INDETERMINATE", False),
        ("aaa", "", "bbb", "INDETERMINATE", False),
        # Unknown target: we cannot assert anything, so we assert failure.
        ("aaa", "bbb", None, "INDETERMINATE", False),
    ],
)
def test_verdict_table(
    vm: ModuleType,
    before: str | None,
    after: str | None,
    target: str | None,
    expected: str,
    ok: bool,
) -> None:
    verdict = vm.verdict(before=before, after=after, target=target)
    assert verdict.name == expected
    assert verdict.ok is ok


def test_did_not_move_is_not_rescued_by_a_successful_command(vm: ModuleType) -> None:
    """There is no argument that turns a failed readback into a pass.

    ``verdict`` deliberately takes no exit status at all. A signature that
    accepted one would let a caller re-introduce exactly the defect this ticket
    exists to remove, so the absence of that parameter is itself the assertion.
    """
    import inspect

    params = set(inspect.signature(vm.verdict).parameters)
    assert params == {"before", "after", "target"}


# --------------------------------------------------------------------------- #
# Venv observation -- no interpreter start
# --------------------------------------------------------------------------- #
def test_observe_reads_version_without_starting_the_interpreter(
    vm: ModuleType, tmp_path: Path
) -> None:
    site_packages = tmp_path / "site-packages"
    _make_dist(site_packages, "omnibase_infra", "0.38.16")

    assert vm.observe_installed_version(site_packages, "omnibase_infra") == "0.38.16"


def test_observe_absent_distribution_is_none_not_an_exception(
    vm: ModuleType, tmp_path: Path
) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    assert vm.observe_installed_version(site_packages, "omnibase_infra") is None


def test_observe_commit_reads_direct_url(vm: ModuleType, tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    _make_dist(site_packages, "omnimarket", "0.4.11", commit="66b7131a3508")
    assert vm.observe_installed_commit(site_packages, "omnimarket") == "66b7131a3508"


def test_observe_commit_is_none_when_direct_url_absent(
    vm: ModuleType, tmp_path: Path
) -> None:
    """A PyPI-installed omnimarket has no ``direct_url.json``.

    That is the exact state the OMN-17190 foreign interpreter was in, and it
    must read as "cannot tell" -- which the verdict table then fails closed on --
    rather than as an empty string that happens to compare unequal to a SHA.
    """
    site_packages = tmp_path / "site-packages"
    _make_dist(site_packages, "omnimarket", "0.4.10")
    assert vm.observe_installed_commit(site_packages, "omnimarket") is None


def test_observe_site_packages_is_resolved_from_the_venv_root(
    vm: ModuleType, tmp_path: Path
) -> None:
    venv = tmp_path / ".venv"
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    assert vm.resolve_site_packages(venv) == site_packages


def test_observe_site_packages_missing_venv_is_none(
    vm: ModuleType, tmp_path: Path
) -> None:
    assert vm.resolve_site_packages(tmp_path / "nope") is None


# --------------------------------------------------------------------------- #
# Clone observation -- the core.bare trap (OMN-17291 AC2, asserted here on the
# observation primitive so every caller inherits it)
# --------------------------------------------------------------------------- #
def test_clone_head_is_readable_on_a_normal_clone(
    vm: ModuleType, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    head = _init_clone(repo)
    assert vm.observe_clone_head(repo) == head


def test_clone_with_core_bare_true_and_a_working_tree_is_reported_unhealthy(
    vm: ModuleType, tmp_path: Path
) -> None:
    """``core.bare=true`` on a clone that has a working tree.

    Proven on `.201` 2026-08-31: ``git fetch`` exits 0 and ``git checkout`` exits
    128, so this clone can never advance while every fetch-shaped health check
    reports it fine. ``observe_clone_health`` must name it as a defect, and the
    reason must say what is wrong -- a refusal that does not is a dead end.
    """
    repo = tmp_path / "repo"
    _init_clone(repo)
    _git(repo, "config", "core.bare", "true")

    health = vm.observe_clone_health(repo)
    assert health.healthy is False
    assert "core.bare" in health.reason
    assert "work tree" in health.reason or "working tree" in health.reason


def test_healthy_clone_reports_healthy(vm: ModuleType, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_clone(repo)
    assert vm.observe_clone_health(repo).healthy is True


def test_missing_clone_is_unhealthy_not_absent(vm: ModuleType, tmp_path: Path) -> None:
    health = vm.observe_clone_health(tmp_path / "nope")
    assert health.healthy is False


# --------------------------------------------------------------------------- #
# Lock targets
# --------------------------------------------------------------------------- #
def test_lock_targets_reads_versions_from_uv_lock(
    vm: ModuleType, tmp_path: Path
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """
version = 1

[[package]]
name = "omnibase-core"
version = "0.46.9"

[[package]]
name = "omnibase-compat"
version = "0.5.6"
""",
        encoding="utf-8",
    )
    targets = vm.lock_targets(lock, ["omnibase-core", "omnibase-compat", "absent-pkg"])
    assert targets["omnibase-core"] == "0.46.9"
    assert targets["omnibase-compat"] == "0.5.6"
    assert "absent-pkg" not in targets


def test_lock_targets_missing_lock_raises(vm: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vm.lock_targets(tmp_path / "no.lock", ["omnibase-core"])


# --------------------------------------------------------------------------- #
# Floor emission -- OMN-17309 depends on this contract
# --------------------------------------------------------------------------- #
def test_floor_document_is_shell_parseable_and_records_what_was_proven(
    vm: ModuleType, tmp_path: Path
) -> None:
    """The floor is consumed by ``scripts/onex`` in awk, with no JSON parser.

    So the emitted shape is part of the contract, not an implementation detail:
    two-space-indented keys inside a ``distributions`` object, and the
    distribution keys spelled exactly as the ``*.dist-info`` prefix (underscores,
    not hyphens) so the wrapper never has to normalise a name.
    """
    out = tmp_path / "floor.json"
    vm.write_floor(
        output=out,
        omni_home=tmp_path,
        distributions={"omnibase_infra": "0.38.16", "omnibase_core": "0.46.9"},
        omnimarket_commit="66b7131a350858309f7833b8c02f97afc2a550e7",
    )

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "onex.workspace.floor.v1"
    assert doc["distributions"]["omnibase_infra"] == "0.38.16"
    assert doc["omnimarket_commit"].startswith("66b7131a")
    assert doc["generated_at"].endswith("Z")

    raw = out.read_text(encoding="utf-8")
    assert '"distributions": {' in raw
    assert '"omnibase_infra": "0.38.16"' in raw


def test_floor_refuses_a_hyphenated_distribution_key(
    vm: ModuleType, tmp_path: Path
) -> None:
    """A hyphenated key would silently never match a ``*.dist-info`` prefix.

    The wrapper would then read "floor does not mention this package" and pass a
    stale venv. Refusing at write time keeps that failure impossible rather than
    invisible.
    """
    with pytest.raises(ValueError, match="omnibase-infra"):
        vm.write_floor(
            output=tmp_path / "floor.json",
            omni_home=tmp_path,
            distributions={"omnibase-infra": "0.38.16"},
            omnimarket_commit="aaaa",
        )


# --------------------------------------------------------------------------- #
# Pinned candidate source provenance -- OMN-18929
# --------------------------------------------------------------------------- #
def _candidate_root(tmp_path: Path) -> tuple[Path, list[str]]:
    root = tmp_path / "candidate"
    names = ["omnibase_infra", "omnibase_core"]
    for name in names:
        _init_clone(root / name)
    return root, names


def test_candidate_manifest_proves_exact_clean_source_set(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    manifest = root / ".onex-candidate-source.json"

    vm.write_candidate_source_manifest(
        output=manifest, omni_home=root, repositories=names
    )
    proof = vm.verify_candidate_source_manifest(
        manifest=manifest, omni_home=root, repositories=names
    )

    assert proof["manifest"] == str(manifest)
    assert len(proof["sha256"]) == 64


def test_candidate_manifest_rejects_changed_or_dirty_source(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    manifest = root / ".onex-candidate-source.json"
    vm.write_candidate_source_manifest(
        output=manifest, omni_home=root, repositories=names
    )

    (root / "omnibase_core" / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        vm.verify_candidate_source_manifest(
            manifest=manifest, omni_home=root, repositories=names
        )

    _git(root / "omnibase_core", "add", "README.md")
    _git(root / "omnibase_core", "commit", "-m", "different source")
    with pytest.raises(ValueError, match="no longer matches"):
        vm.verify_candidate_source_manifest(
            manifest=manifest, omni_home=root, repositories=names
        )


def test_candidate_manifest_rejects_missing_extra_or_escaped_repository(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    manifest = root / ".onex-candidate-source.json"
    vm.write_candidate_source_manifest(
        output=manifest, omni_home=root, repositories=names
    )

    with pytest.raises(ValueError, match="repository set differs"):
        vm.verify_candidate_source_manifest(
            manifest=manifest, omni_home=root, repositories=[*names, "omnimarket"]
        )
    with pytest.raises(ValueError, match="invalid"):
        vm.write_candidate_source_manifest(
            output=manifest, omni_home=root, repositories=["../outside"]
        )


def test_candidate_content_snapshot_attests_dirty_tracked_and_untracked_bytes(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    manifest = root / ".onex-candidate-content-snapshot.json"
    tracked = root / "omnibase_core" / "README.md"
    tracked.write_text("tracked mutation\n", encoding="utf-8")
    untracked = root / "omnibase_core" / "candidate-input.json"
    untracked.write_text('{"exact": true}\n', encoding="utf-8")
    untracked.chmod(0o755)

    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    proof = vm.verify_candidate_content_snapshot(
        manifest=manifest, omni_home=root, repositories=names
    )
    assert len(proof["sha256"]) == 64

    untracked.write_text('{"exact": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )

    untracked.unlink()
    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )


def test_stage_digest_prunes_only_stage_exclusions_and_keeps_included_bytes(
    vm: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, names = _candidate_root(tmp_path)
    repo = root / "omnibase_core"
    (repo / ".gitignore").write_text(
        ".venv/\n**/__pycache__/\n*.egg-info/\n", encoding="utf-8"
    )
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "--quiet", "-m", "ignore generated stage exclusions")

    excluded_payloads = [
        repo / ".venv" / "lib" / "ignored.bin",
        repo / "src" / "package" / "__pycache__" / "ignored.pyc",
        repo / "src" / "package.egg-info" / "ignored.txt",
    ]
    for path in excluded_payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("before", encoding="utf-8")
    _git(
        repo,
        "check-ignore",
        *[str(path.relative_to(repo)) for path in excluded_payloads],
    )

    scanned: list[Path] = []
    original_scandir = vm.os.scandir

    def record_scandir(path: str | os.PathLike[str]) -> object:
        scanned_path = Path(path)
        scanned.append(scanned_path)
        return original_scandir(path)

    manifest = root / ".onex-candidate-content-snapshot.json"
    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    baseline_stage_digest = vm._workspace_stage_tree_sha256(repo)
    excluded_payloads[0].write_text("after-v2", encoding="utf-8")
    excluded_payloads[1].write_text("after-v2", encoding="utf-8")
    excluded_payloads[2].write_text("after-v2", encoding="utf-8")

    with monkeypatch.context() as context:
        context.setattr(vm.os, "scandir", record_scandir)
        excluded_stage_digest = vm._workspace_stage_tree_sha256(repo)
    assert excluded_stage_digest == baseline_stage_digest
    excluded_names = {".git", ".venv", "__pycache__"}
    assert not any(
        excluded_names.intersection(path.parts)
        or any(part.endswith((".pyc", ".egg-info")) for part in path.parts)
        for path in scanned
    )
    assert vm.verify_candidate_content_snapshot(
        manifest=manifest, omni_home=root, repositories=names
    )["sha256"]

    included = repo / "README.md"
    included.write_text("included mutation\n", encoding="utf-8")
    assert vm._workspace_stage_tree_sha256(repo) != baseline_stage_digest
    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )


def test_candidate_content_snapshot_rejects_untracked_mode_change(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    manifest = root / ".onex-candidate-content-snapshot.json"
    untracked = root / "omnibase_core" / "candidate-launcher.sh"
    untracked.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    untracked.chmod(0o644)

    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    untracked.chmod(0o755)

    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )


def test_candidate_content_snapshot_rejects_tracked_mode_change(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    manifest = root / ".onex-candidate-content-snapshot.json"
    tracked = root / "omnibase_core" / "README.md"
    tracked.chmod(0o644)

    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    tracked.chmod(0o755)

    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )


def test_candidate_content_snapshot_attests_relative_internal_symlink(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    repo = root / "omnibase_core"
    hook_dir = repo / "scripts" / "git-hooks"
    hook_dir.mkdir(parents=True)
    (repo / "scripts" / "canonical_clone_guard.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (repo / "scripts" / "alternate_guard.sh").write_text(
        "#!/usr/bin/env bash\nexit 1\n", encoding="utf-8"
    )
    hook = hook_dir / "pre-commit"
    hook.symlink_to("../canonical_clone_guard.sh")
    _git(
        repo,
        "add",
        "scripts/canonical_clone_guard.sh",
        "scripts/alternate_guard.sh",
        "scripts/git-hooks/pre-commit",
    )
    manifest = root / ".onex-candidate-content-snapshot.json"

    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    vm.verify_candidate_content_snapshot(
        manifest=manifest, omni_home=root, repositories=names
    )

    hook.unlink()
    hook.symlink_to("../alternate_guard.sh")
    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )


def test_candidate_content_snapshot_rejects_absolute_internal_symlink(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    repo = root / "omnibase_core"
    hook_dir = repo / "scripts" / "git-hooks"
    hook_dir.mkdir(parents=True)
    target = repo / "scripts" / "canonical_clone_guard.sh"
    target.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    hook = hook_dir / "pre-commit"
    hook.symlink_to(target.resolve())
    _git(
        repo, "add", "scripts/canonical_clone_guard.sh", "scripts/git-hooks/pre-commit"
    )

    with pytest.raises(ValueError, match="absolute staged symlink"):
        vm.write_candidate_content_snapshot(
            output=root / ".onex-candidate-content-snapshot.json",
            omni_home=root,
            repositories=names,
        )


def test_candidate_content_snapshot_rejects_escaping_symlink(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    repo = root / "omnibase_core"
    hook_dir = repo / "scripts" / "git-hooks"
    hook_dir.mkdir(parents=True)
    (root / "outside.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    hook = hook_dir / "pre-commit"
    hook.symlink_to("../../../outside.sh")
    _git(repo, "add", "scripts/git-hooks/pre-commit")

    with pytest.raises(ValueError, match="escapes clone"):
        vm.write_candidate_content_snapshot(
            output=root / ".onex-candidate-content-snapshot.json",
            omni_home=root,
            repositories=names,
        )


@pytest.mark.parametrize("link_kind", ["dangling", "cyclic"])
def test_candidate_content_snapshot_rejects_dangling_or_cyclic_symlink(
    vm: ModuleType, tmp_path: Path, link_kind: str
) -> None:
    root, names = _candidate_root(tmp_path)
    hook_dir = root / "omnibase_core" / "scripts" / "git-hooks"
    hook_dir.mkdir(parents=True)
    first = hook_dir / "pre-commit"
    if link_kind == "dangling":
        first.symlink_to("missing-target.sh")
        _git(root / "omnibase_core", "add", "scripts/git-hooks/pre-commit")
    else:
        second = hook_dir / "pre-merge-commit"
        first.symlink_to("pre-merge-commit")
        second.symlink_to("pre-commit")
        _git(
            root / "omnibase_core",
            "add",
            "scripts/git-hooks/pre-commit",
            "scripts/git-hooks/pre-merge-commit",
        )

    with pytest.raises(ValueError, match="dangling or cyclic staged symlink"):
        vm.write_candidate_content_snapshot(
            output=root / ".onex-candidate-content-snapshot.json",
            omni_home=root,
            repositories=names,
        )


def test_candidate_content_snapshot_rejects_added_or_tampered_bytes(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    manifest = root / ".onex-candidate-content-snapshot.json"
    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "omnibase_core" / "new-source.py").write_text(
        "value = 1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )
    (root / "omnibase_core" / "new-source.py").unlink()
    tracked = root / "omnibase_core" / "README.md"
    tracked.unlink()
    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )
    tracked.write_text("x\n", encoding="utf-8")
    ignored = root / "omnibase_core" / "generated-build-input.txt"
    (root / "omnibase_core" / ".gitignore").write_text(
        "generated-build-input.txt\n", encoding="utf-8"
    )
    ignored.write_text("included by workspace staging\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes no longer match"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest, omni_home=root, repositories=names
        )
    ignored.unlink()
    (root / "omnibase_core" / ".gitignore").unlink()
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest differs"):
        vm.verify_candidate_content_snapshot(
            manifest=manifest,
            omni_home=root,
            repositories=names,
            expected_sha256=digest,
        )


def test_content_snapshot_floor_verifies_only_the_attested_dirty_bytes(
    vm: ModuleType, tmp_path: Path
) -> None:
    root, names = _candidate_root(tmp_path)
    core = root / "omnibase_core"
    (core / "README.md").write_text("attested dirty bytes\n", encoding="utf-8")
    source_package = core / "src" / "omnibase_core"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text(
        "VALUE = 'attested'\n", encoding="utf-8"
    )
    site_packages = tmp_path / "candidate-site"
    shutil.copytree(source_package, site_packages / "omnibase_core")
    runner = tmp_path / "candidate-python"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        f"export PYTHONPATH={site_packages}\n"
        f'exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)
    manifest = root / ".onex-candidate-content-snapshot.json"
    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    proof = vm.verify_candidate_content_snapshot(
        manifest=manifest, omni_home=root, repositories=names
    )
    installations = vm.capture_candidate_content_installation(
        manifest=manifest,
        omni_home=root,
        repositories=names,
        bindings=[f"omnibase_core:src/omnibase_core:omnibase_core:{runner}"],
    )
    vm.write_floor(
        output=root / ".onex-workspace-floor.json",
        omni_home=root,
        distributions={},
        omnimarket_commit=_git(core, "rev-parse", "HEAD"),
        candidate_content_snapshot={**proof, "installations": installations},
    )
    proc = subprocess.run(
        [
            "python3",
            str(_MODULE_PATH),
            "candidate-floor-verify",
            "--floor",
            str(root / ".onex-workspace-floor.json"),
            "--omni-home",
            str(root),
            "--required-mode",
            "content",
            *[item for name in names for item in ("--repo", name)],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    floor = root / ".onex-workspace-floor.json"
    document = json.loads(floor.read_text(encoding="utf-8"))
    document.pop("candidate_content_snapshot")
    floor.write_text(json.dumps(document), encoding="utf-8")
    stripped = subprocess.run(
        [
            "python3",
            str(_MODULE_PATH),
            "candidate-floor-verify",
            "--floor",
            str(floor),
            "--omni-home",
            str(root),
            "--required-mode",
            "content",
            *[item for name in names for item in ("--repo", name)],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert stripped.returncode != 0
    document["candidate_content_snapshot"] = proof
    floor.write_text(json.dumps(document), encoding="utf-8")
    (core / "README.md").write_text("not attested\n", encoding="utf-8")
    refused = subprocess.run(
        [
            "python3",
            str(_MODULE_PATH),
            "candidate-floor-verify",
            "--floor",
            str(root / ".onex-workspace-floor.json"),
            "--omni-home",
            str(root),
            "--required-mode",
            "content",
            *[item for name in names for item in ("--repo", name)],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0


def test_content_snapshot_floor_binds_candidate_interpreter_package_bytes(
    vm: ModuleType, tmp_path: Path
) -> None:
    """A matching base commit is insufficient when installed package bytes drift."""
    root, names = _candidate_root(tmp_path)
    source_package = root / "omnibase_core" / "src" / "omnibase_core"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text(
        "VALUE = 'attested'\n", encoding="utf-8"
    )
    site_packages = tmp_path / "candidate-site"
    installed_package = site_packages / "omnibase_core"
    shutil.copytree(source_package, installed_package)
    runner = tmp_path / "candidate-python"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        f"export PYTHONPATH={site_packages}\n"
        f'exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)
    manifest = root / ".onex-candidate-content-snapshot.json"
    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    proof = vm.verify_candidate_content_snapshot(
        manifest=manifest, omni_home=root, repositories=names
    )
    binding = f"omnibase_core:src/omnibase_core:omnibase_core:{runner}"
    installations = vm.capture_candidate_content_installation(
        manifest=manifest, omni_home=root, repositories=names, bindings=[binding]
    )
    vm.write_floor(
        output=root / ".onex-workspace-floor.json",
        omni_home=root,
        distributions={},
        omnimarket_commit=_git(root / "omnibase_core", "rev-parse", "HEAD"),
        candidate_content_snapshot={**proof, "installations": installations},
    )
    verified = subprocess.run(
        [
            "python3",
            str(_MODULE_PATH),
            "candidate-floor-verify",
            "--floor",
            str(root / ".onex-workspace-floor.json"),
            "--omni-home",
            str(root),
            "--required-mode",
            "content",
            *[item for name in names for item in ("--repo", name)],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr

    (installed_package / "__init__.py").write_text(
        "VALUE = 'stale'\n", encoding="utf-8"
    )
    refused = subprocess.run(
        [
            "python3",
            str(_MODULE_PATH),
            "candidate-floor-verify",
            "--floor",
            str(root / ".onex-workspace-floor.json"),
            "--omni-home",
            str(root),
            "--required-mode",
            "content",
            *[item for name in names for item in ("--repo", name)],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "bytes differ" in refused.stderr


def test_content_binding_preserves_candidate_venv_python_symlink(
    vm: ModuleType, tmp_path: Path
) -> None:
    """A venv interpreter symlink must not be normalized to its base Python."""
    root, names = _candidate_root(tmp_path)
    source_package = root / "omnibase_core" / "src" / "omnibase_core"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text(
        "VALUE = 'candidate-source'\n", encoding="utf-8"
    )
    site_packages = tmp_path / "candidate-site"
    shutil.copytree(source_package, site_packages / "omnibase_core")

    target_python = tmp_path / "base-python-shim"
    invocation_record = tmp_path / "invoked-python-path.txt"
    target_python.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s' \"$0\" > {invocation_record}\n"
        f"export PYTHONPATH={site_packages}\n"
        f'exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    target_python.chmod(0o755)
    candidate_python = tmp_path / "dispatch-venv" / "bin" / "python"
    candidate_python.parent.mkdir(parents=True)
    candidate_python.symlink_to(target_python)

    manifest = root / ".onex-candidate-content-snapshot.json"
    vm.write_candidate_content_snapshot(
        output=manifest, omni_home=root, repositories=names
    )
    vm.capture_candidate_content_installation(
        manifest=manifest,
        omni_home=root,
        repositories=names,
        bindings=[f"omnibase_core:src/omnibase_core:omnibase_core:{candidate_python}"],
    )

    assert invocation_record.read_text(encoding="utf-8") == str(candidate_python)


# --------------------------------------------------------------------------- #
# CLI surface used by the shell reconciler
# --------------------------------------------------------------------------- #
def test_cli_verdict_exits_nonzero_on_did_not_move(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            "python3",
            str(_MODULE_PATH),
            "verdict",
            "--surface",
            "venv:omnimarket",
            "--before",
            "aaa",
            "--after",
            "aaa",
            "--target",
            "bbb",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "DID_NOT_MOVE" in proc.stdout + proc.stderr
    assert "venv:omnimarket" in proc.stdout + proc.stderr


def test_cli_verdict_exits_zero_on_already_at_target() -> None:
    proc = subprocess.run(
        [
            "python3",
            str(_MODULE_PATH),
            "verdict",
            "--surface",
            "clone:omnimarket",
            "--before",
            "bbb",
            "--after",
            "bbb",
            "--target",
            "bbb",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ALREADY_AT_TARGET" in proc.stdout


def test_cli_runs_on_a_bare_python3_with_no_project_venv() -> None:
    """It must run on `.201` outside the project venv.

    Stated as a real assertion rather than a comment: the module imports nothing
    beyond the standard library, so a fresh interpreter with an empty
    ``sys.path[1:]`` can still execute it.
    """
    proc = subprocess.run(
        [
            "python3",
            "-I",
            str(_MODULE_PATH),
            "verdict",
            "--surface",
            "s",
            "--before",
            "a",
            "--after",
            "b",
            "--target",
            "b",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

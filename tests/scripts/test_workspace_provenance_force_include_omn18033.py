# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18033: the content-parity gate must understand wheel force-include.

The OMN-14631 content-parity gate compares the STAGED source package
directory (`sibling-repos/<repo>/src/<repo>`) against the INSTALLED
site-packages directory file-by-file, and treats any file present on one
side and not the other as drift.

A hatch `force-include` maps a file from OUTSIDE `src/` into the wheel's
package directory. Such a file exists in the installed tree by
construction and can never exist in the staged `src/` tree, so the gate
read it as drift and hard-failed the build.

Measured live on the .201 dev lane, 2026-09-19: `omnibase_core#1710`
(OMN-18033, squash `5d562ab4`) added

    [tool.hatch.build.targets.wheel.force-include]
    "architecture-handshakes/gitignore-baseline.yaml" =
        "omnibase_core/data/gitignore-baseline.yaml"

and every BUILD_SOURCE=workspace build from 15:37Z onward died with

    INSTALLED CONTENT DRIFT for 'omnibase-core' (OMN-14631): ...
    differing files (1 total): ['data/gitignore-baseline.yaml']

leaving the dev lane unable to rebuild at all. Deploy-agent jobs
`7afb5fad-9242-4f87-8a86-234fecb20377`, `dfba74d9-6277-40c8-b07c-53c37f9fc09e`
and `a62619cf-f745-4745-a94c-bd29ed63e4ad` all ended there.

The gate is NOT weakened to tolerate extra installed files. It resolves
the staged repo's own declared force-include mappings and compares those
files too, from their real source location in the staged tree -- so a
force-included file whose installed copy has drifted still fails closed,
and an undeclared extra installed file still fails closed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_SCRIPT = (
    REPO_ROOT / "scripts" / "runtime_build" / "compute_workspace_provenance.py"
)

# Synthetic, ticket-scoped names so these tests can never collide with a real
# already-imported package in this repo's own suite.
IMPORT_NAME = "_omn18033_fake_core"
DIST_NAME = "_omn18033-fake-core"

FORCE_INCLUDE_SOURCE = "architecture-handshakes/gitignore-baseline.yaml"
FORCE_INCLUDE_DEST_REL = "data/gitignore-baseline.yaml"
SPEC_TEXT = "version: 1\nblocks:\n  - id: python\n"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cwp_force_include_under_test", PROVENANCE_SCRIPT
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_staged_repo(
    sib_dir: Path,
    *,
    declare_force_include: bool,
    write_force_include_source: bool = True,
) -> Path:
    """Materialize a staged sibling repo, optionally declaring force-include."""
    repo_root = sib_dir / IMPORT_NAME
    staged_pkg = repo_root / "src" / IMPORT_NAME
    staged_pkg.mkdir(parents=True, exist_ok=True)
    (staged_pkg / "__init__.py").write_text("", encoding="utf-8")
    (staged_pkg / "configs").mkdir(exist_ok=True)
    (staged_pkg / "configs" / "routing_tiers.yaml").write_text(
        "tiers: [fresh]\n", encoding="utf-8"
    )

    if write_force_include_source:
        source = repo_root / FORCE_INCLUDE_SOURCE
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(SPEC_TEXT, encoding="utf-8")

    pyproject = f'[project]\nname = "{DIST_NAME}"\nversion = "1.0.0"\n'
    if declare_force_include:
        pyproject += (
            "\n[tool.hatch.build.targets.wheel.force-include]\n"
            f'"{FORCE_INCLUDE_SOURCE}" = "{IMPORT_NAME}/{FORCE_INCLUDE_DEST_REL}"\n'
        )
    (repo_root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (repo_root / "uv.lock").write_text("", encoding="utf-8")
    return repo_root


def _write_installed_package(
    site_packages: Path, *, extra_files: dict[str, str], source_url: str
) -> None:
    """Materialize a pip-style local-path install of the staged repo's wheel."""
    pkg_dir = site_packages / IMPORT_NAME
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "configs").mkdir(exist_ok=True)
    (pkg_dir / "configs" / "routing_tiers.yaml").write_text(
        "tiers: [fresh]\n", encoding="utf-8"
    )
    for rel, content in extra_files.items():
        target = pkg_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    dist_info = site_packages / f"{DIST_NAME.replace('-', '_')}-1.0.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DIST_NAME}\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    (dist_info / "direct_url.json").write_text(
        json.dumps({"url": source_url, "dir_info": {}}), encoding="utf-8"
    )


def _write_vcs_provenance(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "siblings": {
                    IMPORT_NAME: {
                        "vcs_ref": "a" * 40,
                        "vcs_dirty": False,
                        "vcs_branch": "dev",
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _run_main(
    mod: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sib_dir: Path,
    site_packages: Path,
    app_dir: Path,
) -> tuple[int, dict[str, object]]:
    vcs_prov = tmp_path / "sibling-vcs-provenance.json"
    _write_vcs_provenance(vcs_prov)

    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()
    monkeypatch.setenv("VCS_REF", "deadbeef")
    try:
        mod.SIBLING_REPOS_DIR = sib_dir
        mod.VENV_DIR = app_dir / ".venv"
        mod.OUTPUT_MANIFEST = app_dir / "build-provenance.json"
        mod.PIN_COMPARISON_PATH = tmp_path / "no-such-pin.json"
        mod.VCS_PROVENANCE_PATH = vcs_prov
        mod.CONSUMING_REPO = IMPORT_NAME
        mod.WORKSPACE_PACKAGES = {IMPORT_NAME: DIST_NAME}
        # The lock-pin comparison subsystem hardcodes the real sibling names
        # and is not under test here; only the content-parity gate is.
        monkeypatch.setattr(mod, "build_comparisons", lambda **_kwargs: [])
        monkeypatch.setattr(mod, "_host_infra_comparison", lambda _lock_path: None)
        rc = mod.main()
    finally:
        sys.modules.pop(IMPORT_NAME, None)

    manifest: dict[str, object] = json.loads(
        Path(mod.OUTPUT_MANIFEST).read_text(encoding="utf-8")
    )
    return rc, manifest


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    sib_dir = tmp_path / "sibling-repos"
    app_dir = tmp_path / "app"
    site_packages = app_dir / ".venv" / "lib" / "python3.12" / "site-packages"
    app_dir.mkdir(parents=True, exist_ok=True)
    return sib_dir, site_packages, app_dir


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_force_included_file_is_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED before the fix, GREEN after.

    Reproduces the live .201 shape exactly: the installed package carries
    `data/gitignore-baseline.yaml`, force-included from a path outside
    `src/`, and the staged `src/` tree cannot carry it. Before the fix this
    is reported as `content_mismatch` and main() returns 1, which is what
    stopped every dev-lane rebuild on 2026-09-19.
    """
    mod = _load_module()
    sib_dir, site_packages, app_dir = _layout(tmp_path)
    repo_root = _write_staged_repo(sib_dir, declare_force_include=True)
    _write_installed_package(
        site_packages,
        extra_files={FORCE_INCLUDE_DEST_REL: SPEC_TEXT},
        source_url=f"file://{repo_root}",
    )

    rc, manifest = _run_main(
        mod,
        tmp_path,
        monkeypatch,
        sib_dir=sib_dir,
        site_packages=site_packages,
        app_dir=app_dir,
    )

    proofs = list(manifest["proofs"])  # type: ignore[call-overload]
    assert rc == 0, proofs
    assert proofs[0]["status"] == "verified"
    # The manifest states which files were resolved through force-include, so
    # a verifier can tell a declared mapping from an unexplained extra file.
    assert proofs[0]["force_included_files"] == [FORCE_INCLUDE_DEST_REL]


@pytest.mark.unit
def test_force_included_file_with_drifted_content_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix resolves the mapping; it does not stop comparing content.

    A force-included file whose installed copy differs from the staged
    source is exactly the OMN-14631 stale-data-file shape and must still
    hard-fail.
    """
    mod = _load_module()
    sib_dir, site_packages, app_dir = _layout(tmp_path)
    repo_root = _write_staged_repo(sib_dir, declare_force_include=True)
    _write_installed_package(
        site_packages,
        extra_files={FORCE_INCLUDE_DEST_REL: "version: 1\nblocks: [STALE]\n"},
        source_url=f"file://{repo_root}",
    )

    rc, manifest = _run_main(
        mod,
        tmp_path,
        monkeypatch,
        sib_dir=sib_dir,
        site_packages=site_packages,
        app_dir=app_dir,
    )

    proofs = list(manifest["proofs"])  # type: ignore[call-overload]
    assert rc == 1
    assert proofs[0]["status"] == "content_mismatch"
    assert proofs[0]["content_diff_files"] == [FORCE_INCLUDE_DEST_REL]


@pytest.mark.unit
def test_undeclared_extra_installed_file_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: the gate is narrowed, not disarmed.

    An installed file with no force-include declaration behind it is still
    drift -- otherwise the fix would be indistinguishable from deleting the
    `installed - staged` half of the comparison.
    """
    mod = _load_module()
    sib_dir, site_packages, app_dir = _layout(tmp_path)
    repo_root = _write_staged_repo(sib_dir, declare_force_include=True)
    _write_installed_package(
        site_packages,
        extra_files={
            FORCE_INCLUDE_DEST_REL: SPEC_TEXT,
            "configs/leftover_from_a_prior_build.yaml": "b: 2\n",
        },
        source_url=f"file://{repo_root}",
    )

    rc, manifest = _run_main(
        mod,
        tmp_path,
        monkeypatch,
        sib_dir=sib_dir,
        site_packages=site_packages,
        app_dir=app_dir,
    )

    proofs = list(manifest["proofs"])  # type: ignore[call-overload]
    assert rc == 1
    assert proofs[0]["status"] == "content_mismatch"
    assert proofs[0]["content_diff_files"] == [
        "configs/leftover_from_a_prior_build.yaml"
    ]


@pytest.mark.unit
def test_undeclared_force_include_is_still_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declaration is what explains the file, not its path shape.

    Same installed tree as the passing case, but the staged pyproject
    declares no mapping -- the file is unexplained and must fail.
    """
    mod = _load_module()
    sib_dir, site_packages, app_dir = _layout(tmp_path)
    repo_root = _write_staged_repo(sib_dir, declare_force_include=False)
    _write_installed_package(
        site_packages,
        extra_files={FORCE_INCLUDE_DEST_REL: SPEC_TEXT},
        source_url=f"file://{repo_root}",
    )

    rc, manifest = _run_main(
        mod,
        tmp_path,
        monkeypatch,
        sib_dir=sib_dir,
        site_packages=site_packages,
        app_dir=app_dir,
    )

    proofs = list(manifest["proofs"])  # type: ignore[call-overload]
    assert rc == 1
    assert proofs[0]["status"] == "content_mismatch"
    assert proofs[0]["content_diff_files"] == [FORCE_INCLUDE_DEST_REL]


@pytest.mark.unit
def test_declared_force_include_with_missing_source_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mapping whose source is absent from the staged tree is an error.

    The staged tree is then not the tree the wheel was built from, which is
    precisely the condition this proof exists to refuse. It must not be read
    as "no force-include declared".
    """
    mod = _load_module()
    sib_dir, site_packages, app_dir = _layout(tmp_path)
    repo_root = _write_staged_repo(
        sib_dir, declare_force_include=True, write_force_include_source=False
    )
    _write_installed_package(
        site_packages,
        extra_files={FORCE_INCLUDE_DEST_REL: SPEC_TEXT},
        source_url=f"file://{repo_root}",
    )

    rc, _manifest = _run_main(
        mod,
        tmp_path,
        monkeypatch,
        sib_dir=sib_dir,
        site_packages=site_packages,
        app_dir=app_dir,
    )

    assert rc == 1


# ---------------------------------------------------------------------------
# _force_included_files unit coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_force_included_files_resolves_a_directory_source(tmp_path: Path) -> None:
    """A force-include source may be a directory; every file under it maps."""
    mod = _load_module()
    repo_root = tmp_path / "repo"
    (repo_root / "governance" / "nested").mkdir(parents=True)
    (repo_root / "governance" / "a.yaml").write_text("a: 1\n", encoding="utf-8")
    (repo_root / "governance" / "nested" / "b.yaml").write_text(
        "b: 2\n", encoding="utf-8"
    )
    (repo_root / "pyproject.toml").write_text(
        "[tool.hatch.build.targets.wheel.force-include]\n"
        f'"governance" = "{IMPORT_NAME}/data"\n',
        encoding="utf-8",
    )

    errors: list[str] = []
    resolved = mod._force_included_files(repo_root, IMPORT_NAME, errors)

    assert errors == []
    assert resolved == {
        "data/a.yaml": b"a: 1\n",
        "data/nested/b.yaml": b"b: 2\n",
    }


@pytest.mark.unit
def test_force_included_files_ignores_destinations_outside_the_package(
    tmp_path: Path,
) -> None:
    """A mapping into another top-level site-packages entry is not our concern.

    The parity comparison only covers this package's own directory, so a
    mapping elsewhere must neither appear in the expected set nor raise.
    """
    mod = _load_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[tool.hatch.build.targets.wheel.force-include]\n"
        '"LICENSE" = "some_other_package/LICENSE"\n',
        encoding="utf-8",
    )

    errors: list[str] = []
    assert mod._force_included_files(repo_root, IMPORT_NAME, errors) == {}
    assert errors == []


@pytest.mark.unit
def test_force_included_files_reads_the_build_wide_table_too(tmp_path: Path) -> None:
    """`[tool.hatch.build.force-include]` applies to every target, wheel included."""
    mod = _load_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "spec.yaml").write_text("s: 1\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[tool.hatch.build.force-include]\n"
        f'"spec.yaml" = "{IMPORT_NAME}/data/spec.yaml"\n',
        encoding="utf-8",
    )

    errors: list[str] = []
    resolved = mod._force_included_files(repo_root, IMPORT_NAME, errors)

    assert errors == []
    assert resolved == {"data/spec.yaml": b"s: 1\n"}


@pytest.mark.unit
def test_force_included_files_is_empty_without_a_pyproject(tmp_path: Path) -> None:
    mod = _load_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    errors: list[str] = []
    assert mod._force_included_files(repo_root, IMPORT_NAME, errors) == {}
    assert errors == []

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14631: installed-vs-staged content-parity gate in compute_workspace_provenance.

Before this gate existed, a workspace build's "verified" proof only checked
that a sibling package's `direct_url.json` recorded a `file://.../workspace/
sibling-repos/<repo>` source -- i.e. that *some* local-path install happened.
It never proved the resulting installed file CONTENT under site-packages
actually matched the staged source tree. A 2026-07-14 live incident
(OMN-14626/OMN-14625 .201 stability-test readback) found exactly that gap: a
workspace rebuild correctly vendored a branch's Python source while the
package's config YAML data files were stale pre-fix content in the installed
venv, despite the staged source on disk being byte-identical to the branch.

These tests cover the new helpers (`_tracked_files`, `_hash_tree`,
`_installed_package_dir`, `_content_parity_diff`) directly, plus an
end-to-end `main()` proof that a real installed/staged content divergence
hard-fails the build with a `content_mismatch` proof entry, and that a clean
build (installed content == staged content) still passes.
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


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cwp_content_parity_under_test", PROVENANCE_SCRIPT
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Pure-function coverage: _tracked_files / _hash_tree / _content_parity_diff
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hash_tree_matches_for_identical_trees(tmp_path: Path) -> None:
    mod = _load_module()
    tree_a = tmp_path / "a"
    tree_b = tmp_path / "b"
    for tree in (tree_a, tree_b):
        (tree / "configs").mkdir(parents=True)
        (tree / "configs" / "x.yaml").write_text("key: value\n", encoding="utf-8")
        (tree / "__init__.py").write_text("", encoding="utf-8")

    assert mod._hash_tree(tree_a) == mod._hash_tree(tree_b)


@pytest.mark.unit
def test_hash_tree_diverges_on_content_change(tmp_path: Path) -> None:
    mod = _load_module()
    tree_a = tmp_path / "a"
    tree_b = tmp_path / "b"
    (tree_a / "configs").mkdir(parents=True)
    (tree_b / "configs").mkdir(parents=True)
    (tree_a / "configs" / "x.yaml").write_text("key: OLD\n", encoding="utf-8")
    (tree_b / "configs" / "x.yaml").write_text("key: NEW\n", encoding="utf-8")

    assert mod._hash_tree(tree_a) != mod._hash_tree(tree_b)


@pytest.mark.unit
def test_hash_tree_excludes_transient_artifacts(tmp_path: Path) -> None:
    mod = _load_module()
    tree = tmp_path / "pkg"
    tree.mkdir()
    (tree / "real.py").write_text("x = 1\n", encoding="utf-8")
    pycache = tree / "__pycache__"
    pycache.mkdir()
    (pycache / "real.cpython-312.pyc").write_bytes(b"\x00\x01")

    digest_with_cache = mod._hash_tree(tree)
    (pycache / "real.cpython-312.pyc").write_bytes(b"\xff\xff\xff")
    digest_after_mutating_cache = mod._hash_tree(tree)

    # __pycache__ content changes must NOT move the digest.
    assert digest_with_cache == digest_after_mutating_cache


@pytest.mark.unit
def test_content_parity_diff_empty_for_identical_trees(tmp_path: Path) -> None:
    mod = _load_module()
    staged = tmp_path / "staged"
    installed = tmp_path / "installed"
    for tree in (staged, installed):
        (tree / "configs").mkdir(parents=True)
        (tree / "configs" / "routing_tiers.yaml").write_text(
            "tiers: []\n", encoding="utf-8"
        )

    assert mod._content_parity_diff(staged, installed) == []


@pytest.mark.unit
def test_content_parity_diff_reports_changed_missing_and_extra_files(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    staged = tmp_path / "staged"
    installed = tmp_path / "installed"
    (staged / "configs").mkdir(parents=True)
    (installed / "configs").mkdir(parents=True)

    # Changed content -- the exact class of drift from the live incident.
    (staged / "configs" / "routing_tiers.yaml").write_text(
        "tiers: [fresh]\n", encoding="utf-8"
    )
    (installed / "configs" / "routing_tiers.yaml").write_text(
        "tiers: [stale]\n", encoding="utf-8"
    )
    # Present in staged source, missing from the installed tree.
    (staged / "configs" / "only_staged.yaml").write_text("a: 1\n", encoding="utf-8")
    # Present in the installed tree only (e.g. a leftover from a prior build).
    (installed / "configs" / "only_installed.yaml").write_text(
        "b: 2\n", encoding="utf-8"
    )

    diff = mod._content_parity_diff(staged, installed)

    assert diff == [
        "configs/only_installed.yaml",
        "configs/only_staged.yaml",
        "configs/routing_tiers.yaml",
    ]


@pytest.mark.unit
def test_installed_package_dir_resolves_via_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_module()
    site_packages = tmp_path / "site-packages"
    pkg_dir = site_packages / "_omn14631_fake_pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()
    try:
        resolved = mod._installed_package_dir("_omn14631_fake_pkg")
    finally:
        sys.modules.pop("_omn14631_fake_pkg", None)

    assert resolved == pkg_dir


@pytest.mark.unit
def test_installed_package_dir_returns_none_when_unresolvable() -> None:
    mod = _load_module()
    assert mod._installed_package_dir("_omn14631_definitely_not_installed") is None


# ---------------------------------------------------------------------------
# End-to-end main() coverage: the gate must hard-fail on real drift and pass
# clean when installed content matches the staged source.
# ---------------------------------------------------------------------------


def _write_fake_installed_package(
    site_packages: Path,
    *,
    import_name: str,
    dist_name: str,
    files: dict[str, str],
    source_url: str,
) -> None:
    """Materialize a minimal but real pip-style local-path install.

    Writes both the importable package directory (so importlib.util.find_spec
    resolves it) and a `<dist>-<version>.dist-info/` with `direct_url.json`
    (so importlib.metadata.distribution(...) resolves the local-path proof
    `_installed_direct_url` reads).
    """
    pkg_dir = site_packages / import_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    for rel, content in files.items():
        target = pkg_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    dist_info = site_packages / f"{dist_name.replace('-', '_')}-1.0.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    (dist_info / "direct_url.json").write_text(
        json.dumps({"url": source_url, "dir_info": {}}), encoding="utf-8"
    )


def _stage_and_install(
    sib_dir: Path,
    site_packages: Path,
    *,
    import_name: str,
    dist_name: str,
    installed_content: str,
) -> None:
    staged_pkg = sib_dir / import_name / "src" / import_name
    staged_pkg.mkdir(parents=True, exist_ok=True)
    (staged_pkg / "__init__.py").write_text("", encoding="utf-8")
    (staged_pkg / "configs").mkdir(exist_ok=True)
    (staged_pkg / "configs" / "routing_tiers.yaml").write_text(
        "tiers: [fresh]\n", encoding="utf-8"
    )

    _write_fake_installed_package(
        site_packages,
        import_name=import_name,
        dist_name=dist_name,
        files={"configs/routing_tiers.yaml": installed_content},
        source_url=f"file://{sib_dir / import_name}",
    )


def _write_vcs_provenance(path: Path, repos: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "siblings": {
                    repo: {
                        "vcs_ref": "a" * 40,
                        "vcs_dirty": False,
                        "vcs_branch": "dev",
                    }
                    for repo in repos
                }
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def synthetic_workspace_packages() -> dict[str, str]:
    # Synthetic, ticket-scoped names so this test can never collide with a
    # REAL already-imported package (e.g. omnibase_core is virtually
    # guaranteed to already be in sys.modules inside this repo's own test
    # suite, which would make importlib.util.find_spec return the real
    # module's spec instead of discovering our fake site-packages entry).
    return {
        "_omn14631_fake_core": "_omn14631-fake-core",
        "_omn14631_fake_market": "_omn14631-fake-market",
    }


@pytest.mark.unit
def test_main_fails_closed_on_installed_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_workspace_packages: dict[str, str],
) -> None:
    """RED before the fix: reproduces the exact 2026-07-14 incident shape --
    staged source is correct, installed site-packages content is stale."""
    mod = _load_module()
    sib_dir = tmp_path / "sibling-repos"
    site_packages = tmp_path / "site-packages"
    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True)

    packages = synthetic_workspace_packages
    for import_name, dist_name in packages.items():
        # Every package installs content matching its staged source EXCEPT
        # the "market" one, which gets stale (pre-fix) installed content --
        # this is the live incident shape.
        installed_content = (
            "tiers: [STALE-PRE-FIX]\n"
            if import_name == "_omn14631_fake_market"
            else "tiers: [fresh]\n"
        )
        _stage_and_install(
            sib_dir,
            site_packages,
            import_name=import_name,
            dist_name=dist_name,
            installed_content=installed_content,
        )

    (sib_dir / "_omn14631_fake_market" / "uv.lock").write_text("", encoding="utf-8")
    vcs_prov = tmp_path / "sibling-vcs-provenance.json"
    _write_vcs_provenance(vcs_prov, list(packages))

    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()
    monkeypatch.setenv("VCS_REF", "deadbeef")
    try:
        mod.SIBLING_REPOS_DIR = sib_dir
        mod.VENV_DIR = tmp_path / "app" / ".venv"
        mod.OUTPUT_MANIFEST = app_dir / "build-provenance.json"
        mod.PIN_COMPARISON_PATH = tmp_path / "no-such-pin.json"
        mod.VCS_PROVENANCE_PATH = vcs_prov
        mod.CONSUMING_REPO = "_omn14631_fake_market"
        mod.WORKSPACE_PACKAGES = packages
        # Isolate this test from the pre-existing (and unrelated, not
        # parametrized by WORKSPACE_PACKAGES) lock-pin comparison subsystem
        # -- it hardcodes real sibling repo names (omnibase_compat etc.)
        # that this synthetic fixture never stages. Only the OMN-14631
        # content-parity gate is under test here.
        monkeypatch.setattr(mod, "build_comparisons", lambda **_kwargs: [])
        monkeypatch.setattr(mod, "_host_infra_comparison", lambda _lock_path: None)

        rc = mod.main()
    finally:
        for import_name in packages:
            sys.modules.pop(import_name, None)

    assert rc == 1

    manifest = json.loads(mod.OUTPUT_MANIFEST.read_text(encoding="utf-8"))
    proofs_by_repo = {p["repo"]: p for p in manifest["proofs"]}

    stale_proof = proofs_by_repo["_omn14631_fake_market"]
    assert stale_proof["status"] == "content_mismatch"
    assert "configs/routing_tiers.yaml" in stale_proof["content_diff_files"]
    assert (
        stale_proof["staged_package_digest"] != stale_proof["installed_package_digest"]
    )

    clean_proof = proofs_by_repo["_omn14631_fake_core"]
    assert clean_proof["status"] == "verified"
    assert (
        clean_proof["staged_package_digest"] == clean_proof["installed_package_digest"]
    )


@pytest.mark.unit
def test_main_passes_when_installed_content_matches_staged_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_workspace_packages: dict[str, str],
) -> None:
    """GREEN after the fix: a genuinely clean workspace build (installed
    content == vendored source for every sibling) is not penalized by the
    new gate."""
    mod = _load_module()
    sib_dir = tmp_path / "sibling-repos"
    site_packages = tmp_path / "site-packages"
    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True)

    packages = synthetic_workspace_packages
    for import_name, dist_name in packages.items():
        _stage_and_install(
            sib_dir,
            site_packages,
            import_name=import_name,
            dist_name=dist_name,
            installed_content="tiers: [fresh]\n",
        )

    (sib_dir / "_omn14631_fake_market" / "uv.lock").write_text("", encoding="utf-8")
    vcs_prov = tmp_path / "sibling-vcs-provenance.json"
    _write_vcs_provenance(vcs_prov, list(packages))

    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()
    monkeypatch.setenv("VCS_REF", "deadbeef")
    try:
        mod.SIBLING_REPOS_DIR = sib_dir
        mod.VENV_DIR = tmp_path / "app" / ".venv"
        mod.OUTPUT_MANIFEST = app_dir / "build-provenance.json"
        mod.PIN_COMPARISON_PATH = tmp_path / "no-such-pin.json"
        mod.VCS_PROVENANCE_PATH = vcs_prov
        mod.CONSUMING_REPO = "_omn14631_fake_market"
        mod.WORKSPACE_PACKAGES = packages
        # Isolate this test from the pre-existing (and unrelated, not
        # parametrized by WORKSPACE_PACKAGES) lock-pin comparison subsystem
        # -- it hardcodes real sibling repo names (omnibase_compat etc.)
        # that this synthetic fixture never stages. Only the OMN-14631
        # content-parity gate is under test here.
        monkeypatch.setattr(mod, "build_comparisons", lambda **_kwargs: [])
        monkeypatch.setattr(mod, "_host_infra_comparison", lambda _lock_path: None)

        rc = mod.main()
    finally:
        for import_name in packages:
            sys.modules.pop(import_name, None)

    assert rc == 0, mod.OUTPUT_MANIFEST.read_text(encoding="utf-8")

    manifest = json.loads(mod.OUTPUT_MANIFEST.read_text(encoding="utf-8"))
    assert all(p["status"] == "verified" for p in manifest["proofs"])

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the canonical-venv purity fitness assertion (OMN-15620).

Includes the AC5 falsification test: a synthetic undeclared distribution is
installed into a scratch directory (a real ``*.dist-info`` with a real
``entry_points.txt``, discovered by the real ``importlib.metadata`` machinery
-- not a monkeypatched return value) and the check is proven to fire and name
the offending distribution. A check only ever observed passing is not proven
(OMN-15620 AC5).

Related:
    - OMN-15620: canonical .200 venv cross-repo pollution manufactures false REDs
    - omnibase_infra.runtime.venv_purity: implementation under test
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.runtime.venv_purity import (
    ENTRY_POINT_GROUP,
    VenvPurityError,
    assert_venv_purity,
    find_undeclared_onex_providers,
)


def _write_uv_lock(tmp_path: Path, declared_names: list[str]) -> Path:
    """Write a minimal-but-real uv.lock declaring exactly `declared_names`."""
    packages = "\n\n".join(
        f'[[package]]\nname = "{name}"\nversion = "0.0.0"\nsource = {{ registry = "https://pypi.org/simple" }}'
        for name in declared_names
    )
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(f"version = 1\n\n{packages}\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "scratch"\n')
    return lock_path


def _install_fake_distribution(
    site_packages: Path,
    *,
    name: str,
    version: str,
    node_entry_point_names: list[str],
) -> None:
    """Create a real, on-disk ``*.dist-info`` that importlib.metadata will
    discover as an installed distribution -- the closest unit-test-speed
    equivalent to "install one undeclared sibling distribution into a
    scratch copy of the venv" (OMN-15620 AC5's literal falsification
    language): this is a genuine dist-info package structure, not a
    monkeypatched return value.
    """
    dist_info = site_packages / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    entry_points_body = "\n".join(
        f"{ep_name} = {name}.nodes.{ep_name}:Node" for ep_name in node_entry_point_names
    )
    (dist_info / "entry_points.txt").write_text(
        textwrap.dedent(f"""
            [{ENTRY_POINT_GROUP}]
            {entry_points_body}
            """).strip()
        + "\n"
    )


@pytest.mark.unit
class TestFindUndeclaredOnexProviders:
    def test_fails_open_when_lock_path_missing(self, tmp_path: Path) -> None:
        """A nonexistent lockfile path returns empty, never raises."""
        missing = tmp_path / "does-not-exist" / "uv.lock"
        result = find_undeclared_onex_providers(lock_path=missing)
        assert result == ()

    def test_fails_open_when_lock_unparseable(self, tmp_path: Path) -> None:
        bad_lock = tmp_path / "uv.lock"
        bad_lock.write_text("this is not { valid toml ]]]")
        result = find_undeclared_onex_providers(lock_path=bad_lock)
        assert result == ()

    def test_clean_venv_reports_nothing(self, tmp_path: Path) -> None:
        """Every onex.nodes provider present in the scratch venv is declared
        in the scratch lockfile -> no undeclared providers."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        _install_fake_distribution(
            site_packages,
            name="declared-provider",
            version="1.0.0",
            node_entry_point_names=["node_declared_thing"],
        )
        lock_path = _write_uv_lock(tmp_path, ["declared-provider"])

        result = find_undeclared_onex_providers(
            lock_path=lock_path, search_paths=[str(site_packages)]
        )
        assert result == ()

    def test_falsification_undeclared_provider_is_detected_and_named(
        self, tmp_path: Path
    ) -> None:
        """AC5: deliberately install one undeclared sibling distribution and
        confirm the check fires and names the offending distribution."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        _install_fake_distribution(
            site_packages,
            name="declared-provider",
            version="1.0.0",
            node_entry_point_names=["node_declared_thing"],
        )
        # The undeclared sibling -- not in the lockfile below.
        _install_fake_distribution(
            site_packages,
            name="rogue-sibling-pkg",
            version="2.3.4",
            node_entry_point_names=["node_rogue_thing", "node_rogue_other"],
        )
        lock_path = _write_uv_lock(tmp_path, ["declared-provider"])

        result = find_undeclared_onex_providers(
            lock_path=lock_path, search_paths=[str(site_packages)]
        )

        assert len(result) == 1
        offender = result[0]
        assert offender.name == "rogue-sibling-pkg"
        assert offender.version == "2.3.4"
        assert set(offender.entry_point_names) == {
            "node_rogue_thing",
            "node_rogue_other",
        }

    def test_declaration_check_is_normalized(self, tmp_path: Path) -> None:
        """Lockfile declares the hyphenated PyPI name; installed dist-info
        uses the underscored form -- must still be recognized as declared."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        _install_fake_distribution(
            site_packages,
            name="my_underscored_pkg",
            version="1.0.0",
            node_entry_point_names=["node_thing"],
        )
        lock_path = _write_uv_lock(tmp_path, ["my-underscored-pkg"])

        result = find_undeclared_onex_providers(
            lock_path=lock_path, search_paths=[str(site_packages)]
        )
        assert result == ()

    def test_distributions_without_node_entry_points_are_ignored(
        self, tmp_path: Path
    ) -> None:
        """An undeclared distribution that provides NO onex.nodes entry
        points cannot cause DUPLICATE_REGISTRATION and must not be flagged."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        dist_info = site_packages / "harmless-1.0.dist-info"
        dist_info.mkdir(parents=True)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: harmless\nVersion: 1.0\n"
        )
        lock_path = _write_uv_lock(tmp_path, [])

        result = find_undeclared_onex_providers(
            lock_path=lock_path, search_paths=[str(site_packages)]
        )
        assert result == ()


@pytest.mark.unit
class TestAssertVenvPurity:
    def test_passes_silently_on_clean_venv(self, tmp_path: Path) -> None:
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        _install_fake_distribution(
            site_packages,
            name="declared-provider",
            version="1.0.0",
            node_entry_point_names=["node_declared_thing"],
        )
        lock_path = _write_uv_lock(tmp_path, ["declared-provider"])

        assert_venv_purity(lock_path=lock_path, search_paths=[str(site_packages)])

    def test_raises_and_names_offender_on_impure_venv(self, tmp_path: Path) -> None:
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        _install_fake_distribution(
            site_packages,
            name="rogue-sibling-pkg",
            version="9.9.9",
            node_entry_point_names=["node_rogue_thing"],
        )
        lock_path = _write_uv_lock(tmp_path, [])

        with pytest.raises(VenvPurityError) as exc_info:
            assert_venv_purity(lock_path=lock_path, search_paths=[str(site_packages)])

        message = str(exc_info.value)
        assert "rogue-sibling-pkg" in message
        assert "9.9.9" in message
        assert "node_rogue_thing" in message
        assert "uv sync" in message

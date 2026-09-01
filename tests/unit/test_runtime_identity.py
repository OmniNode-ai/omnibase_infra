# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Runtime-identity collection (OMN-17310, epic OMN-17306).

The property under test is honesty, not completeness: an install with no
recoverable commit must SAY it has none, and a distribution that is not
installed must be recorded as ABSENT rather than omitted. Both were rendered as
silence before this module existed, and silence is what let a stale local venv
produce a receipt indistinguishable from a deployed lane's (OMN-17295).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.enums.enum_execution_locus_kind import EnumExecutionLocusKind
from omnibase_core.enums.enum_package_source_kind import EnumPackageSourceKind
from omnibase_core.models.runtime.model_runtime_identity import ModelRuntimeIdentity
from omnibase_infra import runtime_identity as ri

_LANE_MARKET_SHA = "2f123b4c01eabd2c51f7d703491e9cdf36f89bcd"
_VCS_SHA = "66b7131a3508bd2c51f7d703491e9cdf36f89bcd"


class _FakeDist:
    """Minimal importlib.metadata.Distribution stand-in."""

    def __init__(
        self,
        version: str,
        direct_url: str | None,
        location: Path | None = None,
    ) -> None:
        self.version = version
        self._direct_url = direct_url
        self._location = location

    def read_text(self, name: str) -> str | None:
        return self._direct_url if name == "direct_url.json" else None

    def locate_file(self, name: str) -> Path:
        """Where the metadata SAYS this package lives.

        Defaults to wherever the interpreter really imports it from, so a test
        that is not about shadowing reports none. A test that IS about
        shadowing passes an explicit ``location`` that disagrees.
        """
        if self._location is not None:
            return self._location
        real = ri._import_root(str(name))
        return real if real is not None else Path("/nonexistent") / str(name)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """The collector memoises for the process lifetime; tests must not share."""
    ri._build_provenance_siblings.cache_clear()
    ri._collect_packages.cache_clear()
    ri._execution_locus.cache_clear()


def _install(monkeypatch: pytest.MonkeyPatch, dists: dict[str, _FakeDist]) -> None:
    def _distribution(name: str) -> _FakeDist:
        from importlib.metadata import PackageNotFoundError

        if name not in dists:
            raise PackageNotFoundError(name)
        return dists[name]

    monkeypatch.setattr(ri, "distribution", _distribution)


@pytest.mark.unit
class TestPackageResolution:
    def test_git_install_reports_vcs_and_the_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            {
                "omnimarket": _FakeDist(
                    "0.4.11",
                    json.dumps(
                        {
                            "url": "https://github.com/OmniNode-ai/omnimarket.git",
                            "vcs_info": {"vcs": "git", "commit_id": _VCS_SHA},
                        }
                    ),
                )
            },
        )
        entry = ri._package_identity("omnimarket")
        assert entry.source is EnumPackageSourceKind.VCS
        assert entry.commit == _VCS_SHA

    def test_registry_wheel_reports_registry_and_no_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PyPI wheel genuinely has no commit. It must not be labelled vcs."""
        _install(monkeypatch, {"omnibase_core": _FakeDist("0.47.2", None)})
        entry = ri._package_identity("omnibase_core")
        assert entry.source is EnumPackageSourceKind.REGISTRY
        assert entry.commit is None

    def test_absent_distribution_is_recorded_not_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'omnimarket is not installed' is an identity fact (OMN-14060)."""
        _install(monkeypatch, {})
        entry = ri._package_identity("omnimarket")
        assert entry.source is EnumPackageSourceKind.ABSENT
        assert entry.version is None and entry.commit is None

    def test_workspace_install_takes_its_commit_from_the_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The container path: file:// install, SHA only in build-provenance.

        This is the surface that would have exposed OMN-17291 — a lane wearing
        a fresh registry label over week-old vendored content.
        """
        manifest = tmp_path / "build-provenance.json"
        manifest.write_text(
            json.dumps(
                {
                    "per_repo_vcs_provenance": {
                        "siblings": {
                            "omnimarket": {
                                "vcs_ref": _LANE_MARKET_SHA,
                                "vcs_dirty": False,
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(ri, "BUILD_PROVENANCE_PATH", manifest)
        _install(
            monkeypatch,
            {
                "omnimarket": _FakeDist(
                    "0.4.11",
                    json.dumps(
                        {
                            "url": "file:///workspace/sibling-repos/omnimarket",
                            "dir_info": {},
                        }
                    ),
                )
            },
        )
        entry = ri._package_identity("omnimarket")
        assert entry.source is EnumPackageSourceKind.LOCAL_PATH
        assert entry.commit == _LANE_MARKET_SHA

    def test_dirty_staged_tree_yields_unknown_not_a_commit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A dirty tree's HEAD does not identify its content — say so."""
        manifest = tmp_path / "build-provenance.json"
        manifest.write_text(
            json.dumps(
                {
                    "per_repo_vcs_provenance": {
                        "siblings": {
                            "omnimarket": {
                                "vcs_ref": _LANE_MARKET_SHA,
                                "vcs_dirty": True,
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(ri, "BUILD_PROVENANCE_PATH", manifest)
        _install(
            monkeypatch,
            {
                "omnimarket": _FakeDist(
                    "0.4.11",
                    json.dumps({"url": "file:///workspace/x", "dir_info": {}}),
                )
            },
        )
        entry = ri._package_identity("omnimarket")
        assert entry.source is EnumPackageSourceKind.UNKNOWN
        assert entry.commit is None

    def test_unreadable_manifest_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        broken = tmp_path / "build-provenance.json"
        broken.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(ri, "BUILD_PROVENANCE_PATH", broken)
        assert ri._build_provenance_siblings() == {}


@pytest.mark.unit
class TestCollect:
    def test_every_stamped_package_appears(self) -> None:
        identity = ri.collect_runtime_identity()
        assert set(identity.packages) == set(ri.STAMPED_PACKAGES)

    def test_required_minimum_is_covered(self) -> None:
        """The core gate's DEFAULT_REQUIRED_PACKAGES must be a subset."""
        from omnibase_core.validation.validator_receipt_runtime_identity import (
            DEFAULT_REQUIRED_PACKAGES,
        )

        assert set(DEFAULT_REQUIRED_PACKAGES) <= set(ri.STAMPED_PACKAGES)

    def test_locus_is_the_venv_when_not_containerised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ri, "_container_id", lambda: None)
        ri._execution_locus.cache_clear()
        kind, _ = ri._execution_locus()
        assert kind in {
            EnumExecutionLocusKind.VENV,
            EnumExecutionLocusKind.SYSTEM,
        }

    def test_container_locus_is_the_container_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ri, "_container_id", lambda: "9f2c1b0e4a55")
        ri._execution_locus.cache_clear()
        assert ri._execution_locus() == (
            EnumExecutionLocusKind.CONTAINER,
            "9f2c1b0e4a55",
        )

    def test_package_reads_are_memoised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cheap by construction: a second stamp performs no new lookups."""
        calls: list[str] = []

        def _counting(name: str) -> Any:
            calls.append(name)
            raise __import__("importlib.metadata", fromlist=["x"]).PackageNotFoundError(
                name
            )

        monkeypatch.setattr(ri, "distribution", _counting)
        ri.collect_runtime_identity()
        first = len(calls)
        assert first == len(ri.STAMPED_PACKAGES)
        ri.collect_runtime_identity()
        assert len(calls) == first

    def test_stamped_at_is_fresh_per_call(self) -> None:
        """A copied stamp must not be able to re-date itself silently."""
        first = ri.collect_runtime_identity()
        second = ri.collect_runtime_identity()
        assert second.stamped_at >= first.stamped_at

    def test_config_source_is_carried(self) -> None:
        identity = ri.collect_runtime_identity(config_source="/app/contract.yaml")
        assert identity.config_source == "/app/contract.yaml"

    def test_round_trips_through_the_core_model(self) -> None:
        identity = ri.collect_runtime_identity()
        assert (
            ModelRuntimeIdentity.model_validate_json(identity.model_dump_json())
            == identity
        )


@pytest.mark.unit
class TestRender:
    def test_one_line_names_host_locus_and_every_package(self) -> None:
        line = ri.render_identity_line(ri.collect_runtime_identity())
        assert line.startswith("identity: ")
        assert "\n" not in line
        assert "host=" in line and "locus=" in line
        for name in ri.STAMPED_PACKAGES:
            assert f"{name}=" in line


@pytest.mark.unit
class TestShadowedImportDetection:
    """Metadata describes one tree; the interpreter imports another (OMN-17308).

    Reproduced live 2026-08-31 while verifying this module: a stamp collected
    under ``PYTHONPATH=<core-worktree>/src`` reported
    ``omnibase_core=0.47.1@registry`` while 0.47.2 worktree source was what
    actually executed. Every field was individually true and the block as a
    whole was false — which is the precise thing the epic exists to stop.
    """

    def test_import_from_elsewhere_is_reported_shadowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            {
                "omnibase_core": _FakeDist(
                    "0.47.1",
                    None,
                    location=Path("/opt/venv/lib/site-packages/omnibase_core"),
                )
            },
        )
        entry = ri._package_identity("omnibase_core")
        assert entry.source is EnumPackageSourceKind.SHADOWED
        assert entry.version == "0.47.1"
        # The metadata's commit names the tree that LOST, so it is not carried.
        assert entry.commit is None
        assert entry.import_path is not None
        assert "omnibase_core" in entry.import_path

    def test_shadow_wins_over_a_vcs_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A git install that is shadowed must not report its own commit.

        This is the dangerous ordering: without the shadow check first, the
        entry would read `source=vcs commit=<sha>` — maximum apparent
        precision about code that never ran.
        """
        _install(
            monkeypatch,
            {
                "omnibase_core": _FakeDist(
                    "0.47.1",
                    json.dumps(
                        {
                            "url": "https://github.com/OmniNode-ai/omnibase_core.git",
                            "vcs_info": {"vcs": "git", "commit_id": _VCS_SHA},
                        }
                    ),
                    location=Path("/opt/venv/lib/site-packages/omnibase_core"),
                )
            },
        )
        entry = ri._package_identity("omnibase_core")
        assert entry.source is EnumPackageSourceKind.SHADOWED
        assert entry.commit is None

    def test_editable_install_pointing_at_its_own_tree_is_not_shadowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An editable install importing from its declared tree is CORRECT.

        Without this carve-out every editable install in the workspace reports
        SHADOWED and the signal is noise inside a day.
        """
        real_root = ri._import_root("omnibase_core")
        assert real_root is not None
        declared = real_root.parent
        _install(
            monkeypatch,
            {
                "omnibase_core": _FakeDist(
                    "0.47.1",
                    json.dumps({"url": f"file://{declared}", "dir_info": {}}),
                    location=Path("/opt/venv/lib/site-packages/omnibase_core"),
                )
            },
        )
        entry = ri._package_identity("omnibase_core")
        assert entry.source is not EnumPackageSourceKind.SHADOWED

    def test_absent_package_is_never_shadowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, {})
        entry = ri._package_identity("omnimarket")
        assert entry.source is EnumPackageSourceKind.ABSENT
        assert entry.import_path is None

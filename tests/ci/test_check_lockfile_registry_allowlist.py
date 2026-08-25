# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/ci/check_lockfile_registry_allowlist.py (OMN-16516).

Structural regression guard for the 2026-08-23 mirror-leak incident
(OMN-16162/OMN-16413/OMN-16427/OMN-16428/OMN-16431): a bot-generated
`onex_change_control/uv.lock` baked 783 `source = { registry = ... }` lines
pointing at the Tailscale-only devpi mirror
`omninode-pc.tail75df5e.ts.net:3141` -- 100% of the file's registry lines, 0
at `pypi.org`. Any runner without tailnet access hard-fails at `uv sync
--locked` resolution regardless of runner label or cache state.

This is a STRUCTURAL TOML-parse check (`tomllib`), not a regex over raw
lines -- the plan's own corrected mechanism (cloud-ci-offload-plan.md Rev 5
S1-1, correcting Rev 1's regex-shaped first draft).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.check_lockfile_registry_allowlist import (
    ModelLockfileHostFinding,
    check_lockfile,
    main,
)

# The real 2026-08-23 incident signature: a devpi mirror served over the
# tailnet, no TLS, non-standard port.
_TAILNET_MIRROR_URL = "http://omninode-pc.tail75df5e.ts.net:3141/simple"


def _clean_lockfile() -> str:
    return """\
version = 1
revision = 3
requires-python = ">=3.12"

[manifest]
overrides = [
    { name = "omnibase-compat", specifier = "==0.5.6" },
]

[[package]]
name = "aenum"
version = "3.1.16"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/aenum-3.1.16.tar.gz", hash = "sha256:deadbeef" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/aenum-3.1.16-py3-none-any.whl", hash = "sha256:deadbeef" },
]

[[package]]
name = "onex-change-control"
version = "0.5.1"
source = { git = "https://github.com/OmniNode-ai/onex_change_control.git?rev=2dd26ade7caaa7131e532473ec9d8a207d0e77ab#2dd26ade7caaa7131e532473ec9d8a207d0e77ab" }

[[package]]
name = "omnibase-infra"
version = "0.1.0"
source = { editable = "." }
"""


def _poisoned_lockfile() -> str:
    return f"""\
version = 1
revision = 3
requires-python = ">=3.12"

[[package]]
name = "omnibase-core"
version = "0.46.11"
source = {{ registry = "{_TAILNET_MIRROR_URL}" }}
sdist = {{ url = "{_TAILNET_MIRROR_URL}/omnibase-core-0.46.11.tar.gz", hash = "sha256:deadbeef" }}
"""


class TestCheckLockfileCleanFile:
    def test_pypi_registry_source_passes(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_clean_lockfile())
        findings = check_lockfile(lockfile)
        assert findings == ()

    def test_github_git_source_passes(self, tmp_path: Path) -> None:
        """A [tool.uv.sources] internal-package git pin at github.com is a
        legitimate, public, cross-repo dependency -- not a mirror leak."""
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_clean_lockfile())
        findings = check_lockfile(lockfile)
        assert not any(f.package_name == "onex-change-control" for f in findings)

    def test_editable_local_source_has_no_host_to_check(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_clean_lockfile())
        findings = check_lockfile(lockfile)
        assert not any(f.package_name == "omnibase-infra" for f in findings)


class TestCheckLockfilePoisoned:
    def test_tailnet_mirror_registry_is_a_blocking_finding(
        self, tmp_path: Path
    ) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_poisoned_lockfile())
        findings = check_lockfile(lockfile)
        assert len(findings) >= 1
        assert all(isinstance(f, ModelLockfileHostFinding) for f in findings)
        registry_finding = next(
            f for f in findings if f.field_path == "source.registry"
        )
        assert registry_finding.package_name == "omnibase-core"
        assert registry_finding.host == "omninode-pc.tail75df5e.ts.net"

    def test_poisoned_sdist_url_is_also_a_finding(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_poisoned_lockfile())
        findings = check_lockfile(lockfile)
        assert any(f.field_path == "sdist.url" for f in findings)

    def test_incident_replay_62_of_62_registry_lines_all_blocked(
        self, tmp_path: Path
    ) -> None:
        """Replays the exact incident shape: every package in the lockfile
        resolves from the tailnet mirror, 0 from pypi.org (§1.1 FACT)."""
        packages = "\n".join(
            f'[[package]]\nname = "pkg{i}"\nversion = "1.0"\n'
            f'source = {{ registry = "{_TAILNET_MIRROR_URL}" }}\n'
            for i in range(62)
        )
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(f"version = 1\n\n{packages}")
        findings = check_lockfile(lockfile)
        registry_findings = [f for f in findings if f.field_path == "source.registry"]
        assert len(registry_findings) == 62


class TestCheckLockfileAllowlistExtension:
    def test_extra_allowed_host_is_not_a_finding(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(
            '[[package]]\nname = "internal-pkg"\nversion = "1.0"\n'
            'source = { registry = "https://internal-mirror.example.com/simple" }\n'
        )
        findings = check_lockfile(
            lockfile, allowlist=frozenset({"internal-mirror.example.com"})
        )
        assert findings == ()

    def test_default_allowlist_rejects_unlisted_host(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(
            '[[package]]\nname = "internal-pkg"\nversion = "1.0"\n'
            'source = { registry = "https://internal-mirror.example.com/simple" }\n'
        )
        findings = check_lockfile(lockfile)
        assert len(findings) == 1


class TestCli:
    def test_exit_0_on_clean_lockfile(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_clean_lockfile())
        assert main([str(lockfile)]) == 0

    def test_exit_1_on_poisoned_lockfile(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_poisoned_lockfile())
        assert main([str(lockfile)]) == 1

    def test_exit_2_on_malformed_toml(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text("this is not [ valid toml")
        assert main([str(lockfile)]) == 2

    def test_missing_lockfile_is_not_a_failure(self, tmp_path: Path) -> None:
        """A repo with no committed uv.lock (e.g. a docs-only repo) is not a
        gate failure -- there is nothing to check, and the gate must not
        vacuously reward a MISSING lockfile the way it would penalize a
        malformed one."""
        assert main([str(tmp_path / "does-not-exist" / "uv.lock")]) == 0

    def test_anti_vacuity_floor_fails_on_too_few_packages(self, tmp_path: Path) -> None:
        """--min-packages guards the OMN-15538-precedent vacuous-green case:
        a glob typo or a schema-shape drift that makes the gate check
        (near-)nothing must not silently report success forever."""
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_clean_lockfile())  # 3 packages
        assert main([str(lockfile), "--min-packages", "10"]) == 1

    def test_anti_vacuity_floor_passes_when_met(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_clean_lockfile())  # 3 packages
        assert main([str(lockfile), "--min-packages", "2"]) == 0

    def test_allow_host_flag_extends_allowlist(self, tmp_path: Path) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(
            '[[package]]\nname = "internal-pkg"\nversion = "1.0"\n'
            'source = { registry = "https://internal-mirror.example.com/simple" }\n'
        )
        assert main([str(lockfile), "--allow-host", "internal-mirror.example.com"]) == 0

    def test_reports_finding_detail_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        lockfile = tmp_path / "uv.lock"
        lockfile.write_text(_poisoned_lockfile())
        main([str(lockfile)])
        captured = capsys.readouterr()
        assert "omninode-pc.tail75df5e.ts.net" in captured.err
        assert "omnibase-core" in captured.err


# Incident replay (OMN-16162), registered in tests/incident_replays/registry.yaml
# as case `omn16162-lockfile-mirror-leak`. The fixture is the REAL
# onex_change_control/uv.lock committed at d0ada7dc7 -- the exact bytes that
# broke `uv sync --locked` on every runner without tailnet access. This
# guard is the fail-closed backstop this incident retroactively proves is
# necessary: nothing in PR CI at the time could have said no.
_INCIDENT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "omn-16162-lockfile-mirror-leak"
    / "onex_change_control_uv.lock.captured"
)


class TestOmn16162IncidentReplay:
    def test_fixture_is_present_and_real(self) -> None:
        assert _INCIDENT_FIXTURE.is_file()
        assert "tail75df5e" in _INCIDENT_FIXTURE.read_text()

    def test_real_poisoned_occ_lockfile_is_rejected(self) -> None:
        """Would-have-caught-it (R5, false_green): drives the real
        check_lockfile() against the exact committed bytes of
        onex_change_control/uv.lock at d0ada7dc7 -- 783 lines resolving from
        the Tailscale-only devpi mirror, 0 from pypi.org. Before this guard
        existed, nothing in PR CI validated registry hosts at all.
        """
        findings = check_lockfile(_INCIDENT_FIXTURE)
        assert findings, (
            "the real OMN-16162 poisoned onex_change_control/uv.lock must be "
            "REJECTED -- every package in it resolves from a private tailnet "
            "mirror, not a public registry"
        )
        registry_findings = [f for f in findings if f.field_path == "source.registry"]
        assert len(registry_findings) == 62, (
            "the incident's own §1.1 FACT: exactly 62 source.registry lines, "
            "100% of them pointing at the tailnet mirror"
        )
        assert all(f.host == "omninode-pc.tail75df5e.ts.net" for f in registry_findings)

    def test_real_poisoned_lockfile_fails_the_cli(self) -> None:
        assert main([str(_INCIDENT_FIXTURE)]) == 1

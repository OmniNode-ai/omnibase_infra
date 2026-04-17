# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ONEX tick daemon shell scripts and systemd unit files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
TICKS_DIR = SCRIPTS_DIR / "ticks"
SYSTEMD_DIR = SCRIPTS_DIR / "systemd"

TICK_SCRIPTS = ["health-tick.sh", "merge-sweep-tick.sh", "overseer-tick.sh"]
TIMER_UNITS = [
    "onex-tick-health.timer",
    "onex-tick-merge-sweep.timer",
    "onex-tick-overseer.timer",
]
SERVICE_UNITS = [
    "onex-tick-health.service",
    "onex-tick-merge-sweep.service",
    "onex-tick-overseer.service",
]
INSTALLER = SCRIPTS_DIR / "install-tick-daemons.sh"


# ---- Script existence ----


@pytest.mark.parametrize("script", TICK_SCRIPTS)
def test_tick_script_exists(script: str) -> None:
    assert (TICKS_DIR / script).is_file(), (
        f"Missing tick script: scripts/ticks/{script}"
    )


@pytest.mark.parametrize("unit", TIMER_UNITS + SERVICE_UNITS)
def test_systemd_unit_exists(unit: str) -> None:
    assert (SYSTEMD_DIR / unit).is_file(), (
        f"Missing systemd unit: scripts/systemd/{unit}"
    )


def test_installer_exists() -> None:
    assert INSTALLER.is_file(), "Missing scripts/install-tick-daemons.sh"


# ---- Script invariants ----


@pytest.mark.parametrize("script", TICK_SCRIPTS)
def test_tick_script_sources_omnibase_env(script: str) -> None:
    text = (TICKS_DIR / script).read_text()
    active_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    source_lines = [
        ln for ln in active_lines if "source" in ln and ".omnibase/.env" in ln
    ]
    assert source_lines, f"{script} must source ~/.omnibase/.env"


@pytest.mark.parametrize("script", TICK_SCRIPTS)
def test_tick_script_no_hardcoded_secrets(script: str) -> None:
    text = (TICKS_DIR / script).read_text()
    # Passwords/keys must not appear as literals (only via env var references)
    assert "postgresql://" not in text, f"{script}: hardcoded postgres URL"
    assert re.search(r'password\s*=\s*["\'][^$]', text, re.IGNORECASE) is None, (
        f"{script}: hardcoded password literal"
    )


@pytest.mark.parametrize("script", TICK_SCRIPTS)
def test_tick_script_no_hardcoded_absolute_paths(script: str) -> None:
    text = (TICKS_DIR / script).read_text()
    active_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    for line in active_lines:
        # Allow /data/onex (runtime data dir) and /bin /usr /home (system paths)
        # Disallow /Users/ or /Volumes/ (developer-machine-specific)
        assert "/Users/" not in line, f"{script}: hardcoded /Users/ path: {line!r}"
        assert "/Volumes/" not in line, f"{script}: hardcoded /Volumes/ path: {line!r}"


@pytest.mark.parametrize("script", TICK_SCRIPTS)
def test_tick_script_no_session_bound_croncreate(script: str) -> None:
    text = (TICKS_DIR / script).read_text()
    assert "CronCreate" not in text, (
        f"{script}: must not reference session-bound CronCreate"
    )


@pytest.mark.parametrize("script", TICK_SCRIPTS)
def test_tick_script_writes_report_to_disk(script: str) -> None:
    text = (TICKS_DIR / script).read_text()
    assert "/data/onex/ticks/" in text, (
        f"{script}: must write tick report to /data/onex/ticks/"
    )


def test_merge_sweep_uses_graphql_squash_not_gh_merge_auto() -> None:
    text = (TICKS_DIR / "merge-sweep-tick.sh").read_text()
    # Must use GraphQL enablePullRequestAutoMerge, not bare `gh pr merge --auto`
    assert "enablePullRequestAutoMerge" in text, (
        "merge-sweep-tick.sh must use GraphQL enablePullRequestAutoMerge mutation"
    )
    active_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    bad = [ln for ln in active_lines if "gh pr merge" in ln and "--auto" in ln]
    assert not bad, f"merge-sweep-tick.sh: must not use `gh pr merge --auto`: {bad}"


def test_health_tick_probes_required_infra() -> None:
    text = (TICKS_DIR / "health-tick.sh").read_text()
    for host_port in [
        '192.168.86.201" "5436',
        '192.168.86.201" "19092',
        '192.168.86.201" "16379',
    ]:
        assert host_port in text, f"health-tick.sh must probe {host_port}"


def test_health_tick_probes_llm_endpoints() -> None:
    text = (TICKS_DIR / "health-tick.sh").read_text()
    for port in ["8000", "8001", "8100", "8101", "8102"]:
        assert port in text, f"health-tick.sh must probe LLM endpoint port {port}"


def test_overseer_tick_requires_linear_api_key() -> None:
    text = (TICKS_DIR / "overseer-tick.sh").read_text()
    assert "LINEAR_API_KEY" in text, "overseer-tick.sh must reference LINEAR_API_KEY"
    # Must fail-loud if not set
    assert "LINEAR_API_KEY:-}" in text or 'LINEAR_API_KEY"' in text, (
        "overseer-tick.sh must check for empty LINEAR_API_KEY"
    )


def test_overseer_tick_uses_claims_ttl() -> None:
    text = (TICKS_DIR / "overseer-tick.sh").read_text()
    assert "CLAIM_TTL" in text or "claims" in text.lower(), (
        "overseer-tick.sh must implement claims TTL to avoid double-dispatch"
    )


# ---- Systemd unit file invariants ----


@pytest.mark.parametrize("service", SERVICE_UNITS)
def test_service_unit_has_environment_file(service: str) -> None:
    text = (SYSTEMD_DIR / service).read_text()
    assert "EnvironmentFile=" in text, (
        f"{service}: must declare EnvironmentFile= to load ~/.omnibase/.env"
    )


@pytest.mark.parametrize("service", SERVICE_UNITS)
def test_service_unit_is_oneshot(service: str) -> None:
    text = (SYSTEMD_DIR / service).read_text()
    assert "Type=oneshot" in text, f"{service}: must be Type=oneshot"


@pytest.mark.parametrize("service", SERVICE_UNITS)
def test_service_unit_no_root_scope(service: str) -> None:
    text = (SYSTEMD_DIR / service).read_text()
    # --user units must not set User= (that implies system scope)
    assert "User=" not in text, (
        f"{service}: --user scope units must not set User= (only system-scope units need it)"
    )
    # Must not WantedBy multi-user.target (that's system scope)
    assert "multi-user.target" not in text, (
        f"{service}: --user units must WantedBy=default.target, not multi-user.target"
    )


@pytest.mark.parametrize("timer", TIMER_UNITS)
def test_timer_unit_has_persistent(timer: str) -> None:
    text = (SYSTEMD_DIR / timer).read_text()
    assert "Persistent=true" in text, (
        f"{timer}: must set Persistent=true so missed runs fire on restart"
    )


def test_health_timer_schedule() -> None:
    text = (SYSTEMD_DIR / "onex-tick-health.timer").read_text()
    assert "*:03:00" in text, "health timer must fire at :03 each hour"


def test_merge_sweep_timer_schedule() -> None:
    text = (SYSTEMD_DIR / "onex-tick-merge-sweep.timer").read_text()
    assert "*:23:00" in text, "merge-sweep timer must fire at :23 each hour"


def test_overseer_timer_schedule() -> None:
    text = (SYSTEMD_DIR / "onex-tick-overseer.timer").read_text()
    assert "0/15" in text, "overseer timer must fire every 15 minutes"


# ---- Installer invariants ----


def test_installer_targets_user_scope() -> None:
    text = INSTALLER.read_text()
    assert "systemctl --user" in text, (
        "installer must use systemctl --user (not system scope)"
    )
    assert "sudo systemctl" not in text or "# sudo" in text, (
        "installer must not use sudo systemctl (user scope only)"
    )


def test_installer_has_dry_run_mode() -> None:
    text = INSTALLER.read_text()
    assert "dry-run" in text or "DRY_RUN" in text, "installer must support --dry-run"


def test_installer_deploys_all_six_units() -> None:
    text = INSTALLER.read_text()
    for unit in TIMER_UNITS + SERVICE_UNITS:
        assert unit in text, f"installer must reference {unit}"


def test_installer_no_hardcoded_absolute_paths() -> None:
    text = INSTALLER.read_text()
    active_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    for line in active_lines:
        assert "/Users/" not in line, f"installer: hardcoded /Users/ path: {line!r}"
        assert "/Volumes/" not in line, f"installer: hardcoded /Volumes/ path: {line!r}"

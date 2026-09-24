# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19415: a pre-PR slot has a connection budget, and the server is sized for it.

WHY THIS EXISTS
---------------
A pre-PR verify slot reuses the dev lane's Postgres SERVER (epic OMN-18888).
Until this ticket nothing bounded how many connections a slot could open there,
and nothing declared how many the server accepts: it ran the stock
``max_connections`` of 100, three of them reserved for superusers.

Measured on the lab host on 2026-09-24 with a read-only two-second
``pg_stat_activity`` sampler: one booting slot held 63 connections
(``role_omnibase_prepr1`` peaked at 50, four slot containers at 14/14/14/9, most
of them opened by pool floors and never used), the dev lane held 35-45, and the
server sat at 100/100 while the dev runtime logged 1171 refusals in four
minutes. The slot is a guest on that server; the dev lane paid for it. The dev
lane alone, with no slot running, reached 93 at 16:50:07Z just after a runtime
redeploy, so the stock server had no room for any slot at all.

THE FIX, AND WHY IT IS THREE PIECES
-----------------------------------
1. Every slot principal gets an explicit ``CONNECTION LIMIT`` from one budget
   table in ``scripts/provision_db_slot.sh``. A slot that wants more than its
   budget is refused inside the slot ("too many connections for role"), never
   by starving the dev lane.
2. ``--apply`` refuses to provision a slot on a server that is not sized for
   the dev lane plus every pool slot, before it creates anything.
3. ``docker/docker-compose.dev-lane.yml`` declares ``max_connections`` on the
   dev lane's postgres, and this file pins that the declared value covers the
   same arithmetic the provisioner enforces, with the slot count read from the
   pool's own policy table rather than restated. It is declared in the dev-lane
   overlay and NOT in the base ``docker-compose.infra.yml``, because
   stability-test, sim-202 and prod layer their postgres over the base and
   would inherit it.

Ticket: OMN-19415. Parent epic: OMN-18888.
"""

from __future__ import annotations

import importlib.util
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVISIONER = REPO_ROOT / "scripts" / "provision_db_slot.sh"
INFRA_COMPOSE = REPO_ROOT / "docker" / "docker-compose.infra.yml"
DEV_LANE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
SLOT_POLICY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_slot_policy.py"

EXIT_SERVER_UNDERSIZED = 11

# The superuser reservation is a server setting. The dev-lane overlay does not
# override it, so the stock value applies; if it ever sets it, the pin below
# reads the declared value instead.
STOCK_SUPERUSER_RESERVED = 3
STOCK_RESERVED = 0


def _run_scope(slot: str = "prepr1") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PROVISIONER), "--print-scope"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "ONEX_DB_SLOT": slot},
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def _values(stdout: str, key: str) -> list[str]:
    return [
        line.split("=", 1)[1]
        for line in stdout.splitlines()
        if line.startswith(f"{key}=")
    ]


def _scope() -> dict[str, object]:
    result = _run_scope()
    assert result.returncode == 0, result.stderr
    limits: dict[str, int] = {}
    for entry in _values(result.stdout, "connection_limit"):
        role, _, limit = entry.partition(":")
        limits[role] = int(limit)
    (slot_budget,) = _values(result.stdout, "slot_connection_budget")
    (dev_budget,) = _values(result.stdout, "dev_lane_connection_budget")
    (pool_slots,) = _values(result.stdout, "pool_slot_count")
    return {
        "roles": _values(result.stdout, "role"),
        "limits": limits,
        "slot_budget": int(slot_budget),
        "dev_budget": int(dev_budget),
        "pool_slots": int(pool_slots),
    }


def _load_slot_policy() -> object:
    spec = importlib.util.spec_from_file_location(
        "prepr_slot_policy_19415", SLOT_POLICY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ComposeLoader(yaml.SafeLoader):
    """A SafeLoader that tolerates compose's ``!override`` tag.

    The dev-lane overlay uses ``!override``; this file reads one literal command
    list and never merge semantics, so keeping the value and dropping the tag is
    correct here.
    """


def _drop_tag(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise AssertionError(f"unhandled YAML node kind: {type(node).__name__}")


_ComposeLoader.add_multi_constructor("!", _drop_tag)  # type: ignore[no-untyped-call]


def _declared_postgres_settings() -> dict[str, int]:
    loaded = yaml.load(DEV_LANE_COMPOSE.read_text(), Loader=_ComposeLoader)  # noqa: S506 - local tolerant SafeLoader subclass
    services = loaded["services"]
    command = (services.get("postgres") or {}).get("command")
    assert command, (
        "docker-compose.dev-lane.yml declares no postgres command, so the dev "
        "lane's server runs the stock max_connections of 100 -- the undeclared "
        "value this ticket found"
    )
    text = command if isinstance(command, str) else " ".join(command)
    return {key: int(value) for key, value in re.findall(r"-c\s+([a-z_]+)=(\d+)", text)}


class TestEverySlotPrincipalHasABudget:
    def test_every_role_the_slot_creates_has_a_connection_limit(self) -> None:
        scope = _scope()
        roles = scope["roles"]
        limits = scope["limits"]
        assert isinstance(roles, list) and isinstance(limits, dict)
        assert roles, "positive control: the scope must name the slot's roles"
        assert sorted(limits) == sorted(roles), (
            "every slot principal needs exactly one CONNECTION LIMIT; "
            f"roles={sorted(roles)} limits={sorted(limits)}"
        )

    def test_every_limit_is_a_positive_bound(self) -> None:
        limits = _scope()["limits"]
        assert isinstance(limits, dict)
        # -1 is Postgres for "unlimited", which is the defect, and 0 locks the
        # principal out entirely.
        assert all(limit >= 1 for limit in limits.values()), limits

    def test_the_slot_budget_is_the_sum_of_its_limits(self) -> None:
        scope = _scope()
        limits = scope["limits"]
        assert isinstance(limits, dict)
        assert scope["slot_budget"] == sum(limits.values())

    def test_the_heaviest_principal_covers_the_measured_boot_peak(self) -> None:
        """role_omnibase_<slot> peaked at 50 connections on 2026-09-24T16:27Z."""
        limits = _scope()["limits"]
        assert isinstance(limits, dict)
        assert limits["role_omnibase_prepr1"] >= 50

    def test_the_apply_path_sets_and_reads_back_the_limit(self) -> None:
        text = PROVISIONER.read_text()
        assert "CONNECTION LIMIT" in text
        assert "rolconnlimit" in text, (
            "the limit must be read back from pg_roles, not assumed from the "
            "statement that set it"
        )


class TestThePoolSizeIsTheOneThePolicyDeclares:
    def test_the_provisioner_counts_the_same_slots_as_the_pool_policy(self) -> None:
        policy = _load_slot_policy()
        assert _scope()["pool_slots"] == len(policy.SLOTS)  # type: ignore[attr-defined]


class TestTheServerIsSizedForThePool:
    def test_the_dev_lane_overlay_declares_max_connections(self) -> None:
        assert "max_connections" in _declared_postgres_settings()

    def test_the_base_compose_leaves_the_postgres_command_alone(self) -> None:
        """Other lanes merge the base; a base command would recreate their servers."""
        services = yaml.safe_load(INFRA_COMPOSE.read_text())["services"]
        assert "postgres" in services, "positive control: the base defines postgres"
        assert "command" not in services["postgres"]

    def test_the_declared_capacity_covers_the_dev_lane_and_every_slot(self) -> None:
        scope = _scope()
        settings = _declared_postgres_settings()
        required = (
            int(scope["dev_budget"])  # type: ignore[call-overload]
            + int(scope["pool_slots"]) * int(scope["slot_budget"])  # type: ignore[call-overload]
            + settings.get("superuser_reserved_connections", STOCK_SUPERUSER_RESERVED)
            + settings.get("reserved_connections", STOCK_RESERVED)
        )
        assert settings["max_connections"] >= required, (
            f"max_connections={settings['max_connections']} is below the "
            f"{required} the dev lane plus {scope['pool_slots']} slots need"
        )


def _fake_psql(tmp_path: Path, settings: str) -> tuple[dict[str, str], Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "psql.log"
    fake = bindir / "psql"
    fake.write_text(
        "#!/bin/bash\n"
        f'LOG="{log}"\n'
        'printf "ARGS %s\\n" "$*" >> "$LOG"\n'
        'if [ ! -t 0 ]; then cat >> "$LOG"; fi\n'
        'case "$*" in\n'
        "  *rolsuper*current_user*) echo t ;;\n"
        '  *max_connections*) echo "$FAKE_PG_CAPACITY" ;;\n'
        "esac\n"
        "exit 0\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin:/usr/sbin:/sbin",
        "ONEX_DB_SLOT": "prepr1",
        "POSTGRES_PASSWORD": "not-a-real-password",
        "FAKE_PG_CAPACITY": settings,
    }
    return env, log


def _apply(
    tmp_path: Path, settings: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    env, log = _fake_psql(tmp_path, settings)
    result = subprocess.run(
        ["bash", str(PROVISIONER), "--apply", "--env-file", str(tmp_path / "slot.env")],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    return result, log.read_text() if log.exists() else ""


class TestApplyRefusesAnUndersizedServer:
    def test_a_stock_server_is_refused_before_anything_is_created(
        self, tmp_path: Path
    ) -> None:
        result, log = _apply(tmp_path, "100 3 0")
        assert result.returncode == EXIT_SERVER_UNDERSIZED, (
            result.stdout,
            result.stderr,
        )
        assert "CREATE ROLE" not in log
        assert "CREATE DATABASE" not in log
        assert "max_connections" in log, "positive control: the preflight asked"

    def test_a_server_sized_for_the_pool_passes_the_preflight(
        self, tmp_path: Path
    ) -> None:
        settings = _declared_postgres_settings()
        declared = settings["max_connections"]
        result, log = _apply(tmp_path, f"{declared} 3 0")
        # The fake answers nothing past the preflight, so the run fails later;
        # what matters is that it got past the capacity check and began
        # provisioning, which is where the undersized case never arrives.
        assert result.returncode != EXIT_SERVER_UNDERSIZED, result.stderr
        assert "CREATE ROLE" in log

    def test_an_unreadable_capacity_fails_closed(self, tmp_path: Path) -> None:
        result, log = _apply(tmp_path, "")
        assert result.returncode == EXIT_SERVER_UNDERSIZED, (
            result.stdout,
            result.stderr,
        )
        assert "CREATE ROLE" not in log


EXIT_ROLE_WITHOUT_BUDGET = 12


class TestAnUnbudgetedPrincipalFailsClosed:
    def test_a_role_missing_from_the_budget_table_is_refused(
        self, tmp_path: Path
    ) -> None:
        text = PROVISIONER.read_text()
        # The last entry of CONNECTION_BUDGET_MAP, with the map's closing quote.
        entry = '\nchain_canary_reader:2"'
        assert entry in text, "positive control: the budget entry being removed exists"
        mutated = tmp_path / "provision_db_slot.sh"
        mutated.write_text(text.replace(entry, '"', 1))
        result = subprocess.run(
            ["bash", str(mutated), "--print-scope"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "ONEX_DB_SLOT": "prepr1"},
            stdin=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
        assert result.returncode == EXIT_ROLE_WITHOUT_BUDGET, (
            result.stdout,
            result.stderr,
        )
        assert "chain_canary_reader_prepr1" in result.stderr

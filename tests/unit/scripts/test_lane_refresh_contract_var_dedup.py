# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression guard for OMN-14984: lane-identity dedup in the lane-refresh
scripts.

`refresh_stability_lane.sh` / `refresh_dev_lane.sh` used to re-declare their
own `readonly` bash copies of COMPOSE_PROJECT and the health/manifest default
port even though the SAME values already exist as contract-rendered
`STABILITY_TEST_*` / `DEV_*` variables in `docker/runtime-policy.env`
(generated from `contracts/services/runtime_policy.contract.yaml`), which
both scripts already `source` at their own top. That duplication is live
drift risk: re-render the contract and the hardcoded bash literal silently
goes stale.

These tests statically guard the fix:

  1. Neither script re-declares COMPOSE_PROJECT or the manifest/health port
     as a bare hardcoded literal -- both must read the value from the
     already-sourced contract-rendered env var, failing fast BY NAME if it
     is absent (never falling back to a hardcoded default).
  2. The exact env-var names each script reads (`STABILITY_TEST_COMPOSE_
     PROJECT` / `STABILITY_TEST_RUNTIME_MAIN_PORT` / `DEV_COMPOSE_PROJECT` /
     `DEV_RUNTIME_MAIN_PORT`) are cross-checked against the live
     `docker/runtime-policy.env` so this guard fails loudly (not silently)
     if a future contract re-render ever drops one of them.
  3. REDPANDA_CONTAINER / CORE_SERVICES / CORE_CONTAINERS / MIN_CONTRACTS
     remain intentionally hardcoded (verified NOT present in
     `docker/runtime-policy.env` -- migrating them would fabricate a
     contract-rendered source that does not exist, out of scope for this
     ticket) -- documented so a future session does not "fix" them into a
     fabricated read.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STABILITY_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"
DEV_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
RUNTIME_POLICY_ENV = REPO_ROOT / "docker" / "runtime-policy.env"


def _load_runtime_policy_keys() -> set[str]:
    """Parse the KEY= names out of the contract-rendered env file (no eval;
    this only needs the key set, never the values)."""
    text = RUNTIME_POLICY_ENV.read_text()
    keys: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", line)
        if match:
            keys.add(match.group(1))
    return keys


def test_runtime_policy_env_exists_and_is_tracked() -> None:
    assert RUNTIME_POLICY_ENV.is_file(), (
        f"{RUNTIME_POLICY_ENV} not found -- both lane-refresh scripts assume "
        "this contract-rendered file exists and source it unconditionally"
    )


def test_stability_lane_script_reads_compose_project_from_contract_file() -> None:
    text = STABILITY_SCRIPT.read_text()

    # The old hardcoded pattern must be GONE.
    assert 'readonly COMPOSE_PROJECT="omnibase-infra-stability-test"' not in text, (
        "refresh_stability_lane.sh must not re-hardcode COMPOSE_PROJECT -- "
        "read STABILITY_TEST_COMPOSE_PROJECT from the sourced "
        "docker/runtime-policy.env instead"
    )

    # The new fail-fast-by-name read must be present.
    assert "require_contract_var STABILITY_TEST_COMPOSE_PROJECT" in text
    assert 'readonly COMPOSE_PROJECT="${STABILITY_TEST_COMPOSE_PROJECT}"' in text


def test_stability_lane_script_reads_port_from_contract_file() -> None:
    text = STABILITY_SCRIPT.read_text()

    # The old hardcoded port literal in the default URLs must be GONE.
    assert 'MANIFEST_URL="http://${LANE_PROBE_HOST}:18085' not in text
    assert 'HEALTH_URL="http://${LANE_PROBE_HOST}:18085' not in text

    assert "require_contract_var STABILITY_TEST_RUNTIME_MAIN_PORT" in text
    assert "${STABILITY_TEST_RUNTIME_MAIN_PORT}" in text


def test_dev_lane_script_reads_compose_project_from_contract_file() -> None:
    text = DEV_SCRIPT.read_text()

    assert 'readonly COMPOSE_PROJECT="omnibase-infra"' not in text, (
        "refresh_dev_lane.sh must not re-hardcode COMPOSE_PROJECT -- read "
        "DEV_COMPOSE_PROJECT from the sourced docker/runtime-policy.env "
        "instead"
    )

    assert "require_contract_var DEV_COMPOSE_PROJECT" in text
    assert 'readonly COMPOSE_PROJECT="${DEV_COMPOSE_PROJECT}"' in text


def test_dev_lane_script_reads_port_from_contract_file() -> None:
    text = DEV_SCRIPT.read_text()

    assert 'MANIFEST_URL="http://${LANE_PROBE_HOST}:8085' not in text
    assert 'HEALTH_URL="http://${LANE_PROBE_HOST}:8085' not in text

    assert "require_contract_var DEV_RUNTIME_MAIN_PORT" in text
    assert "${DEV_RUNTIME_MAIN_PORT}" in text


def test_contract_var_names_actually_exist_in_runtime_policy_env() -> None:
    """Cross-check: the exact var names both scripts now depend on must be
    real keys in the live contract-rendered file today. If a future contract
    re-render ever drops one of these, this test fails loudly instead of the
    scripts silently hard-failing at runtime with no CI signal."""
    keys = _load_runtime_policy_keys()
    required = {
        "STABILITY_TEST_COMPOSE_PROJECT",
        "STABILITY_TEST_RUNTIME_MAIN_PORT",
        "DEV_COMPOSE_PROJECT",
        "DEV_RUNTIME_MAIN_PORT",
    }
    missing = required - keys
    assert not missing, (
        f"docker/runtime-policy.env no longer renders {missing} -- the "
        "contract (contracts/services/runtime_policy.contract.yaml) must "
        "keep rendering these keys, or refresh_stability_lane.sh / "
        "refresh_dev_lane.sh must be updated in the same change"
    )


def test_require_contract_var_helper_present_in_both_scripts() -> None:
    """Both scripts define the same fail-fast-by-name helper (rule 8: fail
    fast on missing env, never a silent hardcoded fallback)."""
    for script in (STABILITY_SCRIPT, DEV_SCRIPT):
        text = script.read_text()
        assert "require_contract_var()" in text
        assert "exit 64" in text


def test_intentionally_hardcoded_values_are_not_present_in_runtime_policy_env() -> None:
    """Documents (and guards) the honest scope boundary: REDPANDA_CONTAINER /
    CORE_SERVICES / CORE_CONTAINERS / MIN_CONTRACTS stay hardcoded in both
    scripts because docker/runtime-policy.env does not render them today.
    If a future contract change starts rendering one of these keys, this
    test fails so a human decides whether to wire it through rather than the
    duplication silently persisting unnoticed."""
    keys = _load_runtime_policy_keys()
    not_yet_declared = {
        "STABILITY_TEST_REDPANDA_CONTAINER",
        "STABILITY_TEST_MIN_CONTRACTS",
        "DEV_REDPANDA_CONTAINER",
        "DEV_POSTGRES_CONTAINER",
        "DEV_MIN_CONTRACTS",
    }
    present = not_yet_declared & keys
    assert not present, (
        f"docker/runtime-policy.env now renders {present} -- these were "
        "intentionally left hardcoded in refresh_stability_lane.sh / "
        "refresh_dev_lane.sh under OMN-14984 because they did not exist in "
        "the contract-rendered file at the time; now that they do, wire the "
        "scripts to read them instead of leaving this guard to bit-rot"
    )

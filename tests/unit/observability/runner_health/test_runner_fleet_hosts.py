# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner fleet is a declared multi-host inventory (OMN-17477).

Before this, `config/runner_fleet.yaml` described ONE host in scalar fields:
`runner_host`, `runner_name_prefix`, `expected_count`. Every consumer read
those scalars, so a second host was not merely unconfigured -- it was
unrepresentable, and a second host's runners would have collided on container
and runner names with the first host's.

The inventory adds a `hosts:` list without removing the scalars, which stay as
the PRIMARY host's values so that no existing consumer changes behaviour on the
day the list is introduced. What the list buys is the three things a second
host needs and a scalar cannot express: which CPU architecture it is, a name
prefix that cannot collide with another host's, and which runner CLASSES it
carries -- because a host that runs only deploy/verify-class runners must not
be counted toward the action fleet's capacity floor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    ModelRunnerFleetConfig,
    load_runner_fleet_config,
)

REPO_ROOT = Path(__file__).parents[4]
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"


def _config() -> ModelRunnerFleetConfig:
    return load_runner_fleet_config(FLEET_CONFIG)


# The declared hosts, as (address, arch, expected_count). Named here rather
# than spelled inline so the lookups below are dict `.get()` calls: a `"literal"
# in mapping` membership test is indistinguishable from a URL substring check to
# a static analyser, and the two CodeQL high-severity
# `py/incomplete-url-substring-sanitization` alerts this file first raised were
# exactly that false shape. `.get()` says the same thing and does not.
EXPECTED_HOSTS: tuple[tuple[str, str, int], ...] = (
    ("omninode-pc.tail75df5e.ts.net", "amd64", 60),
    ("stickybeatz-2.tail75df5e.ts.net", "arm64", 1),
)


def test_inventory_declares_every_lab_host_with_its_architecture() -> None:
    config = _config()
    by_host = {host.host: host for host in config.hosts}

    for address, arch, expected_count in EXPECTED_HOSTS:
        row = by_host.get(address)
        assert row is not None, (
            f"the inventory must declare the host {address!r}; it declares "
            f"{sorted(by_host)}"
        )
        assert row.arch.value == arch, (
            f"{address} is declared {row.arch.value}, expected {arch}"
        )
        assert row.expected_count == expected_count, (
            f"{address} declares expected_count={row.expected_count}, "
            f"expected {expected_count}"
        )


def test_primary_host_row_agrees_with_the_legacy_scalars() -> None:
    """The scalars are the primary host's values, not a second source of truth.

    If these two ever disagree, a consumer's answer depends on which of the two
    it happens to read -- the exact split-brain the inventory exists to end.
    """
    config = _config()
    primary = config.primary_host()

    assert primary.host == config.runner_host
    assert primary.runner_name_prefix == config.runner_name_prefix
    assert primary.expected_count == config.expected_count


def test_runner_name_prefixes_cannot_collide_across_hosts() -> None:
    """No host's prefix may be a prefix of another's.

    Runner names are `<prefix>-<N>`, and the fleet's own compose test matches
    services with `rf"{prefix}-\\d+"`. A second host prefixed `omninode-runner-101`
    would produce `omninode-runner-101-1`, and the primary host's pattern would
    claim `omninode-runner-101` itself -- one host silently counting another's
    runners as its own.
    """
    config = _config()
    prefixes = [host.runner_name_prefix for host in config.hosts]
    assert len(prefixes) == len(set(prefixes)), f"duplicate prefixes: {prefixes}"
    for a in prefixes:
        for b in prefixes:
            if a is not b and b.startswith(f"{a}-"):
                pytest.fail(f"prefix {b!r} collides with {a!r}")


def test_declared_total_sums_only_the_hosts_carrying_that_class() -> None:
    """Capacity is summed PER CLASS, never across the whole inventory.

    A verify-class runner cannot pick up an action-class job, so counting it in
    the action fleet's declared total would raise the router's degraded floor by
    capacity that can never satisfy it -- the router would read the fleet as
    healthy while the action fleet was short.
    """
    config = _config()

    assert config.declared_total("action") == 60
    assert config.declared_total("verify") >= 1
    # A class nothing declares is zero, not an error and not the whole fleet.
    assert config.declared_total("no-such-class") == 0


def test_host_rows_reject_an_unknown_architecture() -> None:
    raw = yaml.safe_load(FLEET_CONFIG.read_text(encoding="utf-8"))
    raw["hosts"][0]["arch"] = "sparc64"
    with pytest.raises(ValueError):
        ModelRunnerFleetConfig.model_validate(raw)


def test_host_rows_reject_an_empty_class_list() -> None:
    """A host that carries no class is capacity nothing can ever use."""
    raw = yaml.safe_load(FLEET_CONFIG.read_text(encoding="utf-8"))
    raw["hosts"][0]["classes"] = []
    with pytest.raises(ValueError):
        ModelRunnerFleetConfig.model_validate(raw)

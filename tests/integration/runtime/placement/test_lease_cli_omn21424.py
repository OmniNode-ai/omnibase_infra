# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex lease acquire|renew|release|show`` against a three-member etcd (OMN-21424).

Each simulated host talks to its own etcd member, so a grant made through one
host is seen, and refused to a second holder, through another.
"""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from uuid import uuid4

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.handlers.lease.handler_lease_cli import lease_group
from tests.integration.runtime.placement.conftest import EtcdCluster

pytestmark = [pytest.mark.integration]

KINDS = "exclusive:1:30"


def _run(endpoint: str, *args: str) -> Result:
    return CliRunner().invoke(
        lease_group,
        list(args),
        env={"ONEX_LEASE_ETCD_ENDPOINT": endpoint, "ONEX_LEASE_KINDS": KINDS},
        catch_exceptions=False,
    )


def _reply(result: Result) -> dict[str, object]:
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert isinstance(loaded, dict)
    return loaded


def test_the_verb_is_registered_on_the_onex_cli() -> None:
    names = {
        ep.name: ep.value for ep in entry_points(group="onex.cli") if ep.name == "lease"
    }
    assert names == {
        "lease": "omnibase_infra.handlers.lease.handler_lease_cli:lease_group"
    }


def test_acquire_show_renew_release_across_hosts(etcd_cluster: EtcdCluster) -> None:
    host0, host1, host2 = (m.endpoint for m in etcd_cluster.members)
    request_id = str(uuid4())
    acquire = ("acquire", "--kind", "exclusive", "--resource", "pr:cli-omn21424")

    first = _run(host0, *acquire, "--holder", "lane-a", "--request-id", request_id)
    assert first.exit_code == 0, first.output
    granted = _reply(first)
    assert granted["outcome"] == "lease-granted"
    assert int(str(granted["fence"])) > 0

    again = _run(host1, *acquire, "--holder", "lane-a", "--request-id", request_id)
    assert again.exit_code == 0, again.output
    assert _reply(again)["lease_id"] == granted["lease_id"]
    assert _reply(again)["fence"] == granted["fence"]

    other = _run(host1, *acquire, "--holder", "lane-b")
    assert other.exit_code == 3, other.output
    assert _reply(other)["outcome"] == "lease-refused"

    shown = _run(host2, "show", "--kind", "exclusive", "--resource", "pr:cli-omn21424")
    assert shown.exit_code == 0, shown.output
    held = [json.loads(line) for line in shown.stdout.strip().splitlines()]
    assert [(h["lease_id"], h["holder"]) for h in held] == [
        (granted["lease_id"], "lane-a")
    ]

    lease = (
        "--kind",
        "exclusive",
        "--resource",
        "pr:cli-omn21424",
        "--lease-id",
        str(granted["lease_id"]),
    )
    renewed = _run(host2, "renew", *lease)
    assert renewed.exit_code == 0, renewed.output
    assert _reply(renewed)["outcome"] == "lease-renewed"
    assert _reply(renewed)["fence"] == granted["fence"]

    released = _run(host0, "release", *lease)
    assert released.exit_code == 0, released.output
    assert _reply(released)["outcome"] == "lease-released"

    gone = _run(host1, "renew", *lease)
    assert gone.exit_code == 3, gone.output
    assert _reply(gone)["outcome"] == "lease-expired"

    empty = _run(host2, "show", "--kind", "exclusive", "--resource", "pr:cli-omn21424")
    assert empty.exit_code == 0, empty.output
    assert empty.stdout.strip() == ""


def test_an_undeclared_kind_and_a_missing_endpoint_fail_loudly(
    etcd_cluster: EtcdCluster,
) -> None:
    undeclared = _run(
        etcd_cluster.members[0].endpoint,
        "acquire",
        "--kind",
        "nope",
        "--resource",
        "r",
        "--holder",
        "h",
    )
    assert undeclared.exit_code == 2, undeclared.output
    assert "nope" in undeclared.output

    missing = CliRunner().invoke(
        lease_group,
        ["show", "--kind", "exclusive", "--resource", "r"],
        env={"ONEX_LEASE_ETCD_ENDPOINT": "", "ONEX_LEASE_KINDS": KINDS},
    )
    assert missing.exit_code == 2, missing.output
    assert "ONEX_LEASE_ETCD_ENDPOINT" in missing.output

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-repo provisioning-sync pin for the forwarder mirror set (OMN-17201).

WHY THIS FILE EXISTS. ``config.gateway_forwarder.mirror_topics`` in this node's
contract declares which canonical topics cross the tenant gateway boundary. It
does NOT create anything: the physical, TENANT-PREFIXED wire topic is minted by
``omninode_infra/docker/onex-api/kafka_topic_provisioner.py::ensure_tenant_topics``
over ``topic_constants.py::DEFAULT_TENANT_CANONICAL_TOPICS``. That file states
in its own header that it "MUST stay in sync with the P0A forwarder contract
mirror_topics union" -- and until OMN-17201 nothing enforced it.

THE MEASURED COST OF THE UNENFORCED COMMENT. OMN-16204 and OMN-16979 widened
``mirror_topics.outbound`` with the four governed omniclaude hook classes. The
provisioning tuple stayed at eight. ``omninode-dev-msk`` runs
``auto.create.topics.enable=false``, so the four wire topics could not appear
lazily either: on 2026-09-05 the cloud hook-ledger writer crash-looped on
``UnknownTopicOrPartitionError`` against topics that had never been created,
and had to be parked at ``replicas: 0`` (omninode_infra#1164). A broker
readback the same day found all four BARE canonical forms present and all four
tenant-prefixed forms absent.

WHY A PINNED LITERAL LIST AND NOT A CROSS-REPO IMPORT. omninode_infra is not on
this repo's import path in CI, and vendoring its module would make one repo's
tests depend on the other's checkout. Instead BOTH repos pin the same twelve
wire-format literals independently -- this file from the contract's side,
``topic_constants.FORWARDER_MIRROR_TOPIC_UNION`` from the provisioner's side --
so a one-sided edit fails a test in whichever repo made it, which is the
property the comment always claimed and never had.

The pin is deliberately on the UNION (inbound + outbound), not on outbound
alone: a tenant's wire topic has to exist for both directions of the bridge,
and ``ensure_tenant_topics`` provisions one flat set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONTRACT_PATH = (
    Path(__file__).parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)

# The twelve canonical topics that every tenant's wire set must contain.
# Counterpart: omninode_infra docker/onex-api/topic_constants.py
# ``FORWARDER_MIRROR_TOPIC_UNION`` / ``DEFAULT_TENANT_CANONICAL_TOPICS``.
PROVISIONED_TENANT_CANONICAL_TOPICS: frozenset[str] = frozenset(
    {
        "onex.cmd.omnibase-infra.delegation-inference-request.v1",
        "onex.cmd.omnibase-infra.delegation-request.v1",
        "onex.evt.omnibase-infra.gateway-heartbeat.v1",
        "onex.evt.omnibase-infra.inference-response.v1",
        "onex.evt.omnibase-infra.delegation-completed.v1",
        "onex.evt.omnibase-infra.delegation-failed.v1",
        "onex.evt.omniintelligence.llm-call-completed.v1",
        "onex.evt.omnibase-infra.llm-call-completed.v1",
        "onex.evt.omniclaude.session-started.v1",
        "onex.evt.omniclaude.session-ended.v1",
        "onex.evt.omniclaude.tool-executed.v1",
        "onex.evt.omniclaude.prompt-submitted.v1",
    }
)

# The subset OMN-17201 is about: the four governed omniclaude hook classes the
# cloud hook-ledger writer (omnimarket node_projection_hook_ledger) subscribes
# to in their tenant-prefixed form.
HOOK_CLASSES: tuple[str, ...] = (
    "onex.evt.omniclaude.session-started.v1",
    "onex.evt.omniclaude.session-ended.v1",
    "onex.evt.omniclaude.tool-executed.v1",
    "onex.evt.omniclaude.prompt-submitted.v1",
)


def _mirror_topics() -> dict[str, list[str]]:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    return contract["config"]["gateway_forwarder"]["mirror_topics"]


@pytest.mark.unit
def test_mirror_union_equals_the_provisioned_tenant_topic_set() -> None:
    """The whole point: contract union == provisioner set, exactly.

    Set equality in BOTH directions, not containment. A contract topic missing
    from the provisioner is an unprovisionable wire topic (the OMN-17201
    crash-loop). A provisioner topic missing from the contract is a wire topic
    nothing ever writes to or reads from -- broker clutter that reads like
    coverage.
    """
    mirror = _mirror_topics()
    union = set(mirror["inbound"]) | set(mirror["outbound"])
    assert union == set(PROVISIONED_TENANT_CANONICAL_TOPICS)


@pytest.mark.unit
@pytest.mark.parametrize("topic", HOOK_CLASSES)
def test_each_hook_class_is_in_the_provisioned_set(topic: str) -> None:
    """Per-topic proof, so a failure names the class that regressed rather
    than reporting a set difference."""
    mirror = _mirror_topics()
    assert topic in set(mirror["inbound"]) | set(mirror["outbound"])
    assert topic in PROVISIONED_TENANT_CANONICAL_TOPICS


@pytest.mark.unit
def test_pin_is_a_falsifiable_count() -> None:
    """8 (pre-OMN-16204) + 2 (OD-9 pair) + 2 (OMN-16979 governed pair) = 12.

    A count assertion catches the case a set-equality edit would launder: an
    author who "fixes" a failure by editing BOTH sides of the pin at once still
    has to move this number, which is the line a reviewer reads.
    """
    assert len(PROVISIONED_TENANT_CANONICAL_TOPICS) == 12

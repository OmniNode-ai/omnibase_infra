# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18789 AC4: the new broker probe reaches the dev lane and no other.

WHY THIS IS A SEPARATE FILE FROM THE PROBE'S OWN TESTS. The probe being correct
and the probe being correctly SCOPED are two different claims, and the second
one is the one with live blast radius. `docker-compose.infra.yml` is merged by
stability-test and prod; judge and lakshman redeclare `redpanda` wholesale in
their own overlays. A healthcheck edit made in the base file would have flipped
the probe on the STABILITY lane -- the surface the compose path's
`stability-proven` premise is resolved from -- as a side effect, silently.

WHY STATIC YAML AND NOT A RENDER. The paired `docker compose config` render
lives in tests/integration/infra/test_dev_runtime_compose_render.py and needs a
Docker daemon. These assertions need none, so they run on every unit pass and
in every pre-commit. Together they are a complete proof of scope: the base file
is unchanged, no lane overlay but the dev one mentions the probe, and the two
overlays that redeclare `redpanda` still carry the old literal -- so the three
other lanes render byte-identically to what they rendered before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"

#: The healthcheck this change replaces, verbatim. Every lane but dev keeps it.
LEADERLESS_PROBE = "rpk cluster health | grep -q 'Healthy:.*true' || exit 1"

#: What the dev lane runs instead.
PROBE_COMMAND = [
    "CMD",
    "/usr/bin/bash",
    "/usr/local/bin/onex-broker-readiness-probe",
]

#: Compose project -> the overlay that lane merges on top of the base.
OTHER_LANE_OVERLAYS = {
    "omnibase-infra-stability-test": "docker-compose.stability-test.yml",
    "omnibase-infra-judge": "docker-compose.judge.yml",
    "omnibase-infra-lakshman": "docker-compose.lakshman.yml",
}


class _ComposeLoader(yaml.SafeLoader):
    """A SafeLoader that tolerates compose's `!override` and `!!merge` tags.

    This module reads identifiers and literal command strings, never merge
    semantics, so dropping the tag and keeping the value is correct here.
    """


def _drop_tag(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    # Unreachable for any YAML this repo contains. Refuse rather than return
    # None: an unparsed healthcheck reading as absent is exactly the false
    # "this lane is untouched" this file exists to prevent.
    raise AssertionError(f"unhandled YAML node kind: {type(node).__name__}")


_ComposeLoader.add_multi_constructor("!", _drop_tag)  # type: ignore[no-untyped-call]
_ComposeLoader.add_multi_constructor(  # type: ignore[no-untyped-call]
    "tag:yaml.org,2002:", _drop_tag
)


def _load(name: str) -> Any:
    with (DOCKER_DIR / name).open() as handle:
        return yaml.load(handle, Loader=_ComposeLoader)  # noqa: S506 - local tolerant loader


def test_the_dev_lane_overlay_carries_the_probe() -> None:
    redpanda = _load("docker-compose.dev-lane.yml")["services"]["redpanda"]
    assert redpanda["healthcheck"]["test"] == PROBE_COMMAND
    mounts = {entry.split(":")[1] for entry in redpanda["volumes"]}
    assert "/usr/local/bin/onex-broker-readiness-probe" in mounts
    assert "/etc/onex/broker_readiness_declaration.conf" in mounts, (
        "the probe fails closed without its declaration, so an unmounted "
        "declaration is a lane that never goes healthy"
    )


def test_the_dev_lane_mounts_both_files_read_only() -> None:
    redpanda = _load("docker-compose.dev-lane.yml")["services"]["redpanda"]
    for entry in redpanda["volumes"]:
        assert entry.endswith(":ro"), (
            f"{entry}: the broker must not be able to rewrite its own probe"
        )


def test_the_dev_lane_probe_takes_no_credential_on_its_command_line() -> None:
    """`ps` is readable by every process in the container; argv is not private."""
    redpanda = _load("docker-compose.dev-lane.yml")["services"]["redpanda"]
    rendered = " ".join(redpanda["healthcheck"]["test"])
    for leak in ("PASSWORD", "pass=", "RPK_PASS"):
        assert leak not in rendered


def test_the_base_file_still_carries_the_leaderless_probe() -> None:
    """stability-test and prod inherit this one and must keep inheriting it.

    Editing the base is the tempting one-line version of this change and it is
    the wrong one: it would move the STABILITY lane's probe, which is the lane
    the compose path's `stability-proven` premise is resolved from.
    """
    redpanda = _load("docker-compose.infra.yml")["services"]["redpanda"]
    assert redpanda["healthcheck"]["test"] == ["CMD-SHELL", LEADERLESS_PROBE]


@pytest.mark.parametrize(("project", "overlay"), sorted(OTHER_LANE_OVERLAYS.items()))
def test_no_other_lane_is_reached_by_this_change(project: str, overlay: str) -> None:
    services = _load(overlay)["services"]
    redpanda = services.get("redpanda", {})

    declared = redpanda.get("healthcheck")
    if declared is not None:
        assert declared["test"] == ["CMD-SHELL", LEADERLESS_PROBE], (
            f"{project}: its own overlay's broker healthcheck changed. This "
            "ticket's scope is the dev lane."
        )

    raw = (DOCKER_DIR / overlay).read_text()
    assert "onex-broker-readiness-probe" not in raw, (
        f"{project}: the OMN-18789 probe leaked into a non-dev overlay"
    )
    assert "broker_readiness_declaration" not in raw, (
        f"{project}: the OMN-18789 declaration leaked into a non-dev overlay"
    )


def test_the_probe_is_absent_from_the_base_every_other_lane_merges() -> None:
    raw = (DOCKER_DIR / "docker-compose.infra.yml").read_text()
    assert "onex-broker-readiness-probe" not in raw
    assert "broker_readiness_declaration" not in raw

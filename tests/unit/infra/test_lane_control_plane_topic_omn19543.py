# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19543: every lane built on the dev composition creates onex.tenant.events.

``onex.tenant.events`` is a control-plane topic owned by onex-api
(``topic_constants.TENANT_EVENTS_TOPIC`` in the onex-api image). onex-api
creates it before its first publish. On the .201 dev lane onex-api runs, so the
topic exists there. The instance lanes built on the dev composition (dev-202,
and dev-200 and dev-105 after them) disable onex-api, because its image has no
delivery path to those hosts. Nothing else creates the topic: the runtime's
topic provisioner creates contract-declared ONEX-shaped topics only, and a
consumer subscribe does not create one. So ``projection-tenant-registry-writer``
subscribed to a topic that did not exist, and it was unhealthy on dev-200 and
restarting on dev-202 (dev-200-lane TERMINAL, omni_home ledger 2026-09-25).

The fix creates the topic in the dev lane's override of
``redpanda-partition-cap``. That one-shot runs on a cold ``up`` through
depends_on and on every warm redeploy (scripts/deploy-runtime.sh
warm_broker_topic_provisioning). Every instance overlay inherits the command,
because none of them overrides it. These tests run the real script against a
fake ``rpk`` and pin:

* an absent topic is created with one partition, as onex-api creates it;
* an existing topic is left alone (the .201 dev lane, where onex-api created it);
* a create that loses a race to another creator still passes, because the
  read-back finds the topic;
* a create that fails with the topic still absent fails the one-shot (fail
  closed), so the lane does not come up with a writer on a missing topic;
* the writer waits for the one-shot, and the instance overlays inherit it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_DOCKER = Path(__file__).resolve().parents[3] / "docker"
_DEV_LANE = _DOCKER / "docker-compose.dev-lane.yml"
_TOPIC = "onex.tenant.events"
_WRITER = "projection-tenant-registry-writer"
_ONESHOT = "redpanda-partition-cap"

# A fake rpk. State is one topic name per line in $FAKE_RPK_STATE. Every call
# is appended to $FAKE_RPK_LOG so a test can read what the script asked for.
_FAKE_RPK = r"""#!/bin/sh
echo "$*" >> "$FAKE_RPK_LOG"
case "$1 $2" in
  "cluster config")
    case "$3" in
      set) exit 0 ;;
      get)
        case "$4" in
          enable_sasl) echo false ;;
          segment_fallocation_step) echo 1048576 ;;
          log_segment_ms) echo 86400000 ;;
          retention_bytes) echo 1073741824 ;;
          *) echo "unexpected config get $4" >&2; exit 2 ;;
        esac
        exit 0 ;;
    esac ;;
  "topic list")
    echo "NAME PARTITIONS REPLICAS"
    while read -r t; do [ -n "$t" ] && echo "$t 1 1"; done < "$FAKE_RPK_STATE"
    exit 0 ;;
  "topic create")
    if [ "${FAKE_RPK_CREATE_RACE:-0}" = "1" ]; then
      echo "$3" >> "$FAKE_RPK_STATE"
      echo "TOPIC_ALREADY_EXISTS: $3" >&2
      exit 1
    fi
    if [ "${FAKE_RPK_CREATE_FAILS:-0}" = "1" ]; then
      echo "create refused: $3" >&2
      exit 1
    fi
    echo "$3" >> "$FAKE_RPK_STATE"
    exit 0 ;;
esac
echo "unexpected rpk call: $*" >&2
exit 2
"""


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Passthrough constructor for Docker Compose `!override` / `!reset` tags."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that unwraps compose override tags."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)


def _services(path: Path) -> dict[str, Any]:
    compose: dict[str, Any] = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_ComposeLoader,  # noqa: S506
    )
    services: dict[str, Any] = compose["services"]
    return services


def _run_oneshot(
    tmp_path: Path, existing: list[str], **env_flags: str
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    command = _services(_DEV_LANE)[_ONESHOT]["command"]
    # Compose turns `$$` into a literal `$` before the shell sees it.
    script = "\n".join(str(c) for c in command).replace("$$", "$")
    # OMN-19731: the oneshot's scratch files (/tmp/topics.txt, /tmp/cfg.txt)
    # are fixed /tmp paths, private inside its container but shared by every
    # xdist worker on a CI host, so parallel tests read each other's listing.
    # Rewrite these first because tmp_path may itself be under /tmp.
    # Give each test its own copy of every one.
    # The /tmp literals below MATCH the script's paths; nothing opens them.
    container_tmp = r"/tmp/([A-Za-z0-9_.-]+)"  # noqa: S108
    script = re.sub(container_tmp, lambda m: str(tmp_path / m.group(1)), script)
    leftover = script.replace(str(tmp_path), "")
    assert "/tmp/" not in leftover, (  # noqa: S108
        "oneshot script still uses a fixed /tmp path shared across xdist workers"
    )
    fake = tmp_path / "rpk"
    fake.write_text(_FAKE_RPK, encoding="utf-8")
    fake.chmod(0o755)
    script = script.replace("/usr/bin/rpk", str(fake))
    state = tmp_path / "topics"
    state.write_text("".join(f"{t}\n" for t in existing), encoding="utf-8")
    log = tmp_path / "calls"
    log.write_text("", encoding="utf-8")
    sh = shutil.which("sh")
    assert sh is not None
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "FAKE_RPK_STATE": str(state),
        "FAKE_RPK_LOG": str(log),
        "DEV_KAFKA_SASL_USERNAME": "unused",
        "DEV_KAFKA_SASL_PASSWORD": "unused",
        **env_flags,
    }
    result = subprocess.run(
        [sh, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    topics = [t for t in state.read_text(encoding="utf-8").splitlines() if t]
    calls = log.read_text(encoding="utf-8").splitlines()
    return result, topics, calls


def test_an_absent_topic_is_created_with_one_partition(tmp_path: Path) -> None:
    result, topics, calls = _run_oneshot(tmp_path, existing=["onex.evt.x.y.v1"])
    assert result.returncode == 0, result.stderr
    assert topics.count(_TOPIC) == 1
    assert f"topic create {_TOPIC} -p 1" in calls


def test_an_existing_topic_is_left_alone(tmp_path: Path) -> None:
    result, topics, calls = _run_oneshot(tmp_path, existing=[_TOPIC])
    assert result.returncode == 0, result.stderr
    assert topics.count(_TOPIC) == 1
    assert not [c for c in calls if c.startswith("topic create")]


def test_a_create_that_loses_a_race_still_passes(tmp_path: Path) -> None:
    result, topics, _ = _run_oneshot(tmp_path, existing=[], FAKE_RPK_CREATE_RACE="1")
    assert result.returncode == 0, result.stderr
    assert _TOPIC in topics


def test_a_failed_create_with_the_topic_absent_fails_closed(tmp_path: Path) -> None:
    result, topics, _ = _run_oneshot(tmp_path, existing=[], FAKE_RPK_CREATE_FAILS="1")
    assert result.returncode != 0
    assert _TOPIC not in topics
    assert "OMN-19543" in result.stderr


def test_the_tenant_registry_writer_waits_for_the_oneshot() -> None:
    depends_on = _services(_DEV_LANE)[_WRITER]["depends_on"]
    assert depends_on[_ONESHOT] == {"condition": "service_completed_successfully"}


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in _DOCKER.glob("docker-compose.dev-[0-9]*.yml")),
)
def test_instance_overlays_inherit_the_topic_step(overlay: str) -> None:
    services = _services(_DOCKER / overlay)
    assert "command" not in services.get(_ONESHOT, {}), (
        f"{overlay} overrides the {_ONESHOT} command, so its broker would not "
        f"get {_TOPIC}; add the OMN-19543 step to the override"
    )
    assert "depends_on" not in services.get(_WRITER, {}), (
        f"{overlay} replaces the {_WRITER} depends_on and drops the wait on {_ONESHOT}"
    )

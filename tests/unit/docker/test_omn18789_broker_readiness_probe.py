# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18789: the dev-lane broker's readiness verdict comes from a real read.

WHAT THIS PROTECTS
------------------
On 2026-09-18 the ``.201`` dev lane's broker left partition reconciliation stuck
for seventeen minutes. In the 23:29:30Z-23:34:30Z slice alone it logged 5921
``replicated_partition.cc:596 ... error obtaining latest start offset -
{ error_code: not_leader_for_partition [6] }`` lines across essentially every
topic, and a wedged consumer group held ``CURRENT-OFFSET`` 7473 against
``LOG-END-OFFSET`` 7478 byte-identical across two samples. Every liveness
surface anyone watches reported healthy for the whole 97 minutes:

* ``docker ps`` -- ``Up 26 minutes (healthy)``, ``RestartCount`` 0. The
  container healthcheck was ``rpk cluster health | grep -q 'Healthy:.*true'``,
  an ADMIN-API call that never touches the Kafka data path.
* ``GET :9644/v1/cluster/health_overview`` -- ``is_healthy true``,
  ``leaderless_partitions []``, ``leaderless_count 0``. That answer is not even
  wrong on its own terms: on a single-node cluster every partition nominally
  has node 0 as its leader. It simply says nothing about whether a READ against
  that leader returns, which is the only question a client asks.

So the two assertions this file makes are:

1. the verdict is derived from an offset read actually returning numbers, and
2. a consumer group that has committed before and has stopped committing while
   it is behind is NOT READY once it has been frozen longer than the declared
   window.

WHY A STUB ``rpk`` AND NOT A MOCKED PYTHON FUNCTION
---------------------------------------------------
The thing that has to be right is the command Docker runs, in the image Docker
runs it in. ``redpandadata/redpanda:v24.2.7`` has bash, awk, sed, grep, curl and
rpk; it has NO python3 (measured in-container 2026-09-19), which is why the
probe is a bash script and why testing it means running that script. Each test
puts a fixture-driven ``rpk`` first on ``PATH`` and executes the real file, so
what is under test is the healthcheck itself rather than a description of it.

``test_the_pre_change_healthcheck_passes_on_the_outage_stub`` is the positive
control that keeps the rest honest: it runs the command this change REPLACES
against the same stub and asserts it exits 0. Without it, a stub that simply
refused everything would make every assertion below pass for the wrong reason.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = REPO_ROOT / "docker" / "redpanda" / "broker_readiness_probe.sh"
DECLARATION = REPO_ROOT / "docker" / "redpanda" / "broker_readiness_declaration.conf"

#: The pre-OMN-18789 container healthcheck, verbatim from
#: ``docker/docker-compose.infra.yml`` service ``redpanda``.
PRE_CHANGE_HEALTHCHECK = "rpk cluster health | grep -q 'Healthy:.*true' || exit 1"

# ---------------------------------------------------------------------------
# rpk stubs. Each is a bash script dropped on PATH ahead of anything real.
# ---------------------------------------------------------------------------

#: ``rpk cluster health`` / ``health_overview`` as the broker actually answered
#: at 23:29:30Z, paired with the offset read as it actually failed in the same
#: second. This is the outage, reproduced.
STUB_OUTAGE = r"""#!/usr/bin/env bash
case "$1 $2" in
  "cluster health")
    echo "CLUSTER HEALTH OVERVIEW"
    echo "======================="
    echo "Healthy:                     true"
    echo "Unhealthy reasons:           []"
    echo "Leaderless partitions:       []"
    echo "Under-replicated partitions: []"
    exit 0 ;;
  "topic list")
    echo "NAME                PARTITIONS  REPLICAS"
    echo "__consumer_offsets  16          1"
    exit 0 ;;
  "topic describe")
    echo "unable to request metadata: ntp {kafka/__consumer_offsets/0}: error obtaining latest start offset - { error_code: not_leader_for_partition [6] }" >&2
    exit 1 ;;
  "group describe")
    exit 0 ;;
esac
exit 0
"""

#: A broker serving offsets normally, with one consumer group in the exact
#: shape the ticket records: Stable, one member, CURRENT-OFFSET 7473 against
#: LOG-END-OFFSET 7478, lag 5, unchanged between samples.
STUB_FROZEN_GROUP = r"""#!/usr/bin/env bash
case "$1 $2" in
  "cluster health")
    echo "Healthy:                     true"; exit 0 ;;
  "topic list")
    echo "NAME                                       PARTITIONS  REPLICAS"
    echo "__consumer_offsets                         16          1"
    echo "onex.cmd.omnimarket.occ-autobind.v1        1           1"
    exit 0 ;;
  "topic describe")
    echo "PARTITION  LEADER  EPOCH  REPLICAS  LOG-START-OFFSET  HIGH-WATERMARK"
    echo "0          0       179    [0]       0                 28101324"
    exit 0 ;;
  "group describe")
    cat <<'EOF'
GROUP        local.omnimarket.pr_lifecycle_fix_effect.consume.1.0.0
COORDINATOR  0
STATE        Stable
BALANCER     roundrobin
MEMBERS      1
TOTAL-LAG    5

TOPIC                                PARTITION  CURRENT-OFFSET  LOG-START-OFFSET  LOG-END-OFFSET  LAG  MEMBER-ID              CLIENT-ID        HOST
onex.cmd.omnimarket.occ-autobind.v1  0          7473            0                 7478            5    aiokafka-0.13.0-becff  aiokafka-0.13.0  172.19.0.20
EOF
    exit 0 ;;
esac
exit 0
"""

#: The positive control for the group check: byte-identical to
#: ``STUB_FROZEN_GROUP`` except that the group is draining -- CURRENT-OFFSET has
#: advanced 7473 -> 7476 between the two samples. Without this pair the frozen
#: assertion could be satisfied by a probe that simply always fails.
STUB_DRAINING_GROUP = (
    STUB_FROZEN_GROUP.replace("7473", "7476")
    .replace("TOTAL-LAG    5", "TOTAL-LAG    2")
    .replace("7478            5 ", "7478            2 ")
)

#: A broker with nothing wrong: offsets read, and the only group with lag has
#: never committed at all (every CURRENT-OFFSET is ``-``). That is the
#: ``snapshot-cache`` consumer measured live on the dev lane 2026-09-19 -- it
#: tails a topic without ever committing, so its lag grows without bound
#: forever. A probe that called that NOT READY would report the healthy lane
#: broken permanently, which is the failure this control pins.
STUB_NEVER_COMMITTED = r"""#!/usr/bin/env bash
case "$1 $2" in
  "cluster health") echo "Healthy:                     true"; exit 0 ;;
  "topic list")
    echo "NAME                PARTITIONS  REPLICAS"
    echo "__consumer_offsets  16          1"
    exit 0 ;;
  "topic describe")
    echo "PARTITION  LEADER  EPOCH  REPLICAS  LOG-START-OFFSET  HIGH-WATERMARK"
    echo "0          0       179    [0]       0                 28101324"
    exit 0 ;;
  "group describe")
    cat <<'EOF'
GROUP        local.omnimarket-projection-api.snapshot-cache.consume.v1
COORDINATOR  0
STATE        Stable
BALANCER     roundrobin
MEMBERS      1
TOTAL-LAG    11023336

TOPIC                                  PARTITION  CURRENT-OFFSET  LOG-START-OFFSET  LOG-END-OFFSET  LAG       MEMBER-ID  CLIENT-ID        HOST
onex.snapshot.projection.live-events.v1  0        -               0                 11023336        11023336  aiokafka-x  aiokafka-0.13.0  172.19.0.21
EOF
    exit 0 ;;
esac
exit 0
"""

#: A broker that has formed but carries no topics at all -- a fresh
#: ``redpanda_data`` volume, before the lane's first topic is created. There is
#: no partition to read, so there is nothing to prove broken.
STUB_EMPTY_CLUSTER = r"""#!/usr/bin/env bash
case "$1 $2" in
  "cluster health") echo "Healthy:                     true"; exit 0 ;;
  "topic list") echo "NAME  PARTITIONS  REPLICAS"; exit 0 ;;
  "group describe") exit 0 ;;
esac
exit 0
"""

#: Nothing answers on the Kafka port at all.
STUB_UNREACHABLE = r"""#!/usr/bin/env bash
echo "unable to request metadata: dial tcp 172.19.0.2:9092: connect: connection refused" >&2
exit 1
"""


class ProbeRunner(Protocol):
    """Runs the real probe file against a stub ``rpk`` placed first on PATH."""

    def __call__(
        self,
        stub: str,
        *,
        now: int,
        state: Path | None = ...,
        declaration: Path | None = ...,
    ) -> subprocess.CompletedProcess[str]: ...


@pytest.fixture
def probe_env(tmp_path: Path) -> ProbeRunner:
    """Return a runner that executes the real probe against a stub ``rpk``."""

    def _run(
        stub: str,
        *,
        now: int,
        state: Path | None = None,
        declaration: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        rpk = bindir / "rpk"
        rpk.write_text(stub)
        rpk.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
        return subprocess.run(
            [
                "bash",
                str(PROBE),
                "--declaration",
                str(declaration or DECLARATION),
                "--state",
                str(state or (tmp_path / "state")),
                "--now",
                str(now),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )

    return _run


def _declared(key: str) -> str:
    for line in DECLARATION.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{key} is not declared in {DECLARATION}")


# ---------------------------------------------------------------------------
# AC2 -- the verdict comes from a partition read, not from leaderless_count
# ---------------------------------------------------------------------------


def test_pre_change_healthcheck_is_green_on_the_outage_stub(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    rpk = bindir / "rpk"
    rpk.write_text(STUB_OUTAGE)
    rpk.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    result = subprocess.run(
        ["bash", "-c", PRE_CHANGE_HEALTHCHECK],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        "the pre-change healthcheck must be GREEN on the outage stub -- that is "
        "the defect this ticket exists to close. If it is red, the stub no "
        f"longer reproduces the outage.\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_probe_is_not_ready_when_the_offset_read_fails(probe_env: ProbeRunner) -> None:
    """AC2. ``cluster health`` says Healthy; the offset read says not_leader."""
    result = probe_env(STUB_OUTAGE, now=1_000_000)
    assert result.returncode != 0, (
        "the probe must be NOT READY when an offset read returns "
        f"not_leader_for_partition.\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "partition_read_failed" in (result.stdout + result.stderr)


def test_probe_never_consults_cluster_health_or_leaderless_count() -> None:
    """AC2, structurally: a leaderless_count verdict cannot be reintroduced.

    Reverting to the pre-change verdict means reading ``rpk cluster health`` or
    the admin ``health_overview``. Neither string may appear as a command the
    probe runs.
    """
    source = PROBE.read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    for forbidden in ("cluster health", "health_overview", "leaderless"):
        assert forbidden not in code, (
            f"{forbidden!r} appears in executable probe source. The verdict is "
            "derived from a read that succeeds, never from the broker's own "
            "opinion of its leadership."
        )


def test_probe_is_not_ready_when_the_broker_is_unreachable(
    probe_env: ProbeRunner,
) -> None:
    result = probe_env(STUB_UNREACHABLE, now=1_000_000)
    assert result.returncode != 0
    assert "broker_unreachable" in (result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# AC1 -- a group that has stopped committing while behind is NOT READY
# ---------------------------------------------------------------------------


def test_frozen_group_is_not_ready_once_the_window_elapses(
    probe_env: ProbeRunner, tmp_path: Path
) -> None:
    """AC1. Two samples, ``window + 1`` seconds apart, offset unchanged."""
    window = int(_declared("sync_window_seconds"))
    state = tmp_path / "state"

    first = probe_env(STUB_FROZEN_GROUP, now=1_000_000, state=state)
    assert first.returncode == 0, (
        "the first sample establishes the observation; it cannot be NOT READY, "
        f"or a booting lane never goes healthy.\nstderr={first.stderr}"
    )

    second = probe_env(STUB_FROZEN_GROUP, now=1_000_000 + window + 1, state=state)
    assert second.returncode != 0, (
        "a Stable one-member group at CURRENT-OFFSET 7473 / LOG-END-OFFSET 7478 "
        "unchanged across the declared window must be NOT READY.\n"
        f"stdout={second.stdout}\nstderr={second.stderr}"
    )
    assert "group_not_synced" in (second.stdout + second.stderr)


def test_frozen_group_is_still_ready_inside_the_window(
    probe_env: ProbeRunner, tmp_path: Path
) -> None:
    """The window is a bound, not a formality: one second short must pass."""
    window = int(_declared("sync_window_seconds"))
    state = tmp_path / "state"
    probe_env(STUB_FROZEN_GROUP, now=1_000_000, state=state)
    inside = probe_env(STUB_FROZEN_GROUP, now=1_000_000 + window - 1, state=state)
    assert inside.returncode == 0, (
        f"NOT READY before the declared window elapsed.\nstderr={inside.stderr}"
    )


def test_draining_group_is_ready_across_the_same_interval(
    probe_env: ProbeRunner, tmp_path: Path
) -> None:
    """AC1's positive control. Same shape, offset advancing -> READY.

    Without this the frozen assertion above is satisfied by a probe that fails
    unconditionally.
    """
    window = int(_declared("sync_window_seconds"))
    state = tmp_path / "state"
    probe_env(STUB_FROZEN_GROUP, now=1_000_000, state=state)
    advanced = probe_env(STUB_DRAINING_GROUP, now=1_000_000 + window + 1, state=state)
    assert advanced.returncode == 0, (
        "a group whose CURRENT-OFFSET advanced between the samples is draining "
        f"and must be READY.\nstdout={advanced.stdout}\nstderr={advanced.stderr}"
    )


def test_never_committed_group_is_ready_forever(
    probe_env: ProbeRunner, tmp_path: Path
) -> None:
    """Measured live on the dev lane: the snapshot-cache consumer never commits.

    Its lag grows without bound by design. It has no committed offset to
    freeze, so it is outside what this probe can measure -- and calling it NOT
    READY would report a healthy lane broken permanently.
    """
    window = int(_declared("sync_window_seconds"))
    state = tmp_path / "state"
    probe_env(STUB_NEVER_COMMITTED, now=1_000_000, state=state)
    later = probe_env(STUB_NEVER_COMMITTED, now=1_000_000 + (window * 10), state=state)
    assert later.returncode == 0, (
        "a group that has never committed has no sync to measure.\n"
        f"stdout={later.stdout}\nstderr={later.stderr}"
    )


def test_empty_cluster_is_ready(probe_env: ProbeRunner) -> None:
    """A fresh volume has no partition to read, so nothing is proven broken.

    ``redpanda-scram-user`` gates on ``service_healthy``, so a probe that failed
    closed here would make a fresh lane undeployable.
    """
    result = probe_env(STUB_EMPTY_CLUSTER, now=1_000_000)
    assert result.returncode == 0, (
        f"a topic-less cluster must be READY.\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# The declaration is the source of the bound -- there is no env fallback
# ---------------------------------------------------------------------------


def test_declaration_declares_the_window_and_the_probe_topic() -> None:
    assert int(_declared("sync_window_seconds")) > 0
    assert _declared("probe_topic")
    assert _declared("broker")


def test_probe_fails_closed_on_a_missing_declaration(
    probe_env: ProbeRunner, tmp_path: Path
) -> None:
    result = probe_env(
        STUB_FROZEN_GROUP, now=1_000_000, declaration=tmp_path / "absent.conf"
    )
    assert result.returncode != 0
    assert "declaration_unreadable" in (result.stdout + result.stderr)


def test_probe_has_no_environment_fallback_for_the_window() -> None:
    """Operating Rule 8: a silent default is how a wrong bound ships unseen."""
    source = PROBE.read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "sync_window_seconds:-" not in code
    assert "SYNC_WINDOW_SECONDS:-" not in code
    assert "${SYNC_WINDOW_SECONDS:=" not in code


def test_probe_never_places_the_password_on_a_command_line() -> None:
    """`ps` is world-readable in a container; argv is not a secret channel."""
    source = PROBE.read_text()
    assert "-X pass=$" not in source and "-X pass=" not in source, (
        "credentials reach rpk through RPK_PASS/RPK_USER in the environment, "
        "never through argv"
    )


#: `rpk topic list` hides internal topics; `rpk topic list -i` shows them. This
#: stub reproduces that difference exactly, so a probe that drops the flag
#: silently reads some other topic and this test catches it.
STUB_INTERNAL_TOPIC_HIDDEN = r"""#!/usr/bin/env bash
if [ "$1 $2" = "topic list" ]; then
  echo "NAME                PARTITIONS  REPLICAS"
  echo "_schemas            1           1"
  if [ "${3:-}" = "-i" ]; then
    echo "__consumer_offsets  16          1"
  fi
  exit 0
fi
if [ "$1 $2" = "topic describe" ]; then
  echo "PARTITION  LEADER  EPOCH  REPLICAS  LOG-START-OFFSET  HIGH-WATERMARK"
  echo "0          0       179    [0]       0                 28101324"
  exit 0
fi
exit 0
"""


def test_probe_reads_the_declared_internal_topic_not_the_first_visible_one(
    probe_env: ProbeRunner,
) -> None:
    """Regression, found on the live dev lane 2026-09-19 before this shipped.

    ``rpk topic list`` omits internal topics, so the first draft never found
    ``__consumer_offsets`` and fell through to whatever topic happened to be
    listed first -- ``_schemas`` on that lane. The read still succeeded, so the
    probe looked correct while asking about a topic nobody declared. The
    declared target is the one the outage actually left stuck; reading a
    different one is how a probe goes green on the wrong question, which is the
    entire defect this ticket is about.
    """
    result = probe_env(STUB_INTERNAL_TOPIC_HIDDEN, now=1_000_000)
    assert result.returncode == 0, result.stderr
    assert _declared("probe_topic") in result.stdout, (
        "the probe read a topic other than the declared one -- it is not "
        "listing internal topics.\nstdout=" + result.stdout
    )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execute the catalog-local consumer-group probe step against a stub broker (OMN-19496, OMN-20820).

The ``runtime-boot (mode=catalog-local)`` job in
``.github/workflows/reusable-runtime-boot.yml`` has a step that reads
``rpk group list`` and ``rpk group describe`` to prove a live consumer group on
the laptop broker. Until omnibase_infra#5261 it probed the
``node_delegate_skill_orchestrator`` group on the delegate-skill command topic;
that PR removed it, because the local bundle no longer runs an omnimarket
orchestrator. OMN-20820 restores the probe, under this same file name, on a
group omnibase_infra's own runtime opens: its dynamic contract listener on the
node-registration topic.

The probe's first CI run (job 107881660050) failed with exit code 141 within
120 ms of starting, while the runtime was healthy and the group was present:
under ``set -euo pipefail`` a reader that stops early (``awk ... {exit}``,
``head -12``) sends SIGPIPE to the writer still producing output, and pipefail
turns the writer's 141 into the step's exit status.

These tests run the step's own ``run:`` text, unmodified, with the step's own
``env:`` and with ``docker`` and ``sleep`` replaced on PATH by stubs, so the
proof is the shipped script and not a paraphrase of it. The stub broker prints
the listed groups first and then far more than one pipe buffer of further rows,
which is what makes an early reader send SIGPIPE deterministically.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "reusable-runtime-boot.yml"
)
_JOB = "boot-catalog-local"
_STEP = "Runtime dynamic contract listener has a live consumer group"
_PROJECT = "omnibase-infra-local"
_TOPIC = "onex.evt.platform.node-registration.v1"
_LISTENER = "local.omnibase_infra.runtime_config-dynamic-contract-listener"
_GROUP = f"{_LISTENER}.consume.v1.__t.{_TOPIC}"

#: Rows printed after the interesting ones. At ~60 bytes a row this is several
#: times the 64 KiB Linux pipe buffer, so a reader that exits early always
#: leaves the writer blocked on a closed pipe.
_FILLER_ROWS = 6000

_DOCKER_STUB = """#!/usr/bin/env bash
# Stub of `docker exec [-e VAR]... <broker> rpk group list|describe` for the
# probe step. The broker requires SASL (OMN-20820): a call that does not carry
# the superuser's credentials is refused, as the real one is.
#
# STUB_GROUPS holds one "<group> <members>" pair per line: each is listed as
# Stable, and describe reports its member count.
set -u
# A CLI that reads stdin would swallow whatever the caller is looping over, so
# the stub drains it: the step must give docker no stdin it still needs.
cat > /dev/null
[ "$1" = "exec" ] || { echo "stub docker: unexpected $*" >&2; exit 2; }
shift
# docker exec hands the container only the variables -e names. This stub IS the
# docker process, so it drops its own copies first: a credential the step sets
# for docker but forgets to name with -e is then missing, as it is for real.
caller_user="${RPK_USER:-}"
caller_pass="${RPK_PASS:-}"
caller_mechanism="${RPK_SASL_MECHANISM:-}"
unset RPK_USER RPK_PASS RPK_SASL_MECHANISM
while [ "$1" = "-e" ]; do
  case "$2" in
    *=*) export "$2" ;;
    RPK_USER) export RPK_USER="$caller_user" ;;
    RPK_PASS) export RPK_PASS="$caller_pass" ;;
    RPK_SASL_MECHANISM) export RPK_SASL_MECHANISM="$caller_mechanism" ;;
    *) export "$2=$(printenv "$2")" ;;
  esac
  shift 2
done
shift # the container
if [ "${RPK_USER:-}" != "onex-local" ] || [ "${RPK_PASS:-}" != "$STUB_PASSWORD" ] \\
    || [ "${RPK_SASL_MECHANISM:-}" != "SCRAM-SHA-256" ]; then
  echo "unable to request metadata: SASL is required but not provided" >&2
  exit 1
fi
if [ "$1 $2 $3" = "rpk group list" ]; then
  echo "BROKER  GROUP  STATE"
  while read -r name members; do
    [ -n "$name" ] && echo "0  $name  Stable"
  done <<< "$STUB_GROUPS"
  i=0
  while [ "$i" -lt "$STUB_FILLER_ROWS" ]; do
    echo "0  omnibase_infra.node_filler_$i.consume.v1.__t.onex.evt.filler.v1  Stable"
    i=$((i + 1))
  done
  exit 0
fi
if [ "$1 $2 $3" = "rpk group describe" ] && [ $# -eq 4 ]; then
  # one transient failure, as when a group vanishes between list and describe
  if [ -n "${STUB_FLAKY_DESCRIBE:-}" ] && [ ! -e "$STUB_FLAKY_DESCRIBE" ]; then
    : > "$STUB_FLAKY_DESCRIBE"
    echo "unable to describe group $4: GROUP_ID_NOT_FOUND" >&2
    exit 1
  fi
  while read -r name members; do
    if [ "$name" = "$4" ]; then
      echo "GROUP        $name"
      echo "COORDINATOR  0"
      echo "STATE        Stable"
      echo "BALANCER     range"
      echo "MEMBERS      $members"
      echo "TOTAL-LAG    0"
      echo ""
      echo "TOPIC  PARTITION  CURRENT-OFFSET  LOG-START-OFFSET  LOG-END-OFFSET  LAG"
      i=0
      while [ "$i" -lt "$STUB_FILLER_ROWS" ]; do
        echo "$STUB_TOPIC  $i  0  0  0  0"
        i=$((i + 1))
      done
      exit 0
    fi
  done <<< "$STUB_GROUPS"
  echo "unable to describe group $4: GROUP_ID_NOT_FOUND" >&2
  exit 1
fi
echo "stub docker: unexpected exec $*" >&2
exit 2
"""


def _probe_step() -> dict[str, Any]:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][_JOB]["steps"]
    matches = [s for s in steps if s.get("name") == _STEP]
    assert len(matches) == 1, f"expected exactly one step named {_STEP!r}"
    return matches[0]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_probe(
    tmp_path: Path,
    groups: list[tuple[str, int]],
    *,
    env_text: str = "REDPANDA_SUPERUSER_PASSWORD=stub-secret\n",
    flaky_describe: bool = False,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "docker", _DOCKER_STUB)
    # The step retries 30 times with `sleep 10`; a no-op sleep keeps the
    # negative controls fast without changing which branch the script takes.
    _write_executable(bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    step = _probe_step()
    script = tmp_path / "probe.sh"
    script.write_text(step["run"], encoding="utf-8")

    env_file = tmp_path / "local.env"
    env_file.write_text(env_text, encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "LOCAL_ENV_FILE": str(env_file),
        "STUB_PASSWORD": "stub-secret",
        "LOCAL_PROJECT": _PROJECT,
        "STUB_GROUPS": "\n".join(f"{name} {members}" for name, members in groups),
        "STUB_TOPIC": _TOPIC,
        "STUB_FILLER_ROWS": str(_FILLER_ROWS),
    }
    if flaky_describe:
        env["STUB_FLAKY_DESCRIBE"] = str(tmp_path / "describe-failed-once")
    env.update(step.get("env", {}))
    # GitHub runs a `run:` block as `bash -e {0}`; the script sets pipefail itself.
    return subprocess.run(
        ["bash", "-e", str(script)],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _assert_no_live_group(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-2000:]
    assert "no live dynamic-contract-listener consumer group" in result.stdout


def test_probe_passes_on_a_live_group_despite_early_readers(tmp_path: Path) -> None:
    """The job-107881660050 shape: group present, one member, long listings."""
    result = _run_probe(tmp_path, [(_GROUP, 1)])
    assert result.returncode == 0, (
        f"exit {result.returncode}\nstdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert f"live consumer group {_GROUP} (1 member) on {_TOPIC}" in result.stdout


def test_probe_fails_when_no_group_owns_the_topic(tmp_path: Path) -> None:
    _assert_no_live_group(_run_probe(tmp_path, []))


def test_probe_fails_when_the_group_has_no_member(tmp_path: Path) -> None:
    _assert_no_live_group(_run_probe(tmp_path, [(_GROUP, 0)]))


def test_probe_fails_when_the_listener_group_is_on_another_topic(
    tmp_path: Path,
) -> None:
    other = f"{_LISTENER}.consume.v1.__t.onex.evt.platform.node-heartbeat.v1"
    _assert_no_live_group(_run_probe(tmp_path, [(other, 1)]))


def test_probe_fails_when_another_node_owns_the_topic(tmp_path: Path) -> None:
    other = (
        f"local.omnibase_infra.node_registration_orchestrator.consume.v1.__t.{_TOPIC}"
    )
    _assert_no_live_group(_run_probe(tmp_path, [(other, 1)]))


def test_probe_fails_when_the_listener_group_has_another_purpose(
    tmp_path: Path,
) -> None:
    """The listener subscribes with purpose CONSUME; an introspection group is not it."""
    other = f"{_LISTENER}.introspection.v1.__t.{_TOPIC}"
    _assert_no_live_group(_run_probe(tmp_path, [(other, 1)]))


def test_probe_finds_the_live_group_when_an_earlier_match_has_no_member(
    tmp_path: Path,
) -> None:
    """With an instance id per process, two groups match; the first may be dead."""
    dead = f"{_LISTENER}.consume.v1.__i.main.__t.{_TOPIC}"
    live = f"{_LISTENER}.consume.v1.__i.effects.__t.{_TOPIC}"
    result = _run_probe(tmp_path, [(dead, 0), (live, 1)])
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert f"live consumer group {live} (1 member) on {_TOPIC}" in result.stdout


def test_probe_retries_when_a_describe_fails_once(tmp_path: Path) -> None:
    """A group that vanishes between list and describe is retried, not fatal."""
    result = _run_probe(tmp_path, [(_GROUP, 1)], flaky_describe=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert f"rpk group describe {_GROUP} failed; retrying" in result.stdout
    assert f"live consumer group {_GROUP} (1 member) on {_TOPIC}" in result.stdout


def test_probe_prints_every_listed_group_when_none_is_live(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, [])
    _assert_no_live_group(result)
    filler = "omnibase_infra.node_filler_0.consume.v1.__t.onex.evt.filler.v1"
    assert filler in result.stdout


def test_probe_authenticates_every_rpk_call_as_the_local_superuser(
    tmp_path: Path,
) -> None:
    """OMN-20820: against a SASL broker an anonymous listing finds no group."""
    result = _run_probe(tmp_path, [(_GROUP, 1)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stub-secret" not in result.stdout + result.stderr


def test_probe_fails_when_the_credentials_are_stripped(tmp_path: Path) -> None:
    """The authenticated probe made anonymous must turn the step red (AC7)."""
    result = _run_probe(
        tmp_path,
        [(_GROUP, 1)],
        env_text="POSTGRES_PASSWORD=unrelated\n",
    )
    assert result.returncode != 0
    assert "REDPANDA_SUPERUSER_PASSWORD is absent" in result.stdout + result.stderr


def test_probe_fails_when_the_superuser_password_is_wrong(tmp_path: Path) -> None:
    result = _run_probe(
        tmp_path,
        [(_GROUP, 1)],
        env_text="REDPANDA_SUPERUSER_PASSWORD=wrong\n",
    )
    assert result.returncode != 0

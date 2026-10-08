# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19419: real, non-mutating Compose renders inherit one broker profile."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from dotenv import dotenv_values

_ROOT = Path(__file__).resolve().parents[3]
_DOCKER = _ROOT / "docker"
_PROFILE = {
    "segment_fallocation_step": "REDPANDA_SEGMENT_FALLOCATION_STEP",
    "log_segment_ms": "REDPANDA_LOG_SEGMENT_MS",
    "retention_bytes": "REDPANDA_RETENTION_BYTES",
    "memory": "REDPANDA_PROFILE_MEMORY",
}
_MANIFEST = yaml.safe_load(
    (_ROOT / "deploy/lane-census/lane-manifest.yaml").read_text()
)
_LANES = tuple(
    name
    for name, lane in _MANIFEST["lanes"].items()
    if name == "dev"
    or (
        name not in {"judge", "lakshman"}
        and any(service["name"].endswith("-redpanda") for service in lane["services"])
    )
)


def _render(lane: str, overrides: dict[str, str] | None = None) -> dict:
    if lane in {"infra", "ci-bus"}:
        names = [lane]
    elif lane in {"dogfood", "sim-202"}:
        names = ["dogfood"] + (["sim-202"] if lane == "sim-202" else [])
    else:
        names = ["infra", "dev-lane"] if lane != "stability-test" else ["infra"]
        if lane != "dev":
            names.append(lane)
    # Render-only inputs for guarded interpolations, never operator credentials.
    env = {key: os.environ[key] for key in ("PATH", "HOME")}
    for name in names:
        path = _DOCKER / f"docker-compose.{name}.yml"
        for key in re.findall(r"\$\{([A-Z0-9_]+):?\?", path.read_text()):
            env[key] = (
                "1"
                if key.endswith(("_REPLICAS", "_PORT"))
                else str(_ROOT / "tests")
                if key.endswith(("_DIR", "_PATH", "_FILE"))
                else "render-only"
            )
    env.update(
        {
            key: value
            for key, value in dotenv_values(_DOCKER / "runtime-policy.env").items()
            if value is not None
        }
    )
    env.update(
        {
            "OMNI_HOME": str(_ROOT),
            "OMNICLAUDE_SKILLS_DIR": str(_ROOT / "tests"),
            "ONEX_DEPLOY_AGENT_ENV_FILE": str(_ROOT / ".env.example"),
        }
    )
    env.update(overrides or {})
    args = ["docker", "compose", "--env-file", str(_DOCKER / "runtime-policy.env")]
    for name in names:
        args += ["-f", str(_DOCKER / f"docker-compose.{name}.yml")]
    args += [
        "--profile",
        "dogfood" if lane in {"dogfood", "sim-202"} else "runtime",
        "config",
        "--format",
        "json",
    ]
    result = subprocess.run(
        args,
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.unit
@pytest.mark.parametrize("lane", _LANES)
def test_broker_profile_is_shared_in_every_broker_lane(lane: str) -> None:
    services = _render(lane)["services"]
    settings = services["redpanda-partition-cap"]
    env = settings["environment"]
    assert {
        key: env[_PROFILE[key]]
        for key in ("segment_fallocation_step", "log_segment_ms", "retention_bytes")
    } == {
        "segment_fallocation_step": "1048576",
        "log_segment_ms": "86400000",
        "retention_bytes": "1073741824",
    }
    command = " ".join(services["redpanda"]["command"])
    assert f"--memory {env[_PROFILE['memory']]}" in command
    shared = _render("dev")["services"]["redpanda-partition-cap"]["command"]
    assert settings["command"] == shared
    manifest = yaml.safe_load(
        (_ROOT / "deploy/lane-census/lane-manifest.yaml").read_text()
    )
    profile = manifest["lanes"][lane]["broker_profile"]
    assert set(profile["keys"]) == set(_PROFILE)
    assert profile["settings_service"] == "redpanda-partition-cap"


@pytest.mark.unit
def test_broker_profile_can_be_parameterized_per_lane() -> None:
    services = _render(
        "stability-test",
        {
            "REDPANDA_RETENTION_BYTES": "2147483648",
            "STABILITY_TEST_REDPANDA_MEMORY": "6G",
        },
    )["services"]
    env = services["redpanda-partition-cap"]["environment"]
    assert env["REDPANDA_RETENTION_BYTES"] == "2147483648"
    assert env["REDPANDA_PROFILE_MEMORY"] == "6G"
    assert "6G" in services["redpanda"]["command"]


_RPK_STUB = r"""
state_topic_partitions_per_shard=7000
state_topic_memory_per_partition=1048576
state_segment_fallocation_step=33554432
state_log_segment_ms=1209600000
state_retention_bytes=null
rpk() {
  echo "$*" >> "$PROFILE_TRACE"
  if [ "$1 $2" = "cluster config" ]; then
    if [ "$3" = get ]; then
      if [ "$4" = enable_sasl ]; then echo "$PROFILE_SASL"; return; fi
      if [ "$PROFILE_FAILURE" = prior ] && [ "$4" = retention_bytes ]; then return 1; fi
      if [ "$PROFILE_FAILURE" = readback ] && [ "$4" = retention_bytes ] && [ -e "$PROFILE_CHANGED" ]; then echo 0; return; fi
      eval "echo \"\$state_$4\""
    else
      if [ "$PROFILE_FAILURE" = set ]; then return 1; fi
      eval "state_$4=$5"
      touch "$PROFILE_CHANGED"
    fi
  elif [ "$1 $2" = "topic list" ]; then
    if [ "$PROFILE_SASL" = true ]; then
      test "$RPK_USER" = fixture-user && test "$RPK_PASS" = fixture-password && test "$RPK_SASL_MECHANISM" = SCRAM-SHA-256 || return 1
    fi
    echo NAME
    echo onex.dlq.default
    echo onex.dlq.explicit
    if [ -e "$PROFILE_CREATED" ]; then echo onex.tenant.events; fi
  elif [ "$1 $2" = "topic create" ]; then
    touch "$PROFILE_CREATED"
  elif [ "$1 $2" = "topic describe" ]; then
    if [ "$3" = onex.dlq.explicit ]; then
      echo 'retention.bytes 53687091200 DYNAMIC_TOPIC_CONFIG'
      echo 'segment.ms 1209600000 DYNAMIC_TOPIC_CONFIG'
    else
      for key in retention.bytes segment.ms; do
        if [ -e "$PROFILE_PIN.$key" ]; then
          echo "$key -1 DYNAMIC_TOPIC_CONFIG"
        else
          echo "$key null DEFAULT_CONFIG"
        fi
      done
    fi
  elif [ "$1 $2" = "topic alter-config" ]; then
    test "$3" = onex.dlq.default || return 1
    if [ "$PROFILE_FAILURE" != pin ]; then
      touch "$PROFILE_PIN.${5%%=*}"
    fi
  else
    return 1
  fi
}
"""


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["", "prior", "set", "pin", "readback"])
@pytest.mark.parametrize("sasl", ["true", "false"])
def test_broker_profile_executes_with_prior_values_and_refuses_failed_convergence(
    tmp_path: Path, failure: str, sasl: str
) -> None:
    service = _render("dev")["services"]["redpanda-partition-cap"]
    script = service["command"][0].replace("$$", "$").replace("/usr/bin/rpk", "rpk")
    env = {
        "PATH": os.environ["PATH"],
        **service["environment"],
        "RPK_USER": "fixture-user",
        "RPK_PASS": "fixture-password",
        "TMPDIR": str(tmp_path),
        "PROFILE_TRACE": str(tmp_path / "trace"),
        "PROFILE_CHANGED": str(tmp_path / "changed"),
        "PROFILE_CREATED": str(tmp_path / "created"),
        "PROFILE_PIN": str(tmp_path / "pin"),
        "PROFILE_FAILURE": failure,
        "PROFILE_SASL": sasl,
    }
    result = subprocess.run(
        ["/bin/sh", "-c", _RPK_STUB + script],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    trace = (tmp_path / "trace").read_text()
    if failure:
        assert result.returncode != 0, result.stdout
        assert "broker profile read back" not in result.stdout
        if failure in {"prior", "pin"}:
            assert "cluster config set" not in trace
    else:
        assert result.returncode == 0, result.stderr
        assert "prior retention_bytes=null" in result.stdout
        assert "prior segment_fallocation_step=33554432" in result.stdout
        assert "broker profile read back" in result.stdout
        assert (
            trace.index("cluster config get retention_bytes")
            < trace.index("topic alter-config")
            < trace.index("cluster config set")
        )
        assert "topic alter-config onex.dlq.explicit" not in trace
        assert (tmp_path / "created").is_file()
    assert "fixture-password" not in trace + result.stdout + result.stderr

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lab proof lane can require SASL on its broker, opt-in per host (OMN-23363).

The proof lane's overlay had no SASL step for its Redpanda, so the lane could not
require per-machine logins (OMN-21616). The dev lane's model (OMN-18012) is a
SCRAM-user one-shot, then a one-shot that sets the superuser and flips
``enable_sasl``, plus SASL client settings on every broker client.

The overlay is shared by every host that runs the lane, and a host whose broker
has no SASL must keep its clients on PLAINTEXT. An empty-string default is not
inert: ``apply_environment_overrides()`` takes any set variable. So the client
settings come from one host-local env file (``LANE_BROKER_AUTH_ENV_FILE``,
``/dev/null`` when unset), and both one-shots exit 0 without touching the broker
when that file sets no principal.

The overlay is located by the one-shot it declares rather than by name. The clients
are derived from the compose files the lane merges (the base file plus the overlay),
not from a list written here, and the one-shot scripts are executed against a stub
``rpk`` so the guard, the ordering and the refusals are observed, not grepped.
"""

from __future__ import annotations

import copy
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"
COMPOSE_FILES_SH = REPO_ROOT / "scripts" / "runtime_build" / "compose_files.sh"
BROKER_PROFILE = DOCKER_DIR / "docker-compose.broker-profile.yml"
AUTH_ENV_FILE_PATH = "${LANE_BROKER_AUTH_ENV_FILE:-/dev/null}"
SCRAM_USER = "lane-redpanda-scram-user"
SASL_ENABLE = "lane-redpanda-sasl-enable"
NO_AUTH_FILE_EXIT = (
    "no LANE_BROKER_AUTH_ENV_FILE on this host; this one-shot changes nothing"
)

# The clients a reader of this lane knows about. The test derives the real set from
# the merged compose files; this tuple is only the positive control that the
# derivation does not lose any of them.
KNOWN_BROKER_CLIENTS = (
    "omninode-runtime",
    "runtime-effects",
    "projection-api",
    "runtime-worker",
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-live-events-writer",
    "tenant-projection-writer",
    "redpanda-partition-cap",
)
AUTH_KEYS = (
    "KAFKA_SECURITY_PROTOCOL",
    "KAFKA_SASL_MECHANISM",
    "KAFKA_SASL_USERNAME",
    "KAFKA_SASL_PASSWORD",
    "RPK_USER",
    "RPK_PASS",
)
# `environment:` beats `env_file`, so a client that sets any of AUTH_KEYS itself, in
# either compose file, silently ignores the host auth file.
BROKER_ADDRESS = re.compile(
    r"redpanda:(?:9092|19092|19093)|\brpk\b|KAFKA_BOOTSTRAP_SERVERS"
)
# A healthcheck that reaches the Kafka API (or names a Kafka listener) fails once the
# lane's broker requires SASL; the admin API (`rpk cluster health`) stays open.
KAFKA_API_PROBE = re.compile(
    r"rpk\s+(?:topic|group|cluster\s+info|security)|brokers|:(?:9092|19092|19093)\b"
)
# A service whose merged profiles are all inert is never started by this lane: a
# profile that exists only to disable a service, or the carrier's opt-in profile.
INERT_PROFILE = re.compile(r"-disabled$|^profile-carrier-optin$")
LISTENERS = ("redpanda:9092", "redpanda:19092", "redpanda:19093")
PRINCIPAL = "lane-principal"


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """Test-local loader that tolerates Docker Compose merge tags."""


for _tag in ("!override", "!reset"):
    _ComposeLoader.add_constructor(_tag, _construct_compose_value)


def _proof_lane_overlay() -> Path:
    declares = re.compile(rf"^  {re.escape(SASL_ENABLE)}:\s*$", re.MULTILINE)
    found = [
        path
        for path in sorted(DOCKER_DIR.glob("docker-compose.*.yml"))
        if path.name != "docker-compose.dev-lane.yml"
        and declares.search(path.read_text(encoding="utf-8"))
    ]
    assert len(found) == 1, (
        f"expected one overlay declaring {SASL_ENABLE}, found {found}"
    )
    return found[0]


def _services() -> dict[str, dict[str, object]]:
    loader = _ComposeLoader(_proof_lane_overlay().read_text(encoding="utf-8"))
    try:
        compose = loader.get_single_data()
    finally:
        loader.dispose()
    assert isinstance(compose, dict)
    services = compose["services"]
    assert isinstance(services, dict)
    return services


def _env_keys(service: dict[str, object]) -> set[str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        return {str(item).split("=", 1)[0] for item in env}
    assert isinstance(env, dict)
    return {str(key) for key in env}


def _command(name: str) -> str:
    command = _services()[name]["command"]
    assert isinstance(command, list)
    return "\n".join(str(line) for line in command)


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _env_map(service: dict[str, object]) -> dict[str, object]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        return {str(i).split("=", 1)[0]: str(i).split("=", 1)[-1] for i in env}
    assert isinstance(env, dict)
    return dict(env)


def _depends_map(service: dict[str, object]) -> dict[str, dict[str, object]]:
    depends = service.get("depends_on") or {}
    if isinstance(depends, list):
        return {str(name): {"condition": "service_started"} for name in depends}
    assert isinstance(depends, dict)
    return {str(k): dict(v or {}) for k, v in depends.items()}


def _load_services(path: Path) -> dict[str, dict[str, object]]:
    loader = _ComposeLoader(path.read_text(encoding="utf-8"))
    try:
        compose = loader.get_single_data()
    finally:
        loader.dispose()
    assert isinstance(compose, dict)
    services = compose["services"]
    assert isinstance(services, dict)
    return services


@dataclass
class _Merged:
    """The parts of one service's merged definition these checks read."""

    environment: dict[str, object]
    env_file: list[object]
    depends_on: dict[str, dict[str, object]]
    profiles: list[object]
    text: str
    healthcheck: str = ""


def _layer(
    service: dict[str, object], siblings: dict[str, dict[str, object]], directory: Path
) -> _Merged:
    """One service as one file defines it, with its `extends:` target applied first.

    Compose merges the target's ``environment`` (the service's own keys win),
    ``env_file`` and ``depends_on`` under the service's own, so a client that gains
    its broker address or its auth keys through an extends target is read here.
    """
    parent = _Merged({}, [], {}, [], "")
    extends = service.get("extends")
    if isinstance(extends, dict):
        if "file" in extends:
            target_file = directory / str(extends["file"])
            target = _load_services(target_file)[str(extends["service"])]
            parent = _layer(target, _load_services(target_file), target_file.parent)
        else:
            parent = _layer(siblings[str(extends["service"])], siblings, directory)
    own_profiles = _as_list(service.get("profiles"))
    health = service.get("healthcheck")
    return _Merged(
        environment={**parent.environment, **_env_map(service)},
        env_file=[*parent.env_file, *_as_list(service.get("env_file"))],
        depends_on={**parent.depends_on, **_depends_map(service)},
        profiles=own_profiles or parent.profiles,
        text=f"{parent.text} {service}",
        healthcheck=str(health.get("test", "")) if isinstance(health, dict) else "",
    )


def _lane_project() -> str:
    """The lane's compose project, from its overlay's name (docker-compose.<lane>.yml)."""
    lane = (
        _proof_lane_overlay().name.removeprefix("docker-compose.").removesuffix(".yml")
    )
    return f"omnibase-infra-{lane}"


def _lane_compose_files() -> list[Path]:
    """The lane's real ``-f`` list, from the resolver the refresh and the deploy use.

    The host-broker overlay is switched on so the list is the longest one a host runs.
    """
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; declare -a A; '
            'resolve_compose_file_args A "$2" "$3"; printf "%s\\n" "${A[@]}"',
            "_",
            str(COMPOSE_FILES_SH),
            str(REPO_ROOT),
            _lane_project(),
        ],
        env={
            **os.environ,
            "STABILITY_TEST_REDPANDA_ADVERTISED_KAFKA_ADDR": "x",
            "STABILITY_TEST_REDPANDA_ADVERTISED_PANDAPROXY_ADDR": "x",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [Path(token) for token in proc.stdout.splitlines() if token != "-f"]


def _merge_files(files: list[Path]) -> dict[str, _Merged]:
    """The compose files merged in order the way Compose merges them.

    Maps merge by key, ``env_file`` lists append, and a ``profiles:`` a later file
    sets replaces the earlier one. That is the part of Compose's merge these checks read.
    """
    merged: dict[str, _Merged] = {}
    for path in files:
        services = _load_services(path)
        for name, definition in services.items():
            layer = _layer(definition, services, path.parent)
            prior = merged.get(name)
            if prior is None:
                merged[name] = layer
                continue
            merged[name] = _Merged(
                environment={**prior.environment, **layer.environment},
                env_file=[*prior.env_file, *layer.env_file],
                depends_on={**prior.depends_on, **layer.depends_on},
                profiles=layer.profiles or prior.profiles,
                text=f"{prior.text} {layer.text}",
                healthcheck=layer.healthcheck or prior.healthcheck,
            )
    return merged


def _merged_services() -> dict[str, _Merged]:
    files = _lane_compose_files()
    assert _proof_lane_overlay() in files, files
    assert DOCKER_DIR / "docker-compose.infra.yml" in files, files
    return _merge_files(files)


def _derived_clients(merged: dict[str, _Merged]) -> set[str]:
    """Every service, but the broker, that talks to the broker or runs rpk."""
    return {
        name
        for name, service in merged.items()
        if name != "redpanda" and BROKER_ADDRESS.search(service.text)
    }


def _inert(service: _Merged) -> bool:
    profiles = [str(p) for p in service.profiles]
    return bool(profiles) and all(INERT_PROFILE.search(p) for p in profiles)


def _readers(merged: dict[str, _Merged]) -> set[str]:
    return {n for n in _derived_clients(merged) if not _inert(merged[n])}


def _unwired(merged: dict[str, _Merged]) -> list[str]:
    return sorted(
        n for n in _readers(merged) if AUTH_ENV_FILE_PATH not in merged[n].env_file
    )


def _literal_auth(merged: dict[str, _Merged]) -> list[str]:
    return sorted(
        n for n in _readers(merged) if set(merged[n].environment) & set(AUTH_KEYS)
    )


def _fail_open(merged: dict[str, _Merged]) -> list[str]:
    """Readers whose env_file names the auth file as an optional (fail-open) entry."""
    return sorted(
        n
        for n in _readers(merged)
        if any(
            isinstance(entry, dict) and entry.get("required") is False
            for entry in merged[n].env_file
        )
    )


def _without_direct_flip_edge(merged: dict[str, _Merged]) -> list[str]:
    """Readers whose OWN depends_on lacks the flip's completed-successfully edge.

    Direct, not walked: a refresh runs ``up -d --no-deps`` over a selection, which
    drops edges to services outside it, so an edge held only through another service
    (the partition cap) is not an ordering the refresh keeps.
    """
    flip_edge = {"condition": "service_completed_successfully"}
    return sorted(
        n
        for n in _readers(merged) - {SCRAM_USER, SASL_ENABLE}
        if merged[n].depends_on.get(SASL_ENABLE) != flip_edge
    )


def _redpanda_kafka_api_probes(merged: dict[str, _Merged]) -> list[str]:
    return sorted(
        name
        for name in ("redpanda",)
        if KAFKA_API_PROBE.search(merged[name].healthcheck)
    )


@pytest.fixture(scope="module")
def merged() -> dict[str, _Merged]:
    return _merged_services()


@pytest.mark.unit
def test_stability_sasl_derivation_finds_the_known_clients_and_skips_inert_ones(
    merged: dict[str, _Merged],
) -> None:
    readers = _readers(merged)
    assert set(KNOWN_BROKER_CLIENTS) <= readers, sorted(
        set(KNOWN_BROKER_CLIENTS) - readers
    )
    assert {SCRAM_USER, SASL_ENABLE} <= readers
    # Services that talk to the broker but never start in this lane are exempt only
    # because every merged profile of theirs is inert; none of them is a reader.
    exempt = _derived_clients(merged) - readers
    assert "projection-claude-hook-events-writer" in exempt
    assert not exempt & set(KNOWN_BROKER_CLIENTS)


@pytest.mark.unit
def test_stability_sasl_every_derived_client_reads_the_host_auth_env_file(
    merged: dict[str, _Merged],
) -> None:
    assert _unwired(merged) == []
    # Red on a fixture: a client the lane gains and does not wire is reported.
    broken = copy.deepcopy(merged)
    broken["new-kafka-console"] = _Merged(
        environment={},
        env_file=[],
        depends_on={},
        profiles=["runtime"],
        text="KAFKA_BOOTSTRAP_SERVERS=redpanda:9092",
    )
    broken["projection-api"].env_file = []
    assert _unwired(broken) == ["new-kafka-console", "projection-api"]


@pytest.mark.unit
def test_stability_sasl_client_sets_no_literal_auth_in_either_compose_file(
    merged: dict[str, _Merged],
) -> None:
    assert _literal_auth(merged) == [], (
        "environment: beats env_file, so a literal auth key would make the host auth "
        "file be ignored (or put a PLAINTEXT host's clients on SASL)"
    )
    for key in AUTH_KEYS:
        broken = copy.deepcopy(merged)
        broken["omninode-runtime"].environment[key] = "x"
        assert _literal_auth(broken) == ["omninode-runtime"], key


@pytest.mark.unit
def test_stability_sasl_auth_env_file_fails_closed(
    merged: dict[str, _Merged],
) -> None:
    assert _fail_open(merged) == []
    optional = [
        name
        for name, service in _services().items()
        for entry in _as_list(service.get("env_file"))
        if isinstance(entry, dict) and entry.get("required") is False
    ]
    assert optional == [], (
        "an optional env_file turns a mistyped LANE_BROKER_AUTH_ENV_FILE into no auth "
        "variables and both one-shots exiting 0 on a PLAINTEXT broker"
    )
    broken = copy.deepcopy(merged)
    broken["runtime-worker"].env_file = [{"path": "/x", "required": False}]
    assert _fail_open(broken) == ["runtime-worker"]


FLIP_EDGE = {"condition": "service_completed_successfully"}
RUNTIME_FAMILY = ("omninode-runtime", "runtime-effects", "runtime-worker")


@pytest.mark.unit
def test_stability_sasl_every_client_waits_for_the_flip(
    merged: dict[str, _Merged],
) -> None:
    assert _without_direct_flip_edge(merged) == []
    # The flip itself waits for the principal, and the principal for a healthy broker.
    assert merged[SASL_ENABLE].depends_on[SCRAM_USER] == {
        "condition": "service_completed_successfully"
    }
    assert merged[SCRAM_USER].depends_on["redpanda"] == {"condition": "service_healthy"}
    # Red on a fixture: a client that only waits for the broker can start with SCRAM
    # settings before the principal exists.
    broken = copy.deepcopy(merged)
    broken["projection-api"].depends_on.pop(SASL_ENABLE)
    assert _without_direct_flip_edge(broken) == ["projection-api"]
    # The partition cap is not a substitute: dropping the edge from the cap is reported
    # for the cap, and dropping a client's own edge is reported for the client even
    # while the cap still carries it.
    broken = copy.deepcopy(merged)
    broken["redpanda-partition-cap"].depends_on.pop(SASL_ENABLE)
    assert _without_direct_flip_edge(broken) == ["redpanda-partition-cap"]
    for name in RUNTIME_FAMILY:
        broken = copy.deepcopy(merged)
        broken[name].depends_on.pop(SASL_ENABLE)
        assert SASL_ENABLE in merged["redpanda-partition-cap"].depends_on
        assert _without_direct_flip_edge(broken) == [name], name
    # An edge on the wrong condition is no ordering either.
    broken = copy.deepcopy(merged)
    broken["runtime-worker"].depends_on[SASL_ENABLE] = {"condition": "service_started"}
    assert _without_direct_flip_edge(broken) == ["runtime-worker"]


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNTIME_FAMILY)
def test_stability_sasl_runtime_family_carries_its_own_edge_on_the_flip(
    name: str,
) -> None:
    """Read the service's own depends_on array in the overlay, not a walked graph."""
    own = _depends_map(_services()[name])
    assert own.get(SASL_ENABLE) == FLIP_EDGE, own


@pytest.mark.unit
def test_stability_sasl_one_shots_are_ordered_before_the_partition_cap() -> None:
    services = _services()
    scram, enable = services[SCRAM_USER], services[SASL_ENABLE]
    assert scram["depends_on"] == {"redpanda": {"condition": "service_healthy"}}
    assert enable["depends_on"] == {
        SCRAM_USER: {"condition": "service_completed_successfully"}
    }
    cap_depends = services["redpanda-partition-cap"]["depends_on"]
    assert isinstance(cap_depends, dict)
    assert cap_depends.get(SASL_ENABLE) == {
        "condition": "service_completed_successfully"
    }
    for service in (scram, enable):
        assert service.get("env_file") == [AUTH_ENV_FILE_PATH]
        assert service.get("restart") == "no"
        assert "profiles" not in service


@pytest.mark.unit
@pytest.mark.parametrize("name", [SCRAM_USER, SASL_ENABLE])
def test_stability_sasl_one_shot_is_a_no_op_without_the_auth_file(name: str) -> None:
    command = _command(name)
    guard = command.index(NO_AUTH_FILE_EXIT)
    first_rpk = command.index("/usr/bin/rpk")
    assert guard < first_rpk, (
        f"{name} must exit before any rpk call on a PLAINTEXT host"
    )


def _script(name: str, stub: Path, drop: str = "") -> str:
    """The one-shot's shell script as the container runs it, with rpk stubbed.

    Compose writes a literal ``$`` as ``$$``; the container's ``/usr/bin/rpk`` becomes
    the stub.
    """
    script = _command(name).replace("$$", "$").replace("/usr/bin/rpk", str(stub))
    if drop:
        assert drop in script, drop
        script = script.replace(drop, "")
    return script


_STUB_RPK = """#!/bin/sh
{ printf 'rpk'; for a in "$@"; do printf ' %s' "$a"; done
  printf ' RPK_USER=%s\n' "${RPK_USER-unset}"; } >> "$STUB_DIR/calls.log"
case "$*" in
  *"security user list"*)
    echo "USERNAME MECHANISM"
    [ -f "$STUB_DIR/users" ] && sed 's/$/ SCRAM-SHA-256/' "$STUB_DIR/users"
    exit 0 ;;
  *"security user create"*)
    [ -n "${STUB_CREATE_NOOP:-}" ] || echo "$4" >> "$STUB_DIR/users"
    exit 0 ;;
  *"cluster config set"*) exit 0 ;;
  *"cluster config get superusers"*)
    printf '%s\n' "${STUB_SUPERUSERS:-- lane-principal}"; exit 0 ;;
  *"cluster config get enable_sasl"*)
    printf '%s\n' "${STUB_ENABLE_SASL:-true}"; exit 0 ;;
  *"topic list"*)
    for a in "$@"; do case "$a" in brokers=*) broker="${a#brokers=}" ;; esac; done
    # rpk with a user and password and no mechanism set defaults to SCRAM-SHA-256.
    if [ -n "${RPK_USER:-}" ] && [ -n "${RPK_PASS:-}" ] \
       && [ "${RPK_SASL_MECHANISM:-SCRAM-SHA-256}" = SCRAM-SHA-256 ]; then exit 0; fi
    case " ${STUB_OPEN_LISTENERS:-} " in *" $broker "*) exit 0 ;; esac
    exit 1 ;;
esac
exit 99
"""


class _Run:
    def __init__(self, proc: subprocess.CompletedProcess[str], calls: list[str]):
        self.returncode = proc.returncode
        self.stdout = proc.stdout
        self.stderr = proc.stderr
        self.calls = calls

    def call_index(self, needle: str) -> int:
        for index, call in enumerate(self.calls):
            if needle in call:
                return index
        raise AssertionError(f"no rpk call containing {needle!r}: {self.calls}")


def _run_one_shot(name: str, tmp_path: Path, drop: str = "", **env: str) -> _Run:
    stub = tmp_path / "rpk"
    stub.write_text(_STUB_RPK, encoding="utf-8")
    stub.chmod(0o755)
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("KAFKA_", "RPK_", "STUB_"))
    }
    proc = subprocess.run(
        ["sh", "-c", _script(name, stub, drop)],
        env={**clean, "STUB_DIR": str(tmp_path), **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    log = tmp_path / "calls.log"
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return _Run(proc, calls)


# What the auth file gives the container: RPK_USER/RPK_PASS come with it, which is why the
# flip's unauthenticated probe must unset them (`env -u RPK_USER -u RPK_PASS`).
AUTHED = {
    "KAFKA_SASL_USERNAME": PRINCIPAL,
    "KAFKA_SASL_PASSWORD": "pw",
    "RPK_USER": PRINCIPAL,
    "RPK_PASS": "pw",
}


@pytest.mark.unit
@pytest.mark.parametrize("name", [SCRAM_USER, SASL_ENABLE])
def test_stability_sasl_one_shot_exits_zero_without_touching_the_broker(
    name: str, tmp_path: Path
) -> None:
    run = _run_one_shot(name, tmp_path)
    assert run.returncode == 0, run.stderr
    assert NO_AUTH_FILE_EXIT in run.stdout
    assert run.calls == [], "a host with no auth file must make no rpk call"


@pytest.mark.unit
@pytest.mark.parametrize("name", [SCRAM_USER, SASL_ENABLE])
def test_stability_sasl_one_shot_refuses_a_username_with_no_password(
    name: str, tmp_path: Path
) -> None:
    run = _run_one_shot(name, tmp_path, KAFKA_SASL_USERNAME=PRINCIPAL)
    assert run.returncode != 0
    assert "sets a username and no password" in run.stderr
    assert run.calls == []


@pytest.mark.unit
def test_stability_sasl_enable_flips_the_broker_and_proves_both_directions(
    tmp_path: Path,
) -> None:
    command = _command(SASL_ENABLE)
    assert "rpk cluster config set superusers" in command
    assert "rpk cluster config set enable_sasl true" in command
    assert "rpk cluster config get enable_sasl" in command
    assert "UNAUTHENTICATED client reached the Kafka listener" in command
    assert "RPK_SASL_MECHANISM=SCRAM-SHA-256" in command
    scram_command = _command(SCRAM_USER)
    assert "rpk security user create" in scram_command
    assert "--mechanism SCRAM-SHA-256" in scram_command
    run = _run_one_shot(SASL_ENABLE, tmp_path, **AUTHED)
    assert run.returncode == 0, run.stderr
    superusers = run.call_index("cluster config set superusers")
    enable = run.call_index("cluster config set enable_sasl true")
    readback = run.call_index("cluster config get enable_sasl")
    assert superusers < enable < readback, run.calls
    assert "unauthenticated client refused on redpanda:19093" in run.stdout
    for listener in LISTENERS:
        refused = run.call_index(f"topic list -X brokers={listener} RPK_USER=unset")
        accepted = run.call_index(
            f"topic list -X brokers={listener} RPK_USER={PRINCIPAL}"
        )
        assert enable < refused < accepted, run.calls


@pytest.mark.unit
@pytest.mark.parametrize("listener", LISTENERS)
def test_stability_sasl_enable_fails_when_any_listener_admits_an_unauthenticated_client(
    listener: str, tmp_path: Path
) -> None:
    run = _run_one_shot(SASL_ENABLE, tmp_path, STUB_OPEN_LISTENERS=listener, **AUTHED)
    assert run.returncode == 1, run.stdout
    assert f"UNAUTHENTICATED client reached the Kafka listener {listener}" in run.stderr


@pytest.mark.unit
def test_stability_sasl_enable_fails_unless_enable_sasl_reads_back_true(
    tmp_path: Path,
) -> None:
    run = _run_one_shot(SASL_ENABLE, tmp_path, STUB_ENABLE_SASL="false", **AUTHED)
    assert run.returncode != 0
    assert "enable_sasl confirmed true" not in run.stdout
    assert not any("topic list" in call for call in run.calls)


@pytest.mark.unit
def test_stability_sasl_enable_fails_unless_the_principal_is_a_superuser(
    tmp_path: Path,
) -> None:
    run = _run_one_shot(
        SASL_ENABLE, tmp_path, STUB_SUPERUSERS="- someone-else", **AUTHED
    )
    assert run.returncode != 0
    assert "superuser grant confirmed" not in run.stdout


@pytest.mark.unit
def test_stability_sasl_scram_user_creates_an_absent_principal_once(
    tmp_path: Path,
) -> None:
    first = _run_one_shot(SCRAM_USER, tmp_path, **AUTHED)
    assert first.returncode == 0, first.stderr
    assert "SCRAM principal created" in first.stdout
    assert sum("security user create" in c for c in first.calls) == 1
    again = _run_one_shot(SCRAM_USER, tmp_path, **AUTHED)
    assert again.returncode == 0, again.stderr
    assert "SCRAM principal already present" in again.stdout
    assert sum("security user create" in c for c in again.calls) == 1, (
        "no second create"
    )


@pytest.mark.unit
def test_stability_sasl_scram_user_fails_when_the_principal_does_not_read_back(
    tmp_path: Path,
) -> None:
    run = _run_one_shot(SCRAM_USER, tmp_path, STUB_CREATE_NOOP="1", **AUTHED)
    assert run.returncode != 0
    assert "readback confirms the principal exists" not in run.stdout


@pytest.mark.unit
def test_stability_sasl_unauthenticated_probe_unsets_the_auth_files_rpk_credentials(
    tmp_path: Path,
) -> None:
    """The auth file hands the container RPK_USER/RPK_PASS; the probe must not use them.

    With ``env -u RPK_USER -u RPK_PASS`` deleted, the "unauthenticated" probe carries
    the principal's credentials, the broker admits it, and the flip must fail on the
    first listener rather than read as a proof that SASL refuses anonymous clients.
    """
    drop = "env -u RPK_USER -u RPK_PASS "
    for sub in ("ok", "mutant", "blind"):
        (tmp_path / sub).mkdir()
    ok = _run_one_shot(SASL_ENABLE, tmp_path / "ok", **AUTHED)
    assert ok.returncode == 0, ok.stderr
    mutant = _run_one_shot(SASL_ENABLE, tmp_path / "mutant", drop=drop, **AUTHED)
    assert mutant.returncode == 1, mutant.stdout
    assert "UNAUTHENTICATED client reached the Kafka listener" in mutant.stderr
    # ... and the fixture is what makes the deletion visible: without the auth file's
    # RPK variables the same deletion passes silently (the gap this test closes).
    blind = _run_one_shot(
        SASL_ENABLE,
        tmp_path / "blind",
        drop=drop,
        KAFKA_SASL_USERNAME=PRINCIPAL,
        KAFKA_SASL_PASSWORD="pw",
    )
    assert blind.returncode == 0, blind.stderr


def _write_compose(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
def test_stability_sasl_lane_file_list_comes_from_the_resolver_the_refresh_uses(
    merged: dict[str, _Merged],
) -> None:
    files = _lane_compose_files()
    assert [f.name for f in files] == [
        "docker-compose.infra.yml",
        _proof_lane_overlay().name,
        "docker-compose.stability-host-broker.yml",
    ]
    # Services declared in the base file are derived from the same merged view.
    assert "redpanda" in merged and "migration-gate" in merged


@pytest.mark.unit
def test_stability_sasl_derivation_reads_environment_through_extends(
    merged: dict[str, _Merged], tmp_path: Path
) -> None:
    # The partition cap gets its environment from broker-profile.yml via `extends:`.
    assert (
        "REDPANDA_SEGMENT_FALLOCATION_STEP"
        in merged["redpanda-partition-cap"].environment
    )
    # The extends target sets none of the auth keys in environment (which would beat
    # the host auth file), so the file reaches the cap's rpk calls.
    profile_env = _env_keys(_load_services(BROKER_PROFILE)["redpanda-settings"])
    assert not profile_env & set(AUTH_KEYS), profile_env
    assert _literal_auth(merged) == []
    # Red on a fixture: a client that exists only through an extends target with no SASL
    # settings is a client and is unwired.
    ext = _write_compose(
        tmp_path / "ext.yml",
        "services:\n  base-client:\n    environment:\n"
        "      KAFKA_BOOTSTRAP_SERVERS: ${KAFKA_BOOTSTRAP_SERVERS:-redpanda:9092}\n",
    )
    lane = _write_compose(
        tmp_path / "lane.yml",
        f"services:\n  new-client:\n    extends:\n      file: {ext.name}\n"
        "      service: base-client\n",
    )
    fixture = _merge_files([lane])
    assert _unwired(fixture) == ["new-client"]
    # An extends target that sets an auth key in environment is reported too.
    ext.write_text(
        ext.read_text(encoding="utf-8") + "      KAFKA_SASL_USERNAME: literal\n",
        encoding="utf-8",
    )
    assert _literal_auth(_merge_files([lane])) == ["new-client"]


@pytest.mark.unit
def test_stability_sasl_derivation_covers_clients_that_use_the_bootstrap_variable(
    tmp_path: Path,
) -> None:
    lane = _write_compose(
        tmp_path / "lane.yml",
        "services:\n  var-client:\n    environment:\n"
        '      SOMETHING: "${KAFKA_BOOTSTRAP_SERVERS}"\n',
    )
    fixture = _merge_files([lane])
    assert _readers(fixture) == {"var-client"}
    assert _unwired(fixture) == ["var-client"]


@pytest.mark.unit
def test_stability_sasl_redpandas_healthcheck_uses_the_admin_api_not_the_kafka_api(
    merged: dict[str, _Merged],
) -> None:
    """Once enable_sasl is true an anonymous Kafka-API healthcheck fails the broker.

    The healthcheck is `rpk cluster health` (the admin API, open on this broker), so the
    broker stays healthy and every ``service_healthy`` edge on it keeps holding.
    """
    assert merged["redpanda"].healthcheck, "redpanda must declare a healthcheck"
    assert _redpanda_kafka_api_probes(merged) == []
    assert not set(merged["redpanda"].environment) & set(AUTH_KEYS)
    for kafka_api in (
        "rpk cluster info",
        "rpk topic list",
        "rpk cluster health -X brokers=redpanda:9092",
    ):
        broken = copy.deepcopy(merged)
        broken["redpanda"].healthcheck = kafka_api
        assert _redpanda_kafka_api_probes(broken) == ["redpanda"], kafka_api

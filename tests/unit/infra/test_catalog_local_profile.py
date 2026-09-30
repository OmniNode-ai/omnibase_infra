# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The laptop profile: catalog bundle ``local`` (OMN-19496).

A new engineer boots the ONEX stack plus their own runtime with one command and
one env file, with no lab or ops secret. These tests pin the properties that
make that true in the render, so a manifest edit cannot quietly reintroduce a
lab dependency, a shared Docker object name, or a second env source.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog import cli as catalog_cli
from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import DEFAULT_PROJECT, CatalogResolver
from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_CATALOG_DIR = str(_REPO / "docker" / "catalog")
_ENV_TEMPLATE = _REPO / "docker" / "local.env.example"
_OVERLAY_TEMPLATE = _REPO / "docker" / "lane-overlays" / "local.bifrost.example.yaml"
_PROJECT = "omnibase-infra-local"
_OVERLAY_PIN = "/app/config/delegation/local.bifrost.yaml"

#: Every name the laptop profile may ask its operator for. Four local passwords
#: (make local-env generates all four) and the path of the model overlay;
#: nothing else.
_LAPTOP_REQUIRED_ENV = {
    "POSTGRES_PASSWORD",
    "VALKEY_PASSWORD",
    "OMNINODE_RUNTIME_PASSWORD",
    "TENANT_PROJECTION_WRITER_PASSWORD",
    "ONEX_LOCAL_BIFROST_OVERLAY",
}

#: Lab and ops credentials the profile must never require.
_FORBIDDEN_FRAGMENTS = (
    "DEPLOY_AGENT",
    "GITHUB",
    "LINEAR",
    "SLACK",
    "CI_CALLBACK",
    "INFISICAL",
    "KEYCLOAK",
    "SERVICE_CLIENT",
    "LLM_CODER",
    "LLM_DEEPSEEK",
    "LLM_EMBEDDING",
    "OPENROUTER",
    "GEMINI",
)


def test_laptop_required_env_is_four_passwords_and_the_overlay_path() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    assert resolved.required_env == _LAPTOP_REQUIRED_ENV


def test_laptop_required_env_names_no_lab_or_ops_secret() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    offenders = sorted(
        var
        for var in resolved.required_env
        for fragment in _FORBIDDEN_FRAGMENTS
        if fragment in var
    )
    assert offenders == []


def test_laptop_required_env_check_can_fail() -> None:
    """Positive control: the full runtime bundle does require lab and ops secrets."""
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["runtime"])
    assert {"DEPLOY_AGENT_HMAC_SECRET", "GITHUB_TOKEN", "LLM_CODER_URL"} <= set(
        resolved.required_env
    )


def test_local_bundle_runs_both_runtime_kernels_the_writer_and_the_migration_gate() -> (
    None
):
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    assert {
        "postgres",
        "redpanda",
        "valkey",
        "forward-migration",
        "migration-gate",
        "omninode-runtime",
        "runtime-effects",
    } <= set(resolved.manifests)
    runtime = [
        n for n, m in resolved.manifests.items() if m.layer == EnumInfraLayer.RUNTIME
    ]
    assert sorted(runtime) == [
        "consumer-health-projection",
        "omnimarket-projection-delegation",
        "omnimarket-projection-llm-cost",
        "omninode-runtime",
        "projection-api",
        "runtime-effects",
    ]


def test_local_render_scopes_every_docker_object_to_its_project() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    compose = generate_compose(resolved)

    assert compose["name"] == _PROJECT
    services = compose["services"]
    assert isinstance(services, dict)
    for name, svc in services.items():
        assert svc["container_name"] == f"{_PROJECT}-{name}"
        assert svc["networks"] == ["omnibase-infra-network"]
    networks = compose["networks"]
    assert isinstance(networks, dict)
    assert networks["omnibase-infra-network"]["name"] == f"{_PROJECT}-network"
    volumes = compose["volumes"]
    assert isinstance(volumes, dict)
    assert volumes
    for key, spec in volumes.items():
        assert spec["name"] == f"{_PROJECT}-{key}"
    assert services["omninode-runtime"]["image"] == f"{_PROJECT}-runtime:latest"
    assert services["runtime-effects"]["image"] == f"{_PROJECT}-runtime:latest"
    assert "build" in services["omninode-runtime"]


def test_default_project_render_keeps_every_historical_name() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["runtime"])
    assert resolved.project == DEFAULT_PROJECT
    compose = generate_compose(resolved)
    assert compose["name"] == "omnibase-infra"
    services = compose["services"]
    assert isinstance(services, dict)
    assert services["omninode-runtime"]["container_name"] == "omninode-runtime"
    assert services["omninode-runtime"]["image"] == "runtime:latest"
    assert services["postgres"]["container_name"] == "omnibase-infra-postgres"
    networks = compose["networks"]
    assert isinstance(networks, dict)
    assert networks["omnibase-infra-network"]["name"] == "omnibase-infra-network"
    volumes = compose["volumes"]
    assert isinstance(volumes, dict)
    assert volumes["postgres_data"] == {"name": "postgres_data"}


def test_local_runtime_kernels_mount_and_pin_the_local_overlay() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    compose = generate_compose(resolved)
    services = compose["services"]
    assert isinstance(services, dict)
    for name in ("omninode-runtime", "runtime-effects"):
        svc = services[name]
        mounts = [v for v in svc["volumes"] if v.endswith(f":{_OVERLAY_PIN}:ro")]
        assert len(mounts) == 1, f"{name}: {svc['volumes']}"
        assert mounts[0].startswith("${ONEX_LOCAL_BIFROST_OVERLAY:?")
        env = svc["environment"]
        assert env["BIFROST_LANE_OVERLAY_PATH"] == _OVERLAY_PIN
        assert env["DELEGATION_ROUTING_TIERS_PATH"]
        assert env["GITHUB_TOKEN"] == ""
        assert env["DEPLOY_AGENT_HMAC_SECRET"] == ""
    # Infrastructure entries never receive the runtime overlay mount.
    assert not any(
        v.endswith(_OVERLAY_PIN + ":ro") for v in services["postgres"]["volumes"]
    )


def test_local_projection_writer_binds_the_lab_principals_not_the_superuser() -> None:
    """omnimarket's standalone writer attests the connected principal per binding.

    With the superuser DSNs the laptop bundle used to inject, the delegation
    writer crash-looped on a missing topology profile, then on principal
    ``postgres`` where ``omninode_runtime`` was expected, and no
    ``delegation_events`` row was ever written.
    """
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    services = generate_compose(resolved)["services"]
    assert isinstance(services, dict)
    env = services["omnimarket-projection-delegation"]["environment"]
    assert env["ONEX_DATABASE_TOPOLOGY_PROFILE"] == "local"
    assert env["ONEX_TENANT_DB_URL"].startswith(
        "postgresql://tenant_projection_writer:${TENANT_PROJECTION_WRITER_PASSWORD:?"
    )
    assert env["OMNINODE_INTERNAL_DB_URL"].startswith(
        "postgresql://omninode_runtime:${OMNINODE_RUNTIME_PASSWORD:?"
    )
    # forward-migration is what gives those principals a LOGIN credential.
    migration_env = services["forward-migration"]["environment"]
    assert (
        migration_env["OMNINODE_RUNTIME_PASSWORD"] == "${OMNINODE_RUNTIME_PASSWORD:-}"
    )
    assert (
        migration_env["TENANT_PROJECTION_WRITER_PASSWORD"]
        == "${TENANT_PROJECTION_WRITER_PASSWORD:-}"
    )


def test_overlay_template_is_a_typed_lab_overlay_named_for_the_local_lane() -> None:
    overlay = ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(_OVERLAY_TEMPLATE.read_text(encoding="utf-8"))
    )
    assert overlay.lane == "local"
    assert overlay.locale is EnumBifrostLaneLocale.LAB
    endpoints = {binding.endpoint_url for binding in overlay.backends}
    assert len(endpoints) == 1, "one model endpoint setting, shared by both rungs"


def test_env_template_names_every_required_var_and_nothing_secret_from_the_lab() -> (
    None
):
    keys = {
        line.partition("=")[0]
        for line in _ENV_TEMPLATE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    assert keys >= _LAPTOP_REQUIRED_ENV
    assert not [k for k in keys for f in _FORBIDDEN_FRAGMENTS if f in k]


def test_env_file_is_the_only_operator_env_source_and_loads_runtime_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    home_env = tmp_path / "home.env"
    home_env.write_text("OMN19496_LEAK=from-home\n", encoding="utf-8")
    monkeypatch.setattr(catalog_cli, "_HOME_ENV", home_env)
    monkeypatch.setattr(catalog_cli, "_REPO_ENV", tmp_path / "absent-repo.env")
    env_file = tmp_path / "local.env"
    env_file.write_text("OMN19496_PROBE=from-env-file\n", encoding="utf-8")

    assert catalog_cli._load_stack_env(str(env_file)) == 0

    assert os.environ["OMN19496_PROBE"] == "from-env-file"
    assert "OMN19496_LEAK" not in os.environ
    assert os.environ["ONEX_ACTIVE_RUNTIME_PACKAGES"]


def test_env_file_with_template_placeholders_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    assert catalog_cli._load_stack_env(str(_ENV_TEMPLATE)) == 1


def test_missing_env_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    assert catalog_cli._load_stack_env(str(tmp_path / "absent.env")) == 1


def test_two_bundles_naming_different_projects_are_refused(tmp_path: Path) -> None:
    services = tmp_path / "services"
    services.mkdir()
    (tmp_path / "bundles.yaml").write_text(
        yaml.safe_dump(
            {
                "one": {"description": "a", "services": [], "project": "p-one"},
                "two": {"description": "b", "services": [], "project": "p-two"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Compose project conflict"):
        CatalogResolver(catalog_dir=str(tmp_path)).resolve(["one", "two"])


def _fixture_manifest(name: str, layer: str, required: list[str]) -> dict[str, object]:
    return {
        "name": name,
        "description": f"{name} fixture",
        "image": "busybox:1.36",
        "layer": layer,
        "required_env": required,
        "hardcoded_env": {"FIXTURE": "1"},
        "operational_defaults": {},
        "ports": None,
        "healthcheck": None,
        "volumes": [],
        "depends_on": [],
    }


def test_injected_env_satisfies_a_runtime_requirement_but_never_an_infra_one(
    tmp_path: Path,
) -> None:
    services = tmp_path / "services"
    services.mkdir()
    for name, layer, required in (
        ("kernel", "runtime", ["SHARED_VAR", "RUNTIME_ONLY_VAR"]),
        ("store", "infrastructure", ["SHARED_VAR"]),
    ):
        (services / f"{name}.yaml").write_text(
            yaml.safe_dump(_fixture_manifest(name, layer, required)),
            encoding="utf-8",
        )
    (tmp_path / "bundles.yaml").write_text(
        yaml.safe_dump(
            {
                "b": {
                    "description": "fixture",
                    "services": ["kernel", "store"],
                    "inject_env": {"SHARED_VAR": "x", "RUNTIME_ONLY_VAR": "y"},
                }
            }
        ),
        encoding="utf-8",
    )
    resolved = CatalogResolver(catalog_dir=str(tmp_path)).resolve(["b"])
    assert resolved.required_env == {"SHARED_VAR"}


def test_up_precleanup_removes_anonymous_volumes_of_removed_containers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured on .105: without -v every re-up orphaned the one-shots' volumes."""
    services = tmp_path / "catalog" / "services"
    services.mkdir(parents=True)
    (services / "store.yaml").write_text(
        yaml.safe_dump(_fixture_manifest("store", "infrastructure", [])),
        encoding="utf-8",
    )
    (tmp_path / "catalog" / "bundles.yaml").write_text(
        yaml.safe_dump({"b": {"description": "fixture", "services": ["store"]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    monkeypatch.setattr(catalog_cli, "_CATALOG_DIR", str(tmp_path / "catalog"))
    monkeypatch.setattr(catalog_cli, "_DEFAULT_OUTPUT", str(tmp_path / "compose.yml"))
    monkeypatch.setattr(catalog_cli, "_STACK_FILE", str(tmp_path / "stack.yml"))
    env_file = tmp_path / "stack.env"
    env_file.write_text("FIXTURE_ONLY=1\n", encoding="utf-8")
    calls: list[list[str]] = []

    class _Done:
        returncode = 0

    def _fake_run(command: list[str], **_: object) -> _Done:
        calls.append(list(command))
        return _Done()

    monkeypatch.setattr(catalog_cli.subprocess, "run", _fake_run)

    assert catalog_cli.cmd_up(["b", "--env-file", str(env_file)]) == 0

    rm_calls = [c for c in calls if "rm" in c]
    assert rm_calls == [
        [
            "docker",
            "compose",
            "-f",
            str(tmp_path / "compose.yml"),
            "rm",
            "-f",
            "--stop",
            "-v",
        ]
    ]


# --- OMN-19972: the laptop profile publishes on loopback only and names no lab host.
#
# Failure modes these pin (spec, workflow/records/plans/OMN-19972):
#   1. a published port binds all interfaces in the ``local`` render;
#   2. a lab address (the .201 LAN IP, the tailnet domain, the lab hostname)
#      appears anywhere in the ``local`` render;
#   3. the loopback bind or the default override leaks into another bundle,
#      which the lab lanes render from the same shared manifests.

_LOOPBACK = "127.0.0.1"
_LAB_HOST_MARKERS = ("192.168.86.", "tail75df5e", "omninode-pc")


def _published_ports(compose: dict[str, object]) -> dict[str, list[str]]:
    services = compose["services"]
    assert isinstance(services, dict)
    return {
        name: [str(p) for p in svc["ports"]]
        for name, svc in services.items()
        if svc.get("ports")
    }


def _lab_host_hits(compose: dict[str, object]) -> list[str]:
    text = yaml.safe_dump(compose, sort_keys=True)
    return [marker for marker in _LAB_HOST_MARKERS if marker in text]


def test_local_render_publishes_every_port_on_loopback_only() -> None:
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    )
    published = _published_ports(compose)
    # The five host ports the laptop guide names; a render that published none
    # would pass the loopback check vacuously.
    assert {
        "postgres",
        "redpanda",
        "valkey",
        "omninode-runtime",
        "runtime-effects",
    } <= set(published)
    not_loopback = {
        name: ports
        for name, ports in published.items()
        if any(not p.startswith(f"{_LOOPBACK}:") or p.count(":") != 2 for p in ports)
    }
    assert not_loopback == {}


def test_local_render_names_no_lab_host() -> None:
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    )
    assert _lab_host_hits(compose) == []


def test_lab_host_check_can_fail() -> None:
    """Positive control: the shared lab-facing render still carries the .201 default."""
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["core"])
    )
    assert "192.168." in "".join(_lab_host_hits(compose))


@pytest.mark.parametrize("bundle", ["core", "runtime"])
def test_other_bundles_keep_all_interface_ports_and_the_lab_advertise_default(
    bundle: str,
) -> None:
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([bundle])
    )
    published = _published_ports(compose)
    assert published
    for ports in published.values():
        for port in ports:
            assert port.count(":") == 1, port
            assert not port.startswith(f"{_LOOPBACK}:"), port
    services = compose["services"]
    assert isinstance(services, dict)
    command = " ".join(services["redpanda"]["command"])
    assert "${REDPANDA_ADVERTISE_HOST:-192.168.86.201}" in command


# --- OMN-19972 hostile-review follow-ups: env_default_overrides must fail loudly
# rather than render a broken compose file, and must not trip over a non-string
# command part.


def _override_catalog(
    tmp_path: Path, command: list[object], overrides: dict[str, str]
) -> CatalogResolver:
    services = tmp_path / "services"
    services.mkdir()
    manifest = _fixture_manifest("svc", "infrastructure", [])
    manifest["command"] = command
    (services / "svc.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (tmp_path / "bundles.yaml").write_text(
        yaml.safe_dump(
            {
                "b": {
                    "description": "b",
                    "services": ["svc"],
                    "env_default_overrides": overrides,
                }
            }
        ),
        encoding="utf-8",
    )
    return CatalogResolver(catalog_dir=str(tmp_path))


def test_override_of_a_nested_default_is_refused(tmp_path: Path) -> None:
    """``${A:-${B:-x}}`` cannot be rewritten safely; a half-rewrite breaks compose."""
    resolver = _override_catalog(
        tmp_path, ["run", "--addr ${ADV:-${OTHER:-lab}}:1"], {"ADV": "localhost"}
    )
    with pytest.raises(ValueError, match="nested"):
        generate_compose(resolver.resolve(["b"]))


@pytest.mark.parametrize("bad", ["a}b", "a$b"])
def test_override_value_that_would_break_interpolation_is_refused(
    tmp_path: Path, bad: str
) -> None:
    resolver = _override_catalog(tmp_path, ["run"], {"ADV": bad})
    with pytest.raises(ValueError, match="env_default_overrides"):
        resolver.resolve(["b"])


def test_override_leaves_a_non_string_command_part_untouched(tmp_path: Path) -> None:
    resolver = _override_catalog(
        tmp_path, ["sleep", 5, "--addr ${ADV:-lab}"], {"ADV": "localhost"}
    )
    compose = generate_compose(resolver.resolve(["b"]))
    services = compose["services"]
    assert isinstance(services, dict)
    assert services["svc"]["command"] == ["sleep", 5, "--addr ${ADV:-localhost}"]


# --- OMN-19972 demo half (plan T4.1): the services the six pages read ----------
#
# Failure modes these tests are written against, each shown failing on dev
# 83fa0e0c3 before the change:
#   1. an added service is missing from the laptop render, or runs with no
#      health signal, so the CI boot cannot tell it is dead;
#   2. the laptop's projection API keeps port 3002, the lab lanes' port;
#   3. the projection API starts before the kernel has provisioned the exposure
#      topics (measured on the lakshman lane 2026-09-30: it waits 300 s, exits
#      with "no partition metadata" and loops);
#   4. a laptop-only override leaks into another bundle rendered from the same
#      shared manifests;
#   5. the llm-cost writer names an image no catalog render builds.
#   6. the projection API is given no Kafka broker and exits at startup (measured in
#      CI run 36729468068 attempt 2: "projection-api requires Kafka bootstrap
#      servers", 16 restarts, never healthy).

_PAGE_SERVICES = (
    "projection-api",
    "omnimarket-projection-llm-cost",
    "consumer-health-projection",
)
_LAPTOP_PROJECTION_API_PORT = 3102


def _render(*bundles: str) -> dict[str, object]:
    return generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(list(bundles))
    )


def _services(compose: dict[str, object]) -> dict[str, dict[str, object]]:
    services = compose["services"]
    assert isinstance(services, dict)
    return services


def test_local_render_carries_the_services_the_pages_read() -> None:
    assert set(_PAGE_SERVICES) <= set(_services(_render("local")))


def test_every_long_running_local_service_has_a_healthcheck() -> None:
    services = _services(_render("local"))
    unwatched = sorted(
        name
        for name, svc in services.items()
        if svc.get("restart") != "no" and "healthcheck" not in svc
    )
    assert unwatched == []


def test_local_projection_api_publishes_a_laptop_port_not_the_lab_port() -> None:
    ports = _services(_render("local"))["projection-api"]["ports"]
    assert ports == [f"{_LOOPBACK}:{_LAPTOP_PROJECTION_API_PORT}:3002"]


def test_other_bundles_keep_the_projection_api_on_3002() -> None:
    ports = _services(_render("runtime-observability-projections"))["projection-api"][
        "ports"
    ]
    assert ports == ["3002:3002"]


def test_local_projection_api_waits_for_the_kernel_that_provisions_its_topics() -> None:
    depends_on = _services(_render("local"))["projection-api"]["depends_on"]
    assert isinstance(depends_on, dict)
    assert depends_on.get("omninode-runtime") == {"condition": "service_healthy"}


def test_other_bundles_do_not_gain_the_kernel_dependency() -> None:
    depends_on = _services(_render("runtime-observability-projections"))[
        "projection-api"
    ]["depends_on"]
    assert isinstance(depends_on, dict)
    assert "omninode-runtime" not in depends_on


def test_local_projection_api_is_given_the_compose_broker() -> None:
    env = _services(_render("local"))["projection-api"]["environment"]
    assert isinstance(env, dict)
    assert env.get("KAFKA_BROKERS") == "redpanda:9092"


def test_llm_cost_writer_builds_from_the_runtime_image_and_reports_ready() -> None:
    for bundle in ("local", "omnimarket-projections"):
        svc = _services(_render(bundle))["omnimarket-projection-llm-cost"]
        assert "omnimarket-projection:latest" not in str(svc["image"]), bundle
        test = svc["healthcheck"]["test"]  # type: ignore[index]
        assert "/ready" in " ".join(test), bundle
        env = svc["environment"]
        assert isinstance(env, dict)
        assert env.get("PROJECTION_RUNNER_HEALTH_PORT"), bundle


def test_two_bundles_overriding_one_service_port_differently_are_refused(
    tmp_path: Path,
) -> None:
    import shutil

    catalog = tmp_path / "catalog"
    shutil.copytree(_CATALOG_DIR, catalog)
    bundles_file = catalog / "bundles.yaml"
    bundles = yaml.safe_load(bundles_file.read_text())
    bundles["other-laptop"] = {
        "description": "fixture: a second bundle overriding the same port",
        "services": ["projection-api"],
        "port_overrides": {"projection-api": 3999},
    }
    bundles_file.write_text(yaml.safe_dump(bundles, sort_keys=False))
    with pytest.raises(ValueError, match=r"[Pp]ort override conflict"):
        CatalogResolver(catalog_dir=str(catalog)).resolve(["local", "other-laptop"])


# The resolver's other override guards (hostile review, both models: each was
# untested). Each test adds fixture bundles to a copy of the real catalog.


def _catalog_with(tmp_path: Path, extra: dict[str, object]) -> CatalogResolver:
    import shutil

    catalog = tmp_path / "catalog"
    shutil.copytree(_CATALOG_DIR, catalog)
    bundles_file = catalog / "bundles.yaml"
    bundles = yaml.safe_load(bundles_file.read_text())
    bundles.update(extra)
    bundles_file.write_text(yaml.safe_dump(bundles, sort_keys=False))
    return CatalogResolver(catalog_dir=str(catalog))


def test_two_bundles_waiting_differently_on_one_dependency_are_refused(
    tmp_path: Path,
) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "other-laptop": {
                "description": "fixture: waits on the kernel as merely started",
                "services": ["projection-api"],
                "extra_depends_on": {
                    "projection-api": {"omninode-runtime": "service_started"}
                },
            }
        },
    )
    with pytest.raises(ValueError, match="Extra dependency conflict"):
        resolver.resolve(["local", "other-laptop"])


def test_extra_dependency_on_a_service_the_stack_does_not_run_is_refused(
    tmp_path: Path,
) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "probe": {
                "description": "fixture: redpanda alone, waiting on postgres",
                "services": ["redpanda"],
                "extra_depends_on": {"redpanda": {"postgres": "service_healthy"}},
            }
        },
    )
    with pytest.raises(ValueError, match="Extra dependency names 'postgres'"):
        resolver.resolve(["probe"])


def test_port_override_for_a_service_the_stack_does_not_run_is_refused(
    tmp_path: Path,
) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "probe": {
                "description": "fixture: overrides a service it does not run",
                "services": ["redpanda"],
                "port_overrides": {"projection-api": 3999},
            }
        },
    )
    with pytest.raises(ValueError, match="Port override for 'projection-api'"):
        resolver.resolve(["probe"])


def test_port_override_for_a_service_that_publishes_no_port_is_refused(
    tmp_path: Path,
) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "probe": {
                "description": "fixture: overrides a writer that publishes no port",
                "services": ["omnimarket-projection-llm-cost"],
                "port_overrides": {"omnimarket-projection-llm-cost": 3999},
            }
        },
    )
    with pytest.raises(ValueError, match="publishes no port"):
        resolver.resolve(["probe"])


def test_unknown_dependency_condition_is_refused(tmp_path: Path) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "probe": {
                "description": "fixture: a condition compose does not know",
                "services": ["projection-api"],
                "extra_depends_on": {"projection-api": {"redpanda": "service_happy"}},
            }
        },
    )
    # Resolved alone, so no other bundle's condition can conflict with it: only
    # the condition check can refuse (with "local", the conflict guard answered
    # first and this test passed with the condition check removed).
    with pytest.raises(ValueError, match="'service_happy' is not a valid"):
        resolver.resolve(["probe"])

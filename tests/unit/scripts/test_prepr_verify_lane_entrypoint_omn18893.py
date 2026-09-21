# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Task 4 of epic OMN-18888: the pool bring-up entrypoint and its refusals.

What this module pins, and why each pin is behavioural rather than textual
-------------------------------------------------------------------------
Task 4's acceptance criterion says the entrypoint's refusal of every declared
lane is "asserted by a test that reads the policy table rather than matching
text". That phrasing is the whole design constraint: a test that greps
``prepr_verify_lane.sh`` for the word ``stability-test`` passes just as happily
against a comment mentioning the lane as against a branch refusing it. So the
refusals live in ``scripts/runtime_build/prepr_slot_policy.py`` as data, and
the tests below call the same functions the shell calls.

Three further pins exist because the corresponding defect is invisible in a
diff:

* The rendered-slot verifier is exercised against a synthetic render carrying
  each isolation defect ONE at a time, with a passing render as the positive
  control. Without that control, a verifier that returned "no problems" for
  every input would look identical to a working one.
* The compose overlay is parsed and checked against the lane manifest
  OMN-18890 declared, so the twelve container names cannot drift from the set
  the census reconciles against.
* The shell's exit-code literals are compared to the Python constants they
  mirror, because a shell cannot import them and a caller branching on the
  wrong number gets the wrong refusal.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
ENTRYPOINT = RUNTIME_BUILD / "prepr_verify_lane.sh"
POLICY_PATH = RUNTIME_BUILD / "prepr_slot_policy.py"
VERIFIER_PATH = RUNTIME_BUILD / "prepr_verify_rendered_slot.py"
OVERLAY = REPO_ROOT / "docker" / "docker-compose.prepr.yml"
MANIFEST = REPO_ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"

pytestmark = pytest.mark.unit


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = _load("prepr_slot_policy", POLICY_PATH)
verifier = _load("prepr_verify_rendered_slot", VERIFIER_PATH)


# ---------------------------------------------------------------------------
# The refusal table, read as data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("project", sorted(policy.DECLARED_LANE_PROJECTS))
def test_every_declared_lane_is_refused_by_name(project: str) -> None:
    """Rule 24(e): the entrypoint "refuses every declared lane by name"."""
    with pytest.raises(policy.RefusalError) as excinfo:
        policy.assert_target_is_a_pool_slot(project)
    assert excinfo.value.code == policy.EXIT_REFUSED_DECLARED_LANE
    assert project in str(excinfo.value)


def test_the_governed_lanes_are_all_in_the_refusal_table() -> None:
    """A governed lane absent from the table would be refused as *unknown*.

    That is still a refusal, so the slot would not build -- but it would be
    refused for the wrong reason, under a different exit code, with a message
    saying the project is unrecognised rather than that it is governed. The
    distinction matters the day someone reads the exit code to decide whether
    a refusal was a policy decision or a typo.
    """
    for lane_project in (
        "omnibase-infra",
        "omnibase-infra-stability-test",
        "omnibase-infra-judge",
        "omnibase-infra-prod",
        "omnibase-infra-lakshman",
    ):
        assert lane_project in policy.DECLARED_LANE_PROJECTS


def test_an_unknown_project_is_refused_rather_than_created() -> None:
    with pytest.raises(policy.RefusalError) as excinfo:
        policy.assert_target_is_a_pool_slot("omnibase-infra-typo")
    assert excinfo.value.code == policy.EXIT_REFUSED_UNKNOWN_PROJECT


@pytest.mark.parametrize("slot", sorted(policy.SLOTS))
def test_each_pool_slot_resolves_to_itself(slot: int) -> None:
    resolved = policy.resolve_slot(slot)
    assert policy.assert_target_is_a_pool_slot(resolved.compose_project) == resolved


def test_a_slot_number_outside_the_pool_is_refused() -> None:
    with pytest.raises(policy.RefusalError) as excinfo:
        policy.resolve_slot(3)
    assert excinfo.value.code == policy.EXIT_USAGE


def test_the_pool_build_lock_is_not_a_deployable_project() -> None:
    """The build lock's name must not be reachable as a deploy target.

    ``lane_lock.py`` keys a lock on any string, so the pool's build lock name
    is an ordinary compose-project-shaped value. If it were also in ``SLOTS``
    a caller could aim a bring-up at it.
    """
    assert policy.POOL_BUILD_LOCK_PROJECT not in {
        p.compose_project for p in policy.SLOTS.values()
    }
    with pytest.raises(policy.RefusalError) as excinfo:
        policy.assert_target_is_a_pool_slot(policy.POOL_BUILD_LOCK_PROJECT)
    assert excinfo.value.code == policy.EXIT_REFUSED_UNKNOWN_PROJECT


# ---------------------------------------------------------------------------
# The entrypoint declares no lane, force or skip option
# ---------------------------------------------------------------------------


def test_the_entrypoint_declares_no_lane_force_or_skip_option() -> None:
    """Read the script's own option strings, not its prose.

    The parser arms are the only place an option can be accepted, so matching
    ``--lane)`` as a case arm is a behavioural read: a comment or a heredoc
    mentioning ``--lane`` does not create one. Shaped after the prod gate's
    ``test_no_entrypoint_declares_a_health_status_option``.
    """
    text = ENTRYPOINT.read_text(encoding="utf-8")
    arms = set(re.findall(r"^\s*(--[a-z-]+)(?:\s*\|\s*(-[a-z]))?\)", text, re.M))
    declared = {arm for arm, _ in arms}
    for forbidden in ("--lane", "--compose-project", "--force", "--skip", "--cold"):
        assert forbidden not in declared, (
            f"{ENTRYPOINT.name} declares {forbidden}, which rule 24(e) forbids: "
            f"the sanction for building a workspace image is bound to an "
            f"entrypoint that CANNOT be aimed at a declared lane. Adding an "
            f"option that names a lane, or that forces or skips a refusal, "
            f"reopens the path this entrypoint exists to close."
        )
    assert "--slot" in declared and "--worktree" in declared


def test_the_shell_exit_codes_match_the_policy_module() -> None:
    """A shell cannot import a Python constant, so the duplication is pinned."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    for name in [n for n in dir(policy) if n.startswith("EXIT_")]:
        expected = getattr(policy, name)
        match = re.search(rf"^{name}=(\d+)$", text, re.M)
        if match is None:
            continue
        assert int(match.group(1)) == expected, (
            f"{ENTRYPOINT.name} spells {name}={match.group(1)} but "
            f"prepr_slot_policy.{name} is {expected}. A caller branching on "
            f"the exit code would misread which refusal fired."
        )


def test_the_entrypoint_never_writes_to_a_canonical_clone() -> None:
    """No checkout, fetch, reset or clean anywhere in the entrypoint.

    The plan's provenance requirement is that a slot "never checks out,
    fetches into, or otherwise writes to a canonical clone on the host"; the
    existing workspace staging helper does all four, which is why this script
    stages privately instead of calling it.
    """
    text = ENTRYPOINT.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    for verb in ("git checkout", "git fetch", "git reset", "git clean", "git stash"):
        assert verb not in body, (
            f"{ENTRYPOINT.name} runs '{verb}'. A slot build must be read-only "
            f"against every clone on the host: two concurrent slots and the dev "
            f"lane's own builds all read those trees."
        )
    assert "stage_workspace.sh" not in body


# ---------------------------------------------------------------------------
# Attribution: a reason is required, the grant interlock is not extended
# ---------------------------------------------------------------------------


def _attribution_module() -> Any:
    return _load(
        "preflight_lane_deploy_attribution",
        REPO_ROOT / "scripts" / "preflight_lane_deploy_attribution.py",
    )


def test_pool_lane_names_do_not_drift_between_the_two_declarations() -> None:
    """The attribution preflight spells the pool lanes as a literal.

    It does so deliberately -- see its constants block -- so that the live
    stability gate gains no import dependency on a new, ephemeral module. This
    is the pin that makes the duplication safe.
    """
    attribution = _attribution_module()
    assert attribution.POOL_LANES == policy.POOL_LANE_NAMES


def test_a_pool_lane_requires_a_reason_and_is_not_promotion_class() -> None:
    attribution = _attribution_module()
    for lane in sorted(policy.POOL_LANE_NAMES):
        assert lane in attribution.REASON_REQUIRED_LANES, (
            f"{lane} may be built with no stated reason. A slot build mutates a "
            f"shared host and owes attribution exactly as a stability rebuild does."
        )
        assert lane not in attribution.GOVERNED_LANES, (
            f"{lane} is in GOVERNED_LANES, which carries promotion-class meaning "
            f"elsewhere. A slot is a premise for nothing."
        )
        assert lane not in attribution.GRANT_INTERLOCK_LANES, (
            f"{lane} is in GRANT_INTERLOCK_LANES. A slot promotes nothing and "
            f"sources no stability-proven digest, so no rebuild of one can erode "
            f"a live prod grant -- and gating a branch build on resolving the "
            f"change-control repository would push authors off this entrypoint."
        )


def test_the_dev_lane_reason_requirement_is_unchanged() -> None:
    """Negative control on the widening: dev must NOT have become governed."""
    attribution = _attribution_module()
    assert "dev" not in attribution.REASON_REQUIRED_LANES
    assert (
        frozenset({"stability-test", "prod", "judge"}) == attribution.GOVERNED_LANES
    ), "the pool must be added to REASON_REQUIRED_LANES, never to GOVERNED_LANES."


# ---------------------------------------------------------------------------
# The rendered-slot verifier, with a positive control
# ---------------------------------------------------------------------------


def _good_render(slot: int = 1) -> dict[str, Any]:
    """A synthetic render that satisfies every check.

    Built from the policy table rather than typed out, so it cannot silently
    encode the old expectations after a policy change.
    """
    p = policy.resolve_slot(slot)
    suffix = f"_{p.db_slot}"

    def env(extra: dict[str, Any] | None = None) -> dict[str, Any]:
        base = {
            "KAFKA_TOPIC_NAMESPACE": p.topic_namespace,
            "KAFKA_ENVIRONMENT": p.kafka_environment,
            "VALKEY_DB": str(p.valkey_db_index),
            "OMNIBASE_INFRA_DB_URL": f"postgresql://role_omnibase{suffix}:pw@postgres:5432/omnibase_infra{suffix}",
            "OMNIDASH_ANALYTICS_DB_URL": f"postgresql://role_omnidash{suffix}:pw@postgres:5432/omnidash_analytics{suffix}",
        }
        base.update(extra or {})
        return base

    services: dict[str, Any] = {}
    for name in verifier.SLOT_SERVICES:
        kind = (
            "omnimarket" if name.startswith(("projection-", "tenant-")) else "omninode"
        )
        services[name] = {
            "container_name": f"{kind}-prepr-{p.slot}-{name}",
            "environment": env(
                {
                    "KAFKA_CONSUMER_GROUP": f"{p.db_slot}.omnimarket-projections.{name}.consume.v1"
                }
                if name.startswith("projection-")
                else None
            ),
            "ports": [],
            "volumes": [
                {
                    "type": "volume",
                    "source": f"prepr_{name}_logs",
                    "target": "/app/logs",
                }
            ],
        }
    services["omninode-runtime"]["ports"] = [
        {"mode": "ingress", "target": 8085, "published": str(p.runtime_main_port)}
    ]
    return {
        "services": services,
        "volumes": {
            "prepr_runtime_logs": {"name": f"omninode-prepr-{p.slot}-runtime-logs"}
        },
        "networks": {
            "omnibase-infra-network": {
                "name": "omnibase-infra-network",
                "external": True,
            }
        },
    }


def test_positive_control_a_correct_render_reports_no_problems() -> None:
    """Without this, a verifier that never reports anything looks correct."""
    assert verifier.check(_good_render(), policy.resolve_slot(1), False) == []


def _mutate(fn: Any) -> list[str]:
    rendered = _good_render()
    fn(rendered)
    return verifier.check(rendered, policy.resolve_slot(1), False)


def test_an_ambient_kafka_environment_leak_is_caught() -> None:
    """The live defect this gate was written for, measured 2026-09-21.

    Compose resolves an interpolation from the OS environment before
    ``--env-file``, and the lab host's interactive shell exports
    ``KAFKA_ENVIRONMENT=local`` -- the dev lane's own consumer-group token. A
    slot brought up from a terminal inherits it and joins the dev lane's
    consumer groups while every file on disk says it is isolated.
    """
    problems = _mutate(
        lambda r: r["services"]["omninode-runtime"]["environment"].update(
            {"KAFKA_ENVIRONMENT": "local"}
        )
    )
    assert any("KAFKA_ENVIRONMENT" in p for p in problems)


def test_an_appended_dev_port_is_caught() -> None:
    """Compose APPENDS sequences unless the override carries ``!override``."""
    problems = _mutate(
        lambda r: r["services"]["omninode-runtime"]["ports"].append(
            {"mode": "ingress", "target": 8085, "published": "8085"}
        )
    )
    assert any("outside the slot's reserved block" in p for p in problems)


def test_an_inherited_global_volume_is_caught() -> None:
    """The inherited volume names are GLOBAL and shared with the live dev lane."""
    problems = _mutate(
        lambda r: r["services"]["omninode-runtime"]["volumes"].append(
            {"type": "volume", "source": "runtime_logs", "target": "/app/logs"}
        )
    )
    assert any("not slot-scoped" in p for p in problems)


def test_an_inherited_dev_consumer_group_is_caught() -> None:
    problems = _mutate(
        lambda r: r["services"]["projection-delegation-writer"]["environment"].update(
            {
                "KAFKA_CONSUMER_GROUP": "local.omnimarket-projections.delegation-writer.consume.v1"
            }
        )
    )
    assert any("KAFKA_CONSUMER_GROUP" in p for p in problems)


def test_an_unsuffixed_superuser_dsn_is_caught() -> None:
    """The inherited x-runtime-env DSNs resolve to the SUPERUSER by default."""
    problems = _mutate(
        lambda r: r["services"]["omninode-runtime"]["environment"].update(
            {
                "OMNIBASE_INFRA_DB_URL": "postgresql://postgres:pw@postgres:5432/omnibase_infra"
            }
        )
    )
    assert any("does not end in" in p for p in problems)
    assert any("connects as" in p for p in problems)


def test_a_non_external_network_is_caught() -> None:
    problems = _mutate(
        lambda r: r["networks"]["omnibase-infra-network"].update({"external": False})
    )
    assert any("not external" in p for p in problems)


def test_a_missing_slot_service_is_caught() -> None:
    problems = _mutate(lambda r: r["services"].pop("runtime-worker"))
    assert any("missing" in p for p in problems)


def test_an_unfenced_shared_dependency_is_caught() -> None:
    problems = _mutate(lambda r: r["services"].update({"postgres": {}}))
    assert any("not the slot's" in p for p in problems)


def test_the_verifier_fails_closed_on_an_unreadable_render(tmp_path: Path) -> None:
    """A gate that cannot read its own input has not passed, it has not run."""
    missing = tmp_path / "absent.json"
    assert (
        verifier.main(["--rendered", str(missing), "--slot", "1"])
        == policy.EXIT_PROVENANCE_MISMATCH
    )
    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json", encoding="utf-8")
    assert (
        verifier.main(["--rendered", str(malformed), "--slot", "1"])
        == policy.EXIT_PROVENANCE_MISMATCH
    )


# ---------------------------------------------------------------------------
# The overlay, pinned against the lane manifest OMN-18890 declared
# ---------------------------------------------------------------------------


class _TagTolerantLoader(yaml.SafeLoader):
    """Load the overlay without resolving compose's ``!override``/``!reset``."""


def _passthrough(loader: Any, _suffix: str, node: Any) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_TagTolerantLoader.add_multi_constructor("!", _passthrough)
_TagTolerantLoader.add_multi_constructor("tag:yaml.org,2002:merge", _passthrough)


def _overlay_for_slot(slot: int) -> dict[str, Any]:
    text = OVERLAY.read_text(encoding="utf-8")
    text = re.sub(r"\$\{ONEX_PREPR_SLOT[^}]*\}", str(slot), text)
    text = re.sub(r"\$\{ONEX_DB_SLOT[^}]*\}", policy.resolve_slot(slot).db_slot, text)
    # The S506 suppression below is justified: _TagTolerantLoader subclasses
    # SafeLoader and adds only
    # pass-through constructors for compose's own `!override` / `!!merge`
    # tags, so no arbitrary object can be instantiated. safe_load itself
    # cannot be used here -- it raises on those tags, which is the reason this
    # loader exists.
    return yaml.load(text, Loader=_TagTolerantLoader)  # noqa: S506


def _manifest_lane(lane: str) -> dict[str, Any]:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["lanes"][lane]


@pytest.mark.parametrize("slot", sorted(policy.SLOTS))
def test_the_policy_table_matches_the_lane_manifest(slot: int) -> None:
    """OMN-18890 declared the pool before anything was built. Keep them equal.

    A compose project here that the manifest does not declare is a lane the
    census cannot reconcile, so a running slot would report as drift forever.
    """
    lane = _manifest_lane(f"prepr-{slot}")
    assert lane["compose_project"] == policy.resolve_slot(slot).compose_project
    assert lane["compose_file"] == "docker/docker-compose.prepr.yml"
    assert lane["network"] == "omnibase-infra-network"
    assert lane["optional"] is True, (
        "an empty pool is the STEADY state; a non-optional pool lane files a "
        "census ticket on every tick for a pool working as designed."
    )


@pytest.mark.parametrize("slot", sorted(policy.SLOTS))
def test_the_overlay_container_names_match_the_manifest_services(slot: int) -> None:
    declared = {s["name"] for s in _manifest_lane(f"prepr-{slot}")["services"]}
    overlay = _overlay_for_slot(slot)
    rendered_names = {
        svc["container_name"]
        for svc in overlay["services"].values()
        if isinstance(svc, dict) and "container_name" in svc
    }
    missing = declared - rendered_names
    assert not missing, (
        f"the lane manifest declares {sorted(missing)} for slot {slot} and the "
        f"overlay renders no container by that name. The census reconciles a "
        f"running slot against the manifest, so a name that differs reads as "
        f"both a missing service and an unexplained container."
    )


def test_every_shared_dependency_is_fenced_behind_an_unreachable_profile() -> None:
    """The fence, not the caller, is what stops a bare ``up``.

    postgres, redpanda, valkey, keycloak, infisical and the redpanda one-shots
    carry NO profile in the inherited files, so without this they start on a
    bare ``up`` under their hardcoded DEV container names -- the OMN-13581
    failure that destroyed the stability lane's broker for three days.
    """
    overlay = _overlay_for_slot(1)
    fenced_profile = "prepr-never-start-a-shared-dependency"
    for name in (
        "postgres",
        "redpanda",
        "redpanda-partition-cap",
        "redpanda-scram-user",
        "redpanda-sasl-enable",
        "valkey",
        "keycloak",
        "infisical",
        "migration-gate",
        "autoheal",
    ):
        svc = overlay["services"].get(name)
        assert svc is not None, f"{name} is not fenced by the overlay at all."
        assert fenced_profile in (svc.get("profiles") or []), (
            f"{name} is not on the unreachable profile. A bare "
            f"`docker compose -p <slot> up` would create it under the DEV "
            f"container name and displace the dev lane's own container."
        )


def test_the_overlay_replaces_rather_than_appends_every_sequence_field() -> None:
    """``ports``, ``volumes``, ``profiles`` and ``depends_on`` must be replaced.

    Compose merges a sequence by APPENDING unless the entry carries
    ``!override``. An appended ``ports`` binds the dev lane's port; an appended
    ``volumes`` mounts the dev lane's data. Both were reproduced against a real
    render before this assertion was written.
    """
    text = OVERLAY.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        for field in ("ports:", "volumes:", "profiles:", "depends_on:"):
            if stripped.startswith(field) and line.startswith("    "):
                assert "!override" in line, (
                    f"'{stripped}' does not carry !override. Compose would "
                    f"APPEND it to the inherited value rather than replace it."
                )


def test_every_slot_volume_name_carries_the_slot() -> None:
    overlay = _overlay_for_slot(1)
    for key, spec in (overlay.get("volumes") or {}).items():
        name = (spec or {}).get("name", "")
        assert "prepr-1" in name, (
            f"volume '{key}' resolves to the global Docker name {name!r}. The "
            f"inherited names are shared with the RUNNING dev lane, so a slot "
            f"would write into its data and `down -v` would destroy it."
        )


def test_the_slot_network_is_joined_not_created() -> None:
    overlay = _overlay_for_slot(1)
    assert overlay["networks"]["omnibase-infra-network"]["external"] is True


# ---------------------------------------------------------------------------
# The pool cannot be reached through the ordinary deploy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slot", sorted(policy.SLOTS))
def test_deploy_runtime_cannot_resolve_a_pool_project(slot: int) -> None:
    """``resolve_lane_overlay_filename`` must refuse a pool project.

    This is the other half of rule 24(e). The new entrypoint refusing declared
    lanes is worth nothing if the declared-lane deploy path can reach a slot,
    because then a pool arm exists after all -- just in the other direction.
    ``compose_files.sh`` fails closed on an unknown lane, and this asserts that
    nobody has since added a convenience arm for ``prepr-*``.
    """
    project = policy.resolve_slot(slot).compose_project
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{REPO_ROOT}/scripts/runtime_build/compose_files.sh" && '
            f'resolve_lane_overlay_filename "{project}"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, (
        f"scripts/deploy-runtime.sh can resolve an overlay for {project} "
        f"(stdout={result.stdout!r}). The governed deploy path must not be "
        f"able to target a pool slot; the pool has its own entrypoint."
    )


def test_slot_ports_do_not_collide_with_any_live_declared_lane() -> None:
    """A slot must never bind a port a live lane already publishes.

    The one deliberate overlap is the RETIRED compose ``prod`` lane, whose
    28085/28086 block slot 1 reuses -- that is why the block was free
    (OMN-18320). It is named here rather than filtered silently, so that
    restoring a compose prod lane trips this test instead of colliding at
    runtime.
    """
    contract = yaml.safe_load(
        (
            REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
        ).read_text(encoding="utf-8")
    )
    retired = {"prod"}
    live_ports: set[int] = set()
    for name, profile in contract["profiles"].items():
        if name in retired:
            continue
        live_ports.add(int(profile["main_port"]))
        live_ports.add(int(profile["effects_port"]))

    for slot_policy in policy.SLOTS.values():
        for port in (
            slot_policy.runtime_main_port,
            slot_policy.runtime_effects_port,
            slot_policy.gateway_port,
            slot_policy.projection_api_port,
        ):
            assert port not in live_ports, (
                f"slot {slot_policy.slot} publishes {port}, which a live lane "
                f"already declares in the runtime policy contract."
            )


def test_no_slot_uses_the_dev_lanes_valkey_index() -> None:
    """The dev lane's URL carries no index, so it uses logical database 0."""
    for slot_policy in policy.SLOTS.values():
        assert slot_policy.valkey_db_index >= 1


def test_slot_identities_are_pairwise_disjoint() -> None:
    """Two slots sharing any axis is two slots that are one slot."""
    fields = (
        "compose_project",
        "db_slot",
        "topic_namespace",
        "kafka_environment",
        "valkey_db_index",
        "runtime_main_port",
        "runtime_effects_port",
        "gateway_port",
        "projection_api_port",
    )
    for field in fields:
        values = [getattr(p, field) for p in policy.SLOTS.values()]
        assert len(values) == len(set(values)), f"slots share a {field}."


# ---------------------------------------------------------------------------
# The policy CLI the shell actually calls
# ---------------------------------------------------------------------------


def _run_policy(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(POLICY_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_policy_cli_returns_the_named_refusal_codes() -> None:
    governed = _run_policy("--assert-pool-slot", "omnibase-infra-stability-test")
    assert governed.returncode == policy.EXIT_REFUSED_DECLARED_LANE
    unknown = _run_policy("--assert-pool-slot", "omnibase-infra-nope")
    assert unknown.returncode == policy.EXIT_REFUSED_UNKNOWN_PROJECT
    ok = _run_policy("--assert-pool-slot", "omnibase-infra-prepr-1")
    assert ok.returncode == 0
    assert json.loads(ok.stdout)["db_slot"] == "prepr1"


def test_the_slot_env_carries_every_variable_the_overlay_requires() -> None:
    """Each ``:?`` interpolation in the overlay must be satisfiable.

    A slot-scoped variable the policy does not emit means the compose
    invocation aborts at bring-up time rather than here -- after the build has
    already cost twenty-five minutes.
    """
    emitted = set(policy.slot_environment(policy.resolve_slot(1)))
    text = OVERLAY.read_text(encoding="utf-8")
    required = {
        match
        for match in re.findall(r"\$\{([A-Z_][A-Z0-9_]*):\?", text)
        if match.startswith(
            (
                "ONEX_PREPR",
                "PREPR_",
                "KAFKA_TOPIC_NAMESPACE",
                "KAFKA_ENVIRONMENT",
                "ONEX_DB_SLOT",
            )
        )
    }
    # The tenant state directory is created per run by the entrypoint rather
    # than declared in the policy table, because it is a host path and not a
    # slot identity.
    required.discard("ONEX_PREPR_TENANT_STATE_DIR")
    missing = required - emitted
    assert not missing, (
        f"the overlay requires {sorted(missing)} but slot_environment() emits "
        f"none of them, so a bring-up would abort after the build."
    )


def test_the_positive_control_still_carries_the_fields_it_controls() -> None:
    """Guard against the positive control being mutated into vacuity.

    ``_good_render`` is the control every negative test is a mutation of. If it
    ever stopped containing the fields the checks read, every negative test
    would still pass (the mutation would be of an absent field) and every real
    defect would go unnoticed.
    """
    rendered = _good_render()
    runtime = rendered["services"]["omninode-runtime"]
    assert runtime["environment"]["KAFKA_ENVIRONMENT"]
    assert runtime["ports"] and runtime["volumes"]
    assert copy.deepcopy(rendered) == rendered


# ---------------------------------------------------------------------------
# The provisioner must not assume a PostgreSQL client on the host
# ---------------------------------------------------------------------------

PROVISIONER = REPO_ROOT / "scripts" / "provision_db_slot.sh"


def test_the_provisioner_routes_every_psql_call_through_the_resolver() -> None:
    """No bare ``psql`` call may survive outside the resolver itself.

    The lab host has no PostgreSQL client: nothing on PATH, nothing under
    /usr/lib/postgresql, no postgresql-client package. It never needed one,
    because every migration seam in this repository runs psql inside a
    postgres:16-alpine container. The first live slot provisioning therefore
    died on ``psql: command not found`` after the source snapshot had been
    staged and the lane lock taken.

    This is asserted over the source rather than by running the script,
    because the failure is the ABSENCE of a binary and a test host that
    happens to have one cannot reproduce it. Every call site is routed through
    ``psql_run``, which picks the host binary when there is one and a
    throwaway container on the lane network when there is not.
    """
    lines = PROVISIONER.read_text(encoding="utf-8").splitlines()
    offenders: list[str] = []
    in_resolver = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if "psql_run()" in stripped or "command -v psql" in stripped:
            in_resolver = True
        if in_resolver and stripped == "}":
            in_resolver = False
        if in_resolver:
            continue
        if re.search(r"(?<![\w_])psql\s+-", stripped):
            offenders.append(f"{number}: {stripped}")
    assert not offenders, (
        "these call sites invoke psql directly instead of through psql_run, so "
        "they assume a PostgreSQL client on the host and will die with "
        "'command not found' on the lab host:\n  " + "\n  ".join(offenders)
    )


def test_the_provisioner_never_puts_the_password_on_a_command_line() -> None:
    """The container form passes a bare ``-e PGPASSWORD``, not a value.

    An inline ``-e PGPASSWORD=<value>`` would be visible in ``docker inspect``
    and in the host's process list for the life of the call. The bare form
    takes it from the caller's environment instead.
    """
    text = PROVISIONER.read_text(encoding="utf-8")
    assert "-e PGPASSWORD " in text or '-e PGPASSWORD "' in text
    assert "PGPASSWORD=$" not in text.replace('PGPASSWORD="$POSTGRES_PASSWORD"', "")


# ---------------------------------------------------------------------------
# Pinned-sha staging, and no rendered configuration on disk
# ---------------------------------------------------------------------------

CUT_LAB_REF = REPO_ROOT / "scripts" / "runtime_build" / "cut-lab-ref.sh"


def _entrypoint_code() -> str:
    """The entrypoint with comment lines stripped.

    Every assertion below is about what the script DOES. The comments
    describe the defects at length and name the very constructs being
    forbidden, so matching against the raw text would pass or fail on prose.
    """
    return "\n".join(
        line
        for line in ENTRYPOINT.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_snapshot_comes_from_the_object_store_not_a_working_tree() -> None:
    """A copy of a working directory is only the commit when it happens to be clean.

    OMN-19086 measured that failure staying silent: a git directory advanced
    to a new commit over an old working tree, 8,084 files differing, no error
    anywhere. ``git archive`` cannot express that state, because it
    serialises a commit's tree and nothing else.
    """
    body = _entrypoint_code()
    assert "archive --format=tar" in body, (
        "the entrypoint no longer stages from a resolved sha. A working-tree "
        "copy can contain content belonging to no commit, and the receipt "
        "would still name one."
    )
    assert "rsync" not in body, (
        "the entrypoint still rsyncs a source tree into the snapshot. That is "
        "the staging shape the pinned-sha rule refuses."
    )


def test_the_entrypoint_never_writes_a_rendered_configuration() -> None:
    """A rendered config expands every interpolation, credentials included.

    With the operator environment loaded it carries the broker, database and
    Keycloak values in clear, so it is piped into the verifier and never
    persisted. The verifier reads ``-`` for standard input.
    """
    body = _entrypoint_code()
    assert "--rendered -" in body, (
        "the entrypoint does not pipe the render into the verifier."
    )
    # A FILE redirect, not the stderr suppression that legitimately follows
    # the render. `2>/dev/null` discards compose's own diagnostics and
    # persists nothing; `> path` would write the document, credentials and
    # all. Matching a bare ">" would flag the safe form and make this test
    # unpassable, which is how a real gate gets deleted.
    offenders = [
        line.strip()
        for line in body.splitlines()
        if "config --format json" in line
        and re.search(r"(?<![0-9])>\s*\S", line.split("config --format json")[1])
    ]
    assert not offenders, (
        "the entrypoint redirects a rendered configuration to a file:\n  "
        + "\n  ".join(offenders)
        + "\nA render holds every expanded secret; it must not reach disk."
    )
    assert "rendered.json" not in body, (
        "the entrypoint still names a rendered.json path."
    )


def test_a_dirty_sibling_is_refused_rather_than_recorded() -> None:
    """Recording a discrepancy does not make an artifact reproducible."""
    body = _entrypoint_code()
    assert "EXIT_REFUSED_DIRTY_WORKTREE" in body
    assert body.count("EXIT_REFUSED_DIRTY_WORKTREE") >= 2, (
        "only one dirty-tree refusal exists. Both the target and each sibling "
        "must be refused when dirty: a dirty sibling puts content belonging "
        "to no commit into an image whose receipt names one."
    )


@pytest.mark.parametrize("slot", sorted(policy.SLOTS))
def test_cut_lab_ref_refuses_a_pool_lane_by_name(slot: int) -> None:
    """The fast-lane deploy driver must refuse a pool slot, not drive one.

    A pool arm there would be a SECOND path to build a slot, reachable with
    one argument, bypassing the entrypoint's declared-lane refusals, its
    attribution requirement, its build lock, its pinned-sha snapshot and its
    rendered-config gate. Rule 24(e)'s sanction is worth exactly as much as
    the entrypoint being the only way in.

    The refusal is BY NAME rather than falling through to "unknown lane",
    because a pool lane is not unknown: it is declared in the lane manifest
    and deliberately not driven from here. A caller needs to be told where it
    IS driven from, or they go looking for the arm to add.
    """
    lane = f"prepr-{slot}"
    result = subprocess.run(
        ["bash", str(CUT_LAB_REF), "--lane", lane],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "OMNI_HOME": str(REPO_ROOT)},
    )
    assert result.returncode == 2, (
        f"cut-lab-ref.sh did not refuse lane {lane} (exit {result.returncode}). "
        f"A second build path for a pool slot reopens what rule 24(e) closes."
    )
    assert "prepr_verify_lane.sh" in result.stderr, (
        f"the refusal for {lane} does not name the entrypoint that DOES build "
        f"a slot, so a caller is left looking for the arm to add here."
    )


# ---------------------------------------------------------------------------
# Provider and forge credentials must not ride in from the invoking shell
# ---------------------------------------------------------------------------

LAYERED_COMPOSE = (
    REPO_ROOT / "docker" / "docker-compose.infra.yml",
    REPO_ROOT / "docker" / "docker-compose.dev-lane.yml",
    REPO_ROOT / "docker" / "docker-compose.prepr.yml",
)


def _interpolation_count(name: str) -> int:
    """How many times the layered compose files interpolate ``name``."""
    needle = "${" + name
    return sum(f.read_text(encoding="utf-8").count(needle) for f in LAYERED_COMPOSE)


def test_the_interpolation_detector_works_before_any_zero_is_believed() -> None:
    """Positive and negative control on the counter the next test relies on.

    This control exists because the first two readings of the credential
    question returned zero for every name and both were WRONG: one from a
    shell that does not word-split an unquoted variable, one from a bracket
    expression that silently matched nothing. A zero from an unproven matcher
    reads exactly like a clean bill of health, which is the whole reason a
    positive control is mandatory before reporting one.
    """
    assert _interpolation_count("POSTGRES_PASSWORD") > 0, (
        "the detector finds no POSTGRES_PASSWORD interpolation, so it is "
        "broken and every zero it reports below is meaningless."
    )
    assert _interpolation_count("NO_SUCH_VARIABLE_XYZ") == 0


def _scrub_list() -> list[str]:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    match = re.search(r"PREPR_SCRUBBED_CREDENTIAL_VARS=\((.*?)\)", text, re.S)
    assert match, "the entrypoint declares no credential scrub list at all."
    return [n for n in match.group(1).split() if n and not n.startswith("#")]


def test_the_entrypoint_scrubs_credentials_before_reading_the_operator_env() -> None:
    """The unset must precede the source, and the reason is not the obvious one.

    Measured, because the first version of this docstring asserted a mechanism
    that turned out to be wrong. Four arms, a shell exporting a value and a
    file setting one:

    * unset BEFORE the source -> the file's value. Correct.
    * unset AFTER the source  -> UNSET. It strips the file's value too, not
      merely the shell's, so the wrong order breaks the slot by removing
      configuration it legitimately needs. It does NOT leave the shell's
      value behind, which is what this test previously claimed.
    * no unset, file sets the name -> the file's value. ``set -a; source``
      ASSIGNS, so it overwrites what the shell exported and there is no leak
      here at all.
    * no unset, file does NOT set the name -> the SHELL's value. This is the
      only leak shape, and it is the one the scrub exists for.

    So the scrub matters for names the operator env does not itself set, and
    the ordering matters because the wrong order strips the ones it does. Note
    this is a different mechanism from compose's own ``--env-file``, which is
    read by compose rather than assigned into the environment and therefore
    genuinely loses to an ambient value; conflating the two is what produced
    the wrong claim.
    """
    body = _entrypoint_code()
    assert "PREPR_SCRUBBED_CREDENTIAL_VARS" in body
    scrub_at = body.index("PREPR_SCRUBBED_CREDENTIAL_VARS")
    source_at = body.index('source "${OMNIBASE_OPERATOR_ENV_FILE}"')
    assert scrub_at < source_at, (
        "the credential scrub runs AFTER the operator env is sourced. In that "
        "order it strips the operator env's OWN values as well, leaving the "
        "slot without configuration it legitimately needs. Measured, not "
        "reasoned: an unset after the source resolves to nothing at all."
    )


def test_every_scrubbed_name_is_one_the_slot_would_actually_inherit() -> None:
    """The list must not rot into naming variables nothing interpolates.

    A scrub list that names only dead variables passes every test about
    scrubbing while protecting nothing. Two names are carried deliberately
    despite a zero count and are excused by name rather than silently: one
    credential feeds both GitHub spellings, and the other provider is
    reachable through the same resolver family.
    """
    excused = {"GH_TOKEN", "OPENROUTER_API_KEY"}
    live = [n for n in _scrub_list() if n not in excused]
    dead = [n for n in live if _interpolation_count(n) == 0]
    assert not dead, (
        f"these scrubbed names are interpolated nowhere in the layered compose "
        f"files: {dead}. Either they are stale and should go, or the files "
        f"moved and the scrub no longer covers what it was written for."
    )
    assert len(live) >= 5, (
        "the scrub list has shrunk below the set measured for OMN-19076."
    )


# ---------------------------------------------------------------------------
# A branch name is not a pin
# ---------------------------------------------------------------------------


def _stage_fn() -> str:
    """The body of the staging helper, comments stripped."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    start = text.index("stage_commit_tree() {")
    end = text.index("\n}", start)
    return "\n".join(
        line
        for line in text[start:end].splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_staging_helper_refuses_a_ref_that_is_not_a_literal_sha() -> None:
    """A movable ref would be re-resolved by git at extraction time.

    The readback verifies a tree; the archive then has to extract THAT tree.
    If the archive is handed a branch, git resolves it again at extraction
    time and the two can differ. Another lane measured it: two siblings moved
    between builds while its readback still passed.

    Every caller today passes a value resolved once with ``rev-parse HEAD``,
    so the guard never fires in practice. It is asserted anyway because "by
    construction" is a property of the current call sites, not of the
    function, and the failure it prevents is silent.
    """
    body = _stage_fn()
    assert "[0-9a-f]{40}" in body, (
        "the staging helper does not check that its pin is a literal 40-hex "
        "commit. Handed a branch it would archive whatever that branch points "
        "at when the archive runs, which is not what the readback verified."
    )


def test_the_archive_is_given_the_recorded_pin_and_never_a_ref_expression() -> None:
    """The archive's ref argument must be the recorded variable, nothing else."""
    body = _stage_fn()
    archive_lines = [ln.strip() for ln in body.splitlines() if "archive" in ln]
    assert archive_lines, "the staging helper no longer archives anything."
    for line in archive_lines:
        assert '"${commit}"' in line, (
            f"the archive command does not take the recorded pin: {line!r}. "
            f"Anything else here is resolved at extraction time."
        )
        for movable in ("HEAD", "origin/", "FETCH_HEAD", "@{"):
            assert movable not in line, (
                f"the archive command names the movable ref {movable!r}: {line!r}."
            )


def test_every_pin_this_run_records_is_resolved_from_head_exactly_once() -> None:
    """Resolved once, before staging, and reused; never re-resolved per repo."""
    body = _entrypoint_code()
    resolutions = [
        ln.strip() for ln in body.splitlines() if "rev-parse HEAD" in ln and "=" in ln
    ]
    # One for the target, one for each sibling in the resolution loop, and one
    # inside the staging helper, which is the readback rather than a new pin.
    assert len(resolutions) <= 3, (
        f"there are {len(resolutions)} places resolving HEAD into a variable:\n  "
        + "\n  ".join(resolutions)
        + "\nMore than the target, the sibling loop and the staging readback "
        "means a pin is being resolved more than once, and two resolutions of "
        "the same ref can disagree."
    )

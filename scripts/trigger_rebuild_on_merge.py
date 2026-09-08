#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# trigger_rebuild_on_merge.py
#
# Publishes onex.cmd.omnimarket.redeploy-start.v1 (consumed by
# node_redeploy_orchestrator) when a merged PR contains runtime changes. Called
# from the runtime-rebuild-trigger GHA workflow on PR merge to dev or main.
#
# node_redeploy_orchestrator owns the deployment lifecycle (lane policy via the
# prod-gate compute, digest pinning, readiness, rollback) and its deploy
# publish-monitor EFFECT is the SOLE emitter of
# onex.cmd.deploy.rebuild-requested.v1 to the deploy agent. CI publishes a typed
# start command only; it never talks to the deploy agent directly.
#
# Triggers when:
#   - PR had the "runtime_change" label, OR
#   - The canonical deploy-gate classifier identifies a changed runtime path
#
# Lane policy (the triggering ref decides the lane — no hardcoded origin/main):
#   - merge to dev  -> runtime_lane=dev,            source_branch=dev
#   - merge to main -> runtime_lane=stability-test, source_branch=main
#     (dev->main promotion proves the stability lane; prod deploys the
#      stability-proven digest later via node_redeploy_orchestrator, not from CI)
#
# Tickets: OMN-8917 (original auto-trigger), OMN-12573 (re-point to node_redeploy)
#
# Required inputs (when not --dry-run):
#   --bus-lane                -- control-bus lane declared by the overlay
#   --bus-overlay             -- checked-in config/ci_bus_lanes.yaml path
#   --consumer-model          -- canonical strict consumer model source
#
# Optional environment variables:
#   KAFKA_BOOTSTRAP_SERVERS   -- drift guard / from-secret broker only
#   KAFKA_SASL_USERNAME       -- SASL principal; REQUIRED when the lane the
#                                overlay declares uses a SASL_* security
#                                protocol, unused otherwise (OMN-18012)
#   KAFKA_SASL_PASSWORD       -- that principal's secret, same condition
#   RUNNER_ENVIRONMENT        -- GitHub Actions runner class, forwarded by the
#                                workflow as --runner-environment (OMN-17888)
#
# TRANSPORT IS DECLARED, NEVER INFERRED (OMN-18012): the lane overlay carries
# `security_protocol` (and `sasl_mechanism` for a SASL protocol) beside each
# lane's broker, and this publisher READS them. The previous body chose
# SASL_SSL/PLAIN whenever both SASL credentials happened to be present in the
# environment. Credential presence is not a statement about transport: when the
# .201 dev-lane Redpanda external listener began requiring SASL/SCRAM-SHA-256
# over PLAINTEXT on 2026-09-07 and the org KAFKA_SASL_* secrets were injected,
# that inference picked TLS against a listener that speaks none.
#
# RUNNER REACHABILITY (OMN-17888): the dev control-bus lane declares a LAN /
# tailnet broker. A GitHub-hosted runner is not on that network, so librdkafka
# cannot even resolve the name and the 30-second flush expires with the command
# undelivered. That is a ROUTING defect, not a broker outage, and it must say so
# in one line rather than as a DNS error buried under a flush timeout. This
# script therefore refuses to attempt a live publish from a github-hosted runner
# and names the remedy (route the job to the LAN fleet).
#
# Usage:
#   python scripts/trigger_rebuild_on_merge.py \
#     --changed-files "src/omnimarket/nodes/foo/handler.py,README.md" \
#     --labels "runtime_change,bug" \
#     --base-branch "dev" \
#     --source-sha "<merge_commit_sha>" \
#     [--dry-run]

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import click
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

# CI publishes the node_redeploy_orchestrator start command; the orchestrator's
# deploy publish-monitor effect is the sole emitter of the deploy-agent rebuild
# command downstream.
TOPIC = "onex.cmd.omnimarket.redeploy-start.v1"

_RUNTIME_LABEL = "runtime_change"

# Lane-declared transport vocabulary (OMN-18012). The overlay
# (omnimarket config/ci_bus_lanes.yaml) declares `security_protocol` and, for a
# SASL protocol, `sasl_mechanism` beside each lane's broker. This publisher READS
# them; it never derives them from the shape of the environment. Credential
# PRESENCE is not a statement about transport -- inferring SASL_SSL from injected
# credentials is precisely what took every OCC companion mint down on 2026-09-07
# against a listener that speaks SASL over PLAINTEXT with no TLS.
_INMEMORY_BROKER = "inmemory"
_SASL_PROTOCOLS = frozenset({"SASL_PLAINTEXT", "SASL_SSL"})
_NON_SASL_PROTOCOLS = frozenset({"PLAINTEXT", "SSL"})
_VALID_SECURITY_PROTOCOLS = _SASL_PROTOCOLS | _NON_SASL_PROTOCOLS

RuntimePathClassifier = Callable[[list[str]], list[str]]

# Maps the merged PR's base branch to a runtime lane. Values match
# deploy_agent.events.EnumRuntimeLane (dev | stability-test | prod). prod is not
# triggerable from CI: production deploys the stability-proven digest through
# node_redeploy_orchestrator's promotion gate, never from a merge event.
_BASE_BRANCH_LANES: dict[str, str] = {
    "dev": "dev",
    "main": "stability-test",
}


class ModelCiBusLane(BaseModel):
    """One checked-in CI control-bus lane declaration.

    Carries the lane's broker AND its transport. The transport half is the
    OMN-18012 contract, documented in the overlay itself
    (omnimarket ``config/ci_bus_lanes.yaml``) and enforced field-for-field here:

    * ``security_protocol`` -- one of PLAINTEXT | SSL | SASL_PLAINTEXT |
      SASL_SSL, spelled as librdkafka spells it. REQUIRED on any lane whose
      broker is a concrete ``host:port`` or ``from-secret``; an ``inmemory``
      lane publishes nothing cross-process, so it needs none.
    * ``sasl_mechanism`` -- REQUIRED when ``security_protocol`` is SASL_*, and
      contradictory (rejected) beside a non-SASL protocol or with no protocol
      at all.

    ``extra="forbid"`` stays: an unknown key in this overlay is a typo that
    would otherwise route a publisher to a default it never declared. That
    strictness is also why this model had to learn the two transport keys --
    once the overlay declared them, a model that knew only ``broker`` rejected
    the live overlay outright and took the dev-lane redeploy trigger red on
    every runtime PR (omnibase_infra run 34160709151).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    broker: str
    security_protocol: str | None = None
    sasl_mechanism: str | None = None

    @field_validator("broker")
    @classmethod
    def validate_broker(cls, value: str) -> str:
        """Reject empty broker declarations at the contract boundary."""
        if not value:
            raise ValueError("broker must not be empty")
        return value

    @field_validator("security_protocol")
    @classmethod
    def validate_security_protocol(cls, value: str | None) -> str | None:
        """Accept only a librdkafka security protocol, normalised to upper case."""
        if value is None:
            return None
        protocol = value.upper()
        if protocol not in _VALID_SECURITY_PROTOCOLS:
            raise ValueError(
                f"security_protocol {value!r} is not a librdkafka security "
                f"protocol; valid values: {sorted(_VALID_SECURITY_PROTOCOLS)}"
            )
        return protocol

    @field_validator("sasl_mechanism")
    @classmethod
    def validate_sasl_mechanism(cls, value: str | None) -> str | None:
        """Reject an empty mechanism declaration outright rather than ignoring it."""
        if value is None:
            return None
        if not value:
            raise ValueError("sasl_mechanism must not be empty when declared")
        return value

    @model_validator(mode="after")
    def validate_transport_declaration(self) -> ModelCiBusLane:
        """Enforce the overlay's own transport contract, never a guess."""
        publishes_cross_process = self.broker != _INMEMORY_BROKER
        if publishes_cross_process and self.security_protocol is None:
            raise ValueError(
                f"lane declares broker {self.broker!r} but no security_protocol; "
                f"declare one of {sorted(_VALID_SECURITY_PROTOCOLS)} beside it. "
                "Refusing to guess the transport (OMN-18012)"
            )
        if self.security_protocol in _SASL_PROTOCOLS and self.sasl_mechanism is None:
            raise ValueError(
                f"security_protocol {self.security_protocol} requires a "
                "sasl_mechanism (e.g. SCRAM-SHA-256); none was declared"
            )
        if self.security_protocol not in _SASL_PROTOCOLS and self.sasl_mechanism:
            raise ValueError(
                f"sasl_mechanism {self.sasl_mechanism!r} is declared beside "
                f"security_protocol {self.security_protocol!r}, which carries no "
                "SASL; fix the overlay rather than let the publisher pick a half"
            )
        return self


class ModelCiBusOverlay(BaseModel):
    """Typed contract for the authoritative lane-to-broker overlay."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    default: str
    lanes: dict[str, ModelCiBusLane]

    @field_validator("default")
    @classmethod
    def validate_default(cls, value: str) -> str:
        """The current overlay contract has an explicit in-memory default."""
        if value != "inmemory":
            raise ValueError("default must be 'inmemory'")
        return value

    @field_validator("lanes")
    @classmethod
    def validate_lanes(
        cls, value: dict[str, ModelCiBusLane]
    ) -> dict[str, ModelCiBusLane]:
        """A bus overlay without declared lanes cannot route a producer."""
        if not value:
            raise ValueError("lanes must not be empty")
        if any(not lane.strip() for lane in value):
            raise ValueError("lane names must not be empty")
        return value


# The script is also loaded directly from its file path by hermetic tests. Give
# Pydantic the explicit namespace so postponed annotations resolve without
# depending on the module having first been inserted into sys.modules.
ModelCiBusOverlay.model_rebuild(_types_namespace={"ModelCiBusLane": ModelCiBusLane})


class ModelRedeployStartCommandWire(BaseModel):
    """Strict producer-side subset of the redeploy orchestrator command."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    correlation_id: UUID
    scope: Literal["full"] = "full"
    git_ref: str
    runtime_lane: Literal["dev", "stability-test"]
    build_source: Literal["workspace", "release"]
    requested_by: str

    @field_validator("git_ref")
    @classmethod
    def validate_git_ref(cls, value: str) -> str:
        """The live trigger carries an exact hexadecimal merge commit SHA."""
        is_hex = all(character in "0123456789abcdef" for character in value)
        if not (7 <= len(value) <= 64) or not is_hex:
            raise ValueError(
                "git_ref must be a 7-64 character lowercase hex commit SHA"
            )
        return value

    @field_validator("requested_by")
    @classmethod
    def validate_requested_by(cls, value: str) -> str:
        if not value:
            raise ValueError("requested_by must not be empty")
        return value


ModelRedeployStartCommandWire.model_rebuild(
    _types_namespace={"Literal": Literal, "UUID": UUID}
)


def load_ci_bus_overlay(path: Path) -> ModelCiBusOverlay:
    """Load and strictly validate the checked-in CI bus overlay."""
    if not path.is_file():
        raise ValueError(f"CI bus overlay does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ModelCiBusOverlay.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"Invalid CI bus overlay {path}: {exc}") from exc


def resolve_ci_bus_broker(
    *,
    overlay: ModelCiBusOverlay,
    lane: str,
    injected_broker: str,
) -> str:
    """Resolve one control-bus lane without allowing opaque target drift."""
    lane_key = lane.strip()
    if not lane_key:
        raise ValueError("CI bus lane must not be empty")
    declaration = overlay.lanes.get(lane_key)
    if declaration is None:
        raise ValueError(
            f"CI bus lane {lane_key!r} is not declared; "
            f"declared lanes: {sorted(overlay.lanes)}"
        )

    broker = declaration.broker
    if broker == "inmemory":
        raise ValueError(
            f"CI bus lane {lane_key!r} declares 'inmemory'; a post-merge "
            "redeploy command requires a cross-process broker"
        )
    if broker == "from-secret":
        if not injected_broker:
            raise ValueError(
                f"CI bus lane {lane_key!r} declares 'from-secret', but "
                "KAFKA_BOOTSTRAP_SERVERS is empty"
            )
        return injected_broker
    if injected_broker and injected_broker != broker:
        raise ValueError(
            "LANE BUS DRIFT: injected KAFKA_BOOTSTRAP_SERVERS does not match "
            f"the checked-in broker for lane {lane_key!r} ({broker})"
        )
    return broker


def resolve_ci_bus_security(
    *,
    overlay: ModelCiBusOverlay,
    lane: str,
) -> tuple[str, str]:
    """Resolve one lane's DECLARED ``(security_protocol, sasl_mechanism)``.

    OMN-18012. The transport is config, declared in the overlay beside the
    broker it belongs to -- the same "one reviewable source of truth" principle
    OMN-14801 applied to the broker, applied to the other half of the
    connection. This function reads the overlay and nothing else: it never
    consults the environment and never infers.

    Only meaningful on the lanes that actually publish; ``resolve_ci_bus_broker``
    has already refused an ``inmemory`` lane by the time this is called, and the
    model has already refused a publishing lane that declares no protocol.
    """
    lane_key = lane.strip()
    declaration = overlay.lanes.get(lane_key)
    if declaration is None:
        raise ValueError(
            f"CI bus lane {lane_key!r} is not declared; "
            f"declared lanes: {sorted(overlay.lanes)}"
        )
    protocol = declaration.security_protocol
    if protocol is None:
        raise ValueError(
            f"CI bus lane {lane_key!r} declares no security_protocol; a "
            "publishing lane must declare its transport (OMN-18012)"
        )
    return protocol, declaration.sasl_mechanism or ""


def build_kafka_producer_config(
    bootstrap_servers: str,
    username: str,
    password: str,
    security_protocol: str,
    sasl_mechanism: str,
) -> dict[str, str | int | float | bool]:
    """Build the producer transport from the LANE-DECLARED protocol (OMN-18012).

    ``security_protocol`` / ``sasl_mechanism`` are required arguments resolved
    from the overlay by :func:`resolve_ci_bus_security`. A default here would be
    the guess this function exists to delete: the body this replaced selected
    ``SASL_SSL`` / ``PLAIN`` whenever both SASL credentials happened to be
    present in the environment, and plaintext otherwise. On 2026-09-07 the .201
    dev-lane Redpanda EXTERNAL listener began requiring SASL/SCRAM-SHA-256 over
    PLAINTEXT -- no TLS at all -- while the KAFKA_SASL_* org secrets were
    injected, and that inference picked TLS against a listener that speaks none.
    """
    protocol = security_protocol.strip().upper()
    mechanism = sasl_mechanism.strip()
    if protocol not in _VALID_SECURITY_PROTOCOLS:
        raise ValueError(
            f"security_protocol {security_protocol!r} is not a librdkafka "
            f"security protocol; valid values: {sorted(_VALID_SECURITY_PROTOCOLS)}"
        )
    if bool(username) != bool(password):
        raise ValueError(
            "KAFKA_SASL_USERNAME and KAFKA_SASL_PASSWORD must both be set "
            "or both be empty"
        )

    config: dict[str, str | int | float | bool] = {
        "bootstrap.servers": bootstrap_servers,
        "security.protocol": protocol,
    }

    if protocol not in _SASL_PROTOCOLS:
        if username or password:
            # Not an error -- the declared transport is authoritative, and
            # dropping credentials it cannot carry is safe. It IS drift worth
            # naming, so say so rather than let a silently-unauthenticated
            # publish look identical to a correctly-configured one.
            click.echo(
                "WARNING: SASL credentials are present in this environment but "
                f"the lane declares security_protocol={protocol}, which carries "
                "none. Publishing unauthenticated per the lane declaration and "
                "IGNORING the injected credentials. If the broker now requires "
                "SASL, the overlay is stale -- fix the overlay, never the "
                "inference (OMN-18012).",
                err=True,
            )
        return config

    if not mechanism:
        raise ValueError(
            f"security_protocol {protocol} requires a sasl_mechanism; none was "
            "declared for this lane"
        )
    if not username or not password:
        raise ValueError(
            f"the lane declares security_protocol={protocol} / "
            f"sasl_mechanism={mechanism}, but KAFKA_SASL_USERNAME and/or "
            "KAFKA_SASL_PASSWORD are not set in this job's environment. A SASL "
            "lane cannot be published to without credentials: wire them through "
            "the workflow rather than downgrading the declared transport"
        )
    config.update(
        {
            "sasl.mechanisms": mechanism,
            "sasl.username": username,
            "sasl.password": password,
        }
    )
    return config


def _field_call_is_required(value: ast.expr | None) -> bool:
    """Return whether one annotated consumer field has no default."""
    if value is None:
        return True
    if not isinstance(value, ast.Call):
        return False
    if value.args and isinstance(value.args[0], ast.Constant):
        if value.args[0].value is Ellipsis:
            return True
    for keyword in value.keywords:
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is Ellipsis
    return False


def _class_declares_extra_forbid(node: ast.ClassDef) -> bool:
    for statement in node.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "model_config"
            for target in statement.targets
        ):
            continue
        if not isinstance(statement.value, ast.Call):
            return False
        return any(
            keyword.arg == "extra"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "forbid"
            for keyword in statement.value.keywords
        )
    return False


def load_consumer_model_contract(
    model_path: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    """Read field and required-field truth from the canonical consumer model."""
    if not model_path.is_file():
        raise ValueError(f"consumer model does not exist: {model_path}")
    try:
        tree = ast.parse(
            model_path.read_text(encoding="utf-8"), filename=str(model_path)
        )
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"invalid consumer model {model_path}: {exc}") from exc

    model_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ModelRedeployStartCommand"
        ),
        None,
    )
    if model_node is None:
        raise ValueError(
            f"consumer model {model_path} does not define ModelRedeployStartCommand"
        )
    if not _class_declares_extra_forbid(model_node):
        raise ValueError(
            "ModelRedeployStartCommand must declare ConfigDict(extra='forbid')"
        )

    fields: set[str] = set()
    required: set[str] = set()
    for statement in model_node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue
        if not isinstance(statement.target, ast.Name):
            continue
        name = statement.target.id
        fields.add(name)
        if _field_call_is_required(statement.value):
            required.add(name)
    if not fields:
        raise ValueError("ModelRedeployStartCommand declares no annotated fields")
    return frozenset(fields), frozenset(required)


def assert_consumer_model_accepts_payload(
    *, payload: dict[str, object], model_path: Path
) -> None:
    """Fail before publish if the strict consumer cannot accept these keys."""
    fields, required = load_consumer_model_contract(model_path)
    payload_fields = frozenset(payload)
    extras = sorted(payload_fields - fields)
    missing = sorted(required - payload_fields)
    if extras:
        raise ValueError(
            f"consumer rejects extra fields from producer payload: {extras}"
        )
    if missing:
        raise ValueError(f"producer payload misses required consumer fields: {missing}")


def build_redeploy_start_payload(
    *,
    runtime_lane: str,
    build_source: str,
    source_sha: str,
    correlation_id: str,
    requested_by: str,
) -> dict[str, object]:
    """Build exactly the strict command shape consumed by node_redeploy."""
    command = ModelRedeployStartCommandWire(
        correlation_id=correlation_id,
        scope="full",
        git_ref=source_sha,
        runtime_lane=runtime_lane,
        build_source=build_source,
        requested_by=requested_by,
    )
    return command.model_dump(mode="json")


def load_runtime_path_classifier(path: Path) -> RuntimePathClassifier:
    """Load the exact deploy-gate runtime-path classifier used by hosted CI.

    Runtime deployment scope has one owner: omniclaude's deploy-gate validator.
    Loading its ``find_runtime_paths`` callable keeps the post-merge publisher
    aligned with the required deploy gate instead of maintaining a second path
    allowlist that can silently drift.
    """
    if not path.is_file():
        raise ValueError(f"runtime path validator does not exist: {path}")

    module_name = "_canonical_deploy_path_classifier"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load runtime path validator: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ValueError(f"invalid runtime path validator {path}: {exc}") from exc

    classifier = getattr(module, "find_runtime_paths", None)
    if not callable(classifier):
        raise ValueError(
            f"runtime path validator {path} does not define find_runtime_paths"
        )
    return cast("RuntimePathClassifier", classifier)


def classify_runtime_paths(
    changed_files: list[str], classifier: RuntimePathClassifier
) -> list[str]:
    """Run and validate the canonical classifier's output fail-closed."""
    runtime_paths = classifier(changed_files)
    if not isinstance(runtime_paths, list) or any(
        not isinstance(path, str) or not path.strip() for path in runtime_paths
    ):
        raise ValueError("runtime path validator returned an invalid path list")
    return runtime_paths


def should_trigger(runtime_paths: list[str], labels: list[str]) -> bool:
    """Return True for a runtime label or canonical deploy-path hit."""
    return _RUNTIME_LABEL in labels or bool(runtime_paths)


def lane_for_base_branch(base_branch: str) -> str:
    """Map a merged PR's base branch to a node_redeploy_orchestrator runtime lane.

    Fails closed on unmapped branches: a misconfigured trigger must not silently
    pick a default lane and rebuild the wrong runtime.
    """
    lane = _BASE_BRANCH_LANES.get(base_branch)
    if lane is None:
        allowed = ", ".join(sorted(_BASE_BRANCH_LANES))
        msg = (
            f"No runtime lane mapping for base branch {base_branch!r}; "
            f"allowed base branches: {allowed}"
        )
        raise ValueError(msg)
    return lane


def build_source_for_base_branch(base_branch: str) -> str:
    """Map branch lineage to the deploy agent's provenance mode."""
    if base_branch == "dev":
        return "workspace"
    if base_branch == "main":
        return "release"
    raise ValueError(f"No build source mapping for base branch {base_branch!r}")


def publish_redeploy_start_event(
    bootstrap_servers: str,
    username: str,
    password: str,
    security_protocol: str,
    sasl_mechanism: str,
    runtime_lane: str,
    build_source: str,
    source_sha: str,
    correlation_id: str,
    requested_by: str,
) -> tuple[int, str]:
    """Publish a strict redeploy-start command to node_redeploy_orchestrator.

    Returns ``(delivered, coordinates)`` where ``delivered`` is the number of
    commands the broker acknowledged (``1`` on success) and ``coordinates`` is
    the broker-assigned ``partition=<p> offset=<o>`` proof of that acceptance.
    Raises on any delivery failure so the caller can assert a non-zero emit
    count — a producer that delivers nothing must fail closed, never report
    success. A drained queue that produced no delivery callback is also a
    failure: with no broker-assigned offset there is no proof of publication
    (the OMN-17378 green-but-silent shape).
    """
    from confluent_kafka import Producer

    payload = build_redeploy_start_payload(
        runtime_lane=runtime_lane,
        build_source=build_source,
        source_sha=source_sha,
        correlation_id=correlation_id,
        requested_by=requested_by,
    )

    producer = Producer(
        build_kafka_producer_config(
            bootstrap_servers,
            username,
            password,
            security_protocol,
            sasl_mechanism,
        )
    )

    delivery_error: BaseException | None = None
    delivery_coordinates: str | None = None

    def _on_delivery(err: object, msg: object) -> None:
        nonlocal delivery_error, delivery_coordinates
        if err is not None:
            delivery_error = RuntimeError(str(err))
            return
        partition = getattr(msg, "partition", None)
        offset = getattr(msg, "offset", None)
        partition_value = partition() if callable(partition) else partition
        offset_value = offset() if callable(offset) else offset
        delivery_coordinates = f"partition={partition_value} offset={offset_value}"

    message = json.dumps(payload, default=str).encode("utf-8")
    key = f"gha-redeploy/{correlation_id}".encode()

    producer.produce(
        topic=TOPIC,
        key=key,
        value=message,
        on_delivery=_on_delivery,
    )
    remaining = producer.flush(timeout=30)

    if delivery_error is not None:
        raise RuntimeError(f"Kafka delivery failed: {delivery_error}") from None
    if remaining and remaining > 0:
        raise RuntimeError(
            f"Kafka delivery timed out: {remaining} message(s) remain "
            "undelivered after the 30 second flush"
        )
    if delivery_coordinates is None:
        raise RuntimeError(
            "Kafka flush drained without a delivery callback: no "
            "broker-assigned offset, so publication is unproven"
        )

    # Exactly one redeploy-start command was delivered; the caller asserts N>0.
    return 1, delivery_coordinates


def emit_github_output(published: bool, source_sha: str, runtime_lane: str) -> None:
    """Record the publish DECISION where a downstream job can read it (OMN-17888 AC4).

    AC4 asks that a delivered-but-not-applied redeploy surface as lane staleness.
    The convergence job that does that must only run when this script actually
    published a command — a no-op run has no redeploy to wait for. That fact
    lives here and nowhere else, so it is written out rather than re-derived by
    a second copy of the classifier.

    Additive and inert outside GitHub Actions: with GITHUB_OUTPUT unset this
    writes nothing and changes no exit code.
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"published={str(published).lower()}\n")
        handle.write(f"source_sha={source_sha}\n")
        handle.write(f"runtime_lane={runtime_lane}\n")


@click.command()
@click.option(
    "--changed-files",
    default="",
    help="Comma-separated list of changed file paths",
)
@click.option(
    "--labels",
    default="",
    help="Comma-separated list of PR label names",
)
@click.option(
    "--base-branch",
    required=True,
    help="Merged PR base branch (dev | main) — decides the runtime lane",
)
@click.option(
    "--source-sha",
    required=True,
    help="Merge commit SHA of the triggering PR (the ref node_redeploy_orchestrator rebuilds)",
)
@click.option(
    "--requested-by",
    default="gha-runtime-rebuild-trigger",
    help="Identifier for who is requesting the redeploy",
)
@click.option(
    "--correlation-id",
    default="",
    help="Correlation ID (auto-generated if not provided)",
)
@click.option(
    "--bus-lane",
    default="",
    help="Control-bus lane id declared by the CI bus overlay (normally 'dev')",
)
@click.option(
    "--bus-overlay",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the authoritative checked-in config/ci_bus_lanes.yaml",
)
@click.option(
    "--consumer-model",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to omnimarket's canonical ModelRedeployStartCommand source",
)
@click.option(
    "--runtime-path-validator",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to omniclaude's canonical deploy-gate validator source",
)
@click.option(
    "--runner-environment",
    default="",
    help=(
        "GitHub Actions RUNNER_ENVIRONMENT of the executing job. "
        "'github-hosted' cannot reach the LAN control bus and is refused "
        "before a live publish is attempted (OMN-17888)."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Check trigger conditions and print decision without publishing",
)
def main(
    changed_files: str,
    labels: str,
    base_branch: str,
    source_sha: str,
    requested_by: str,
    correlation_id: str,
    bus_lane: str,
    bus_overlay: Path | None,
    consumer_model: Path | None,
    runtime_path_validator: Path,
    runner_environment: str,
    dry_run: bool,
) -> None:
    """Publish a node_redeploy_orchestrator start command if a PR contains runtime changes.

    Triggers when the PR had the runtime_change label or the canonical deploy
    gate classifies a changed path as runtime-scoped. The triggering base branch
    decides the runtime lane; the merge SHA is the ref node_redeploy_orchestrator rebuilds.
    """
    files: list[str] = (
        [f.strip() for f in changed_files.split(",") if f.strip()]
        if changed_files
        else []
    )
    label_list: list[str] = (
        [lb.strip() for lb in labels.split(",") if lb.strip()] if labels else []
    )

    corr_id = correlation_id or str(uuid.uuid4())

    runtime_lane = lane_for_base_branch(base_branch)
    build_source = build_source_for_base_branch(base_branch)

    try:
        classifier = load_runtime_path_classifier(runtime_path_validator)
        runtime_paths = classify_runtime_paths(files, classifier)
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    if not should_trigger(runtime_paths, label_list):
        click.echo(
            "No rebuild trigger: no runtime_change label or runtime path changes detected."
        )
        emit_github_output(False, source_sha, runtime_lane)
        sys.exit(0)

    # OMN-17888: this line records the DECISION, never the delivery. It is
    # emitted before a 30-second flush that can time out with the command
    # undelivered, so wording it as "Redeploy triggered" made an intent read as
    # a receipt. The only publication evidence is the final "Published
    # redeploy-start ..." line, which carries the broker-assigned coordinates.
    click.echo(
        f"Runtime change detected (delivery NOT yet confirmed): "
        f"runtime_lane={runtime_lane} source_branch={base_branch} "
        f"source_sha={source_sha} correlation_id={corr_id} labels={label_list} "
        f"files_matched={runtime_paths}"
    )

    if dry_run:
        click.echo("(dry-run: skipping Kafka publish)")
        emit_github_output(False, source_sha, runtime_lane)
        sys.exit(0)

    # A live publish targets the lane's LAN / tailnet broker. A github-hosted
    # runner is not on that network: the name does not resolve, the flush
    # expires, and the run reds 30 seconds later with a DNS error that reads
    # like a broker outage. Refuse first and name the real defect, so a runner
    # re-route (Operating Rule 14) cannot silently relocate this publisher off
    # the fleet again (OMN-17888, same class as OMN-17378).
    if runner_environment.strip() == "github-hosted":
        click.echo(
            "ERROR: refusing to publish from a github-hosted runner — the "
            f"{bus_lane or 'dev'} control-bus broker is LAN/tailnet-only and is "
            "unreachable from GitHub-hosted compute. Route this job to the "
            "self-hosted omnibase-ci fleet (OMNI_RUNTIME_REBUILD_RUNS_ON_JSON) "
            "rather than the shared trusted-CI runner seam.",
            err=True,
        )
        sys.exit(1)

    injected_broker = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    username = os.environ.get("KAFKA_SASL_USERNAME", "")
    password = os.environ.get("KAFKA_SASL_PASSWORD", "")

    if bus_overlay is None or not bus_lane.strip() or consumer_model is None:
        click.echo(
            "ERROR: --bus-lane, --bus-overlay, and --consumer-model are "
            "required for a live redeploy publish",
            err=True,
        )
        sys.exit(1)

    try:
        overlay = load_ci_bus_overlay(bus_overlay)
        bootstrap_servers = resolve_ci_bus_broker(
            overlay=overlay,
            lane=bus_lane,
            injected_broker=injected_broker,
        )
        # The transport is DECLARED by the lane, never inferred here (OMN-18012).
        security_protocol, sasl_mechanism = resolve_ci_bus_security(
            overlay=overlay,
            lane=bus_lane,
        )
        # Validate the transport pair before the producer is constructed.
        build_kafka_producer_config(
            bootstrap_servers,
            username,
            password,
            security_protocol,
            sasl_mechanism,
        )
        candidate_payload = build_redeploy_start_payload(
            runtime_lane=runtime_lane,
            build_source=build_source,
            source_sha=source_sha,
            correlation_id=corr_id,
            requested_by=requested_by,
        )
        assert_consumer_model_accepts_payload(
            payload=candidate_payload,
            model_path=consumer_model,
        )
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    # A runtime change was detected and this is not a dry run, so the job's
    # PURPOSE is now to publish exactly one redeploy-start command. If any
    # precondition for publishing is absent (resolved broker),
    # this producer CANNOT emit — that is zero output and MUST fail closed
    # (RT-5 / OMN-14467), never "skip publish" green. The dead-producer bug this
    # replaces printed "KAFKA_BOOTSTRAP_SERVERS is not set -- skipping publish"
    # and exited 0, so the deploy trigger was green-but-silent on every merge —
    # it had never published once (run 29189239291).
    from omnibase_infra.utils.util_producer_effect_assertion import (
        ProducerZeroOutputError,
        assert_producer_emitted,
        require_producer_preconditions,
    )

    try:
        require_producer_preconditions(
            artifact=TOPIC,
            preconditions={
                "CI_BUS_BROKER": bootstrap_servers,
            },
        )
    except ProducerZeroOutputError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        delivered, delivery_coordinates = publish_redeploy_start_event(
            bootstrap_servers=bootstrap_servers,
            username=username,
            password=password,
            security_protocol=security_protocol,
            sasl_mechanism=sasl_mechanism,
            runtime_lane=runtime_lane,
            build_source=build_source,
            source_sha=source_sha,
            correlation_id=corr_id,
            requested_by=requested_by,
        )
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Delivery error: {exc}", err=True)
        sys.exit(1)

    # "Produced N>0, and here it is": a completed publish that delivered zero
    # commands is a silent-producer failure and must go red, not report success.
    try:
        assert_producer_emitted(
            delivered, artifact=TOPIC, detail=f"correlation_id={corr_id}"
        )
    except ProducerZeroOutputError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(
        f"Published redeploy-start to {TOPIC} "
        f"(correlation_id={corr_id}, delivered={delivered}, "
        f"{delivery_coordinates})"
    )

    # Only reached once the broker has ACKed the command. Publication is the
    # ONLY thing this records; whether the lane then applies it is the question
    # the downstream convergence job answers (OMN-17888 AC4).
    emit_github_output(True, source_sha, runtime_lane)


if __name__ == "__main__":
    main()

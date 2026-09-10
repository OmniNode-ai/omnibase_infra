# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Operator entry point: publish a signed rebuild command to the control bus.

THE COMMAND IS BUILT FROM THE CONTRACT, NOT BESIDE IT (OMN-16442)
-----------------------------------------------------------------

``deploy-agent-trigger.sh`` used to hand-write the command JSON in an embedded
Python snippet::

    envelope = {
        "correlation_id": ..., "git_ref": ..., "reason": ...,
        "requested_by": ..., "scope": "runtime", "services": [],
    }

and the agent that consumes it validates against
:class:`deploy_agent.events.ModelRebuildRequested`, which **requires**
``runtime_lane`` and **forbids** ``reason``. The two drifted apart and nothing
noticed, because nothing compared them. Measured on the .201 dev lane at
2026-09-08T12:14:27Z, verbatim from the agent's journal::

    Invalid payload (correlation_id=2b017562-...): 2 validation errors for
    ModelRebuildRequested / runtime_lane / Field required / reason / Extra
    inputs are not permitted
    Rejected command: invalid_payload

So no operator invocation of the sanctioned script could ever be accepted --
and the script's own header forbids hand-building the JSON (correctly: the
HMAC is computed over exact bytes). The entry point was closed in both
directions.

The fix is structural rather than a field edit: this module constructs a
``ModelRebuildRequested`` and serialises **it**. A field added to, removed
from, or renamed on the model changes what this publishes in the same commit.
A drift of the old kind is no longer expressible.

TRANSPORT IS RESOLVED THE WAY THE AGENT RESOLVES IT
---------------------------------------------------

The old script read unprefixed ``KAFKA_SASL_USERNAME`` /
``KAFKA_SASL_PASSWORD``, while the .201 lane env file carries the dev SCRAM
principal under ``DEV_``-prefixed names -- so its SASL block was silently
skipped -- and both its ``kcat`` and ``rpk`` branches then hardcoded
``SASL_SSL`` / ``PLAIN`` against a broker that has been ``SASL_PLAINTEXT`` +
``SCRAM-SHA-256`` since 2026-09-07 (OMN-18012 Phase B). Neither tool is on
PATH on .201, so resolution fell through to a third branch that published
through ``docker exec ... rpk --brokers localhost:9092`` with **no
authentication at all**.

This module calls :func:`deploy_agent.kafka_config.load_deploy_agent_kafka_config_from_env`
-- the same loader the agent itself starts from, prefix support included. There
is one transport, declared once, read by both ends. The unauthenticated branch
is gone: a missing prerequisite is a refusal that names it.

COMPRESSION IS EXPLICIT, AND THAT IS NOT COSMETIC
-------------------------------------------------

``rpk topic produce`` defaults to snappy. The agent's ``kafka-python`` client
has no snappy codec installed, so one snappy record on
``onex.cmd.deploy.rebuild-requested.v1`` killed the agent on every poll --
twelve crashes in sixty seconds, then systemd's start limit. This producer
declares ``compression_type=None`` so the wire format is a property of the
publisher's own configuration rather than of whichever CLI happened to be
installed. The consumer-side quarantine (``deploy_agent.consumer``) is the
other half: this half stops us minting the pill, that half stops one from
stalling the control plane.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from enum import StrEnum
from typing import Any

from deploy_agent.auth import sign_envelope
from deploy_agent.events import (
    TOPIC_REBUILD_REQUESTED,
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.kafka_config import (
    ModelDeployAgentKafkaConfig,
    load_deploy_agent_kafka_config_from_env,
)
from deploy_agent.lane_policy import resolve_default_runtime_lane_from_env
from deploy_agent.tracking_ref import (
    ENV_TRACKING_REF,
    load_tracking_remote_ref_from_env,
)

logger = logging.getLogger(__name__)

ENV_HMAC_SECRET = "DEPLOY_AGENT_HMAC_SECRET"

PUBLISH_FLUSH_TIMEOUT_SECONDS = 30


class TriggerRefusedError(RuntimeError):
    """A precondition for publishing a command is missing or contradictory."""


def build_rebuild_command(
    *,
    git_ref: str | None,
    runtime_lane: EnumRuntimeLane | None,
    scope: Scope,
    build_source: BuildSource,
    requested_by: str,
    correlation_id: uuid.UUID,
    services: list[str],
    image_ref: str | None = None,
    image_digest: str | None = None,
) -> ModelRebuildRequested:
    """Build the command as the contract model, resolving declared defaults.

    ``git_ref`` falls back to ``origin/<DEPLOY_AGENT_TRACKING_REF>`` and
    ``runtime_lane`` to :func:`resolve_default_runtime_lane_from_env` -- both
    from declared environment, neither from a literal. Validation (services
    subset of scope, prod requires a digest) happens here, in the operator's
    own terminal, instead of as a rejection line in a journal on the host.
    """
    return ModelRebuildRequested(
        correlation_id=correlation_id,
        requested_by=requested_by,
        scope=scope,
        runtime_lane=runtime_lane or resolve_default_runtime_lane_from_env(),
        build_source=build_source,
        services=services,
        git_ref=git_ref or load_tracking_remote_ref_from_env(),
        image_ref=image_ref,
        image_digest=image_digest,
    )


def command_to_signed_envelope(
    command: ModelRebuildRequested, secret: str
) -> dict[str, Any]:
    """Serialise the contract model and attach the HMAC the agent verifies.

    ``mode="json"`` is what makes the round trip exact: UUIDs and StrEnums land
    as the JSON scalars the wire carries, so
    ``ModelRebuildRequested.model_validate(envelope_without_signature)``
    reconstructs the same command. ``tests/unit/test_trigger_contract_round_trip.py``
    asserts that on every published field.
    """
    if not secret:
        raise TriggerRefusedError(
            f"{ENV_HMAC_SECRET} is not set. An unsigned command is silently "
            "dropped by deploy_agent.auth.verify_command, so publishing one "
            "would look like success and do nothing. Source the operator env "
            "file that carries it."
        )
    return sign_envelope(command.model_dump(mode="json"), secret)


def masked_envelope_json(envelope: dict[str, Any]) -> str:
    """Render the envelope for the audit log with the signature truncated."""
    shown = dict(envelope)
    signature = str(shown.get("_signature", ""))
    shown["_signature"] = f"{signature[:8]}...<masked>"
    return json.dumps(shown, indent=2, sort_keys=True)


def publish_signed_command(
    envelope: dict[str, Any],
    kafka_config: ModelDeployAgentKafkaConfig,
) -> None:
    """Publish one command to the control bus, uncompressed and authenticated.

    ``compression_type=None`` is declared rather than inherited: see the module
    docstring for what a snappy record did to the agent.
    """
    try:
        from kafka import KafkaProducer
    except ImportError as exc:
        raise TriggerRefusedError(
            "kafka-python is not installed in this interpreter, so the command "
            "cannot be published. Install the agent venv beside the trigger "
            "script, or point DEPLOY_AGENT_PYTHON at an interpreter that has "
            "it. There is no unauthenticated fallback transport."
        ) from exc

    correlation_id = envelope["correlation_id"]
    producer = KafkaProducer(
        **kafka_config.producer_kwargs(),
        compression_type=None,
        value_serializer=lambda v: json.dumps(v, separators=(",", ":")).encode("utf-8"),
        key_serializer=lambda v: str(v).encode("utf-8"),
    )
    try:
        producer.send(
            TOPIC_REBUILD_REQUESTED,
            key=f"manual-{correlation_id}",
            value=envelope,
        )
        producer.flush(timeout=PUBLISH_FLUSH_TIMEOUT_SECONDS)
    finally:
        producer.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deploy-agent-trigger",
        description=(
            "Publish one signed rebuild-requested command to "
            f"{TOPIC_REBUILD_REQUESTED}."
        ),
    )
    parser.add_argument(
        "--git-ref",
        default=None,
        help=(
            "Ref the agent resets its deploy-source clone to, e.g. origin/dev. "
            "Defaults to origin/$DEPLOY_AGENT_TRACKING_REF."
        ),
    )
    # No argparse `choices` on the enum-valued flags, deliberately: argparse
    # renders its own "invalid choice" text and exits 2, which would replace the
    # refusal wording #3323 landed and its end-to-end binding test pins. The
    # accepted set is still the enum's own members -- see _resolve_enum below --
    # so a new lane or build source cannot be added to the model without this
    # entry point learning about it.
    parser.add_argument(
        "--runtime-lane",
        default=None,
        help=(
            "Lane to deploy. Defaults to the single lane named by "
            "$DEPLOY_AGENT_ALLOWED_LANES, else $DEPLOY_AGENT_TRACKING_REF when "
            "that branch name is also a lane name."
        ),
    )
    parser.add_argument(
        "--scope",
        default=Scope.RUNTIME.value,
        help="Service scope to rebuild (default: runtime).",
    )
    parser.add_argument(
        "--build-source",
        default=None,
        help=(
            "Build from a released artifact or from the workspace. Defaults to "
            "the build source the target lane can actually build: workspace on "
            "dev, release on stability-test and prod (OMN-16442)."
        ),
    )
    parser.add_argument(
        "--service",
        dest="services",
        action="append",
        default=[],
        help=(
            "Restrict the rebuild to one service; repeatable. Must be within "
            "--scope, which is checked here rather than on the host."
        ),
    )
    parser.add_argument("--image-ref", default=None, help="Optional image ref.")
    parser.add_argument(
        "--image-digest",
        default=None,
        help="Pinned image digest. REQUIRED for --runtime-lane prod.",
    )
    parser.add_argument("--requested-by", default="operator-manual")
    parser.add_argument("--correlation-id", default=None)
    parser.add_argument(
        "--reason",
        default=None,
        help=(
            "Free-text audit note. Printed locally and NOT published: "
            "ModelRebuildRequested forbids extra fields, and sending this was "
            "half of why every manual command was rejected (OMN-16442)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and sign the command, print it, publish nothing.",
    )
    return parser


def _resolve_enum[EnumT: StrEnum](raw: str, enum_type: type[EnumT], flag: str) -> EnumT:
    """Coerce a flag value to its enum member, refusing in the flag's own words.

    The accepted set is the enum's members, read at call time, so a lane or
    build source added to the model is accepted here without a second list to
    keep in step.
    """
    try:
        return enum_type(raw)
    except ValueError as exc:
        accepted = ", ".join(member.value for member in enum_type)
        raise TriggerRefusedError(
            f"unknown {flag} '{raw}'.\n       Accepted: {accepted} "
            f"({enum_type.__name__})."
        ) from exc


def _resolve_runtime_lane(raw: str | None) -> EnumRuntimeLane:
    """Resolve the lane from the flag, else from a declaration, else refuse.

    DIVERGENCE FROM #3323, STATED RATHER THAN SLIPPED IN: that PR made
    ``--runtime-lane`` unconditionally required. This resolves it from the
    lane's own declared environment first -- a single-lane
    ``DEPLOY_AGENT_ALLOWED_LANES``, or ``DEPLOY_AGENT_TRACKING_REF`` when the
    branch name is also a lane name -- because the trigger runs inside a lane's
    environment which already declares that lane twice over. It is NOT a
    weakening of that PR's rule: nothing but an explicit declaration resolves
    it, there is no literal default anywhere, and an undeclared lane refuses in
    the same words with the same exit code, which is what the binding test
    pins.
    """
    if raw:
        return _resolve_enum(raw, EnumRuntimeLane, "--runtime-lane")
    try:
        return resolve_default_runtime_lane_from_env()
    except RuntimeError as exc:
        accepted = " | ".join(lane.value for lane in EnumRuntimeLane)
        raise TriggerRefusedError(
            f"--runtime-lane is required ({accepted}).\n"
            "       It has no default: the lane selects the compose overlay,\n"
            "       compose project and health ports the deploy acts on.\n"
            f"       {exc}"
        ) from exc


#: OMN-16442: the build source each lane can actually build, named per lane
#: rather than carried as one literal default. ``release`` cannot carry
#: un-released merged-dev code (CLAUDE.md, "Cold vs warm lane bring-up"), which
#: is why the sanctioned dev refresh — scripts/runtime_build/refresh_dev_lane.sh
#: — has always run BUILD_SOURCE=workspace, and why a release-mode dev build is
#: refused by the prod promotion-lineage gate on every day rather than on a bad
#: day. The two sanctioned dev paths disagreed about the build source and the
#: agent path had picked the one that cannot work.
DEFAULT_BUILD_SOURCE_FOR_LANE: dict[EnumRuntimeLane, BuildSource] = {
    EnumRuntimeLane.DEV: BuildSource.WORKSPACE,
    EnumRuntimeLane.STABILITY_TEST: BuildSource.RELEASE,
    EnumRuntimeLane.PROD: BuildSource.RELEASE,
}


def _resolve_build_source(
    raw: str | None, runtime_lane: EnumRuntimeLane
) -> BuildSource:
    """Resolve the build source from the flag, else from the lane.

    An explicit ``--build-source`` is still honoured everywhere it can be
    satisfied. The one combination refused is ``release`` on the dev lane, and
    it is refused HERE — before a command is signed and published — because the
    executor would refuse it on the host anyway, with a message about prod
    promotion lineage that names neither the lane nor the reason.
    """
    if raw is None:
        return DEFAULT_BUILD_SOURCE_FOR_LANE[runtime_lane]
    build_source = _resolve_enum(raw, BuildSource, "--build-source")
    if build_source is BuildSource.RELEASE and runtime_lane is EnumRuntimeLane.DEV:
        raise TriggerRefusedError(
            "--build-source release is not buildable on --runtime-lane dev.\n"
            "       A release build must come from a tree whose HEAD is an\n"
            "       ancestor of the release-synced origin/main, and a dev head\n"
            "       is by construction not one, so the deploy agent's prod\n"
            "       promotion-lineage gate refuses it on every day, not on a\n"
            "       bad day. release also cannot carry un-released merged-dev\n"
            "       code, which is the whole point of a dev rebuild.\n"
            "       Use --build-source workspace (the dev default), or target\n"
            "       --runtime-lane stability-test."
        )
    return build_source


def _resolve_git_ref(raw: str | None) -> str:
    if raw:
        return raw
    try:
        return load_tracking_remote_ref_from_env()
    except RuntimeError as exc:
        raise TriggerRefusedError(
            f"no --git-ref given and {ENV_TRACKING_REF} is not set.\n"
            f"       Pass --git-ref origin/<branch>, or export\n"
            f"       {ENV_TRACKING_REF}=<branch> (e.g. dev) to supply the "
            f"default.\n       {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)

    correlation_id = (
        uuid.UUID(args.correlation_id) if args.correlation_id else uuid.uuid4()
    )

    try:
        runtime_lane = _resolve_runtime_lane(args.runtime_lane)
        # Prod deploys a pinned, stability-proven digest and never rebuilds from
        # a ref. The model enforces this too, but refusing here means the
        # operator is told before a command is signed rather than after the
        # agent rejects it.
        if runtime_lane is EnumRuntimeLane.PROD and not args.image_digest:
            raise TriggerRefusedError(
                "--runtime-lane prod requires --image-digest sha256:...\n"
                "       Production deploys the exact stability-proven digest."
            )
        command = build_rebuild_command(
            git_ref=_resolve_git_ref(args.git_ref),
            runtime_lane=runtime_lane,
            scope=_resolve_enum(args.scope, Scope, "--scope"),
            build_source=_resolve_build_source(args.build_source, runtime_lane),
            requested_by=args.requested_by,
            correlation_id=correlation_id,
            services=list(args.services),
            image_ref=args.image_ref,
            image_digest=args.image_digest,
        )
        envelope = command_to_signed_envelope(
            command, os.environ.get(ENV_HMAC_SECRET, "")
        )
    except (TriggerRefusedError, RuntimeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    print("=== deploy-agent-trigger ===")
    print(f"topic:          {TOPIC_REBUILD_REQUESTED}")
    print(f"git_ref:        {command.git_ref}")
    print(f"runtime_lane:   {command.runtime_lane.value}")
    print(f"scope:          {command.scope.value}")
    print(f"build_source:   {command.build_source.value}")
    print(f"correlation_id: {command.correlation_id}")
    print(f"requested_by:   {command.requested_by}")
    # Printed, not signed: see the NOTE ON --reason in the wrapper's header.
    print(
        f"reason:         {args.reason or '(none)'}"
        "  [local audit only, not in envelope]"
    )
    print("payload (sig masked):")
    print(masked_envelope_json(envelope))

    if args.dry_run:
        print("\n(dry-run: skipping the publish)")
        return 0

    try:
        kafka_config = load_deploy_agent_kafka_config_from_env()
    except (RuntimeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nPublishing to {kafka_config.bootstrap_servers} "
        f"({kafka_config.security_protocol}"
        + (f"/{kafka_config.sasl_mechanism}" if kafka_config.sasl_mechanism else "")
        + ") ..."
    )
    try:
        publish_signed_command(envelope, kafka_config)
    except Exception as exc:  # noqa: BLE001 — the operator needs the reason, not a traceback
        print(f"PUBLISH FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Published. correlation_id={command.correlation_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

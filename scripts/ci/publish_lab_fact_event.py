#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Publish one lab FACT document to the bus (OMN-18769, AC1 and AC3).

TWO CALLERS, ONE TRANSPORT. ``runtime-rebuild-trigger.yml`` publishes the
lab-pass receipt's verdict; ``lane-census-refresh.yml`` publishes the
census-OBSERVED document. Both are facts about a lab lane that a projection
folds onto that lane's row, both are emitted from a job on the .201 runner,
and both have to speak the same SASL transport. A second copy of that
resolution is a second thing to get wrong -- the topic is read OFF the
document rather than passed in, so this script can never publish a fact
onto a topic its own producer did not name.

WHAT THIS IS FOR. ``scripts/ci/lab_pass_receipt.py emit --event-out`` writes the
event document beside the artifact it has already written. That artifact stays
the durable evidence and the rule-24(b) delivery gate keeps reading it; nothing
here changes that. What this script adds is RENDERABILITY: an Actions artifact
is reachable only by an exact-name REST query carrying ``actions: read``, which
a dashboard panel cannot make, so a lab-pass verdict has been invisible to every
surface except the gate that consumes it. ``node_projection_lab_lane_health``
folds what this publishes onto the lane's row.

WHY IT IS A SEPARATE SCRIPT. ``lab_pass_receipt.py`` is a pure document builder
and a test pins it that way -- it opens no broker connection, because the four
jobs that emit a receipt differ in whether they can reach one. The transport
belongs beside the emitter, in its own file, not folded into the receipt model's
module.

WHY IT REPLACED A BARE ``rpk topic produce``. The first revision of OMN-18769
published with ``rpk topic produce --brokers "$KAFKA_BOOTSTRAP_SERVERS"``, and
both halves of that were wrong on the job it ran in. Nothing in
``runtime-rebuild-trigger.yml`` sets ``KAFKA_BOOTSTRAP_SERVERS``, so the step
could only ever take its own "unset" branch and warn; and the .201 dev-lane
Redpanda EXTERNAL listener has required SASL/SCRAM-SHA-256 over PLAINTEXT since
OMN-18012 Phase B, which a bare ``--brokers`` invocation does not speak. A step
that can only warn is not a mechanism, it is a comment that runs.

WHY IT NEVER INFERS THE TRANSPORT. The protocol and mechanism are resolved from
the checked-in lane overlay (omnimarket ``config/ci_bus_lanes.yaml``) by the
helpers ``trigger_rebuild_on_merge.py`` already owns, and from nowhere else.
Credential PRESENCE is not a statement about transport: the inference this
deliberately does not reproduce picked ``SASL_SSL`` whenever SASL credentials
happened to be in the environment and took down every OCC companion mint on
2026-09-07.

WHY EVERY FAILURE EXITS 0. This script reports a FACT about a lab pass that has
already finished. A broker that is unreachable is not a lane that failed, and
failing the job here would convert an observability publish into a delivery
outage. Every publish-side failure is loud on stderr and exits 0; the artifact,
which is the authority, is unaffected either way. That is not a swallowed error
-- there is no caller for whom this script's exit code is the answer to any
question about the lab.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

#: The one exit code this script has. See the module docstring.
EXIT_OK = 0

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIGGER_REBUILD = REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"

#: Bounded rather than unbounded: a hung broker must not hold a CI job open to
#: its own timeout. 30s matches the flush timeout the repo's other publishers
#: use (``scripts/publish_pr_merged_event.py``).
FLUSH_TIMEOUT_SECONDS = 30.0


def _warn(message: str) -> None:
    """Emit a GitHub Actions warning annotation and a plain stderr line.

    Both, deliberately: the annotation is what a human scanning the run sees,
    and the plain line is what survives into a downloaded log.
    """
    print(f"::warning::{message}")
    print(f"publish_lab_fact_event: {message}", file=sys.stderr)


def _load_trigger_rebuild() -> Any:
    """Import the lane-overlay helpers by path.

    ``scripts/`` is not a package, so this is a file-location import rather
    than a module import. The helpers -- ``load_ci_bus_overlay``,
    ``resolve_ci_bus_broker``, ``resolve_ci_bus_security``,
    ``build_kafka_producer_config`` -- are reused rather than re-implemented
    precisely so this publisher cannot drift from the transport rules the
    rebuild trigger already enforces.
    """
    spec = importlib.util.spec_from_file_location(
        "omn18769_trigger_rebuild_on_merge", TRIGGER_REBUILD
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError(f"cannot load {TRIGGER_REBUILD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event",
        required=True,
        type=Path,
        help=(
            "the fact document to publish: `lab_pass_receipt.py emit "
            "--event-out`, or `lane-census-check.sh --observed-out`"
        ),
    )
    parser.add_argument(
        "--bus-lane",
        default="dev",
        help=(
            "the CI bus lane id to publish on, as declared in the overlay. "
            "A fact about the .201 dev lane belongs on `dev`."
        ),
    )
    parser.add_argument(
        "--bus-overlay",
        type=Path,
        default=None,
        help=(
            "path to the checked-in omnimarket config/ci_bus_lanes.yaml, "
            "sparse-checked-out by the calling job. Absent means there is no "
            "declared transport to publish on, which is reported and skipped."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.event.is_file():
        _warn(f"no lab fact document at {args.event} — nothing to publish")
        return EXIT_OK

    try:
        payload: dict[str, Any] = json.loads(args.event.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"lab fact document at {args.event} is unreadable: {exc}")
        return EXIT_OK

    topic = payload.get("topic")
    if not isinstance(topic, str) or not topic:
        _warn("lab fact document names no topic — refusing to guess one")
        return EXIT_OK

    if args.bus_overlay is None or not args.bus_overlay.is_file():
        _warn(
            f"no CI bus overlay at {args.bus_overlay} — the lane's declared "
            "transport is unknown and this publish is SKIPPED rather than "
            "guessed (OMN-18012)"
        )
        return EXIT_OK

    try:
        trigger = _load_trigger_rebuild()
        overlay = trigger.load_ci_bus_overlay(args.bus_overlay)
        broker = trigger.resolve_ci_bus_broker(overlay=overlay, lane=args.bus_lane)
        protocol, mechanism = trigger.resolve_ci_bus_security(
            overlay=overlay, lane=args.bus_lane
        )
    except Exception as exc:  # noqa: BLE001 — see the module docstring on exit 0
        _warn(
            f"cannot resolve the declared transport for lane {args.bus_lane!r}: {exc}"
        )
        return EXIT_OK

    if not broker:
        _warn(
            f"CI bus lane {args.bus_lane!r} declares no cross-process broker "
            "(in-memory lane) — the fact is NOT published"
        )
        return EXIT_OK

    import os

    try:
        config = trigger.build_kafka_producer_config(
            broker,
            os.environ.get("KAFKA_SASL_USERNAME", ""),
            os.environ.get("KAFKA_SASL_PASSWORD", ""),
            protocol,
            mechanism,
        )
    except ValueError as exc:
        _warn(f"cannot build a producer for lane {args.bus_lane!r}: {exc}")
        return EXIT_OK

    try:
        from confluent_kafka import Producer
    except ImportError as exc:  # pragma: no cover - the CI image carries it
        _warn(f"confluent_kafka is unavailable: {exc}")
        return EXIT_OK

    # The message KEY is the lane, matching the projection's key_columns, so a
    # lane's facts land on one partition and their order is the order the
    # reducer sees. A census-observed document names no single lane (it covers
    # the whole host), so it keys null and takes the default partitioner --
    # correct, because its fold is per-finding and already guarded on
    # `observed_at`. The reducer guards on observed_at in SQL regardless -- this
    # makes the common case ordered rather than relying on that guard.
    key = payload.get("lane")
    try:
        producer = Producer(config)
        producer.produce(
            topic,
            key=key.encode() if isinstance(key, str) else None,
            value=json.dumps(payload).encode(),
        )
        remaining = producer.flush(timeout=FLUSH_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 — see the module docstring on exit 0
        _warn(f"lab fact publish to {topic} FAILED: {exc}")
        return EXIT_OK

    if remaining:
        _warn(
            f"lab fact publish to {topic} did not flush within "
            f"{FLUSH_TIMEOUT_SECONDS}s ({remaining} message(s) still queued)"
        )
        return EXIT_OK

    # Named from the document's own fields, and only the ones it carries: a
    # lab-pass receipt has `result`/`lane`/`sha`, a census-observed document has
    # `event_type`/`host`/`drift_count`, and printing a missing field as `None`
    # is how a log line starts asserting something the fact never said.
    described = ", ".join(
        f"{field}={payload[field]!r}"
        for field in ("event_type", "lane", "sha", "result", "host", "drift_count")
        if field in payload
    )
    print(
        f"published lab fact ({described}) to {topic} on lane {args.bus_lane!r} ({protocol})"
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

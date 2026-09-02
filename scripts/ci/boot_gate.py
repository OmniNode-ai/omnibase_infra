#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17534 — the moving parts of the candidate boot gate that deserve a test.

``deliver-dev-candidate-to-staging.yml``'s ``candidate-boot-gate`` job boots the
just-built runtime candidate on an ephemeral kind cluster, against the REAL
onex-dev manifests rendered through ``omninode_infra``'s ``k8s/onex-lab``
overlay, before the candidate may be announced to ``omninode_infra``.

Three pieces live here rather than in workflow YAML, because each has a way of
being subtly wrong that a shell one-liner would hide:

``pin-image``
    Writes the caller's image override into the overlay's kustomization. Getting
    this wrong silently boots a MIXTURE of the candidate and whatever the tree
    pinned, and reports the result as if it were one image.

``wait``
    Polls every runtime-family Deployment and the projection snapshot topic, and
    fails closed. A ``kubectl rollout status`` loop reports only the FIRST
    Deployment that fails and hides the rest, which is the opposite of what a
    diagnostic gate should do.

``redact``
    Strips Secret payloads out of the rendered manifest before it is uploaded as
    an artifact. The render contains generated Secret data; OMN-17534 AC-5 says
    no credential value may reach the logs, and "we generated it so it does not
    matter" is not a reason to publish one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

#: The topic the projection plane must have provisioned for
#: omnimarket-projection-api to reach readiness. Its absence is what left the
#: API 0/1 on UnknownTopicOrPartitionError during the 2026-09-02 staging
#: incident, so "every Deployment is Ready" alone is not a sufficient gate.
REQUIRED_TOPIC = "onex.snapshot.projection.consumer-flow.v1"

#: Emitted in place of every Secret value in the uploaded render.
REDACTED = "<redacted by boot_gate.py — OMN-17534 AC-5>"


# ---------------------------------------------------------------------------
# pin-image
# ---------------------------------------------------------------------------
def pin_image(kustomization: Path, name: str, digest: str, new_name: str | None) -> int:
    """Append an ``images:`` override to a kustomization, in place.

    Equivalent to ``kustomize edit set image``, done here so the gate needs only
    ``kubectl kustomize`` and not a second binary, and so the behaviour is
    covered by a unit test instead of trusted.

    The overlay commits NO ``images:`` block of its own: OMN-17533 AC-5 forbids a
    committed ``sha256:`` digest there while AC-4 requires caller override, and
    both hold only if the pin arrives from outside. An existing block is
    therefore a signal that something changed upstream, not a merge case to
    silently handle -- refuse rather than guess which pin wins.
    """
    if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
        print(f"error: {digest!r} is not a sha256 digest", file=sys.stderr)
        return 1
    text = kustomization.read_text()
    document = yaml.safe_load(text)
    if document.get("images"):
        print(
            f"error: {kustomization} already declares an `images:` block. The "
            "onex-lab overlay is expected to commit none (OMN-17533 AC-4/AC-5); "
            "refusing to guess which pin should win.",
            file=sys.stderr,
        )
        return 1
    block = [
        "",
        "# Appended by scripts/ci/boot_gate.py (OMN-17534). Never committed:",
        "# this file is edited only inside the gate's ephemeral checkout.",
        "images:",
        f"  - name: {name}",
        f"    newName: {new_name or name}",
        f"    digest: {digest}",
        "",
    ]
    kustomization.write_text(text.rstrip("\n") + "\n" + "\n".join(block))
    print(f"pinned {name} -> {new_name or name}@{digest}")
    return 0


# ---------------------------------------------------------------------------
# redact
# ---------------------------------------------------------------------------
def redact_render(source: Path, destination: Path) -> int:
    """Copy a rendered manifest with every Secret payload replaced.

    Key NAMES are kept: they are the useful diagnostic ("the Secret exists and
    carries OMNINODE_INTERNAL_DB_URL") and they are already public in the
    manifests. Only the values go.
    """
    documents = list(yaml.safe_load_all(source.read_text()))
    kept: list[dict[str, Any]] = []
    redacted_count = 0
    for document in documents:
        if not document:
            continue
        if document.get("kind") == "Secret":
            for field in ("data", "stringData"):
                if document.get(field):
                    document[field] = dict.fromkeys(document[field], REDACTED)
                    redacted_count += 1
        kept.append(document)
    destination.write_text(yaml.safe_dump_all(kept, sort_keys=False))
    print(f"redacted {redacted_count} Secret payload(s) into {destination}")
    return 0


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------
def _kubectl_json(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", *args, "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload: dict[str, Any] = json.loads(result.stdout)
    return payload


def _deployment_rows(namespace: str) -> list[tuple[str, int, int, str]]:
    payload = _kubectl_json(["get", "deployments", "-n", namespace])
    rows: list[tuple[str, int, int, str]] = []
    for item in payload.get("items", []):
        name = item["metadata"]["name"]
        status = item.get("status", {})
        desired = item.get("spec", {}).get("replicas", 0)
        ready = status.get("readyReplicas", 0)
        reason = ""
        for condition in status.get("conditions", []) or []:
            if (
                condition.get("type") == "Available"
                and condition.get("status") != "True"
            ):
                reason = (
                    f"{condition.get('reason', '')}: {condition.get('message', '')}"
                )
        rows.append((name, ready, desired, reason))
    return sorted(rows)


def _topic_exists(namespace: str, broker_deployment: str, topic: str) -> bool:
    result = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            namespace,
            f"deployment/{broker_deployment}",
            "--",
            "rpk",
            "topic",
            "list",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return any(
        line.split()[0] == topic for line in result.stdout.splitlines() if line.split()
    )


def wait_for_boot(
    *,
    namespace: str,
    timeout_seconds: int,
    poll_seconds: int,
    lane_prefix: str,
    broker_deployment: str,
    require_topic: bool,
) -> int:
    """Poll until EVERY runtime-family Deployment is Ready and the topic exists.

    Reports the whole table on every failure, not just the first offender. When
    three of fifteen Deployments are down for one reason and a fourth is down for
    another, a gate that names one of them costs a second run to find the rest --
    which is exactly the cost OMN-17519 paid.

    The lane's own stand-ins (broker, Postgres, Valkey) are excluded from the
    runtime-family roster by name prefix and waited on separately, so a slow
    broker is never reported as a candidate defect.
    """
    deadline = time.monotonic() + timeout_seconds
    last_rows: list[tuple[str, int, int, str]] = []
    topic_seen = False
    while True:
        last_rows = _deployment_rows(namespace)
        runtime_rows = [row for row in last_rows if not row[0].startswith(lane_prefix)]
        lane_rows = [row for row in last_rows if row[0].startswith(lane_prefix)]
        if not runtime_rows:
            print(
                f"error: no runtime-family Deployment found in namespace {namespace}. "
                "The render applied nothing, or applied it somewhere else.",
                file=sys.stderr,
            )
            return 1
        # A Deployment the manifests scale to zero is Ready by definition --
        # there is nothing to run. Four of the onex-dev runtime family are
        # `replicas: 0` in the committed manifests (omnibase-intelligence-api,
        # omninode-agent-actions-consumer, omninode-contract-resolver,
        # omninode-skill-lifecycle-consumer), so requiring `desired > 0` per row
        # made this gate structurally un-passable: those four report 0/0 forever
        # and no candidate could satisfy it. Found by the first real run
        # (33674463837), which is what a first run is for.
        #
        # The `desired > 0` guard is not deleted, only moved: it applies ONCE to
        # the roster as a whole below, so a render that scaled the entire plane
        # to zero still fails rather than passing vacuously.
        if not any(desired > 0 for _n, _ready, desired, _r in runtime_rows):
            print(
                "error: every runtime-family Deployment in namespace "
                f"{namespace} declares zero replicas. Nothing was booted, so "
                "there is nothing for this gate to have proven.",
                file=sys.stderr,
            )
            return 1
        runtime_ready = all(ready == desired for _n, ready, desired, _r in runtime_rows)
        lane_ready = all(ready == desired for _n, ready, desired, _r in lane_rows)
        if runtime_ready and lane_ready:
            if not require_topic:
                topic_seen = True
            elif not topic_seen:
                topic_seen = _topic_exists(namespace, broker_deployment, REQUIRED_TOPIC)
            if topic_seen:
                _print_table(last_rows, topic_seen)
                print(
                    "\nBOOT GATE PASS: every Deployment Ready and the topic is provisioned."
                )
                return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        not_ready = [name for name, ready, desired, _r in last_rows if ready != desired]
        print(
            f"[{int(remaining)}s left] waiting on "
            f"{len(not_ready)} Deployment(s): {', '.join(sorted(not_ready)) or 'none'}"
            + (
                ""
                if topic_seen or not require_topic
                else f"; topic {REQUIRED_TOPIC} not yet present"
            )
        )
        time.sleep(min(poll_seconds, max(1, int(remaining))))

    _print_table(last_rows, topic_seen)
    print(
        "\nBOOT GATE FAIL: the candidate did not reach a healthy runtime plane "
        f"within {timeout_seconds}s.",
        file=sys.stderr,
    )
    if not topic_seen and require_topic:
        print(
            f"  - topic {REQUIRED_TOPIC} was never provisioned. "
            "omnimarket-projection-api's /ready is fail-closed on it, so this is a "
            "cause and not only a symptom.",
            file=sys.stderr,
        )
    for name, ready, desired, reason in last_rows:
        if ready != desired:
            print(
                f"  - {name}: {ready}/{desired} Ready. {reason}".rstrip(),
                file=sys.stderr,
            )
    print(
        "\nFull pod logs, --previous logs, describes and the redacted render are "
        "in this job's uploaded artifact.",
        file=sys.stderr,
    )
    return 1


def _print_table(rows: list[tuple[str, int, int, str]], topic_seen: bool) -> None:
    width = max((len(name) for name, *_ in rows), default=10)
    print("\n" + "DEPLOYMENT".ljust(width) + "  READY  STATE")
    for name, ready, desired, reason in rows:
        if ready != desired:
            state = "NOT READY"
        elif desired == 0:
            state = "Ready (scaled to zero by the manifests)"
        else:
            state = "Ready"
        print(f"{name.ljust(width)}  {ready}/{desired}    {state}")
        if reason:
            print(" " * (width + 2) + f"       {reason}")
    print(f"\n{REQUIRED_TOPIC}: {'present' if topic_seen else 'ABSENT'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pin = sub.add_parser(
        "pin-image", help="append an images: override to a kustomization"
    )
    pin.add_argument("--kustomization", type=Path, required=True)
    pin.add_argument("--name", required=True)
    pin.add_argument("--new-name", default=None)
    pin.add_argument("--digest", required=True)

    red = sub.add_parser(
        "redact", help="strip Secret payloads from a rendered manifest"
    )
    red.add_argument("--source", type=Path, required=True)
    red.add_argument("--destination", type=Path, required=True)

    wait = sub.add_parser(
        "wait", help="poll until the runtime plane is up, fail closed"
    )
    wait.add_argument("--namespace", default="onex-dev")
    wait.add_argument("--timeout-seconds", type=int, default=900)
    wait.add_argument("--poll-seconds", type=int, default=15)
    wait.add_argument("--lane-prefix", default="onex-lab-")
    wait.add_argument("--broker-deployment", default="onex-lab-redpanda")
    wait.add_argument(
        "--skip-topic-check",
        action="store_true",
        help=(
            "diagnostic use only; the gate never passes this. The topic check is "
            "half of what makes a green run mean anything."
        ),
    )

    args = parser.parse_args()
    if args.command == "pin-image":
        return pin_image(args.kustomization, args.name, args.digest, args.new_name)
    if args.command == "redact":
        return redact_render(args.source, args.destination)
    return wait_for_boot(
        namespace=args.namespace,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
        lane_prefix=args.lane_prefix,
        broker_deployment=args.broker_deployment,
        require_topic=not args.skip_topic_check,
    )


if __name__ == "__main__":
    raise SystemExit(main())

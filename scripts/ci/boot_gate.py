#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17534 — the moving parts of the candidate boot gate that deserve a test.

``deliver-dev-candidate-to-staging.yml``'s ``candidate-boot-gate`` job boots the
just-built runtime candidate on an ephemeral kind cluster, against the REAL
onex-dev manifests rendered through ``omninode_infra``'s ``k8s/onex-lab``
overlay, before the candidate may be announced to ``omninode_infra``.

Four pieces live here rather than in workflow YAML, because each has a way of
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

``prepare-host-paths``
    Creates the render's static hostPath directories on the ephemeral node,
    already owned by the uid the render says will run there. `hostPath` is the
    one volume type the kubelet does not apply `fsGroup` to, so a non-root pod
    meets a root-owned directory on a fresh cluster and crash-loops. Ownership
    is derived from the render, never restated here (OMN-18364).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
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
# prepare-host-paths
# ---------------------------------------------------------------------------
#: Mode applied to a prepared hostPath directory. A database data root is
#: private to its own uid on the persistent lane (`k8s/onex-lab/substitutions/
#: postgres.yaml` records it as "already owned by 70 at mode 0700"), and this
#: step exists to give the ephemeral cluster the same starting condition.
DEFAULT_HOST_PATH_MODE = "0700"

#: Workload kinds carrying a pod template that can mount a PersistentVolumeClaim.
_POD_TEMPLATE_KINDS = ("Deployment", "StatefulSet")


class HostPathOwnershipError(RuntimeError):
    """A hostPath PersistentVolume whose required ownership cannot be derived."""


def _pod_template(document: dict[str, Any]) -> dict[str, Any]:
    spec = document.get("spec") or {}
    template = spec.get("template") or {}
    pod_spec = template.get("spec") or {}
    return pod_spec if isinstance(pod_spec, dict) else {}


def resolve_host_path_plans(
    documents: list[dict[str, Any]], mode: str = DEFAULT_HOST_PATH_MODE
) -> list[tuple[str, int, int, str]]:
    """Return ``(path, uid, gid, mode)`` for every hostPath PV in a render.

    OMN-18364. `hostPath` is the ONE volume type the kubelet does not apply
    `fsGroup` to: for `type: DirectoryOrCreate` it creates the directory as
    root at 0755 and never re-owns an existing one. That was harmless while
    the lane's Postgres ran as root and its own entrypoint dropped privileges
    and chowned; OMN-18765 then set `runAsNonRoot: true` / `runAsUser: 70` on
    the pod to match the namespace's restricted Pod Security Standard, and the
    two changes are individually correct and jointly fatal on a FRESH cluster.

    Measured 2026-09-19: on the persistent lab lane the directory already
    existed owned by 70, so it kept working and nothing surfaced. On the boot
    gate's per-run kind cluster it is created fresh every time, so
    `onex-lab-postgres` crash-looped on

        mkdir: can't create directory '/var/lib/postgresql/data/pgdata':
        Permission denied

    with `Restart Count: 5`, the Deployment blew its 180s
    `progressDeadlineSeconds`, and every delivery since 09:29Z failed at
    `error: deployment "onex-lab-postgres" exceeded its progress deadline`.

    The namespace enforces `restricted`, so the manifest cannot fix this for
    itself: an initContainer running as root to chown its own volume is
    refused at admission. The directory has to arrive already owned, which is
    what the persistent lane has by history and the ephemeral cluster must be
    given deliberately.

    Ownership is DERIVED from the render rather than asserted here, so a change
    to the pod's uid moves this step with it instead of leaving a second copy
    of the number to drift. The chain is
    PersistentVolume -> claimRef/PersistentVolumeClaim -> the workload whose
    pod template mounts that claim -> that pod's `securityContext`.

    Fail-closed at every link: an unclaimed volume, a claim no workload mounts,
    two workloads disagreeing about the uid, a pod that declares no
    `runAsUser`, and an empty result set are each an error. An empty set is an
    error because a preparation step that silently prepared nothing is
    indistinguishable from one that worked, which is the failure this whole
    gate exists to refuse.
    """
    volumes: dict[str, str] = {}
    claim_to_volume: dict[str, str] = {}
    for document in documents:
        if not document:
            continue
        kind = document.get("kind")
        name = (document.get("metadata") or {}).get("name")
        spec = document.get("spec") or {}
        if kind == "PersistentVolume":
            host_path = (spec.get("hostPath") or {}).get("path")
            if host_path and name:
                volumes[name] = host_path
                claim_ref = (spec.get("claimRef") or {}).get("name")
                if claim_ref:
                    claim_to_volume[claim_ref] = name
        elif kind == "PersistentVolumeClaim":
            volume_name = spec.get("volumeName")
            if volume_name and name:
                claim_to_volume[name] = volume_name

    # volume name -> {(uid, gid)} contributed by each workload mounting it
    owners: dict[str, set[tuple[int, int]]] = {name: set() for name in volumes}
    claimants: dict[str, list[str]] = {name: [] for name in volumes}
    for document in documents:
        if not document or document.get("kind") not in _POD_TEMPLATE_KINDS:
            continue
        workload = (document.get("metadata") or {}).get("name") or "<unnamed>"
        pod_spec = _pod_template(document)
        security = pod_spec.get("securityContext") or {}
        for volume in pod_spec.get("volumes") or []:
            claim = (volume.get("persistentVolumeClaim") or {}).get("claimName")
            if not claim:
                continue
            volume_name = claim_to_volume.get(claim)
            if volume_name not in volumes:
                continue
            claimants[volume_name].append(workload)
            run_as_user = security.get("runAsUser")
            if run_as_user is None:
                raise HostPathOwnershipError(
                    f"{workload} mounts hostPath volume {volume_name!r} "
                    f"({volumes[volume_name]}) but its pod securityContext "
                    "declares no runAsUser, so the directory's required owner "
                    "cannot be derived from the render (OMN-18364)."
                )
            run_as_group = security.get("runAsGroup", security.get("fsGroup"))
            if run_as_group is None:
                raise HostPathOwnershipError(
                    f"{workload} mounts hostPath volume {volume_name!r} "
                    f"({volumes[volume_name]}) but its pod securityContext "
                    "declares neither runAsGroup nor fsGroup (OMN-18364)."
                )
            owners[volume_name].add((int(run_as_user), int(run_as_group)))

    plans: list[tuple[str, int, int, str]] = []
    for volume_name, host_path in sorted(volumes.items(), key=lambda item: item[1]):
        found = owners[volume_name]
        if not found:
            raise HostPathOwnershipError(
                f"hostPath volume {volume_name!r} ({host_path}) is declared but "
                "no Deployment or StatefulSet in the render mounts its claim, "
                "so the kubelet would create it root-owned and any non-root "
                "consumer added later would crash-loop on it (OMN-18364)."
            )
        if len(found) > 1:
            raise HostPathOwnershipError(
                f"hostPath volume {volume_name!r} ({host_path}) is mounted by "
                f"{sorted(claimants[volume_name])} with disagreeing pod "
                f"securityContexts {sorted(found)}. One directory cannot be "
                "owned by two uids (OMN-18364)."
            )
        uid, gid = found.pop()
        plans.append((host_path, uid, gid, mode))

    if not plans:
        raise HostPathOwnershipError(
            "the render declares no hostPath PersistentVolume, so this step "
            "prepared nothing. That is indistinguishable from a step that "
            "worked, so it fails closed: if the lane genuinely stopped using a "
            "static hostPath volume, delete this step deliberately (OMN-18364)."
        )
    return plans


def _docker_exec(node: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", node, *command],
        capture_output=True,
        text=True,
        check=False,
    )


def prepare_host_paths(
    render: Path,
    node: str,
    mode: str = DEFAULT_HOST_PATH_MODE,
    *,
    runner: Callable[[str, list[str]], subprocess.CompletedProcess[str]] = _docker_exec,
) -> int:
    """Create every hostPath directory the render needs, already owned.

    `runner` is a keyword-only seam so the resolution and the readback are
    testable without a cluster; argv cannot reach it.
    """
    try:
        plans = resolve_host_path_plans(
            list(yaml.safe_load_all(render.read_text())), mode=mode
        )
    except HostPathOwnershipError as exc:
        print(f"::error::{exc}")
        return 1

    failures = 0
    for path, uid, gid, file_mode in plans:
        created = runner(
            node,
            ["install", "-d", "-m", file_mode, "-o", str(uid), "-g", str(gid), path],
        )
        if created.returncode != 0:
            print(
                f"::error::could not prepare {path} on {node}: "
                f"{created.stderr.strip() or created.stdout.strip()}"
            )
            failures += 1
            continue
        # Read the ownership back off the node. `install` reporting success is
        # not proof the directory is owned as asked -- an existing directory on
        # a filesystem that refuses chown would pass the command and fail the
        # pod, minutes later, as the permission error this step exists to stop.
        readback = runner(node, ["stat", "-c", "%u:%g:%a", path])
        observed = readback.stdout.strip()
        expected = f"{uid}:{gid}:{int(file_mode, 8):o}"
        if readback.returncode != 0 or observed != expected:
            print(
                f"::error::{path} on {node} reads back as "
                f"{observed or readback.stderr.strip()!r}, expected {expected!r}"
            )
            failures += 1
            continue
        print(f"prepared {path} on {node}: {observed}")

    return 1 if failures else 0


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

    prep = sub.add_parser(
        "prepare-host-paths",
        help="create the render's hostPath dirs on a node, already owned",
    )
    prep.add_argument("--render", type=Path, required=True)
    prep.add_argument("--node", required=True)
    prep.add_argument("--mode", default=DEFAULT_HOST_PATH_MODE)

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
    if args.command == "prepare-host-paths":
        return prepare_host_paths(args.render, args.node, args.mode)
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

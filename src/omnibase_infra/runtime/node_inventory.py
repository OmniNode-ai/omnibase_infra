# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The node inventory an image carries, and the readback that resolves it.

OMN-18708 (mechanism M1 of
``beta/plans/2026-09-18-program-board-capabilities-and-value.md``).

The missing hop
---------------
Nothing in the chain from a contract to a deployed image named a node. The
lab-pass receipt is keyed by commit and lane; the runtime change classifier
reduces a diff to one repository-wide boolean; the rebuild is whole-lane; the
promotion grant's match key is a lane, a target kind and an image digest. The
image label block was the decisive negative -- ``docker/Dockerfile.runtime``
stamped an OCI revision from the build's commit, and no label anywhere named a
node, a contract, or a contract hash.

The one place the link already existed is the runtime's own introspection
manifest, which carries every discovered contract alongside the image identity.
That is a live-process read: a promotion gate running with no cluster access
cannot reach it. This module puts the same facts somewhere ``docker inspect``
can read them.

The triple, and why exactly these three
---------------------------------------
``name`` alone answers "is this node in the image". ``name`` + ``node_version``
answers "which declared version of it". Neither detects a contract whose BODY
was rewritten with its name and version untouched, which is the drift class the
inventory exists to catch -- so the third field is the contract's **content**
hash, in the canonical form
:mod:`omnibase_infra.runtime.util_contract_content_hash` declares. There is
deliberately no second hasher here: this module calls that function, and a
build step with no Python reproduces it with the shell one-liner that module
documents.

Truthful by construction, not by assertion
------------------------------------------
A Dockerfile cannot turn a ``RUN`` step's output into a ``LABEL``, so the value
has to arrive as a build argument -- which on its own would mean the label says
whatever the build script was told to say. The build therefore does two things
with this module, and the second is what makes the first trustworthy:

1. ``emit`` runs inside the built image, over the image's OWN discovery pass,
   and prints the label value. The build script captures it and re-runs the
   build passing it as ``NODE_INVENTORY`` (every earlier layer is a cache hit,
   because the ``ARG`` is declared immediately before the ``LABEL`` at the end
   of the runtime stage).
2. ``verify`` runs inside the image in that second pass and re-derives the
   inventory from the same discovery pass, refusing the build unless the
   stamped value is byte-identical. A build script that passed a stale,
   hand-written or empty value fails HERE, inside the image it is describing.

So the label is not a claim the build context made about the image; it is the
image's own answer, checked against itself.

Failure behaviour is uniformly fail-closed
------------------------------------------
An empty inventory, an unparseable one, a duplicate name, a hash that is not 64
lowercase hex, and a contract the discovery pass found with no content hash are
all :class:`NodeInventoryError`. None of them degrades to an empty label. A
gate reading an empty label would learn "this image contains nothing", which is
never true and is indistinguishable from "nobody stamped this image": an absent
label is a visible gap, an empty one is a false fact.

Related:
    - OMN-18709: the contract content hash (the third field).
    - OMN-18714 (T3): the lab rung that consumes the readback.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # fixed argv, no shell, trusted docker binary
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.errors.error_node_inventory import NodeInventoryError
from omnibase_infra.runtime.util_contract_content_hash import contract_content_hash

if TYPE_CHECKING:
    from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
        ModelAutoWiringManifest,
    )

__all__ = [
    "NODE_INVENTORY_LABEL",
    "NODE_INVENTORY_SCHEMA",
    "ModelNodeInventoryEntry",
    "build_node_inventory",
    "parse_node_inventory_label",
    "read_image_node_inventory",
    "render_node_inventory_label",
    "resolve_entry_to_contract_path",
]

#: The OCI label key. ``com.omninode.*`` is this repository's existing custom
#: prefix -- ``build_source``, ``promotion_class``, ``non_main_lineage`` and
#: ``workspace_provenance_manifest`` are already stamped under it by
#: ``docker/Dockerfile.runtime`` -- so a reader enumerating OmniNode facts about
#: an image finds this one in the same namespace rather than a second one.
NODE_INVENTORY_LABEL: Final[str] = "com.omninode.node_inventory"

#: Carried INSIDE the label value. A consumer that does not recognise the
#: schema refuses the label rather than best-effort reading a list whose entry
#: shape it is guessing at. Bump when a field is added or a meaning changes.
NODE_INVENTORY_SCHEMA: Final[str] = "node_inventory.v1"

#: 64 lowercase hex, the rendering
#: :func:`~omnibase_infra.runtime.util_contract_content_hash.contract_content_hash`
#: declares. An abbreviated or uppercase digest is refused rather than
#: normalised: two contracts can share a prefix, and a reader that normalises is
#: a reader that can match the wrong contract.
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelNodeInventoryEntry(BaseModel):
    """One node the image ships, named so it can be tied back to its contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, description="Node name from the contract")
    node_version: str = Field(
        ..., min_length=1, description="Declared node version string"
    )
    contract_content_hash: str = Field(
        ...,
        description=(
            "Canonical content hash of the contract file, in the form declared "
            "by omnibase_infra.runtime.util_contract_content_hash."
        ),
    )

    def model_post_init(self, _context: object, /) -> None:
        if not _HASH_RE.match(self.contract_content_hash):
            msg = (
                f"contract {self.name!r}: contract_content_hash="
                f"{self.contract_content_hash!r} is not 64 lowercase hex "
                "characters. The canonical form is declared in "
                "util_contract_content_hash; an abbreviated or uppercase "
                "digest is refused rather than normalised, because a reader "
                "that normalises can match the wrong contract."
            )
            raise NodeInventoryError(msg, contract_name=self.name)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "node_version": self.node_version,
            "contract_content_hash": self.contract_content_hash,
        }


def build_node_inventory(
    manifest: ModelAutoWiringManifest,
) -> tuple[ModelNodeInventoryEntry, ...]:
    """Build the inventory from a discovery manifest, in a stable order.

    Sorted by ``(name, node_version)`` so two builds of one commit render the
    same string and a byte comparison of two labels is a comparison of two node
    sets, not of two iteration orders.

    Args:
        manifest: The manifest the image's own discovery pass produced.

    Returns:
        One entry per discovered contract.

    Raises:
        NodeInventoryError: The manifest discovered no contracts, a contract
            carries no content hash (it was not read off disk, so there is
            nothing to hash and inventing a value would be the exact silent
            failure this ticket removes), or two contracts share a name.
    """
    if not manifest.contracts:
        msg = (
            "discovery found no contracts, so there is no inventory to stamp. "
            "Refusing to emit an empty label: a reader would learn 'this image "
            "contains nothing', which is never true and is indistinguishable "
            "from 'nobody stamped this image'."
        )
        raise NodeInventoryError(msg)

    entries: list[ModelNodeInventoryEntry] = []
    for contract in manifest.contracts:
        if not contract.contract_content_hash:
            msg = (
                f"contract {contract.name!r} carries no content hash, so its "
                "triple cannot be completed. A contract reaches the inventory "
                "only via the discovery pass, which hashes the file it parsed; "
                "a null here means this manifest was not produced by that pass."
            )
            raise NodeInventoryError(msg, contract_name=contract.name)
        entries.append(
            ModelNodeInventoryEntry(
                name=contract.name,
                node_version=contract.node_version,
                contract_content_hash=contract.contract_content_hash,
            )
        )

    _refuse_duplicate_names(entries)
    return tuple(sorted(entries, key=lambda e: (e.name, e.node_version)))


def _refuse_duplicate_names(entries: Sequence[ModelNodeInventoryEntry]) -> None:
    """Two rows with one name means one of them is unreadable.

    A consumer resolving a name to a contract would have to pick, and whichever
    it picked would look authoritative.
    """
    names = [e.name for e in entries]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        msg = (
            f"duplicate node name(s) {duplicates} in the inventory. A consumer "
            "resolving a name to a contract would have to pick one, and "
            "whichever it picked would look authoritative."
        )
        raise NodeInventoryError(msg, contract_name=duplicates[0])


def render_node_inventory_label(
    entries: Iterable[ModelNodeInventoryEntry],
) -> str:
    """Render the label value: one compact JSON object, no newlines.

    Newlines are excluded because the value crosses a ``--build-arg`` boundary
    and a ``GITHUB_OUTPUT`` one, both of which are line-oriented.

    Raises:
        NodeInventoryError: The inventory is empty, or two entries share a name.
    """
    ordered = tuple(sorted(entries, key=lambda e: (e.name, e.node_version)))
    if not ordered:
        msg = (
            "refusing to render an empty node inventory label. An empty label "
            "is a false fact, not a missing one: it asserts the image ships no "
            "nodes. A build that cannot determine its inventory must fail."
        )
        raise NodeInventoryError(msg)
    _refuse_duplicate_names(ordered)
    return json.dumps(
        {
            "schema": NODE_INVENTORY_SCHEMA,
            "nodes": [e.to_dict() for e in ordered],
        },
        separators=(",", ":"),
    )


def parse_node_inventory_label(raw: str | None) -> tuple[ModelNodeInventoryEntry, ...]:
    """Parse a stamped label value back into entries, refusing anything unclear.

    Absent, blank, non-JSON, wrong-schema, not-an-object, empty-node-list, and
    an entry with an unknown or missing field are all refusals. There is no
    best-effort branch: a reader that guesses at an unfamiliar label is how a
    gate goes green on a record it did not understand.

    Raises:
        NodeInventoryError: On every one of the above.
    """
    if raw is None or not raw.strip():
        msg = (
            f"the {NODE_INVENTORY_LABEL} label is absent or blank. This image "
            "was not stamped with a node inventory, so nothing can be resolved "
            "from it. Rebuild through the sanctioned build path."
        )
        raise NodeInventoryError(msg)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"the {NODE_INVENTORY_LABEL} label is not valid JSON: {exc}"
        raise NodeInventoryError(msg) from exc
    if not isinstance(payload, dict):
        msg = (
            f"the {NODE_INVENTORY_LABEL} label must be a JSON object, got "
            f"{type(payload).__name__}."
        )
        raise NodeInventoryError(msg)
    unknown = sorted(set(payload) - {"schema", "nodes"})
    if unknown:
        msg = f"unknown {NODE_INVENTORY_LABEL} field(s) {unknown}."
        raise NodeInventoryError(msg)
    schema = payload.get("schema")
    if schema != NODE_INVENTORY_SCHEMA:
        msg = (
            f"the {NODE_INVENTORY_LABEL} label declares schema {schema!r}, not "
            f"{NODE_INVENTORY_SCHEMA!r}. Refusing to interpret an inventory "
            "written against a different contract."
        )
        raise NodeInventoryError(msg)
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        msg = f"the {NODE_INVENTORY_LABEL} label's 'nodes' must be a list."
        raise NodeInventoryError(msg)
    if not nodes:
        msg = (
            f"the {NODE_INVENTORY_LABEL} label carries an empty node list, "
            "which asserts the image ships no nodes."
        )
        raise NodeInventoryError(msg)

    entries: list[ModelNodeInventoryEntry] = []
    for row in nodes:
        if not isinstance(row, dict):
            msg = f"a node inventory entry must be an object, got {type(row).__name__}."
            raise NodeInventoryError(msg)
        try:
            entries.append(ModelNodeInventoryEntry.model_validate(row))
        except NodeInventoryError:
            raise
        except Exception as exc:
            msg = f"a node inventory entry is unreadable: {exc}"
            raise NodeInventoryError(msg) from exc

    _refuse_duplicate_names(entries)
    return tuple(sorted(entries, key=lambda e: (e.name, e.node_version)))


# ---------------------------------------------------------------------------
# readback -- docker inspect, no running container and no cluster access
# ---------------------------------------------------------------------------
def read_image_node_inventory(
    image_ref: str,
    *,
    docker_binary: str = "docker",
) -> tuple[ModelNodeInventoryEntry, ...]:
    """Read one image's inventory with ``docker inspect``.

    This is the R5 path: it inspects an image's config, so it needs neither a
    running container nor cluster access. A promotion gate holding only a digest
    can answer "which nodes, at which contract bodies, does this image ship".

    Raises:
        NodeInventoryError: ``docker inspect`` failed, returned something this
            function does not understand, or the image carries no usable label.
    """
    try:
        completed = subprocess.run(
            [
                docker_binary,
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                image_ref,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"docker inspect could not be run for {image_ref}: {exc}"
        raise NodeInventoryError(msg, image_ref=image_ref) from exc
    if completed.returncode != 0:
        msg = (
            f"docker inspect {image_ref} exited {completed.returncode}: "
            f"{completed.stderr.strip() or '<no stderr>'}"
        )
        raise NodeInventoryError(msg, image_ref=image_ref)
    try:
        labels = json.loads(completed.stdout.strip() or "null")
    except json.JSONDecodeError as exc:
        msg = f"docker inspect {image_ref} printed unparseable label JSON: {exc}"
        raise NodeInventoryError(msg, image_ref=image_ref) from exc
    if not isinstance(labels, dict):
        msg = (
            f"image {image_ref} reports no label map "
            f"(got {type(labels).__name__}), so it carries no inventory."
        )
        raise NodeInventoryError(msg, image_ref=image_ref)
    try:
        return parse_node_inventory_label(labels.get(NODE_INVENTORY_LABEL))
    except NodeInventoryError as exc:
        raise NodeInventoryError(str(exc), image_ref=image_ref) from exc


def resolve_entry_to_contract_path(
    entry: ModelNodeInventoryEntry,
    search_roots: Sequence[Path],
) -> Path:
    """Resolve one inventory entry back to the contract file it was built from.

    This is the hop the ticket exists to close, walked in the reverse
    direction: an image label names a content hash, and this finds the
    ``contract.yaml`` in a checkout whose canonical bytes hash to it. The match
    is on the HASH, never on the name -- a contract that was renamed still
    resolves, and a contract whose body drifted does not, which is the whole
    point of hashing the content.

    Args:
        entry: The inventory entry to resolve.
        search_roots: Directories to walk for ``contract.yaml`` files.

    Returns:
        The path whose content hash equals the entry's.

    Raises:
        NodeInventoryError: No contract under any root hashes to that value, so
            the image ships a contract body this checkout does not contain.
    """
    seen = 0
    for root in search_roots:
        for candidate in sorted(Path(root).rglob("contract.yaml")):
            seen += 1
            try:
                if contract_content_hash(candidate) == entry.contract_content_hash:
                    return candidate
            except Exception:  # noqa: BLE001 - an unreadable candidate is not a match
                continue
    msg = (
        f"no contract under {[str(r) for r in search_roots]} hashes to "
        f"{entry.contract_content_hash} for node {entry.name!r} "
        f"({seen} contract file(s) examined). The image ships a contract body "
        "this checkout does not contain -- which is a real finding, not a "
        "lookup failure: the running image and this tree disagree."
    )
    raise NodeInventoryError(msg, contract_name=entry.name)


# ---------------------------------------------------------------------------
# CLI -- the two halves the build calls, and the two a resolver calls
# ---------------------------------------------------------------------------
def _discovered_inventory() -> tuple[ModelNodeInventoryEntry, ...]:
    """Run the real discovery pass and build the inventory from it.

    Imported inside the function because ``discover_contracts`` walks entry
    points, which is work no importer of this module should pay for unless it
    asked for it.
    """
    from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

    return build_node_inventory(discover_contracts())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omnibase_infra.runtime.node_inventory",
        description=(
            "Emit, verify, read back and resolve the node inventory an image "
            "carries (OMN-18708)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "emit",
        help=(
            "Run this process's discovery pass and print the label value. Run "
            "INSIDE the built image; the build script stamps what it prints."
        ),
    )

    verify = sub.add_parser(
        "verify",
        help=(
            "Re-derive the inventory from this process's discovery pass and "
            "refuse unless it is byte-identical to --expect. This is the "
            "Dockerfile guard: it runs inside the image being described."
        ),
    )
    verify.add_argument(
        "--expect",
        required=True,
        help="The stamped label value, as passed to the build.",
    )

    inspect_image = sub.add_parser(
        "inspect-image",
        help="Read one image's inventory with docker inspect. No cluster access.",
    )
    inspect_image.add_argument("image", help="Image reference or digest.")
    inspect_image.add_argument(
        "--json",
        action="store_true",
        help="Print the parsed inventory as JSON instead of a table.",
    )

    resolve = sub.add_parser(
        "resolve",
        help=(
            "Read an image's inventory and resolve every entry back to the "
            "contract file in a checkout whose content hashes to it."
        ),
    )
    resolve.add_argument("image", help="Image reference or digest.")
    resolve.add_argument(
        "--root",
        action="append",
        required=True,
        type=Path,
        help="A directory to search for contract.yaml files. Repeatable.",
    )
    resolve.add_argument(
        "--node",
        action="append",
        default=None,
        help="Resolve only these node names. Repeatable; default is all.",
    )
    return parser


def _print_table(entries: Sequence[ModelNodeInventoryEntry], out: TextIO) -> None:
    width = max(len(e.name) for e in entries)
    for entry in entries:
        print(
            f"{entry.name:<{width}}  {entry.node_version:<10}  "
            f"{entry.contract_content_hash}",
            file=out,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = sys.stdout
    err = sys.stderr

    try:
        if args.command == "emit":
            print(render_node_inventory_label(_discovered_inventory()), file=out)
            return 0

        if args.command == "verify":
            actual = render_node_inventory_label(_discovered_inventory())
            expected = args.expect.strip()
            if actual != expected:
                # Name the difference as node sets, not as two long strings: a
                # build log showing two 40KB values tells nobody what changed.
                actual_names = {e.name for e in parse_node_inventory_label(actual)}
                try:
                    expected_names = {
                        e.name for e in parse_node_inventory_label(expected)
                    }
                except NodeInventoryError as exc:
                    print(
                        f"NODE_INVENTORY does not parse, so it cannot describe "
                        f"this image: {exc}",
                        file=err,
                    )
                    return 1
                print(
                    "NODE_INVENTORY disagrees with this image's own discovery "
                    f"pass. Stamped but absent from the image: "
                    f"{sorted(expected_names - actual_names)}. In the image but "
                    f"not stamped: {sorted(actual_names - expected_names)}. "
                    "Same node set with a different value means a contract "
                    "body or version moved.",
                    file=err,
                )
                return 1
            print(
                f"NODE_INVENTORY verified against this image's discovery pass: "
                f"{len(parse_node_inventory_label(actual))} node(s).",
                file=out,
            )
            return 0

        if args.command == "inspect-image":
            entries = read_image_node_inventory(args.image)
            if args.json:
                print(
                    json.dumps([e.to_dict() for e in entries], indent=2),
                    file=out,
                )
            else:
                _print_table(entries, out)
            return 0

        if args.command == "resolve":
            entries = read_image_node_inventory(args.image)
            wanted = set(args.node) if args.node else None
            selected = [e for e in entries if wanted is None or e.name in wanted]
            if wanted:
                missing = sorted(wanted - {e.name for e in selected})
                if missing:
                    print(
                        f"image {args.image} carries no inventory entry for {missing}.",
                        file=err,
                    )
                    return 1
            for entry in selected:
                path = resolve_entry_to_contract_path(entry, args.root)
                print(f"{entry.name}\t{entry.node_version}\t{path}", file=out)
            return 0
    except NodeInventoryError as exc:
        print(f"node inventory: {exc}", file=err)
        return 1

    print(f"unknown command {args.command!r}", file=err)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

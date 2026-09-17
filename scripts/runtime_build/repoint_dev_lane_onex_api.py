#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Advance the compose dev lane's ``ONEX_API_IMAGE`` pin to a lab-built onex-api
image, as a scripted, validated, reversible write (OMN-18113).

What this closes, and what it does not.

``onex-api`` on the ``.201`` compose dev lane (project ``omnibase-infra``) is
TAG-REFERENCED, not lane-built: ``docker/docker-compose.dev-lane.yml`` renders
``image: ${ONEX_API_IMAGE:?...}`` and the value is resolved from the operator env
file on the lab host. Nothing in either repository ever wrote that key, so the
lane kept whatever tag a human last typed there -- measured 2026-09-17, an
unlabelled hand build from 2026-09-08, eight days and four ``docker/onex-api``
commits behind ``omninode_infra`` ``origin/dev``. A governed refresh RECREATED
the container faithfully every time and recreated it on the same stale image,
which is the quiet half of the defect: the lane looks refreshed.

THE IMAGE HALF OF THIS WAS NEVER MISSING. ``deploy_agent.lab_overlay._derive_pins``
already builds ``onex-lab/omnicloud-core:<omninode_infra sha8>-<stamp>`` from
``docker/onex-api/Dockerfile`` -- the same Dockerfile and the same context
``build-and-push-onex-api.yml`` uses -- on EVERY lab-overlay apply, and the build
is a plain ``docker build`` on the lab host's own daemon, so the result is
resident for compose to use with no registry and no pull. Those images were being
built for the k3s ``onex-lab`` lane's four pins and the compose lane was never
pointed at them. This script is the missing delivery step and nothing more: it
does not build, it does not recreate, and it never invents a tag.

Why it resolves from the daemon and not from the lab-overlay record.

``<state_dir>/lab-overlay/*.json`` is the lane-state glob a reader would reach
for first, and it is the wrong source here for a stated reason:
``ModelLabOverlayRecord`` carries ``sha`` = the merged **omnibase_infra** sha, and
the api image reference appears only inside the free-text ``evidence`` of the
``images_pinned`` check. Recovering a pin by substring-parsing an evidence string
that is explicitly truncated at 240 characters would be a pin resolved from prose.
The tag itself is the structured fact -- ``<omninode_infra sha8>-<stamp>`` is
written by ``_derive_pins`` from ``manifest_sha`` -- so the daemon's own tag list
is read instead, and the sha in the tag is then verified against a real
``omninode_infra`` commit before anything is written.

RESIDENCY IS CHECKED, NOT ASSUMED. ``lab_overlay._derive_pins`` records the
measured fact that an image resident at one moment is not guaranteed resident at
the next -- image GC runs on this host. A pin naming a collected image is a lane
that cannot start, so the chosen reference is inspected immediately before the
write and a missing one is a named refusal rather than a write.

NO DEFAULTS (rule 8). ``--env-file`` and ``--omninode-clone`` are both required.
A default env-file path would let this write a file the caller did not name, and
a default clone path would let provenance be verified against the wrong tree.
Every ambiguous condition below refuses and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

#: The repository the lab-overlay applier publishes the onex-api image under.
#: Kept equal to ``deploy_agent.lab_overlay.API_IMAGE_NAME`` by
#: ``test_repoint_dev_lane_onex_api_omn18113`` rather than by comment, because a
#: rename there that was not mirrored here would make this script resolve zero
#: candidates and read as "the applier has not run" forever.
API_IMAGE_NAME = "onex-lab/omnicloud-core"

#: The env key the dev-lane compose file renders ``onex-api``'s image from.
PIN_KEY = "ONEX_API_IMAGE"

#: ``_derive_pins`` builds the api tag as ``f"{manifest_sha[:8]}-{stamp}"`` where
#: ``stamp`` is a ``%Y%m%dT%H%M%SZ`` UTC timestamp. Anchored on both ends: a tag
#: that merely CONTAINS this shape is not one this script will pin, because a
#: hand build is free to name itself anything and the whole point is to stop
#: pinning hand builds.
TAG_RE = re.compile(r"^(?P<sha8>[0-9a-f]{8})-(?P<stamp>\d{8}T\d{6}Z)$")

#: OCI label the build may carry naming the omninode_infra commit. Absent on
#: every image built before OMN-18113 -- so its absence is NOT a refusal, and its
#: DISAGREEMENT with the tag is. Provenance that contradicts itself is worse than
#: provenance that is missing, because a reader believes it.
REVISION_LABEL = "org.opencontainers.image.revision"

#: Exit codes. Distinct so a caller can tell "nothing to do" from "refused" --
#: ``refresh_dev_lane.sh`` records a refusal in its receipt and keeps refreshing,
#: which it could not do if a refusal were indistinguishable from a crash.
#: ``EXIT_USAGE`` is argparse's own code for a missing required argument; it is
#: named and pinned by a test so a future custom ``parser.error`` that changed it
#: is a red test rather than a caller silently misreading a refusal as usage.
EXIT_OK = 0
EXIT_REFUSED = 3
EXIT_USAGE = 2


class RepointRefusalError(Exception):
    """A condition under which nothing is written and the reason is named."""


@dataclass(frozen=True)
class ModelCandidate:
    """One resident lab-built onex-api image and the lineage its tag asserts."""

    reference: str
    sha8: str
    stamp: str

    @property
    def sort_key(self) -> str:
        # The stamp is fixed-width UTC ISO-basic, so string order is time order.
        # Sorting by stamp and not by sha means "newest build wins" across shas,
        # which is what advancing the lane means when omninode_infra has moved
        # more than once since the last repoint.
        return self.stamp


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def list_candidates(*, docker: str) -> tuple[list[ModelCandidate], int]:
    """Every resident image under ``API_IMAGE_NAME`` whose tag is applier-shaped.

    Returns the candidates AND the total tag count seen, because a zero-candidate
    result has two very different causes -- the applier has never run, or the
    daemon could not be read -- and rule 16 forbids reporting the first without a
    positive control that distinguishes it from the second.
    """
    completed = _run(
        [
            docker,
            "images",
            "--format",
            "{{.Repository}}:{{.Tag}}",
            API_IMAGE_NAME,
        ]
    )
    if completed.returncode != 0:
        msg = (
            f"could not list {API_IMAGE_NAME} images: `{docker} images` exited "
            f"{completed.returncode}: {completed.stderr.strip() or '<no stderr>'}. "
            "This is an unreadable daemon, NOT an absence of images; nothing was "
            "written."
        )
        raise RepointRefusalError(msg)

    seen = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    candidates: list[ModelCandidate] = []
    for reference in seen:
        _, _, tag = reference.partition(":")
        match = TAG_RE.match(tag)
        if match is None:
            continue
        candidates.append(
            ModelCandidate(
                reference=reference,
                sha8=match.group("sha8"),
                stamp=match.group("stamp"),
            )
        )
    return candidates, len(seen)


def choose(
    candidates: list[ModelCandidate],
    *,
    total_seen: int,
    sha: str | None,
) -> ModelCandidate:
    """The newest applier-built image, optionally restricted to one lineage."""
    pool = candidates
    if sha is not None:
        pool = [c for c in candidates if c.sha8 == sha[:8]]
    if not pool:
        scope = f" for omninode_infra sha {sha[:8]}" if sha is not None else ""
        msg = (
            f"positive control: the daemon lists {total_seen} {API_IMAGE_NAME} "
            f"tag(s), of which {len(candidates)} are applier-built, so this is an "
            f"absence and not an unreadable daemon; none of them{scope} can be "
            "pinned. The lab-overlay applier has not produced an image this lane "
            "can advance to; nothing was written."
        )
        raise RepointRefusalError(msg)
    return max(pool, key=lambda c: c.sort_key)


def inspect_resident(reference: str, *, docker: str) -> tuple[str, dict[str, str]]:
    """Image id and labels for a reference that must be resident RIGHT NOW.

    Between the listing above and the write below is the window in which image GC
    can collect the very tag being pinned. A pin naming a collected image renders
    a compose service that cannot start, so residency is re-established here
    rather than inferred from the listing.

    NO ``--format``, DELIBERATELY. The first revision asked for
    ``{{json .Config.Labels}}`` and it failed on the lab host against the image the
    lane was actually running: an image built without labels carries no ``Labels``
    KEY at all, and Go's template engine errors on a missing map key rather than
    rendering null -- ``map has no entry for key "Labels"``. That error arrives on
    a non-zero exit, so a resident image read as collected and the script refused
    to advance a lane it should have advanced. Parsing the full inspect document
    cannot have that failure mode: an absent key is an absent key.
    """
    completed = _run([docker, "image", "inspect", reference])
    if completed.returncode != 0:
        msg = (
            f"{reference} was listed but is not resident now "
            f"(`{docker} image inspect` exited {completed.returncode}): "
            f"{completed.stderr.strip() or '<no stderr>'}. Pinning a collected "
            "image would render a service that cannot start; nothing was written."
        )
        raise RepointRefusalError(msg)
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        msg = (
            f"`{docker} image inspect {reference}` returned output that is not "
            f"JSON ({exc}); nothing was written."
        )
        raise RepointRefusalError(msg) from exc
    if not isinstance(document, list) or not document:
        msg = (
            f"`{docker} image inspect {reference}` returned an empty document; "
            "nothing was written."
        )
        raise RepointRefusalError(msg)
    entry = document[0]
    image_id = str(entry.get("Id", ""))
    raw = (entry.get("Config") or {}).get("Labels")
    labels = {str(k): str(v) for k, v in (raw or {}).items()}
    return image_id, labels


def verify_provenance(
    candidate: ModelCandidate,
    *,
    labels: dict[str, str],
    clone: Path,
    git: str,
) -> dict[str, object]:
    """The tag's sha must name a real ``omninode_infra`` commit, and any label
    that states a revision must agree with it.

    The tag is the provenance carrier today: ``_derive_pins`` writes the overlay's
    ``manifest_sha`` into it, and no image built before OMN-18113 carries a label
    at all. So an absent label is not a finding. A label that names a DIFFERENT
    commit is, because two disagreeing provenance claims on one artifact mean the
    reader cannot tell which build this is.
    """
    if not (clone / ".git").exists():
        msg = (
            f"--omninode-clone {clone} is not a git clone (no .git). Provenance "
            "cannot be verified against a tree that is not there; nothing was "
            "written."
        )
        raise RepointRefusalError(msg)

    resolved = _run(
        [
            git,
            "-c",
            f"safe.directory={clone}",
            "-C",
            str(clone),
            "rev-parse",
            f"{candidate.sha8}^{{commit}}",
        ]
    )
    if resolved.returncode != 0:
        msg = (
            f"the tag {candidate.reference} claims omninode_infra commit "
            f"{candidate.sha8}, which does not resolve in {clone}: "
            f"{resolved.stderr.strip() or '<no stderr>'}. An image whose asserted "
            "lineage is not a commit in this repository is not a governed build; "
            "nothing was written."
        )
        raise RepointRefusalError(msg)
    full_sha = resolved.stdout.strip()

    label_revision = labels.get(REVISION_LABEL)
    # A label may legitimately carry an abbreviated sha, so the comparison is
    # "is the label a prefix of the resolved commit", not string equality.
    if label_revision and not full_sha.startswith(label_revision):
        msg = (
            f"{candidate.reference} carries {REVISION_LABEL}={label_revision} but "
            f"its tag claims {full_sha}. Two disagreeing provenance claims on one "
            "image; nothing was written."
        )
        raise RepointRefusalError(msg)

    # Informational, never a refusal: whether this build is at or behind the
    # clone's own origin/dev. A clone that has not fetched recently is a stale
    # READER, not a bad image, so this must not gate the write.
    tip = _run(
        [
            git,
            "-c",
            f"safe.directory={clone}",
            "-C",
            str(clone),
            "rev-parse",
            "origin/dev^{commit}",
        ]
    )
    origin_dev = tip.stdout.strip() if tip.returncode == 0 else None
    is_tip = origin_dev == full_sha if origin_dev else None

    return {
        "omninode_infra_sha": full_sha,
        "omninode_infra_origin_dev": origin_dev,
        "is_origin_dev_tip": is_tip,
        "revision_label": label_revision,
    }


def read_pin(env_file: Path) -> tuple[list[str], int, str]:
    """The env file's lines, the index of its single pin line, and its value.

    A missing key and a duplicated key are both refusals. The duplicate matters
    more than it looks: a shell sourcing the file takes the LAST assignment, so
    rewriting the first one produces a file that reads as changed and a lane that
    is not.
    """
    if not env_file.is_file():
        msg = f"--env-file {env_file} does not exist; nothing was written."
        raise RepointRefusalError(msg)
    text = env_file.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(f"{PIN_KEY}=")]
    if not hits:
        msg = (
            f"{env_file} carries no {PIN_KEY}= line. This script advances an "
            "existing pin; it does not decide that this lane should have one. "
            "Nothing was written."
        )
        raise RepointRefusalError(msg)
    if len(hits) > 1:
        lineno = ", ".join(str(i + 1) for i in hits)
        msg = (
            f"{env_file} carries {len(hits)} {PIN_KEY}= lines (lines {lineno}). A "
            "sourcing shell takes the last one, so rewriting any single line "
            "would change the file without changing the lane. Nothing was written."
        )
        raise RepointRefusalError(msg)
    index = hits[0]
    value = lines[index].split("=", 1)[1].strip()
    return lines, index, value


def write_pin(
    env_file: Path,
    *,
    lines: list[str],
    index: int,
    reference: str,
    stamp: str,
) -> Path:
    """Replace exactly one line, keeping a backup, and prove it afterwards.

    The readback is the point of this function. It asserts the file still has the
    same number of lines, still has exactly one pin line, that the pin line now
    holds the requested value, and that exactly ONE line differs from the backup.
    A write that changed a second line is restored from the backup and refused --
    the operator env file on this host carries every other lane's configuration
    and a broad edit here is not recoverable from the receipt.
    """
    backup = env_file.with_suffix(env_file.suffix + f".bak.omn18113-{stamp}")
    shutil.copy2(env_file, backup)

    trailing = "\n" if lines[index].endswith("\n") else ""
    updated = list(lines)
    updated[index] = f"{PIN_KEY}={reference}{trailing}"
    env_file.write_text("".join(updated), encoding="utf-8")

    after = env_file.read_text(encoding="utf-8").splitlines(keepends=True)
    before = backup.read_text(encoding="utf-8").splitlines(keepends=True)
    differing = [
        i
        for i in range(max(len(before), len(after)))
        if before[i : i + 1] != after[i : i + 1]
    ]
    pins_after = [line for line in after if line.startswith(f"{PIN_KEY}=")]
    ok = (
        len(after) == len(before)
        and len(differing) == 1
        and len(pins_after) == 1
        and pins_after[0].split("=", 1)[1].strip() == reference
    )
    if not ok:
        env_file.write_text("".join(before), encoding="utf-8")
        msg = (
            f"readback refused the write to {env_file}: "
            f"{len(before)} line(s) before, {len(after)} after, "
            f"{len(differing)} line(s) differ, {len(pins_after)} {PIN_KEY}= line(s) "
            f"after. The file was restored from {backup}."
        )
        raise RepointRefusalError(msg)
    return backup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repoint_dev_lane_onex_api.py",
        description=(
            "Advance the compose dev lane's ONEX_API_IMAGE pin to the newest "
            "lab-overlay-built onex-api image resident on this host."
        ),
    )
    # Both required and neither defaulted -- see the module docstring, rule 8.
    parser.add_argument(
        "--env-file",
        required=True,
        type=Path,
        help="operator env file holding ONEX_API_IMAGE. No default, deliberately.",
    )
    parser.add_argument(
        "--omninode-clone",
        required=True,
        type=Path,
        help="omninode_infra clone the image's asserted lineage is verified against. No default.",
    )
    parser.add_argument(
        "--sha",
        default=None,
        help="restrict to images built from this omninode_infra sha; default is the newest, whatever its lineage.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write the pin. Without it nothing is written and the resolution is printed.",
    )
    parser.add_argument("--docker", default="docker", help="docker executable")
    parser.add_argument("--git", default="git", help="git executable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    try:
        candidates, total_seen = list_candidates(docker=args.docker)
        chosen = choose(candidates, total_seen=total_seen, sha=args.sha)
        image_id, labels = inspect_resident(chosen.reference, docker=args.docker)
        provenance = verify_provenance(
            chosen, labels=labels, clone=args.omninode_clone, git=args.git
        )
        lines, index, previous = read_pin(args.env_file)

        changed = previous != chosen.reference
        backup: Path | None = None
        if changed and args.execute:
            backup = write_pin(
                args.env_file,
                lines=lines,
                index=index,
                reference=chosen.reference,
                stamp=stamp,
            )
    except RepointRefusalError as exc:
        print(json.dumps({"result": "REFUSED", "reason": str(exc)}, indent=2))
        return EXIT_REFUSED

    result = {
        "result": "WRITTEN"
        if (changed and args.execute)
        else ("PLANNED" if changed else "UNCHANGED"),
        "env_file": str(args.env_file),
        "key": PIN_KEY,
        "pin_before": previous,
        "pin_after": chosen.reference if (changed and args.execute) else previous,
        "proposed": chosen.reference,
        "tag_advanced": bool(changed and args.execute),
        "image_id": image_id,
        "candidates_considered": len(candidates),
        "tags_seen": total_seen,
        "backup": str(backup) if backup else None,
        **provenance,
    }
    print(json.dumps(result, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

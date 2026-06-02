#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prod runtime image promotion-lineage guard (OMN-12626, R1).

The promotion gate protects the ``main`` *branch*, but nothing has pinned the
prod runtime *image* to a promoted ``main`` digest. The prod 0.36.1 image was
built locally from a source clone 12 commits behind ``origin/main`` with
uncommitted edits, with no ``org.opencontainers.image.revision`` label and no
build-provenance manifest. "Prod == promoted main digest" was not mechanically
guaranteed.

This module is a **pre-build/deploy guard**. It refuses to build or deploy the
prod runtime image unless:

  (a) the build source working tree is CLEAN (no tracked, staged, or untracked
      edits) — :func:`working_tree_clean`,
  (b) HEAD is an ancestor-of/equal-to ``origin/main`` (promoted lineage) — a
      dev-only or dirty tree is rejected — :func:`head_is_promoted`,
  (c) the build bakes ``org.opencontainers.image.revision=<git sha>`` plus a
      build-provenance manifest (the guard hands the build the required
      ``--build-arg`` tokens via :func:`promoted_build_args`; the Dockerfile
      stamps the ``revision`` label and writes ``/app/build-provenance.json``),
      and
  (d) the deployed image carries that revision so "prod == promoted main
      digest" is assertable from a ``docker inspect`` —
      :func:`assert_image_revision_matches`.

The guard is **fail-fast**: a missing ``origin/main`` ref, an unreadable repo,
or a missing/unknown image revision raises :class:`ProdLineageError`. There is
no silent default that would let an unpromoted artifact reach prod.

This module is build/deploy *tooling*: it inspects git and an image's
``docker inspect`` output. It never rebuilds or restarts a live runtime.

Usage::

    # Pre-build/deploy lineage check on a source clone (exits non-zero on fail)
    uv run python scripts/check_prod_promotion_lineage.py --repo /path/to/clone

    # Also assert the deployed/built image carries the promoted revision
    uv run python scripts/check_prod_promotion_lineage.py \\
        --repo /path/to/clone --image omninode-prod-runtime

Exit codes:
    0 — clean, promoted HEAD (and, if --image given, matching revision)
    1 — guard failed (stderr has a structured reason)
    2 — usage / environment error
"""

from __future__ import annotations

import argparse
import enum
import json
import subprocess
import sys
from pathlib import Path

# The OCI annotation the Dockerfile stamps from the GIT_SHA/VCS_REF build args.
REVISION_LABEL = "org.opencontainers.image.revision"
# The label the runtime image carries pointing at the baked provenance manifest.
PROVENANCE_LABEL = "com.omninode.workspace_provenance_manifest"
# Sentinel values that mean "no real revision was baked in".
_REVISION_SENTINELS = frozenset({"", "unknown", "dev", "none", "null"})
# The promoted lineage anchor. Prod may only build from main.
PROMOTED_REF = "origin/main"


class EnumProdLineageFailure(enum.Enum):
    """Structured reasons a prod build/deploy is rejected.

    Carried on :class:`ProdLineageError` so callers (and tests) can assert the
    exact failure mode instead of string-matching messages.
    """

    DIRTY_TREE = "dirty_tree"
    NOT_PROMOTED = "not_promoted"
    NO_PROMOTED_REF = "no_promoted_ref"
    GIT_ERROR = "git_error"
    MISSING_REVISION = "missing_revision"
    REVISION_MISMATCH = "revision_mismatch"
    MISSING_PROVENANCE = "missing_provenance"


class ProdLineageError(RuntimeError):
    """Raised when a prod runtime build/deploy fails the promotion-lineage guard.

    Fails the build/deploy CLOSED. The ``reason`` attribute is an
    :class:`EnumProdLineageFailure` describing the exact failure mode.
    """

    def __init__(self, reason: EnumProdLineageFailure, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in ``repo_dir`` and return the completed process."""
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def working_tree_clean(repo_dir: Path) -> bool:
    """Return ``True`` only when the working tree has zero pending changes.

    ``git status --porcelain`` reports modified-tracked, staged, AND untracked
    files (untracked are the ``??`` rows). Any non-empty output means the build
    source does not byte-match a commit, so the baked SHA would lie about the
    image contents — exactly the R1 failure mode.

    Raises :class:`ProdLineageError` (``GIT_ERROR``) when git cannot read the
    repository; never returns a silent default.
    """
    result = _git(repo_dir, "status", "--porcelain", "--untracked-files=all")
    if result.returncode != 0:
        raise ProdLineageError(
            EnumProdLineageFailure.GIT_ERROR,
            f"git status failed in {repo_dir}: "
            f"{result.stderr.strip() or result.stdout.strip()}",
        )
    return result.stdout.strip() == ""


def head_is_promoted(repo_dir: Path) -> bool:
    """Return ``True`` when HEAD is an ancestor-of/equal-to ``origin/main``.

    Promoted lineage means the exact source commit has reached ``main`` through
    the promotion gate. ``git merge-base --is-ancestor HEAD origin/main`` is
    ``True`` when HEAD == origin/main or HEAD is a strict ancestor; it is
    ``False`` for a dev-only commit that diverged after the last promotion.

    Raises :class:`ProdLineageError` (``NO_PROMOTED_REF``) when ``origin/main``
    cannot be resolved — a prod build with no visible promoted ref is rejected,
    not silently allowed.
    """
    rev = _git(
        repo_dir, "rev-parse", "--verify", "--quiet", f"{PROMOTED_REF}^{{commit}}"
    )
    if rev.returncode != 0 or not rev.stdout.strip():
        raise ProdLineageError(
            EnumProdLineageFailure.NO_PROMOTED_REF,
            f"cannot resolve {PROMOTED_REF} in {repo_dir}; prod builds require a "
            "fetched origin/main to prove promoted lineage. Run "
            "`git fetch origin main` on the build source clone.",
        )
    ancestry = _git(repo_dir, "merge-base", "--is-ancestor", "HEAD", PROMOTED_REF)
    # `merge-base --is-ancestor` exits 0 (ancestor), 1 (not ancestor), or >1 on
    # error. Treat any non-0/1 exit as a hard git error rather than "not
    # promoted" so a broken repo cannot masquerade as a clean rejection.
    if ancestry.returncode == 0:
        return True
    if ancestry.returncode == 1:
        return False
    raise ProdLineageError(
        EnumProdLineageFailure.GIT_ERROR,
        f"git merge-base --is-ancestor failed in {repo_dir}: "
        f"{ancestry.stderr.strip() or ancestry.stdout.strip()}",
    )


def resolve_head_sha(repo_dir: Path) -> str:
    """Return the full 40-char HEAD SHA, or raise on a git error."""
    result = _git(repo_dir, "rev-parse", "HEAD")
    sha = result.stdout.strip()
    if result.returncode != 0 or not sha:
        raise ProdLineageError(
            EnumProdLineageFailure.GIT_ERROR,
            f"git rev-parse HEAD failed in {repo_dir}: "
            f"{result.stderr.strip() or result.stdout.strip()}",
        )
    return sha


def assert_prod_build_promoted(repo_dir: Path) -> str:
    """Assert the prod build source is clean AND promoted; return the HEAD SHA.

    This is the (a)+(b) gate. It raises :class:`ProdLineageError` with the
    precise :class:`EnumProdLineageFailure` reason on any violation:

      - ``DIRTY_TREE``    — uncommitted/untracked edits in the build source,
      - ``NOT_PROMOTED``  — HEAD is dev-only (not an ancestor of origin/main),
      - ``NO_PROMOTED_REF`` — origin/main is not fetched/resolvable.

    On success it returns the promoted HEAD SHA so the caller can bake it via
    :func:`promoted_build_args` and later verify the deployed image with
    :func:`assert_image_revision_matches`.
    """
    if not working_tree_clean(repo_dir):
        raise ProdLineageError(
            EnumProdLineageFailure.DIRTY_TREE,
            f"prod build rejected: working tree in {repo_dir} has uncommitted or "
            "untracked changes; the baked revision would not match the image "
            "contents. Commit or stash, then rebuild from a clean clone.",
        )
    if not head_is_promoted(repo_dir):
        head = resolve_head_sha(repo_dir)
        raise ProdLineageError(
            EnumProdLineageFailure.NOT_PROMOTED,
            f"prod build rejected: HEAD {head} in {repo_dir} is not an "
            f"ancestor-of/equal-to {PROMOTED_REF}; prod may only build a "
            "promoted main commit, never a dev-only or local tree.",
        )
    return resolve_head_sha(repo_dir)


def promoted_build_args(sha: str) -> list[str]:
    """Return the ``--build-arg`` tokens that bake the promoted revision.

    The Dockerfile stamps ``org.opencontainers.image.revision`` from ``GIT_SHA``
    (builder stage) and ``VCS_REF`` (runtime stage), and emits the build
    provenance manifest; ``RUNTIME_SOURCE_HASH`` is the env stamp the runtime
    banner and ``attest-source-hash`` gate verify. Passing all three from a
    single promoted SHA keeps (c) consistent end-to-end.
    """
    return [
        "--build-arg",
        f"GIT_SHA={sha}",
        "--build-arg",
        f"VCS_REF={sha}",
        "--build-arg",
        f"RUNTIME_SOURCE_HASH={sha}",
    ]


def assert_image_revision_matches(
    inspect: dict[str, object], *, expected_sha: str
) -> None:
    """Assert a ``docker inspect`` payload carries the promoted revision (c)+(d).

    ``inspect`` is one element of ``docker inspect <image>`` (parsed JSON). The
    image must carry:

      - a non-sentinel ``org.opencontainers.image.revision`` label that matches
        ``expected_sha`` (short-SHA prefix match is accepted), and
      - the ``com.omninode.workspace_provenance_manifest`` label pointing at the
        baked ``build-provenance.json`` manifest.

    Raises :class:`ProdLineageError` (``MISSING_REVISION`` / ``REVISION_MISMATCH``
    / ``MISSING_PROVENANCE``) on any violation so a deployed image that does not
    pin to the promoted digest fails closed.
    """
    config = inspect.get("Config")
    labels: dict[str, object] = {}
    if isinstance(config, dict):
        raw_labels = config.get("Labels")
        if isinstance(raw_labels, dict):
            labels = raw_labels

    revision = str(labels.get(REVISION_LABEL, "")).strip()
    if revision.lower() in _REVISION_SENTINELS:
        raise ProdLineageError(
            EnumProdLineageFailure.MISSING_REVISION,
            f"image is missing a real {REVISION_LABEL} label "
            f"(found {revision!r}); cannot prove prod == promoted main digest.",
        )

    expected = expected_sha.strip().lower()
    found = revision.lower()
    if not (expected.startswith(found) or found.startswith(expected)):
        raise ProdLineageError(
            EnumProdLineageFailure.REVISION_MISMATCH,
            f"image {REVISION_LABEL}={revision!r} does not match the promoted "
            f"HEAD {expected_sha!r}; prod is serving an artifact that is not the "
            "promoted main digest.",
        )

    provenance = str(labels.get(PROVENANCE_LABEL, "")).strip()
    if not provenance:
        raise ProdLineageError(
            EnumProdLineageFailure.MISSING_PROVENANCE,
            f"image is missing the {PROVENANCE_LABEL} label; the build did not "
            "bake a build-provenance manifest.",
        )


def inspect_image(image: str) -> dict[str, object]:
    """Return the first ``docker inspect`` element for ``image``.

    Raises :class:`ProdLineageError` (``GIT_ERROR``) when docker cannot inspect
    the image — there is no fallback that would let an unverifiable image pass.
    """
    result = subprocess.run(
        ["docker", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProdLineageError(
            EnumProdLineageFailure.GIT_ERROR,
            f"docker inspect {image} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}",
        )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or not payload:
        raise ProdLineageError(
            EnumProdLineageFailure.GIT_ERROR,
            f"docker inspect {image} returned no objects.",
        )
    first = payload[0]
    if not isinstance(first, dict):
        raise ProdLineageError(
            EnumProdLineageFailure.GIT_ERROR,
            f"docker inspect {image} returned a non-object element.",
        )
    return first


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_prod_promotion_lineage.py",
        description=(
            "Refuse to build/deploy the prod runtime image unless the source "
            "tree is clean and HEAD is a promoted origin/main commit (OMN-12626)."
        ),
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the build source git clone (default: current directory).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Optional image ref/name to assert carries the promoted revision "
            "label + provenance manifest (e.g. omninode-prod-runtime)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 pass, 1 fail, 2 usage)."""
    args = _parse_args(argv)
    repo_dir = Path(args.repo).resolve()
    if not (repo_dir / ".git").exists():
        print(
            f"[prod-lineage] ERROR: {repo_dir} is not a git repository "
            "(no .git). Point --repo at the build source clone.",
            file=sys.stderr,
        )
        return 2

    try:
        sha = assert_prod_build_promoted(repo_dir)
        print(
            f"[prod-lineage] OK: build source clean and promoted; "
            f"HEAD {sha} is an ancestor-of/equal-to {PROMOTED_REF}."
        )
        if args.image is not None:
            inspect = inspect_image(args.image)
            assert_image_revision_matches(inspect, expected_sha=sha)
            print(
                f"[prod-lineage] OK: image {args.image} carries "
                f"{REVISION_LABEL}={sha} + a build-provenance manifest."
            )
    except ProdLineageError as exc:
        print(f"[prod-lineage] FAIL ({exc.reason.value}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

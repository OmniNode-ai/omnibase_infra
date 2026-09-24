#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ONE runtime-change path classifier for this repository (OMN-18664).

WHY THIS MODULE EXISTS SEPARATELY
---------------------------------
Two callers now ask the same question -- "does this set of changed files change
what the dev lane RUNS?" -- and they must never be able to disagree about the
answer.

1. ``scripts/trigger_rebuild_on_merge.py`` asks it about a merged pull request,
   to decide whether to publish a redeploy start command. A ``true`` there is
   what eventually produces a ``lab-pass-receipt-compose-dev-<sha>`` artifact.
2. ``scripts/ci/release_train.py`` asks it about each commit between a receipted
   commit and the default branch's head, to decide whether that receipt still
   describes the runtime HEAD carries. A ``false`` there is what lets the train
   inherit an ancestor's receipt.

The two answers are the same fact read from opposite ends, so a second path list
would mean the train could inherit a receipt across a commit the trigger
considered runtime-affecting -- a cut on a lane proof that does not describe the
code being cut. There is exactly one list, and it lives here.

The same holds for the whole predicate, not just the list (OMN-19318): the
trigger also fires on the merged pull request's ``runtime_change`` label, and
until OMN-19318 the train did not read it, so a label-only merge was walked
past. :func:`is_runtime_affecting` is the one union of both signals, and both
callers load this module under ONE ``sys.modules`` name
(``_omnibase_infra_runtime_change_classifier``) so they hold the same function
object rather than two copies of it.

WHY IT IS STDLIB-ONLY, DELIBERATELY
-----------------------------------
``trigger_rebuild_on_merge.py`` imports click and pydantic at module scope and
runs under ``uv run`` with the project installed. ``release_train.py`` is
stdlib-plus-pyyaml by design and runs on bare ``python3`` before any project
install. A train that had to import the trigger module to reach the classifier
would inherit that dependency and stop running before it decided anything. So
the shared surface is carved out into a module both can import, and the trigger
re-exports every name from here so its own callers and tests are unchanged.

The canonical deploy-path classifier itself is still omniclaude's deploy-gate
validator, loaded by path. Nothing about the union changes here; it only moved.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

#: A callable taking the changed-file list and returning the runtime-relevant
#: subset. Structurally the signature of omniclaude's ``find_runtime_paths``.
RuntimePathClassifier = Callable[[list[str]], list[str]]


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


# OMN-18072: lane-STATE paths the canonical deploy-gate classifier does not
# match, and is right not to match. Its question is "does this PR need deploy
# EVIDENCE"; this trigger's question is the wider "does this merge change what
# the lane RUNS". A migration runner, the migration corpus it applies and the
# runtime policy env are all lane state that a rebuild applies and that nothing
# else applies.
#
# MEASURED: omnibase_infra#3352 (0e9106da) changed
# scripts/run-forward-migrations.sh and deleted files under
# docker/migrations/_blocked/. The trigger ran on that merge (run 34330947198)
# and reported "No rebuild trigger: no runtime_change label or runtime path
# changes detected", so the seam has never executed on the dev lane.
#
# This supplements the canonical list HERE rather than widening it in
# omniclaude, because that list is also the required deploy gate on four repos
# and answers a different question. The canonical result is unioned, never
# replaced or narrowed.
LANE_STATE_PATH_PATTERNS: tuple[str, ...] = (
    # The forward-migration runner every compose up executes against the lane DB.
    "scripts/run-forward-migrations.sh",
    # The migration corpus that runner applies, including the _blocked/ holding
    # area -- moving a file out of it is exactly what changes the lane.
    "docker/migrations/**",
    # The compose model itself (already canonical for *.yml; declared here so
    # the trigger does not depend on that overlap staying true) and the lane's
    # runtime policy env, which the runtime reads at start.
    "docker/docker-compose*.yml",
    "docker/docker-compose*.yaml",
    "docker/runtime-policy.env",
    # The deploy agent's own source and launcher (OMN-18200). The process that
    # builds, recreates and verifies the lane is lane state by the same argument
    # the migration runner is. Without this, a fix to the agent cannot reach the
    # agent: no command is published, so it takes no job, and self_update has
    # only PRE_ACCEPT and POST_TERMINAL boundaries -- both job-driven, neither
    # reached at startup. Measured on omnibase_infra#3520 (``ead1f59b``), the
    # fix to the agent's own lab_overlay build: run 34815067432 declined, and
    # the lab host's clone stayed at ``8fd25217``, behind that fix. Restarting
    # the unit does not help, because it re-execs the same stale clone.
    #
    # Deliberately the package and the launcher, not ``scripts/deploy-agent/**``:
    # the agent's own tests change no lane behaviour, and every match here costs
    # a full dev-lane rebuild.
    "scripts/deploy-agent/deploy_agent/**",
    "scripts/deploy-agent/deploy/**",
    # OMN-18572. The dev lane's `onex-api` service, in the OMNINODE_INFRA tree.
    #
    # This one matches a path that does not exist in this repository, and that
    # is deliberate rather than a mistake: the publisher is shared, and for a
    # caller whose --source-repo is omninode_infra these are the paths that
    # decide what the lane runs. The lane resolves `image: ${ONEX_API_IMAGE}`,
    # and the lab-overlay applier builds that image from exactly this directory
    # in the archived overlay tree (`lab_overlay.API_DOCKERFILE`/`API_CONTEXT`).
    #
    # The canonical classifier is right to miss it. Its `RUNTIME_PATH_PATTERNS`
    # carry `docker/Dockerfile*` and `docker/**/*.Dockerfile`, and neither
    # matches `docker/onex-api/Dockerfile` -- three segments against a
    # two-segment pattern, and no `.Dockerfile` suffix -- while nothing at all
    # matches `docker/onex-api/main.py`. That list answers "does this PR need
    # deploy EVIDENCE" for the CLOUD plane, whose onex-api image is built and
    # pinned by a separate workflow entirely.
    #
    # Measured cost of its absence: omninode_infra#1523 merged 2026-09-17
    # 09:00:53Z and the lane was still running that squash's PARENT at 11:15Z,
    # with tenant creation on the lab impossible throughout.
    "docker/onex-api/**",
    # OMN-18671: the dependency manifests and locks the runtime image is BUILT
    # from. A floor bump IS the change, not a description of one.
    #
    # MEASURED: omnimarket#2629 (``d4fb6ddd``, 2026-09-18T08:11:09Z) raised the
    # omnibase-infra floor 0.38.30 -> 0.38.31, which swaps the installed lab
    # binding table (Qwen3.6-35B -> Qwen3.8-27B). Run 35323097013 reported "no
    # runtime_change label or runtime path changes detected", skipped verify and
    # announce, and produced no lab-pass receipt for that sha -- the exact-name
    # query returns 0 against a positive control of 4.
    #
    # The canonical classifier is right to miss these for its own question. Its
    # list is the required deploy gate on four repositories, so carrying these
    # there would demand deploy evidence of every dependency bump in all four;
    # and because this trigger pins omniclaude's ``main``, which advances only
    # on a release fast-forward, a widening there would not reach this trigger
    # until a release cut. This file is this repository's own and takes effect
    # on merge.
    #
    # Root-anchored deliberately, never ``**/pyproject.toml``: omnimarket
    # carries ``experiments/adk_eval/track_a_adk/pyproject.toml``, which no
    # runtime image installs from, and every match here costs a full dev-lane
    # rebuild. The three repositories that publish through this trigger --
    # omnibase_infra, omnimarket, omninode_infra -- all declare their runtime
    # dependencies in a root ``pyproject.toml`` with a root ``uv.lock``.
    # omninode_infra's ``docker/onex-api/requirements*.txt`` are already covered
    # by the ``docker/onex-api/**`` entry above.
    # Deliberately NOT ``scripts/deploy-agent/pyproject.toml`` or its lock.
    # OMN-18200 weighed the deploy agent's subtree and admitted two named
    # directories rather than opening ``scripts/deploy-agent/**``, pinning the
    # exclusion in tests/scripts/test_trigger_deploy_agent_source_omn18200.py.
    # The agent is a systemd unit on the lab host, not a layer of the lane
    # image, so its manifests are a different argument from the three
    # repository roots above. Overturning that decision belongs to a ticket
    # that makes it, not to this one -- stated as a residual rather than taken.
    "pyproject.toml",
    "uv.lock",
    # OMN-19383: omnimarket's shared events package. Measured on
    # omnimarket#2813's change to runtime_deployment.py (run 35961924174),
    # which read "No rebuild trigger" although that module is imported by
    # all ten canonical redeploy-node handlers -- orchestrator,
    # deploy-publish-monitor effect, FSM reducer, prod-promotion-gate
    # compute, grant-resolver effect, health-fact-resolver effect. The gap
    # is the directory, not that one file: every other module under
    # src/omnimarket/events/ has the same handler fan-in by construction
    # (it exists so one node does not import another node's private
    # package -- see runtime_deployment.py's own module docstring), and a
    # spot-measured import count against src/omnimarket/nodes/*/handlers/*.py
    # at omnimarket dev HEAD (2026-09-24) found at least one handler
    # importer for every file in the directory (__init__.py 252,
    # topics.py 160, delegation.py 80, verification.py 51, generation.py
    # 52, github.py 49, ledger.py 31, runtime_deployment.py 10, and 30
    # more). The canonical deploy-gate classifier is right to miss it for
    # its own question (deploy EVIDENCE, not lane state), which is why
    # this goes in the supplement rather than widening that list -- same
    # reasoning as the pyproject.toml/uv.lock entries above (OMN-18671).
    "src/omnimarket/events/**",
    # OMN-19378: the rest of the packaged source trees the lane image installs
    # WHOLE. The entry above covers events/; the canonical list names a fixed
    # set of subtrees (``nodes/``, ``runtime/``, ``handlers/``, ``services/``
    # ...) and so also misses every ``models/``, ``enums/``, ``adapters/``,
    # ``delegation/`` and ``configs/`` file: 939 of 4393 files under
    # src/omnimarket and 1162 of 3322 under src/omnibase_infra (``models/``,
    # ``event_bus/``, ``errors/`` ...), replayed over both dev branches on
    # 2026-09-24.
    #
    # The lane does not select modules. It installs omnimarket from the staged
    # clone (``omnimarket @ file:///workspace/sibling-repos/omnimarket``, read
    # back from the running container's ``direct_url.json``) and omnibase_infra
    # as the image's own project, so every file under either tree is in the
    # image. Which of them a given process imports is an import-graph question
    # this trigger cannot answer cheaply, and guessing wrong is the direction
    # that ships an unrebuilt lane. Replayed over the last 60 merges to dev this
    # adds 9 omnimarket rebuilds and 1 omnibase_infra rebuild.
    "src/omnimarket/**",
    "src/omnibase_infra/**",
)

#: What a matched path is attributed to when no pattern in this module claims
#: it -- i.e. the canonical deploy-gate classifier matched it. Re-deriving WHICH
#: canonical pattern matched would mean a second implementation of that
#: repository's matcher here, which is the divergence this module exists to
#: prevent, so the SOURCE is named and the pattern is not guessed.
CANONICAL_CLASSIFIER_SOURCE = "canonical-deploy-gate"


def _matches_pattern(path: str, pattern: str) -> bool:
    """Segment-wise, repo-root-anchored glob match.

    ``PurePosixPath.match`` is right-anchored (it matches a SUFFIX of the path)
    and does not treat ``**`` as recursive, so it both over-matches a nested
    ``a/b/docker/runtime-policy.env`` and under-matches
    ``docker/migrations/_blocked/README.md``. Both are wrong for a path list
    that decides whether a lane gets rebuilt, so the match is spelled out:
    a trailing ``/**`` matches everything beneath that directory, and every
    other pattern matches segment for segment from the repo root.
    """
    if pattern.endswith("/**"):
        return path.startswith(pattern[: -len("**")])
    pattern_parts = pattern.split("/")
    path_parts = path.split("/")
    if len(pattern_parts) != len(path_parts):
        return False
    return all(
        fnmatch.fnmatchcase(actual, expected)
        for actual, expected in zip(path_parts, pattern_parts, strict=True)
    )


def find_lane_state_paths(changed_files: list[str]) -> list[str]:
    """Return the changed files that are lane STATE this trigger must rebuild for.

    Order-preserving and de-duplicated, so the union below reads as the
    canonical hits followed by the supplementary ones.
    """
    hits: list[str] = []
    for path in changed_files:
        if not isinstance(path, str) or not path.strip():
            continue
        candidate = path.strip()
        if (
            any(_matches_pattern(candidate, p) for p in LANE_STATE_PATH_PATTERNS)
            and candidate not in hits
        ):
            hits.append(candidate)
    return hits


def attribute_runtime_paths(runtime_paths: list[str]) -> list[tuple[str, str]]:
    """Pair each matched path with the pattern that claimed it (OMN-18671).

    The decision line used to print the matched FILES only, which says that a
    rebuild was triggered but not on what grounds. When a merge is classified
    runtime-affecting for a reason nobody expected -- or, as in omnimarket#2629,
    when it is NOT and should have been -- the pattern is the fact that settles
    it, and reading it out of the run log beats re-deriving it by hand.

    A path no pattern here claims came from the canonical deploy-gate classifier
    and is attributed to ``CANONICAL_CLASSIFIER_SOURCE``. The first matching
    pattern wins, so the attribution is stable under list order rather than
    under dict iteration.
    """
    attributed: list[tuple[str, str]] = []
    for path in runtime_paths:
        pattern = next(
            (p for p in LANE_STATE_PATH_PATTERNS if _matches_pattern(path, p)),
            CANONICAL_CLASSIFIER_SOURCE,
        )
        attributed.append((path, pattern))
    return attributed


def format_runtime_path_attribution(attributed: list[tuple[str, str]]) -> str:
    """Render the attribution for one log line: ``path <- pattern`` per hit."""
    return ", ".join(f"{path} <- {pattern}" for path, pattern in attributed)


#: The pull request label that marks a merge runtime-affecting whatever its
#: paths say (OMN-19318). Spelled once, here; the trigger and the train read it.
RUNTIME_CHANGE_LABEL = "runtime_change"

#: A label source: the labels themselves, or a zero-argument reader that
#: fetches them. A reader is only called when the paths do not already decide.
LabelSource = Sequence[str] | Callable[[], Sequence[str]]


class LabelReadError(RuntimeError):
    """The merged pull request's labels could not be read.

    Runtime-affecting is then UNDECIDED, and this is the explicit token for
    that. A caller must not read it as "no label": that is the fail-OPEN
    direction, in which the proof-subject walk inherits an older receipt across
    a merge the trigger rebuilt for.
    """


def is_runtime_affecting(runtime_paths: Sequence[str], labels: LabelSource) -> bool:
    """THE runtime-affecting predicate (OMN-19318, plan row D15, PS-1).

    A merge is runtime-affecting when the path rule marks it (``runtime_paths``
    is the output of :func:`classify_runtime_paths`) OR the merged pull request
    that produced it carries :data:`RUNTIME_CHANGE_LABEL`.

    Both callers use this one function object: the rebuild trigger's
    ``should_trigger`` IS this function, and the release train's per-commit
    predicate (which ``resolve_lab_candidate`` walks with) calls it. Before
    OMN-19318 the train read paths only, so a label-only merge was rebuilt and
    verified by the trigger while the walk stepped past it to an older subject.

    ``labels`` may be a reader. It is not called when a runtime path already
    decides the answer. A reader that raises, or returns something other than a
    list of strings, raises :class:`LabelReadError` rather than answering.
    """
    if runtime_paths:
        return True
    if callable(labels):
        try:
            read = labels()
        except LabelReadError:
            raise
        except Exception as exc:
            msg = f"the merged pull request's labels could not be read: {exc}"
            raise LabelReadError(msg) from exc
    else:
        read = labels
    if isinstance(read, str) or not all(isinstance(label, str) for label in read):
        msg = (
            "the merged pull request's labels could not be read: expected a list "
            f"of label names, got {read!r}"
        )
        raise LabelReadError(msg)
    return RUNTIME_CHANGE_LABEL in {label.strip() for label in read}


#: OMN-19375: packages whose OWN ``[project].version`` is proven not to reach
#: what the lane runs, so a merge that changes only that string in the root
#: ``pyproject.toml`` and the package's own ``uv.lock`` entry is not lane state.
#:
#: MEASURED: every omnimarket release is followed by a bot merge that changes
#: exactly those two lines, and each one published a full dev-lane rebuild (jobs
#: ``0805d076``, ``e4d36317``, ``ff4dc5de`` on 2026-09-24; the last recreated
#: ``omninode-runtime`` nine minutes before C15 run 35971574919, which failed).
#: For omnimarket the string is inert: the lane installs it ``--no-deps`` from
#: the staged source tree (``file:///workspace/sibling-repos/omnimarket`` in the
#: running container's ``direct_url.json``), ``omnimarket.__version__`` is a
#: literal ``0.1.0``, and the one runtime reader of the distribution version
#: (``version_handshake.check_plugin_compat``) has no caller.
#:
#: omnibase-infra is deliberately absent: ``service_kernel.KERNEL_VERSION``,
#: ``overlay_config_resolver`` and ``version_compatibility`` read its version at
#: runtime. A package joins this set only with that proof made for it.
VERSION_INERT_PACKAGES: frozenset[str] = frozenset({"omnimarket"})

#: Reads one repository-root file at the merged commit's first parent and at
#: the merged commit, as ``(before, after)``; ``None`` for a side where the file
#: is absent. Raises when the commits themselves cannot be read.
ManifestReader = Callable[[str], tuple[str | None, str | None]]

_MANIFESTS: tuple[str, str] = ("pyproject.toml", "uv.lock")


def _normalize_package_name(name: str) -> str:
    """PEP 503 normalization, the form ``uv.lock`` records names in."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _without_project_version(document: dict[str, object]) -> dict[str, object]:
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml has no [project] table")
    return {**document, "project": {**project, "version": None}}


def _without_own_lock_version(
    document: dict[str, object], package: str
) -> dict[str, object]:
    """The lock with the root package's own ``version`` blanked, and nothing else.

    The root package is the one entry whose name is ``package`` and whose source
    is the project directory itself. Exactly one must exist; anything else is a
    lock this function does not understand, and it raises.
    """
    entries = document.get("package")
    if not isinstance(entries, list):
        raise ValueError("uv.lock has no [[package]] entries")
    own = [
        index
        for index, entry in enumerate(entries)
        if isinstance(entry, dict)
        and _normalize_package_name(str(entry.get("name", ""))) == package
        and entry.get("source") in ({"editable": "."}, {"virtual": "."})
    ]
    if len(own) != 1:
        raise ValueError(f"uv.lock names {len(own)} root entries for {package}")
    blanked = list(entries)
    blanked[own[0]] = {**entries[own[0]], "version": None}
    return {**document, "package": blanked}


def inert_version_bump_paths(
    changed_files: Sequence[str], manifest_reader: ManifestReader
) -> list[str]:
    """The root manifests whose change is only a version-inert package's own version.

    ``pyproject.toml`` qualifies when the parsed document differs only in
    ``[project].version``; ``uv.lock`` when it differs only in the root
    package's own ``version``. Either way the package, named by the merged
    ``pyproject.toml``, must be in :data:`VERSION_INERT_PACKAGES`.

    Fails CLOSED: a reader that raises, a manifest that is absent on either side
    or does not parse, and a lock this module does not understand all return
    nothing exempt, so the path stays lane state exactly as before OMN-19375.
    """
    candidates = [path for path in _MANIFESTS if path in changed_files]
    if not candidates:
        return []
    try:
        pyproject_before, pyproject_after = manifest_reader("pyproject.toml")
        if pyproject_before is None or pyproject_after is None:
            return []
        before = tomllib.loads(pyproject_before)
        after = tomllib.loads(pyproject_after)
        package = _normalize_package_name(str(after["project"]["name"]))
        if package not in VERSION_INERT_PACKAGES:
            return []
        inert: list[str] = []
        if "pyproject.toml" in candidates and _without_project_version(
            before
        ) == _without_project_version(after):
            inert.append("pyproject.toml")
        if "uv.lock" in candidates:
            lock_before, lock_after = manifest_reader("uv.lock")
            if lock_before is not None and lock_after is not None:
                if _without_own_lock_version(
                    tomllib.loads(lock_before), package
                ) == _without_own_lock_version(tomllib.loads(lock_after), package):
                    inert.append("uv.lock")
        return inert
    # A reader failure, a TOML parse error (a ValueError), a missing
    # [project].name or a lock this module does not understand: every one keeps
    # the manifests lane state.
    except (OSError, ValueError, KeyError, TypeError):
        return []


def git_manifest_reader(repo: Path, sha: str) -> ManifestReader:
    """A :data:`ManifestReader` over a clone holding ``sha`` and its first parent.

    A file absent from a commit reads as ``None``; a commit the clone does not
    hold raises ``OSError``, which :func:`inert_version_bump_paths` turns into
    "not exempt".
    """

    def _show(rev: str, path: str) -> str | None:
        shown = subprocess.run(
            ["git", "-C", str(repo), "show", f"{rev}:{path}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if shown.returncode == 0:
            return shown.stdout
        resolves = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{rev}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if resolves.returncode != 0:
            msg = f"{repo} does not hold commit {rev}: {shown.stderr.strip()}"
            raise OSError(msg)
        return None

    def _read(path: str) -> tuple[str | None, str | None]:
        return _show(f"{sha}^1", path), _show(sha, path)

    return _read


def classify_runtime_paths(
    changed_files: list[str],
    classifier: RuntimePathClassifier,
    manifest_reader: ManifestReader | None = None,
) -> list[str]:
    """Run and validate the canonical classifier's output fail-closed.

    OMN-18072: the canonical result is then UNIONED with the lane-state
    supplement above. The validation stays ahead of the union so a broken
    canonical classifier still fails closed rather than being papered over by a
    supplementary hit.

    OMN-19375: with a ``manifest_reader``, a root manifest whose change is only
    a version-inert package's own version is left out of the supplement's half.
    The canonical half is never narrowed. Without a reader nothing is exempt.
    """
    runtime_paths = classifier(changed_files)
    if not isinstance(runtime_paths, list) or any(
        not isinstance(path, str) or not path.strip() for path in runtime_paths
    ):
        raise ValueError("runtime path validator returned an invalid path list")
    inert = (
        set(inert_version_bump_paths(changed_files, manifest_reader))
        if manifest_reader is not None
        else set()
    )
    combined = list(runtime_paths)
    for path in find_lane_state_paths(changed_files):
        if path not in combined and path not in inert:
            combined.append(path)
    return combined

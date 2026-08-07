#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every cross-repo pin must name a commit REACHABLE FROM A PROTECTED BRANCH (OMN-15538).

The confusion this exists to kill: **immutable is not reachable.**
-------------------------------------------------------------------
A feature-branch head SHA is immutable -- nothing will ever change what that
object contains. It is also *unreferenced*: GitHub deletes the branch when the
PR squash-merges, the object becomes reachable from nothing, and every consumer
that pinned it is permanently broken. Shape validation ("it's 40 hex chars, so
it's immutable, so it's safe") is precisely the reasoning that produced two
production incidents in a single day on 2026-07-30:

* **Workflow pin.** ``omnibase_infra@dev .github/workflows/ci.yml`` pinned
  ``OmniNode-ai/omnimarket/.github/workflows/merge-hold-gate-reusable.yml@879d6fc6``
  -- the head of an omnimarket PR branch. The branch was deleted on merge,
  ``ci.yml`` began failing at *parse* time, ``CI Summary`` (infra dev's sole
  required context) never reported, and every open PR in the repo became
  unmergeable for ~2.5h (OMN-15536).
* **Dependency pin.** ``omnimarket@dev pyproject.toml`` pins
  ``omnibase-core`` at ``5a907b71`` -- the live head of the still-open
  ``jonah/omn-15392-evidence-execution-scope`` branch. Five lines of prose
  directly above it say the branch "MUST NOT merge with a feature-branch pin".
  It merged anyway. A prose comment is not a mechanism.

Both passed every gate, because no gate checked reachability.

Why "reachable from a PROTECTED branch", not "reachable at all"
---------------------------------------------------------------
The pre-existing platform check (``omnimarket/scripts/ci/
check_uv_lock_pin_reachability.py``, OMN-14449) asks ``git branch -r --contains
<rev>`` and accepts a non-empty answer -- i.e. *reachable from ANY remote
branch*. ``5a907b71`` IS contained by its own live feature branch, so that
oracle returns OK on it today and only turns RED once the branch is deleted:
it reports the bug **after** the outage rather than before it. The invariant
that discriminates a durable pin from a time bomb is membership in the history
of a branch nobody can delete:

    reachable  <=>  compare(<protected>...<ref>).status in {"behind", "identical"}

``behind``    the pinned ref is an ANCESTOR of the protected branch  -> durable
``identical`` the pinned ref IS the protected branch head            -> durable
``ahead``     the protected branch is an ancestor of the ref         -> NOT durable
              (a descendant commit: an unlanded branch head)
``diverged``  no ancestry either way                                 -> NOT durable

``dev`` OR ``main`` both count. That union is load-bearing, not defensive:
``omnibase_infra`` legitimately pins ``onex_change_control@2dd26ade``, which is
``diverged`` from OCC ``dev`` and ``behind`` OCC ``main``. A dev-only oracle
would false-RED a correct pin, and a gate that cries wolf gets bypassed.

Existence is not the same probe, and it gives the wrong answer
--------------------------------------------------------------
``GET /repos/{o}/{r}/commits/{sha}`` **still serves unreferenced objects**, so
an existence probe PASSES on a broken pin. That is why this module uses the
compare endpoint, and only falls back to the commits endpoint in the failure
path -- to say *which* kind of dead a dead pin is ("exists but unreferenced"
vs "gone entirely"), never to grant a pass.

Fail-closed posture
-------------------
Rate-limit, transport failure, or any non-definitive HTTP status yields
``UNDETERMINED``, which is a FAILURE. ``--allow-undetermined`` downgrades that
to a loud warning for offline local use and is **refused outright when ``CI``
is set**, so the enforcing surface can never be weakened into a silent skip
(the optional-input-means-the-check-does-not-exist trap).

Auth: all OmniNode-ai repos referenced here are public, so the compare endpoint
works unauthenticated (60 req/hr). ``GH_TOKEN``/``GITHUB_TOKEN`` is attached as
a bearer token when present purely for rate-limit headroom (5000 req/hr); in
GitHub Actions the job passes the ambient ``secrets.GITHUB_TOKEN``. Probes are
deduplicated by ``(repo, ref)``, so pinning the same SHA twice (``uses:`` plus
``vocabulary_ref``) costs one request. ``gh`` is deliberately NOT used: the CI
python steps carry no ``gh`` login and ``gh`` hard-refuses unauthenticated use,
which would make the gate permanently red for transport reasons rather than for
broken pins (same finding as OMN-14941).

Exit codes: ``0`` every pin durable | ``1`` at least one pin not durable, or
undetermined | ``2`` misuse (e.g. ``--allow-undetermined`` under CI).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import yaml

_ORG = "OmniNode-ai"
_ORG_PREFIX = f"{_ORG}/"
# The reachability oracle is a property of github.com itself, so there is no
# ONEX routing authority to resolve it from; wiring a CI-only build-hygiene
# gate through runtime service discovery would couple it to infrastructure it
# must be able to run without. Same posture as the PyPI index literal in
# scripts/ci/publish_with_retry.py.
_GITHUB_API = "https://api.github.com"  # url-authority-ok: fixed public REST API, no ONEX routing authority
_DEFAULT_PROTECTED: tuple[str, ...] = ("dev", "main")
# Raised from 10.0 (2026-08-06, defect fix): a single 10s-timeout, zero-retry
# GET was redding this gate on a pin that was verifiably reachable live (the
# onex_change_control @ 2dd26ade... compare call) -- 10s is not a realistic
# GitHub REST compare-endpoint budget under transient load. 30s per attempt.
_REQUEST_TIMEOUT_SECONDS = 30.0
# Bounded retry for transient transport failures (timeout/connection errors,
# HTTP 5xx, HTTP 429, and a rate-limit-signaled 403). A definitive HTTP 4xx is
# never retried -- it will not change on a second try, and this job runs on
# every PR so added latency is never free. Worst case for one fully-exhausted
# call has TWO ceilings depending on backoff class (corrected 2026-08-06,
# terminal adversarial verify round 2 -- the fixed-schedule figure below was
# previously mis-cited as THE worst case; it is only the cheaper of the two):
#   fixed-schedule (non-rate-limit 5xx, no server backoff header):
#     3 * _REQUEST_TIMEOUT_SECONDS + sum(_API_RETRY_BACKOFF_SECONDS) = 96s
#   rate-limited (429 / rate-limited 403 with a server Retry-After header,
#   which is PREFERRED over the fixed schedule and capped at
#   _MAX_RATE_LIMIT_BACKOFF_SECONDS -- see ``_rate_limit_backoff_seconds``):
#     3 * _REQUEST_TIMEOUT_SECONDS + 2 * _MAX_RATE_LIMIT_BACKOFF_SECONDS = 150s
#
# That 150s (the true ceiling, not 96s) bounds ONE call, not the run. The
# circuit breaker below resets its consecutive-failure counter on ANY
# HTTP-status-bearing response -- including a retried-and-still-failing
# 503/429 -- so a run whose transport failures are intermittent rather than
# sustained can pay the full per-call ceiling on every pin without the
# breaker ever tripping (defect found 2026-08-06: measured max 944s / 1174s
# across 21-pin trials with 19-23/30 exceeding the 600s CI job timeout).
# ``_RUN_DEADLINE_SECONDS`` below is the actual run-wide bound; the breaker
# remains a fast-path for the sustained-outage case it was built for, but is
# no longer the thing standing between this gate and the job timeout.
_API_MAX_ATTEMPTS = 3
_API_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (2.0, 4.0)
_TRANSIENT_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
# A server-requested rate-limit backoff (``Retry-After`` or
# ``x-ratelimit-reset``) is honored up to this cap, never unbounded -- an
# hour-long primary-rate-limit reset must not turn one pin into an hour-long
# CI job.
_MAX_RATE_LIMIT_BACKOFF_SECONDS = 30.0
# Stop probing after this many consecutive transport (non-HTTP-status) failures
# and report the remainder as undetermined -- a fast path for a SUSTAINED
# outage. It does not, by itself, bound a run with INTERMITTENT failures; see
# ``_RUN_DEADLINE_SECONDS``.
_TRANSPORT_FAILURE_CIRCUIT_BREAKER = 3
# Run-wide wall-clock budget across every pin resolved by one ``_Resolver``.
# Checked before each per-branch compare call AND before ``_explain``'s own
# commits-endpoint lookup (defect found 2026-08-06: ``_explain`` used to run
# an unguarded second call after the last passing check, doubling the
# post-deadline tail to 192s). With both call sites guarded, the actual
# worst case is this deadline plus at most one already-in-flight call's tail.
#
# Lowered from 480.0 to 220.0 (2026-08-06, terminal adversarial verify round
# 2, PR #2679 comment 5209929069): the prior ``480 + 96 = 576s`` bound used
# the WRONG per-call ceiling and ignored job setup entirely.
#
# * The 96s figure only covered the fixed-schedule backoff class
#   (2.0 + 4.0 = 6s of sleep). It is NOT the worst case: a 429 / rate-limited
#   403 with a server ``Retry-After`` prefers the server-provided delay over
#   the fixed schedule, capped at ``_MAX_RATE_LIMIT_BACKOFF_SECONDS`` (30s)
#   -- see ``_rate_limit_backoff_seconds``. One fully-exhausted call in that
#   class costs ``_API_MAX_ATTEMPTS`` (3) timeouts of
#   ``_REQUEST_TIMEOUT_SECONDS`` (30s) each, PLUS a capped 30s server backoff
#   after each of the first two attempts:
#     3 * 30.0 + 2 * 30.0 = 90 + 60 = 150s   (measured/derived from code, not
#                                              the 96s this comment used to say)
# * 600s is the CI **job**'s ``timeout-minutes`` (.github/workflows/ci.yml),
#   not the script's own budget. Measured on this PR's own prior CI head
#   (job 92716571663): checkout + ``uv`` setup (cache disabled) costs ~196s
#   BEFORE the script starts, leaving the script itself only ~404s of the
#   600s job budget -- not 600s. The old ``480 + 96 = 576s`` "margin" never
#   accounted for that ~196s prefix, so real worst case was
#   ``196 + 480 + 150 = 826s``: the job gets cancelled by GitHub at 600s and
#   the script's own fail-closed UNDETERMINED print never happens -- the
#   exact wedged-red-without-a-message failure mode this file exists to
#   remove.
#
# New bound, real margin:
#     ~200s measured setup + 220s deadline + 150s max post-deadline tail
#       = ~570s < 600s job timeout  (~30s margin)
#
# 220s still comfortably bounds a *sustained* cross-pin failure run (which is
# fail-closed UNDETERMINED anyway, not a false pass) while leaving the script
# room to print its own diagnostic before the job's hard cancellation.
_RUN_DEADLINE_SECONDS = 220.0

_SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")

# `omnibase-core @ git+https://github.com/OmniNode-ai/omnibase_core.git@<ref>`
_PEP508_GIT_RE = re.compile(
    r"git\+https://github\.com/"
    + _ORG
    + r"/(?P<repo>[\w.-]+?)(?:\.git)?@(?P<ref>[^\s#\]]+)"
)
# uv.lock: `git = "https://github.com/OmniNode-ai/<repo>.git?rev=<ref>#<resolved>"`
_UV_LOCK_GIT_RE = re.compile(
    r"https://github\.com/"
    + _ORG
    + r"/(?P<repo>[\w.-]+?)(?:\.git)?\?(?:rev|branch|tag)=(?P<ref>[^\"#&]+)"
    r"(?:#(?P<resolved>[0-9a-f]{40}))?"
)
# `git = "https://github.com/OmniNode-ai/<repo>.git"` in a [tool.uv.sources] table
_SOURCE_URL_RE = re.compile(
    r"\Ahttps://github\.com/" + _ORG + r"/(?P<repo>[\w.-]+?)(?:\.git)?/?\Z"
)


class Verdict(str, Enum):
    """Outcome of one reachability resolution."""

    REACHABLE = "REACHABLE"
    UNREACHABLE = "UNREACHABLE"
    UNDETERMINED = "UNDETERMINED"


class PinRef(NamedTuple):
    """One cross-repo pin found in a tracked file.

    ``source`` is the file it came from, ``locus`` the addressable position
    inside it (so a failure names the exact line to edit), ``kind`` the pin
    form, ``repo`` the target OmniNode-ai repo and ``ref`` the pinned git ref.
    """

    source: str
    locus: str
    kind: str
    repo: str
    ref: str


@dataclass(frozen=True)
class Resolution:
    """A verdict plus the evidence that produced it."""

    verdict: Verdict
    detail: str


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _split_uses(value: str) -> tuple[str, str] | None:
    """Split ``OmniNode-ai/<repo>[/<path>]@<ref>`` into ``(repo, ref)``.

    Returns ``None`` for anything out of scope: local (``./``) pins, docker
    pins, third-party actions, and any value with no ``@`` ref.
    """
    if not value.startswith(_ORG_PREFIX) or "@" not in value:
        return None
    spec, _, ref = value.rpartition("@")
    remainder = spec[len(_ORG_PREFIX) :]
    repo, _, _path = remainder.partition("/")
    if not repo or not ref:
        return None
    return repo, ref


def _is_literal(value: object) -> bool:
    """True for a plain scalar; ``${{ ... }}`` expressions are not resolvable."""
    return isinstance(value, str) and "${{" not in value


def extract_workflow_pins(path: Path) -> list[PinRef]:
    """Extract cross-repo pins from one GitHub Actions workflow file.

    Covers three surfaces:

    * ``jobs.<id>.uses`` -- reusable-workflow pins (the OMN-15536 shape),
    * ``jobs.<id>.steps[i].uses`` -- step-level action pins,
    * ``jobs.<id>.with.<key>`` -- any 40-hex literal input on a job that calls
      a cross-repo reusable workflow, attributed to that same repo.

    That third surface is why this is not just a ``uses:`` checker: the wedging
    infra job pinned the identical dead SHA **twice**, once as ``uses:`` and
    once as ``vocabulary_ref``, and repointing only the first would have left a
    second dead reference behind (OMN-15538 AC-5). Keying on "40-hex literal
    under ``with:``" rather than on the name ``vocabulary_ref`` means a renamed
    or newly-added sibling input is covered without editing this file.

    YAML parsing (not a line regex) means commented-out ``uses:`` lines are
    structurally invisible -- several infra workflows carry documentation
    examples that a regex would extract and fail on.
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return []
    jobs = loaded.get("jobs")
    if not isinstance(jobs, dict):
        return []

    pins: list[PinRef] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue

        job_uses = job.get("uses")
        job_target: str | None = None
        if _is_literal(job_uses):
            assert isinstance(job_uses, str)  # nosec B101 - narrowed by _is_literal
            split = _split_uses(job_uses)
            if split is not None:
                repo, ref = split
                job_target = repo
                pins.append(
                    PinRef(
                        source=str(path),
                        locus=f"jobs.{job_id}.uses",
                        kind="workflow-uses",
                        repo=repo,
                        ref=ref,
                    )
                )

        # Sibling ref inputs on a cross-repo caller (vocabulary_ref and kin).
        job_with = job.get("with")
        if job_target is not None and isinstance(job_with, dict):
            for key, value in job_with.items():
                if _is_literal(value) and _SHA40_RE.match(str(value)):
                    pins.append(
                        PinRef(
                            source=str(path),
                            locus=f"jobs.{job_id}.with.{key}",
                            kind="workflow-with",
                            repo=job_target,
                            ref=str(value),
                        )
                    )

        steps = job.get("steps")
        if isinstance(steps, list):
            for index, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                step_uses = step.get("uses")
                if not _is_literal(step_uses):
                    continue
                assert isinstance(step_uses, str)  # nosec B101 - narrowed above
                split = _split_uses(step_uses)
                if split is None:
                    continue
                repo, ref = split
                pins.append(
                    PinRef(
                        source=str(path),
                        locus=f"jobs.{job_id}.steps[{index}].uses",
                        kind="workflow-step-uses",
                        repo=repo,
                        ref=ref,
                    )
                )
    return pins


def _pins_from_source_table(
    source: Path, name: str, table: dict[str, Any]
) -> list[PinRef]:
    """Pins from one ``[tool.uv.sources.<name>]`` entry.

    A ``git`` URL with no ``rev``/``branch``/``tag`` tracks the target's default
    branch, which is protected by construction and cannot evaporate -- nothing
    to check.
    """
    git_url = table.get("git")
    if not isinstance(git_url, str):
        return []
    match = _SOURCE_URL_RE.match(git_url.strip())
    if match is None:
        return []
    repo = match["repo"]
    pins: list[PinRef] = []
    for key in ("rev", "branch", "tag"):
        ref = table.get(key)
        if isinstance(ref, str) and ref:
            pins.append(
                PinRef(
                    source=str(source),
                    locus=f"tool.uv.sources.{name}.{key}",
                    kind="pyproject-source",
                    repo=repo,
                    ref=ref,
                )
            )
    return pins


def _pep508_pins(source: Path, locus: str, requirements: object) -> list[PinRef]:
    """Pins from a PEP 508 requirement list (``pkg @ git+https://...@<ref>``)."""
    if not isinstance(requirements, list):
        return []
    pins: list[PinRef] = []
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, str):
            continue
        for match in _PEP508_GIT_RE.finditer(requirement):
            pins.append(
                PinRef(
                    source=str(source),
                    locus=f"{locus}[{index}]",
                    kind="pyproject-pep508",
                    repo=match["repo"],
                    ref=match["ref"],
                )
            )
    return pins


def extract_pyproject_pins(path: Path) -> list[PinRef]:
    """Extract cross-repo git pins from a ``pyproject.toml``.

    Two forms, both live on the platform today:

    * ``[tool.uv.sources]`` -- ``{ git = "...", rev|branch|tag = "..." }``.
      This is the omnimarket instance. Note the SHA lives in a *separate key*
      from the URL, which is exactly why the OMN-14449 uv.lock regex
      (``...git?rev=<sha>`` in one string) cannot see it.
    * PEP 508 direct references in ``project.dependencies``,
      ``project.optional-dependencies.*`` and ``dependency-groups.*``.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    pins: list[PinRef] = []

    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    if isinstance(sources, dict):
        for name, entry in sources.items():
            if isinstance(entry, dict):
                pins.extend(_pins_from_source_table(path, name, entry))
            elif isinstance(entry, list):
                # uv permits a list of conditional sources per package.
                for element in entry:
                    if isinstance(element, dict):
                        pins.extend(_pins_from_source_table(path, name, element))

    project = data.get("project", {})
    if isinstance(project, dict):
        pins.extend(
            _pep508_pins(path, "project.dependencies", project.get("dependencies"))
        )
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for extra, requirements in optional.items():
                pins.extend(
                    _pep508_pins(
                        path, f"project.optional-dependencies.{extra}", requirements
                    )
                )

    groups = data.get("dependency-groups")
    if isinstance(groups, dict):
        for group, requirements in groups.items():
            pins.extend(_pep508_pins(path, f"dependency-groups.{group}", requirements))

    return pins


def extract_uv_lock_pins(path: Path) -> list[PinRef]:
    """Extract cross-repo git pins from a ``uv.lock``.

    Prefers the resolved commit in the URL fragment when present -- that is the
    object a build actually fetches -- and falls back to the query-parameter ref.
    """
    text = path.read_text(encoding="utf-8")
    seen: set[tuple[str, str]] = set()
    pins: list[PinRef] = []
    for match in _UV_LOCK_GIT_RE.finditer(text):
        repo = match["repo"]
        ref = match["resolved"] or match["ref"]
        if (repo, ref) in seen:
            continue
        seen.add((repo, ref))
        pins.append(
            PinRef(
                source=str(path),
                locus=f"package source for {repo}",
                kind="uv-lock",
                repo=repo,
                ref=ref,
            )
        )
    return pins


def _expand_targets(paths: Sequence[Path]) -> list[Path]:
    """Expand directories to the pin-bearing files inside them."""
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.yml")))
            expanded.extend(sorted(path.glob("*.yaml")))
            for name in ("pyproject.toml", "uv.lock"):
                candidate = path / name
                if candidate.is_file():
                    expanded.append(candidate)
        elif path.is_file():
            expanded.append(path)
    return expanded


def extract_pins(paths: Iterable[Path]) -> list[PinRef]:
    """Extract every cross-repo pin from the given files/directories."""
    pins: list[PinRef] = []
    for path in _expand_targets(list(paths)):
        name = path.name
        if name == "pyproject.toml":
            pins.extend(extract_pyproject_pins(path))
        elif name == "uv.lock":
            pins.extend(extract_uv_lock_pins(path))
        elif path.suffix in {".yml", ".yaml"}:
            pins.extend(extract_workflow_pins(path))
    return pins


# ---------------------------------------------------------------------------
# Reachability resolution
# ---------------------------------------------------------------------------


def status_is_reachable(status: str) -> bool:
    """Map a GitHub compare ``status`` to durable-reachability.

    ``behind`` (ancestor of the protected branch) and ``identical`` (it IS the
    protected branch head) are durable. ``ahead`` -- the protected branch is an
    ancestor of the pin, i.e. the pin is a descendant commit that has not
    landed -- and ``diverged`` are not.
    """
    return status in {"behind", "identical"}


def _is_transient_http_status(status: int) -> bool:
    """True for an HTTP status that is unconditionally worth retrying: 429
    (rate limit) or any 5xx.

    404, 400, 401, 410, ... are definitive regardless of headers: a second
    identical GET will not produce a different answer, so retrying only adds
    latency to a gate that runs on every PR.

    403 is deliberately NOT in this set -- a plain 403 (bad/missing auth,
    genuinely forbidden) is just as definitive as a 404. But GitHub also uses
    403 to signal BOTH the primary rate limit (``x-ratelimit-remaining: 0``)
    and the secondary rate limit (a ``Retry-After`` header), and those two
    ARE transient. Discriminating requires the response headers, which this
    function -- status-code-only, by design, so it stays a trivial pure
    predicate -- does not receive. See :func:`_is_rate_limited_403`, which
    ``_api_get`` consults separately for the 403 case.
    """
    return status in _TRANSIENT_HTTP_STATUS


def _is_rate_limited_403(headers: Any) -> bool:
    """True when a 403's headers carry GitHub's rate-limit signal.

    Primary rate limit: ``x-ratelimit-remaining: 0``. Secondary rate limit:
    a ``Retry-After`` header. A 403 with neither is a genuine authorization
    failure -- definitive, not transient (see :func:`_is_transient_http_status`).
    """
    if headers is None:
        return False
    if headers.get("x-ratelimit-remaining") == "0":
        return True
    return headers.get("Retry-After") is not None


def _rate_limit_backoff_seconds(headers: Any) -> float | None:
    """Read a server-provided rate-limit backoff off response headers.

    Prefers ``Retry-After`` (seconds); falls back to ``x-ratelimit-reset``
    (unix epoch seconds), converted to a delta from now. Always capped at
    ``_MAX_RATE_LIMIT_BACKOFF_SECONDS`` -- honoring the signal is better than
    a blind fixed schedule, but honoring it UNBOUNDED would let one rate
    limit turn a single pin into an hour-long CI job. Returns ``None`` when
    neither header is present or parseable, so the caller falls back to the
    fixed schedule.
    """
    if headers is None:
        return None
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), _MAX_RATE_LIMIT_BACKOFF_SECONDS))
        except ValueError:
            pass
    reset_at = headers.get("x-ratelimit-reset")
    if reset_at is not None:
        try:
            delta = float(reset_at) - time.time()
        except ValueError:
            return None
        if delta > 0:
            return min(delta, _MAX_RATE_LIMIT_BACKOFF_SECONDS)
    return None


def _api_get(
    url: str, *, sleep: Callable[[float], None] | None = None
) -> tuple[int | None, dict[str, Any] | None, str]:
    """GET a GitHub REST endpoint with bounded retry on transient failures.

    Returns ``(status, body, detail)``. ``status is None`` means the request
    could not be performed at all after exhausting ``_API_MAX_ATTEMPTS``.

    Retries ONLY the transient failure classes: a transport-level failure
    (timeout, connection error -- no HTTP status at all), an HTTP 429/5xx
    (see :func:`_is_transient_http_status`), or a rate-limit-signaled 403
    (see :func:`_is_rate_limited_403`) -- up to ``_API_MAX_ATTEMPTS`` total
    attempts. Backoff prefers the server-provided rate-limit signal
    (:func:`_rate_limit_backoff_seconds`, capped) and otherwise falls back to
    the short fixed schedule (``_API_RETRY_BACKOFF_SECONDS``, never
    unbounded). A definitive HTTP 4xx returns immediately on the first
    attempt -- no retry, no added latency.

    If every attempt fails on a transient class, the final failing result is
    returned unchanged from today's single-shot behavior: the caller's
    existing fail-closed UNDETERMINED mapping is untouched. This function
    eliminates FALSE failures caused by one unlucky transient hiccup; it does
    not, and must not, weaken the fail-closed posture for a genuinely
    unresolvable pin.
    """
    effective_sleep = sleep or time.sleep
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "omnibase-infra-pin-reachability-gate (OMN-15538)",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - fixed https host

    last_status: int | None = None
    last_body: dict[str, Any] | None = None
    last_detail = ""
    for attempt in range(1, _API_MAX_ATTEMPTS + 1):
        server_backoff: float | None = None
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed https host
                request, timeout=_REQUEST_TIMEOUT_SECONDS
            ) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
                body = payload if isinstance(payload, dict) else None
                return response.status, body, f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            detail = f"HTTP {exc.code}"
            try:
                error_body = json.loads(exc.read().decode("utf-8", errors="replace"))
                message = (
                    error_body.get("message", "")
                    if isinstance(error_body, dict)
                    else ""
                )
                if message:
                    detail = f"HTTP {exc.code}: {message}"
            except (ValueError, OSError):
                pass
            rate_limited_403 = exc.code == 403 and _is_rate_limited_403(exc.headers)
            if not _is_transient_http_status(exc.code) and not rate_limited_403:
                return exc.code, None, detail
            last_status, last_body, last_detail = exc.code, None, detail
            if exc.code == 429 or rate_limited_403:
                server_backoff = _rate_limit_backoff_seconds(exc.headers)
        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
            ValueError,
            http.client.HTTPException,
        ) as exc:
            last_status, last_body, last_detail = None, None, f"transport error: {exc}"

        if attempt < _API_MAX_ATTEMPTS:
            if server_backoff is not None:
                delay = server_backoff
            else:
                # Clamp rather than index directly: if _API_MAX_ATTEMPTS is
                # ever raised without extending _API_RETRY_BACKOFF_SECONDS to
                # match, reuse the last known backoff instead of raising
                # IndexError inside a required CI gate.
                schedule_index = min(attempt - 1, len(_API_RETRY_BACKOFF_SECONDS) - 1)
                delay = _API_RETRY_BACKOFF_SECONDS[schedule_index]
            effective_sleep(delay)

    return last_status, last_body, last_detail


def _compare_url(repo: str, base: str, ref: str) -> str:
    quoted_base = urllib.parse.quote(base, safe="/")
    quoted_ref = urllib.parse.quote(ref, safe="/")
    return f"{_GITHUB_API}/repos/{_ORG}/{repo}/compare/{quoted_base}...{quoted_ref}"


class _Resolver:
    """Resolves pins against the live GitHub API, with dedup + a circuit breaker.

    Two independent bounds protect the CI job timeout, for two different
    failure shapes:

    * ``tripped`` (the pre-existing consecutive-transport-failure breaker) is
      a fast path for a SUSTAINED outage -- every call failing with no HTTP
      status at all.
    * ``_deadline_exceeded`` (the run-wide wall-clock budget) is what
      actually bounds an INTERMITTENT-failure run: the breaker's counter
      resets on any HTTP-status-bearing response (even a retried-and-still-
      failing one), so a pattern that alternates success/failure across pins
      never trips it, and every pin can independently pay the full per-call
      retry ceiling. See ``_RUN_DEADLINE_SECONDS``.
    """

    def __init__(
        self, protected: Sequence[str], *, now: Callable[[], float] | None = None
    ) -> None:
        self._protected = tuple(protected)
        self._cache: dict[tuple[str, str], Resolution] = {}
        self._consecutive_transport_failures = 0
        self._now = now or time.monotonic
        self._deadline_at = self._now() + _RUN_DEADLINE_SECONDS

    @property
    def tripped(self) -> bool:
        return (
            self._consecutive_transport_failures >= _TRANSPORT_FAILURE_CIRCUIT_BREAKER
        )

    @property
    def _deadline_exceeded(self) -> bool:
        return self._now() >= self._deadline_at

    def resolve(self, repo: str, ref: str) -> Resolution:
        key = (repo, ref)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        resolution = self._resolve_uncached(repo, ref)
        self._cache[key] = resolution
        return resolution

    def _resolve_uncached(self, repo: str, ref: str) -> Resolution:
        # A pin that names a protected branch itself is durable by definition:
        # the branch cannot be deleted, and no API call can make that truer.
        if ref in self._protected:
            return Resolution(
                Verdict.REACHABLE, f"pins the protected branch {ref!r} directly"
            )

        if self.tripped:
            return Resolution(
                Verdict.UNDETERMINED,
                "skipped: transport-failure circuit breaker already tripped",
            )

        not_found: list[str] = []
        for base in self._protected:
            if self._deadline_exceeded:
                return Resolution(
                    Verdict.UNDETERMINED,
                    f"skipped: run-wide {_RUN_DEADLINE_SECONDS:.0f}s deadline "
                    "exceeded -- remaining pins reported undetermined "
                    "(fail-closed) to stay within the CI job timeout",
                )
            status, body, detail = _api_get(_compare_url(repo, base, ref))
            if status is None:
                self._consecutive_transport_failures += 1
                if self.tripped:
                    return Resolution(
                        Verdict.UNDETERMINED,
                        f"{detail} (circuit breaker tripped after "
                        f"{_TRANSPORT_FAILURE_CIRCUIT_BREAKER} consecutive failures)",
                    )
                return Resolution(Verdict.UNDETERMINED, detail)
            self._consecutive_transport_failures = 0
            if status == 200 and body is not None:
                compare_status = str(body.get("status", ""))
                if status_is_reachable(compare_status):
                    return Resolution(
                        Verdict.REACHABLE,
                        f"compare {base}...{ref[:12]} = {compare_status}",
                    )
                not_found.append(f"{base}...{ref[:12]} = {compare_status}")
                continue
            if status == 404:
                not_found.append(f"{base}...{ref[:12]} = HTTP 404")
                continue
            # 403 / 429 (rate limit) and anything else: cannot determine.
            return Resolution(Verdict.UNDETERMINED, detail)

        return Resolution(Verdict.UNREACHABLE, self._explain(repo, ref, not_found))

    def _explain(self, repo: str, ref: str, observations: list[str]) -> str:
        """Say WHICH kind of dead the pin is.

        The commits endpoint serves unreferenced objects, so a 200 here proves
        the OMN-14447 shape exactly: the object exists and is reachable from
        nothing protected. This call never grants a pass -- it only sharpens
        the failure message.

        Guarded by the same run-wide deadline as the per-branch compare
        calls above: without this check, this call could run AFTER the
        deadline had already been exceeded by the last compare call,
        doubling the documented 96s post-deadline tail to 192s (defect found
        2026-08-06). Once the deadline is exceeded, skip the lookup entirely
        and return the plain observations -- the verdict is UNREACHABLE
        either way; this only sharpens detail text, never a pass/fail
        outcome, so skipping it costs nothing but explanation detail.
        """
        joined = "; ".join(observations)
        if self._deadline_exceeded:
            return joined
        status, _body, _detail = _api_get(
            f"{_GITHUB_API}/repos/{_ORG}/{repo}/commits/"
            f"{urllib.parse.quote(ref, safe='/')}"
        )
        if status == 200:
            return (
                f"{joined} -- the object EXISTS but is reachable from no protected "
                "branch (unlanded or deleted feature-branch head)"
            )
        if status == 404:
            return f"{joined} -- the ref does not resolve at all (object gone or repo/ref typo)"
        return joined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_FIX_GUIDANCE = """
  Fix: re-pin to the SQUASH-MERGED commit on dev (or main). Squash-merged
       commits are reachable forever; feature-branch heads evaporate the moment
       the PR lands, and an UNLANDED branch head is a time bomb that detonates
       on merge -- it resolves right up until it doesn't.

         gh api repos/OmniNode-ai/<repo>/compare/dev...<sha> --jq .status
           behind | identical -> durable, safe to pin
           ahead  | diverged  -> NOT durable, do not pin

  Do NOT "verify" a pin with `gh api .../commits/<sha>`: GitHub serves
  unreferenced objects, so that probe passes on exactly the pins that break.
"""


def _default_targets(root: Path) -> list[Path]:
    targets: list[Path] = []
    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        targets.append(workflows)
    for name in ("pyproject.toml", "uv.lock"):
        candidate = root / name
        if candidate.is_file():
            targets.append(candidate)
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reject any cross-repo pin whose commit is not reachable from a "
            "protected branch of the target repo."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Files or directories to scan. Defaults to .github/workflows/, "
            "pyproject.toml and uv.lock under --root."
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=Path(), help="Repo root for default targets."
    )
    parser.add_argument(
        "--protected",
        action="append",
        default=None,
        metavar="BRANCH",
        help=f"Protected branch to resolve against (repeatable; default: {' '.join(_DEFAULT_PROTECTED)}).",
    )
    parser.add_argument(
        "--min-pins",
        type=int,
        default=0,
        help=(
            "Fail if fewer than N pins were extracted. Guards the "
            "vacuous-green case where a glob typo or schema drift silently "
            "makes the gate check nothing."
        ),
    )
    parser.add_argument(
        "--allow-undetermined",
        action="store_true",
        help=(
            "Downgrade UNDETERMINED (offline / rate-limited) from failure to a "
            "warning. Local use only -- REFUSED when CI is set."
        ),
    )
    args = parser.parse_args(argv)

    in_ci = bool(os.environ.get("CI"))
    if args.allow_undetermined and in_ci:
        print(
            "error: --allow-undetermined is refused under CI. The enforcing "
            "surface must fail closed on unresolvable pins; an undetermined "
            "result is not a pass.",
            file=sys.stderr,
        )
        return 2

    protected = tuple(args.protected) if args.protected else _DEFAULT_PROTECTED
    targets = list(args.paths) if args.paths else _default_targets(args.root)
    pins = extract_pins(targets)

    if len(pins) < args.min_pins:
        print(
            f"FAIL: extracted {len(pins)} pin(s) from {len(targets)} target(s), "
            f"expected at least {args.min_pins}. The extractor is more likely "
            "broken than the tree is clean -- a gate that checks nothing "
            "reports green forever.",
            file=sys.stderr,
        )
        return 1

    if not pins:
        print(f"no cross-repo {_ORG} pins found in {len(targets)} target(s)")
        return 0

    resolver = _Resolver(protected)
    unreachable: list[tuple[PinRef, Resolution]] = []
    undetermined: list[tuple[PinRef, Resolution]] = []

    print(
        f"Resolving {len(pins)} cross-repo pin(s) against protected branches "
        f"{'/'.join(protected)}:"
    )
    for pin in sorted(pins):
        resolution = resolver.resolve(pin.repo, pin.ref)
        marker = {
            Verdict.REACHABLE: "  OK          ",
            Verdict.UNREACHABLE: "  UNREACHABLE ",
            Verdict.UNDETERMINED: "  UNDETERMINED",
        }[resolution.verdict]
        print(
            f"{marker} {pin.repo} @ {pin.ref[:12]}  [{pin.kind}] "
            f"{pin.source}::{pin.locus}  ({resolution.detail})"
        )
        if resolution.verdict is Verdict.UNREACHABLE:
            unreachable.append((pin, resolution))
        elif resolution.verdict is Verdict.UNDETERMINED:
            undetermined.append((pin, resolution))

    # Keep the human-readable per-pin listing above the failure block in a
    # merged 2>&1 CI log; unflushed stdout otherwise lands after stderr.
    sys.stdout.flush()

    if undetermined and not args.allow_undetermined:
        print(
            f"\nFAIL: {len(undetermined)} pin(s) could not be resolved. An "
            "unresolvable pin is not a passing pin -- this gate fails closed so "
            "a network blip or rate limit can never mint a false green.\n",
            file=sys.stderr,
        )
        for pin, resolution in undetermined:
            print(
                f"  {pin.repo} @ {pin.ref}  {pin.source}::{pin.locus}  "
                f"({resolution.detail})",
                file=sys.stderr,
            )
        print(
            "\n  Set GH_TOKEN/GITHUB_TOKEN for rate-limit headroom. Offline, "
            "re-run with --allow-undetermined (local only).",
            file=sys.stderr,
        )
        return 1

    if undetermined:
        print(
            f"\nWARNING: {len(undetermined)} pin(s) UNDETERMINED and "
            "--allow-undetermined was passed. This run PROVED NOTHING about "
            "those pins; CI is the enforcing surface."
        )

    if not unreachable:
        print(f"\nPin reachability OK: {len(pins)} cross-repo pin(s) checked")
        return 0

    print(
        f"\nFAIL: {len(unreachable)} pin(s) are NOT reachable from any protected "
        f"branch ({'/'.join(protected)}) of their target repo:\n",
        file=sys.stderr,
    )
    for pin, resolution in unreachable:
        print(f"  {pin.source}::{pin.locus}", file=sys.stderr)
        print(f"    {pin.repo} @ {pin.ref}", file=sys.stderr)
        print(f"    {resolution.detail}", file=sys.stderr)
    print(_FIX_GUIDANCE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

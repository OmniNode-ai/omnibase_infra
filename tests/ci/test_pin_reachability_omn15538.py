# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hermetic half of the protected-branch pin-reachability gate (OMN-15538).

The live half — which resolves the two real incident vectors against the
GitHub API — is ``tests/integration/ci/test_pin_reachability_live_omn15538.py``,
under ``tests/integration`` for the same reasons as its OMN-14941 sibling: the
pre-push selector always ignores that tree, and the gate's enforcement surface
is CI (plus the dedicated ``Pin Reachability (OMN-15538)`` ci.yml job).

What is pinned here, and why each assertion exists
--------------------------------------------------
Every test below corresponds to a way one of the two 2026-07-30 incidents got
through, or to a way a naive "fix" would silently un-fix it:

* ``status_is_reachable`` — the discriminating oracle. ``ahead`` must be
  REJECTED. Instance B (``omnimarket@dev`` -> ``omnibase_core@5a907b71``) is
  ``ahead`` of dev: it is a descendant commit on a still-open branch, so it
  resolves today and dies on merge. Accepting ``ahead`` would make this gate
  green on exactly the pin it was built for.
* ``with:`` extraction — Instance A pinned the same dead SHA twice
  (``uses:`` and ``vocabulary_ref``). A ``uses:``-only checker leaves the
  second one behind.
* ``pyproject`` ``[tool.uv.sources]`` extraction — the pre-existing OMN-14449
  reachability check regexes the uv.lock ``?rev=`` URL form only, so the
  ``git = ...`` / ``rev = ...`` split-key form is structurally invisible to
  it. That blind spot IS Instance B.
* ``--allow-undetermined`` refusal under CI — an escape hatch that survives
  into the enforcing surface converts a fail-closed gate into a silent skip.
* ``--min-pins`` — a gate that extracts nothing passes forever.
"""

from __future__ import annotations

import email.message
import http.client
import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from scripts.ci.check_pin_reachability import (
    PinRef,
    Verdict,
    _api_get,
    _is_transient_http_status,
    _Resolver,
    extract_pins,
    extract_pyproject_pins,
    extract_uv_lock_pins,
    extract_workflow_pins,
    main,
    status_is_reachable,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# The exact pin that wedged omnibase_infra CI for ~2.5h on 2026-07-30
# (OMN-15536): the head of an omnimarket PR branch, deleted on squash-merge.
INCIDENT_A_DEAD_SHA = "879d6fc6825f876458c6d45ed670c8715de8ac95"
INCIDENT_A_GOOD_SHA = "454c429f328e68e19300f33ffb8121f1bccc7f86"
# The pin live on omnimarket@dev pyproject.toml: the head of the still-open
# jonah/omn-15392-evidence-execution-scope branch.
INCIDENT_B_DEAD_SHA = "5a907b71c5cf321ed6407cd0509ec406afc81ff5"
INCIDENT_B_GOOD_SHA = "3f2998b3337e4050b4758e0dd2a0fe1061ce0d98"


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("behind", True),  # pin is an ancestor of the protected branch
        ("identical", True),  # pin IS the protected branch head
        ("ahead", False),  # unlanded descendant — Instance B's shape
        ("diverged", False),  # no ancestry — Instance A's shape
        ("", False),
        ("unknown", False),
    ],
)
def test_status_is_reachable_maps_compare_status(status: str, expected: bool) -> None:
    assert status_is_reachable(status) is expected


@pytest.mark.unit
def test_ahead_is_rejected_not_merely_diverged() -> None:
    """Falsification control for the single most tempting wrong simplification.

    "Unreachable means diverged" is the intuitive reading, and it is wrong: an
    unlanded feature-branch head descends from dev, so it compares ``ahead``.
    A gate that only rejects ``diverged`` catches Instance A and waves through
    Instance B — which is precisely the half-fix this ticket exists to prevent.
    """
    assert status_is_reachable("diverged") is False
    assert status_is_reachable("ahead") is False


# ---------------------------------------------------------------------------
# Workflow extraction
# ---------------------------------------------------------------------------


_WORKFLOW_FIXTURE = f"""\
name: fixture
on:
  pull_request:
# uses: OmniNode-ai/omniclaude/.github/workflows/commented-out.yml@main
jobs:
  merge-hold-gate:
    uses: OmniNode-ai/omnimarket/.github/workflows/merge-hold-gate-reusable.yml@{INCIDENT_A_DEAD_SHA}
    with:
      vocabulary_ref: {INCIDENT_A_DEAD_SHA}
      context_name: merge-hold-gate / evaluate
  dynamic-caller:
    uses: OmniNode-ai/omnibase_core/.github/workflows/occ-preflight.yml@dev
    with:
      core-ref: ${{{{ github.sha }}}}
  step-user:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
      - uses: ./.github/actions/setup-python-uv
      - uses: OmniNode-ai/onex_change_control/.github/actions/thing@main
"""


@pytest.mark.unit
def test_workflow_extraction_covers_uses_with_and_steps(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.yml"
    fixture.write_text(_WORKFLOW_FIXTURE, encoding="utf-8")
    pins = extract_workflow_pins(fixture)

    assert [(p.locus, p.kind, p.repo, p.ref) for p in pins] == [
        (
            "jobs.merge-hold-gate.uses",
            "workflow-uses",
            "omnimarket",
            INCIDENT_A_DEAD_SHA,
        ),
        (
            "jobs.merge-hold-gate.with.vocabulary_ref",
            "workflow-with",
            "omnimarket",
            INCIDENT_A_DEAD_SHA,
        ),
        ("jobs.dynamic-caller.uses", "workflow-uses", "omnibase_core", "dev"),
        (
            "jobs.step-user.steps[2].uses",
            "workflow-step-uses",
            "onex_change_control",
            "main",
        ),
    ]


@pytest.mark.unit
def test_workflow_extraction_catches_the_sibling_ref_incident_a_pinned_twice(
    tmp_path: Path,
) -> None:
    """OMN-15538 AC-5: repointing only ``uses:`` leaves a second dead ref."""
    fixture = tmp_path / "fixture.yml"
    fixture.write_text(_WORKFLOW_FIXTURE, encoding="utf-8")
    dead = [p for p in extract_workflow_pins(fixture) if p.ref == INCIDENT_A_DEAD_SHA]
    assert len(dead) == 2, "both uses: and vocabulary_ref must be extracted"
    assert {p.kind for p in dead} == {"workflow-uses", "workflow-with"}


@pytest.mark.unit
def test_workflow_extraction_skips_expressions_commented_lines_and_third_party(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture.yml"
    fixture.write_text(_WORKFLOW_FIXTURE, encoding="utf-8")
    pins = extract_workflow_pins(fixture)
    # `core-ref: ${{ github.sha }}` is not a resolvable literal.
    assert not any(p.locus.endswith("with.core-ref") for p in pins)
    # A commented-out uses: line is not YAML structure.
    assert not any("commented-out" in p.ref for p in pins)
    # actions/checkout and ./local composites are out of scope.
    assert all(
        p.repo in {"omnimarket", "omnibase_core", "onex_change_control"} for p in pins
    )


@pytest.mark.unit
def test_workflow_extraction_ignores_non_workflow_yaml(tmp_path: Path) -> None:
    fixture = tmp_path / "not-a-workflow.yml"
    fixture.write_text("just: a mapping\n", encoding="utf-8")
    assert extract_workflow_pins(fixture) == []


# ---------------------------------------------------------------------------
# pyproject / uv.lock extraction
# ---------------------------------------------------------------------------


_PYPROJECT_FIXTURE = f"""\
[project]
name = "fixture"
version = "0.0.0"
dependencies = [
    "requests>=2",
    "omnibase-spi @ git+https://github.com/OmniNode-ai/omnibase_spi.git@deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
]

[project.optional-dependencies]
dev = ["omniclaude @ git+https://github.com/OmniNode-ai/omniclaude@feature/nope"]

[dependency-groups]
lint = ["omnidash @ git+https://github.com/OmniNode-ai/omnidash.git@v1.2.3"]

[tool.uv.sources]
omnibase-core = {{ git = "https://github.com/OmniNode-ai/omnibase_core.git", rev = "{INCIDENT_B_DEAD_SHA}" }}
omnibase-infra = {{ git = "https://github.com/OmniNode-ai/omnibase_infra.git", branch = "dev" }}
onex-change-control = {{ git = "https://github.com/OmniNode-ai/onex_change_control.git" }}
third-party = {{ git = "https://github.com/someone-else/thing.git", rev = "0123456789abcdef0123456789abcdef01234567" }}
"""


@pytest.mark.unit
def test_pyproject_extraction_covers_uv_sources_and_pep508(tmp_path: Path) -> None:
    fixture = tmp_path / "pyproject.toml"
    fixture.write_text(_PYPROJECT_FIXTURE, encoding="utf-8")
    pins = extract_pyproject_pins(fixture)

    assert {(p.locus, p.repo, p.ref) for p in pins} == {
        (
            "project.dependencies[1]",
            "omnibase_spi",
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        ),
        ("project.optional-dependencies.dev[0]", "omniclaude", "feature/nope"),
        ("dependency-groups.lint[0]", "omnidash", "v1.2.3"),
        ("tool.uv.sources.omnibase-core.rev", "omnibase_core", INCIDENT_B_DEAD_SHA),
        ("tool.uv.sources.omnibase-infra.branch", "omnibase_infra", "dev"),
    }


@pytest.mark.unit
def test_pyproject_extraction_is_the_blind_spot_the_uv_lock_gate_has(
    tmp_path: Path,
) -> None:
    """Instance B lives in the split-key form the OMN-14449 regex cannot match.

    ``omnimarket/scripts/ci/check_uv_lock_pin_reachability.py`` matches only
    ``git = "https://github.com/<org>/<repo>.git?rev=<sha>"`` — one string
    carrying both URL and rev. A ``[tool.uv.sources]`` table puts the rev in a
    separate key, so that regex finds nothing. This asserts the new extractor
    does not inherit the blind spot.
    """
    fixture = tmp_path / "pyproject.toml"
    fixture.write_text(_PYPROJECT_FIXTURE, encoding="utf-8")
    revs = {p.ref for p in extract_pyproject_pins(fixture)}
    assert INCIDENT_B_DEAD_SHA in revs


@pytest.mark.unit
def test_pyproject_extraction_ignores_foreign_org_and_default_branch_sources(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "pyproject.toml"
    fixture.write_text(_PYPROJECT_FIXTURE, encoding="utf-8")
    pins = extract_pyproject_pins(fixture)
    # Not an OmniNode-ai repo — out of scope.
    assert not any(p.repo == "thing" for p in pins)
    # No rev/branch/tag: tracks the default branch, which cannot evaporate.
    assert not any(
        p.locus.startswith("tool.uv.sources.onex-change-control") for p in pins
    )


_UV_LOCK_FIXTURE = f"""\
[[package]]
name = "omnibase-core"
source = {{ git = "https://github.com/OmniNode-ai/omnibase_core.git?rev={INCIDENT_B_DEAD_SHA}#{INCIDENT_B_DEAD_SHA}" }}

[[package]]
name = "omnibase-spi"
source = {{ git = "https://github.com/OmniNode-ai/omnibase_spi.git?branch=dev#{INCIDENT_B_GOOD_SHA}" }}

[[package]]
name = "requests"
source = {{ registry = "https://pypi.org/simple" }}
"""


@pytest.mark.unit
def test_uv_lock_extraction_prefers_the_resolved_commit(tmp_path: Path) -> None:
    fixture = tmp_path / "uv.lock"
    fixture.write_text(_UV_LOCK_FIXTURE, encoding="utf-8")
    pins = extract_uv_lock_pins(fixture)
    assert {(p.repo, p.ref) for p in pins} == {
        ("omnibase_core", INCIDENT_B_DEAD_SHA),
        # `?branch=dev` resolves to a concrete commit in the fragment; the
        # fragment is the object a build actually fetches, so it is what must
        # be reachable — a branch name alone would hide a since-force-pushed ref.
        ("omnibase_spi", INCIDENT_B_GOOD_SHA),
    }


@pytest.mark.unit
def test_extract_pins_dispatches_by_filename(tmp_path: Path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "fixture.yml").write_text(
        _WORKFLOW_FIXTURE, encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT_FIXTURE, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(_UV_LOCK_FIXTURE, encoding="utf-8")

    pins = extract_pins([tmp_path / ".github" / "workflows", tmp_path])
    kinds = {p.kind for p in pins}
    assert kinds == {
        "workflow-uses",
        "workflow-with",
        "workflow-step-uses",
        "pyproject-source",
        "pyproject-pep508",
        "uv-lock",
    }


# ---------------------------------------------------------------------------
# Vacuity + fail-closed posture
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_real_tree_extraction_is_nonempty() -> None:
    """Guard the vacuous-green case: a glob typo makes the gate check nothing.

    This repo is known to carry cross-repo workflow pins plus git-sourced
    sibling deps; if extraction collapses, the live gate passes on everything.
    """
    pins = extract_pins(
        [WORKFLOWS_DIR, REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"]
    )
    assert len(pins) >= 10, (
        f"expected >=10 cross-repo pins in this tree, got {len(pins)} — "
        "the extractor is more likely broken than the tree is clean"
    )
    assert {p.kind for p in pins} >= {"workflow-uses", "pyproject-source", "uv-lock"}


@pytest.mark.unit
def test_min_pins_fails_when_extraction_collapses(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\n', encoding="utf-8"
    )
    assert main([str(tmp_path / "pyproject.toml"), "--min-pins", "1"]) == 1


@pytest.mark.unit
def test_allow_undetermined_is_refused_under_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape hatch must not survive into the enforcing surface.

    An ``--allow-undetermined`` that CI honours turns a fail-closed gate into
    a rate-limit-shaped silent skip — the optional-input trap that makes a
    check exist on paper and nowhere else.
    """
    monkeypatch.setenv("CI", "true")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\n', encoding="utf-8"
    )
    assert main([str(tmp_path / "pyproject.toml"), "--allow-undetermined"]) == 2


@pytest.mark.unit
def test_no_pins_is_success_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CI", raising=False)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\n', encoding="utf-8"
    )
    assert main([str(tmp_path / "pyproject.toml")]) == 0


@pytest.mark.unit
def test_pinref_is_orderable_for_stable_output() -> None:
    """Report ordering must be deterministic so CI log diffs are meaningful."""
    a = PinRef("f.yml", "jobs.a.uses", "workflow-uses", "omnimarket", "a" * 40)
    b = PinRef("f.yml", "jobs.b.uses", "workflow-uses", "omnimarket", "b" * 40)
    assert sorted([b, a]) == [a, b]


@pytest.mark.unit
def test_verdict_values_are_stable_strings() -> None:
    assert Verdict.REACHABLE.value == "REACHABLE"
    assert Verdict.UNREACHABLE.value == "UNREACHABLE"
    assert Verdict.UNDETERMINED.value == "UNDETERMINED"


# ---------------------------------------------------------------------------
# Transport retry (defect fix, 2026-08-06): one unretried 10s timeout on the
# onex_change_control @ 2dd26ade... compare call was redding essentially every
# open omnibase_infra dev PR, even though the pin is verifiably reachable
# live. ``_api_get`` must absorb a transient hiccup with a bounded retry
# instead of handing the caller a single-shot UNDETERMINED -- while a
# genuinely exhausted retry ceiling must still fail closed (this is a false-RED
# fix, not a weakening of the fail-closed gate), and a definitive 4xx must
# never be retried (it will not change on a second try; retrying it only adds
# latency to a job that runs on every PR).
# ---------------------------------------------------------------------------


def _http_error(code: int, message: str = "") -> urllib.error.HTTPError:
    body = json.dumps({"message": message}).encode("utf-8") if message else b"{}"
    return urllib.error.HTTPError(
        url="https://api.github.com/x",
        code=code,
        msg=message or "error",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )


class _FakeResponse:
    """Minimal stand-in for the ``http.client.HTTPResponse`` context manager."""

    def __init__(self, status: int, payload: dict[str, object]) -> None:
        self.status = status
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _scripted_urlopen(
    effects: list[Exception | _FakeResponse],
) -> tuple[object, list[int]]:
    """Return a fake ``urllib.request.urlopen`` that replays ``effects`` in order.

    ``calls`` records one entry per invocation so tests can assert exactly how
    many HTTP attempts were made -- the whole point of the retry-vs-no-retry
    split.
    """
    calls: list[int] = []

    def fake_urlopen(request: object, timeout: float | None = None) -> _FakeResponse:
        index = len(calls)
        calls.append(index)
        effect = effects[index]
        if isinstance(effect, Exception):
            raise effect
        return effect

    return fake_urlopen, calls


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (404, False),
        (403, False),
        (400, False),
        (401, False),
        (410, False),
    ],
)
def test_is_transient_http_status_covers_only_5xx_and_429(
    status: int, expected: bool
) -> None:
    assert _is_transient_http_status(status) is expected


@pytest.mark.unit
def test_api_get_retries_transient_timeout_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) attempt 1 transient, attempt 2 succeeds -> the successful result, no UNDETERMINED."""
    fake_urlopen, calls = _scripted_urlopen(
        [TimeoutError("timed out"), _FakeResponse(200, {"status": "behind"})]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, body, detail = _api_get("https://api.github.com/x", sleep=slept.append)

    assert status == 200
    assert body == {"status": "behind"}
    assert detail == "HTTP 200"
    assert len(calls) == 2, "must retry exactly once after the transient timeout"
    assert slept == [2.0], "one bounded backoff sleep before the successful retry"


@pytest.mark.unit
def test_api_get_retries_transient_http_5xx_and_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both no-status transport errors AND HTTP 5xx/429 are retried."""
    fake_urlopen, calls = _scripted_urlopen(
        [
            _http_error(503, "Service Unavailable"),
            _http_error(429, "rate limited"),
            _FakeResponse(200, {"status": "identical"}),
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, body, _detail = _api_get("https://api.github.com/x", sleep=slept.append)

    assert status == 200
    assert body == {"status": "identical"}
    assert len(calls) == 3
    assert slept == [2.0, 4.0], "bounded exponential backoff across two retries"


@pytest.mark.unit
def test_api_get_exhausts_retries_and_stays_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) fail-closed preserved: exhausting the ceiling is still a FAILURE, never a pass."""
    fake_urlopen, calls = _scripted_urlopen(
        [
            TimeoutError("timed out"),
            TimeoutError("timed out"),
            TimeoutError("timed out"),
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, body, detail = _api_get("https://api.github.com/x", sleep=slept.append)

    assert status is None, "exhausted transient failure must not resolve to a pass"
    assert body is None
    assert "transport error" in detail
    assert len(calls) == 3, "retries exactly _API_MAX_ATTEMPTS times, no more"
    assert slept == [2.0, 4.0]


@pytest.mark.unit
def test_resolver_stays_undetermined_after_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) end-to-end through ``_Resolver``: exhausted retries -> UNDETERMINED, not REACHABLE.

    This is the assertion that actually matters: a caller consuming
    ``_Resolver.resolve`` (as ``main()`` does) must see the same fail-closed
    UNDETERMINED verdict it always has -- the retry only removes the FALSE
    reds, never the true ones.
    """
    fake_urlopen, calls = _scripted_urlopen(
        [
            TimeoutError("timed out"),
            TimeoutError("timed out"),
            TimeoutError("timed out"),
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    resolution = _Resolver(("dev", "main")).resolve("omnimarket", "a" * 40)

    assert resolution.verdict is Verdict.UNDETERMINED
    assert len(calls) == 3, (
        "one compare call, fully retried, is enough to fail closed -- the "
        "resolver must not need to exhaust every protected branch"
    )


@pytest.mark.unit
def test_api_get_does_not_retry_a_definitive_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) a definitive 4xx maps to its existing meaning immediately, no retry."""
    fake_urlopen, calls = _scripted_urlopen([_http_error(404, "Not Found")])
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, body, detail = _api_get("https://api.github.com/x", sleep=slept.append)

    assert status == 404
    assert body is None
    assert detail == "HTTP 404: Not Found"
    assert len(calls) == 1, "a definitive 404 must not be retried"
    assert slept == [], "no backoff sleep for a non-retried definitive failure"


@pytest.mark.unit
def test_api_get_does_not_retry_a_definitive_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) 403 without rate-limit semantics (429) is definitive, not transient."""
    fake_urlopen, calls = _scripted_urlopen([_http_error(403, "Forbidden")])
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, _body, _detail = _api_get("https://api.github.com/x", sleep=slept.append)

    assert status == 403
    assert len(calls) == 1, "a definitive 403 must not be retried"
    assert slept == []


@pytest.mark.unit
def test_api_get_backoff_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """(d) backoff is bounded -- a persistent transient failure sleeps a fixed,
    short, known schedule, never an unbounded or growing-without-limit one."""
    fake_urlopen, _calls = _scripted_urlopen(
        [
            TimeoutError("t"),
            TimeoutError("t"),
            TimeoutError("t"),
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    _api_get("https://api.github.com/x", sleep=slept.append)

    assert len(slept) == 2, "at most _API_MAX_ATTEMPTS - 1 backoff sleeps"
    assert all(0 < delay <= 10.0 for delay in slept), (
        f"each backoff sleep must be short and bounded, got {slept}"
    )
    assert sum(slept) <= 30.0, f"total backoff time must be bounded, got {slept}"


# ---------------------------------------------------------------------------
# Regression (defect fix, 2026-08-06): bounded-worst-case-under-CI-timeout
# ---------------------------------------------------------------------------
# The per-call retry ceiling above (150s, the TRUE worst case -- see the
# rate-limited-class correction below) only bounds ONE call. The
# pre-existing consecutive-transport-failure circuit breaker resets to 0 on
# ANY HTTP-status-bearing response (including a retried-and-still-failing
# 503/429), so an intermittent-transport run can pay the full per-call
# ceiling on every pin without the breaker ever tripping -- the run-wide
# wall time was unbounded. ``_Resolver`` must enforce its own run-wide
# deadline independent of the breaker.


def _headers(pairs: dict[str, str] | None = None) -> email.message.Message:
    msg = email.message.Message()
    for key, value in (pairs or {}).items():
        msg[key] = value
    return msg


def _http_error_with_headers(
    code: int, message: str = "", headers: dict[str, str] | None = None
) -> urllib.error.HTTPError:
    body = json.dumps({"message": message}).encode("utf-8") if message else b"{}"
    return urllib.error.HTTPError(
        url="https://api.github.com/x",
        code=code,
        msg=message or "error",
        hdrs=_headers(headers),
        fp=io.BytesIO(body),
    )


@pytest.mark.unit
def test_resolver_enforces_a_run_wide_deadline_independent_of_the_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the injected clock is already past the run-wide deadline, a fresh
    resolution must fail closed to UNDETERMINED WITHOUT making any network
    call -- proving the deadline is checked before spending any more wall
    time, not merely reported after the fact."""
    from scripts.ci.check_pin_reachability import _RUN_DEADLINE_SECONDS

    def fail_if_called(request: object, timeout: float | None = None) -> None:
        raise AssertionError(
            "no network call should be attempted once the run-wide deadline "
            "is already exceeded"
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail_if_called)

    clock = {"t": _RUN_DEADLINE_SECONDS + 1.0}
    resolver = _Resolver(("dev", "main"), now=lambda: clock["t"])
    # Force the deadline to be recorded as already-elapsed regardless of the
    # constructor's own start-time read.
    resolver._deadline_at = 0.0

    resolution = resolver.resolve("omnimarket", "b" * 40)

    assert resolution.verdict is Verdict.UNDETERMINED
    assert "deadline" in resolution.detail.lower()


@pytest.mark.unit
def test_resolver_aggregate_wall_time_is_bounded_under_intermittent_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through ``_Resolver.resolve``, using a pin-boundary-aligned
    TRUE-WORST-CASE pattern: a fast ``dev`` compare followed by a ``main``
    compare that fully exhausts the RATE-LIMITED retry ceiling -- 429 with a
    server ``Retry-After`` header on the first two attempts (capped at
    ``_MAX_RATE_LIMIT_BACKOFF_SECONDS``, and PREFERRED over the fixed 2s/4s
    schedule per ``_rate_limit_backoff_seconds``) -- before finally answering
    a definitive 404 on the third attempt:

        3 * _REQUEST_TIMEOUT_SECONDS + 2 * _MAX_RATE_LIMIT_BACKOFF_SECONDS
          = 90 + 60 = 150s

    That lands the run-wide deadline crossing exactly inside ``main``'s
    still-in-flight compare call, with ``_explain``'s own lookup guarded
    against running afterward.

    Corrected 2026-08-06 (terminal adversarial verify round 2, PR #2679
    comment 5209929069): a prior version of this test exercised only the
    header-less-503 fixed-schedule backoff class, whose per-call ceiling is
    96s -- cheaper than, and therefore not discriminating against, the
    150s rate-limited ceiling this module's own retry logic actually pays
    when GitHub signals a rate limit. A test asserting ``<= 96.0 + 1.0`` on
    THIS construction would fail (undercounts the true tail by 54s) --
    proof the old assertion was pinning the wrong number, not merely a
    smaller one."""
    from scripts.ci.check_pin_reachability import (
        _API_MAX_ATTEMPTS,
        _MAX_RATE_LIMIT_BACKOFF_SECONDS,
    )

    clock = {"t": 0.0}

    def fake_sleep(seconds: float) -> None:
        clock["t"] += seconds

    attempt_counts: dict[str, int] = {}
    calls: list[str] = []

    def fake_urlopen(request: object, timeout: float | None = None) -> object:
        url = getattr(request, "full_url", "")
        calls.append(url)
        if "/compare/dev" in url:
            # dev-branch compare: fast, definitive "not reachable" -- no
            # latency tax.
            return _FakeResponse(200, {"status": "diverged"})
        if "/commits/" in url:
            # _explain's own lookup: must never be reached once the
            # run-wide deadline is already exceeded by main's compare call.
            raise AssertionError(
                "_explain must not issue its own unguarded network call "
                "past the deadline"
            )
        # main-branch compare: fully exhaust the RATE-LIMITED retry ceiling
        # (429 + Retry-After, capped and preferred over the fixed schedule)
        # before finally answering a definitive 404 on the last attempt.
        clock["t"] += timeout or 0.0
        attempt_counts[url] = attempt_counts.get(url, 0) + 1
        if attempt_counts[url] < _API_MAX_ATTEMPTS:
            raise _http_error_with_headers(
                429,
                "rate limited",
                {"Retry-After": str(_MAX_RATE_LIMIT_BACKOFF_SECONDS)},
            )
        raise _http_error_with_headers(404, "not found")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    resolver = _Resolver(("dev", "main"), now=lambda: clock["t"])
    # Land the deadline exactly between "dev's fast compare completes" (~0s)
    # and "main's 150s compare completes" (~150s) -- the per-branch loop's
    # OWN deadline check (before main's call) must still pass here, so
    # main's call is already in flight when the deadline is crossed; it is
    # ``_explain``'s guard that must catch the crossing afterward.
    deadline_at = 50.0
    resolver._deadline_at = deadline_at

    resolution = resolver.resolve("omnimarket", "b" * 40)

    assert resolution.verdict is Verdict.UNREACHABLE
    assert not any("/commits/" in url for url in calls), (
        "the deadline was already exceeded by main's compare call -- "
        "_explain must not issue its own unguarded network call past the "
        "deadline"
    )
    # True bound: the deadline plus at most ONE already-in-flight call's
    # REAL 150s rate-limited ceiling -- not the cheaper 96s fixed-schedule
    # figure, and not doubled by an unguarded _explain call.
    real_ceiling = 3 * 30.0 + 2 * _MAX_RATE_LIMIT_BACKOFF_SECONDS
    assert real_ceiling == 150.0, "sanity: this test's own ceiling arithmetic"
    assert clock["t"] <= real_ceiling + 1.0, (
        f"aggregate simulated wall time {clock['t']}s must stay bounded by "
        f"one already-in-flight call's {real_ceiling}s rate-limited ceiling "
        "past the deadline, not double-charged by an unguarded _explain call"
    )
    # Post-deadline tail specifically (the portion that runs AFTER the
    # deadline was already crossed) must not exceed the real per-call
    # ceiling either -- it is bounded by "one already-in-flight call", full
    # stop, regardless of when within that call the deadline landed.
    post_deadline_tail = clock["t"] - deadline_at
    assert post_deadline_tail <= real_ceiling + 1.0, (
        f"post-deadline tail {post_deadline_tail}s must stay within the "
        f"real {real_ceiling}s per-call ceiling"
    )
    # Discriminate against the old, wrong 96s figure: this construction's
    # true cost must exceed it, proving a test that only asserted <= 96s
    # would have been fooled by a cheaper backoff class than the one that
    # actually governs the rate-limited path.
    assert clock["t"] > 96.0, (
        f"aggregate simulated wall time {clock['t']}s must exceed the old "
        "(wrong) 96s fixed-schedule ceiling -- otherwise this test cannot "
        "discriminate the rate-limited class from the cheaper one"
    )


# ---------------------------------------------------------------------------
# Regression (defect fix, 2026-08-06): rate-limit-aware 403/429 handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_api_get_retries_a_primary_rate_limited_403_honoring_reset_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 carrying GitHub's primary-rate-limit signal
    (``x-ratelimit-remaining: 0``) is transient and must be retried, backing
    off by the server-provided ``x-ratelimit-reset`` delta rather than a
    blind fixed schedule."""
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    fake_urlopen, calls = _scripted_urlopen(
        [
            _http_error_with_headers(
                403,
                "API rate limit exceeded",
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1005"},
            ),
            _FakeResponse(200, {"status": "behind"}),
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, body, _detail = _api_get("https://api.github.com/x", sleep=slept.append)

    assert status == 200
    assert body == {"status": "behind"}
    assert len(calls) == 2, (
        "a rate-limited 403 must be retried, not treated as definitive"
    )
    assert slept == [5.0], "backoff must follow the server-provided reset delta"


@pytest.mark.unit
def test_api_get_caps_rate_limit_backoff_instead_of_honoring_it_unbounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server-requested backoff longer than the cap must be clamped, never
    honored unbounded -- an hour-long primary-rate-limit reset must not turn
    one pin into an hour-long CI job."""
    fake_urlopen, _calls = _scripted_urlopen(
        [
            _http_error_with_headers(
                429, "rate limited", headers={"Retry-After": "3600"}
            ),
            _FakeResponse(200, {"status": "identical"}),
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    _api_get("https://api.github.com/x", sleep=slept.append)

    assert slept, "must still back off"
    assert slept[0] <= 30.0, f"rate-limit backoff must be capped, got {slept}"


@pytest.mark.unit
def test_api_get_does_not_retry_a_plain_403_even_with_unrelated_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 with headers present but no rate-limit signal is still
    definitive -- discrimination is by signal, not by header presence."""
    fake_urlopen, calls = _scripted_urlopen(
        [
            _http_error_with_headers(
                403, "Forbidden", headers={"x-github-request-id": "abc"}
            )
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, _body, _detail = _api_get("https://api.github.com/x", sleep=slept.append)

    assert status == 403
    assert len(calls) == 1, "a 403 without rate-limit headers must not be retried"
    assert slept == []


# ---------------------------------------------------------------------------
# Regression (defect fix, 2026-08-06): backoff-schedule/max-attempts coupling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_api_get_does_not_index_error_when_max_attempts_exceeds_backoff_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_API_RETRY_BACKOFF_SECONDS`` has 2 entries for the shipped
    ``_API_MAX_ATTEMPTS = 3``. Raising ``_API_MAX_ATTEMPTS`` without
    extending the tuple must degrade gracefully (reuse the last known
    backoff), never raise ``IndexError`` inside a required CI gate."""
    import scripts.ci.check_pin_reachability as module

    monkeypatch.setattr(module, "_API_MAX_ATTEMPTS", 5)
    fake_urlopen, calls = _scripted_urlopen(
        [TimeoutError("t")] * 5,
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, body, detail = module._api_get(
        "https://api.github.com/x", sleep=slept.append
    )

    assert status is None
    assert body is None
    assert "transport error" in detail
    assert len(calls) == 5
    assert len(slept) == 4, "sleeps between all 5 attempts, no IndexError"


# ---------------------------------------------------------------------------
# Regression (defect fix, 2026-08-06): http.client.HTTPException coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: http.client.IncompleteRead(b""),
        lambda: http.client.BadStatusLine("garbage"),
    ],
    ids=["IncompleteRead", "BadStatusLine"],
)
def test_api_get_retries_http_client_transport_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    make_exc: Callable[[], Exception],
) -> None:
    """``http.client.IncompleteRead`` (raised by ``response.read()`` on a
    truncated/chunked-abort body) and ``http.client.BadStatusLine``
    (re-raised by urllib's ``do_open`` from ``h.getresponse()``) are neither
    ``OSError`` nor any of the other pre-fix caught classes -- their MRO is
    ``(HTTPException, Exception, BaseException, object)``. Pre-fix, both
    escaped ``_api_get`` uncaught: zero retries, an uncaught traceback, on
    exactly the transient-transport class this gate exists to retry."""
    fake_urlopen, calls = _scripted_urlopen(
        [make_exc(), _FakeResponse(200, {"status": "behind"})]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []

    status, body, _detail = _api_get("https://api.github.com/x", sleep=slept.append)

    assert status == 200
    assert body == {"status": "behind"}
    assert len(calls) == 2, "a transport HTTPException must be retried, not raised"


# ---------------------------------------------------------------------------
# Regression (defect fix, 2026-08-06): negative Retry-After must not crash
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rate_limit_backoff_floors_a_negative_retry_after_at_zero() -> None:
    """A malformed/adversarial ``Retry-After: -5`` header must not produce a
    negative delay -- the sibling ``x-ratelimit-reset`` branch already
    guards with ``if delta > 0``; this branch must match."""
    from scripts.ci.check_pin_reachability import _rate_limit_backoff_seconds

    delay = _rate_limit_backoff_seconds(_headers({"Retry-After": "-5"}))

    assert delay is not None
    assert delay >= 0.0


@pytest.mark.unit
def test_api_get_does_not_crash_on_a_negative_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end, using the REAL ``time.sleep`` (no ``sleep=`` override) --
    exactly where the pre-fix bug crashed: ``effective_sleep(delay)`` at
    :640 is OUTSIDE any try/except, so ``time.sleep(-5.0)`` raises
    ``ValueError: sleep length must be non-negative`` unconditionally, an
    unguarded crash in a required CI gate. Fails closed only in the sense
    that a crash halts the job; it must instead retry with a floored,
    non-negative backoff."""
    fake_urlopen, calls = _scripted_urlopen(
        [
            _http_error_with_headers(
                429, "rate limited", headers={"Retry-After": "-5"}
            ),
            _FakeResponse(200, {"status": "identical"}),
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    status, body, _detail = _api_get("https://api.github.com/x")

    assert status == 200
    assert body == {"status": "identical"}
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Regression (workflow-level fix, 2026-08-07, PR #2679 comment 5211687650):
# the script-side deadline alone cannot bound this job -- a 38-run fleet
# measurement found job setup (job start -> script step start) had median
# 317s and max 613s under the old ``cache-enabled: "false"``, with 6/38 runs
# killed by ``timeout-minutes: 10`` (600s) DURING setup, before the script
# ever started. No value of ``_RUN_DEADLINE_SECONDS`` fixes a job whose setup
# alone can exceed the job timeout. This binding test parses ci.yml as DATA
# (not a copied literal) so a future edit to either the job's
# ``timeout-minutes`` or the script's ``_RUN_DEADLINE_SECONDS`` is checked
# against the real measured budget, in both directions:
#
#   * too small -- the job timeout no longer covers
#     measured-setup + script-deadline + worst-case-tail, so GitHub kills the
#     job mid-run exactly like the pre-fix incident;
#   * too large -- "raise timeout-minutes" gamed into an absurd value that
#     trivially satisfies the lower-bound arithmetic without being a real
#     fix, defeating a CI gate's fail-fast purpose.
# ---------------------------------------------------------------------------

# Measured worst-case job-start -> script-step-start wall time across the
# 38-run fleet (PR #2679 comment 5211687650), under the OLD
# ``cache-enabled: "false"``. Enabling the uv cache (the ci.yml half of this
# same fix) lowers the TYPICAL case well below this, but the GitHub Actions
# cache is a best-effort, evictable store (lockfile hash change, 7-day
# no-access eviction, cold first run) -- a cache MISS can still cost the full
# no-cache setup time, so this stays the conservative bound the arithmetic
# below is built on, independent of whether the cache actually hits on any
# given run.
_MEASURED_SETUP_BUDGET_SECONDS = 613.0


def _job_timeout_seconds(job_id: str) -> float:
    """Parse ``timeout-minutes`` for ``job_id`` out of the real ci.yml, as
    data -- never a copied/hardcoded literal, so this test tracks the live
    workflow file."""
    loaded = yaml.safe_load((WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8"))
    job = loaded["jobs"][job_id]
    timeout_minutes = job["timeout-minutes"]
    assert isinstance(timeout_minutes, int | float), (
        f"jobs.{job_id}.timeout-minutes must be a plain number, got {timeout_minutes!r}"
    )
    return float(timeout_minutes) * 60.0


def _assert_pin_reachability_timeout_budget_ok(timeout_seconds: float) -> None:
    """Both directions of the budget check.

    Lower bound: the job timeout must cover measured setup + the script's
    own run-wide deadline + the worst-case post-deadline tail (one
    fully-exhausted rate-limited call:
    ``_API_MAX_ATTEMPTS * _REQUEST_TIMEOUT_SECONDS +
    (_API_MAX_ATTEMPTS - 1) * _MAX_RATE_LIMIT_BACKOFF_SECONDS`` = 150s, per
    the derivation in ``check_pin_reachability.py``'s
    ``_RUN_DEADLINE_SECONDS`` comment).

    Upper bound: the job timeout must not be absurdly large. This job makes
    a handful of bounded, retried GitHub REST calls -- nothing legitimate
    ever needs more than 30 minutes. Without this half, "raise
    timeout-minutes" could be satisfied by setting it to something enormous,
    which trivially passes the lower-bound arithmetic without fixing
    anything and defeats the point of having a bounded CI gate at all.
    """
    from scripts.ci.check_pin_reachability import (
        _API_MAX_ATTEMPTS,
        _MAX_RATE_LIMIT_BACKOFF_SECONDS,
        _REQUEST_TIMEOUT_SECONDS,
        _RUN_DEADLINE_SECONDS,
    )

    worst_case_tail_seconds = (
        _API_MAX_ATTEMPTS * _REQUEST_TIMEOUT_SECONDS
        + (_API_MAX_ATTEMPTS - 1) * _MAX_RATE_LIMIT_BACKOFF_SECONDS
    )
    required_seconds = (
        _MEASURED_SETUP_BUDGET_SECONDS + _RUN_DEADLINE_SECONDS + worst_case_tail_seconds
    )
    assert required_seconds < timeout_seconds, (
        f"pin-reachability job timeout ({timeout_seconds:.0f}s) does not "
        f"cover measured setup ({_MEASURED_SETUP_BUDGET_SECONDS:.0f}s) + "
        f"script deadline ({_RUN_DEADLINE_SECONDS:.0f}s) + worst-case tail "
        f"({worst_case_tail_seconds:.0f}s) = {required_seconds:.0f}s -- "
        "raise jobs.pin-reachability.timeout-minutes in ci.yml"
    )

    upper_bound_seconds = 30 * 60.0
    assert timeout_seconds <= upper_bound_seconds, (
        f"pin-reachability job timeout ({timeout_seconds:.0f}s) exceeds the "
        f"sane upper bound ({upper_bound_seconds:.0f}s) for a job that makes "
        "a handful of bounded, retried GitHub REST calls -- an absurdly "
        "large timeout is not a real fix for the setup-budget problem and "
        "defeats CI's fail-fast purpose"
    )


@pytest.mark.unit
def test_pin_reachability_job_timeout_covers_measured_budget() -> None:
    """Binding test: the real ci.yml ``timeout-minutes`` for the
    ``pin-reachability`` job must cover measured setup + the script's own
    deadline + the worst-case post-deadline tail, with real margin, and must
    not be an absurdly large non-fix. Fails on the pre-fix ``timeout-minutes:
    10`` (600s < 913s required)."""
    _assert_pin_reachability_timeout_budget_ok(_job_timeout_seconds("pin-reachability"))


@pytest.mark.unit
def test_pin_reachability_job_timeout_budget_kills_too_small_mutant() -> None:
    """A too-small job timeout (480s, echoing the old broken ``480.0``
    magnitude that used to be ``_RUN_DEADLINE_SECONDS`` before it was found
    to overshoot the job timeout) must fail the lower-bound check -- this is
    the exact class of defect this test exists to catch: a job whose setup
    alone can exceed 480s gets killed mid-setup, exactly as measured pre-fix."""
    with pytest.raises(AssertionError, match="does not cover"):
        _assert_pin_reachability_timeout_budget_ok(480.0)


@pytest.mark.unit
def test_pin_reachability_job_timeout_budget_kills_absurd_upper_bound_mutant() -> None:
    """An absurdly large job timeout (99999s, ~27.8h) must fail the
    upper-bound sanity check even though it trivially satisfies the
    lower-bound arithmetic -- proving the test cannot be satisfied by gaming
    ``timeout-minutes`` to an unreasonable value instead of a real fix."""
    with pytest.raises(AssertionError, match="exceeds the sane upper bound"):
        _assert_pin_reachability_timeout_budget_ok(99999.0)

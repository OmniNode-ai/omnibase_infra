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

import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

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

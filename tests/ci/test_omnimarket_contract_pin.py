# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: the grants gate's derivation input must be committed repo state.

``Application Database Domain Enforcement (OMN-15361)`` derives the ``TABLE``
grants in ``src/omnibase_infra/topology/instances/*.yaml`` from omnimarket node
contracts. OMN-15703 made that derivation read **omnimarket's live ``dev``
HEAD** at CI-run time, to kill the OMN-15701 stale-pin failure mode.

That fixed staleness and created a worse coupling: because omnibase_infra ``dev``
enforces through the single required ``CI Summary`` umbrella (OMN-4497), an
omnimarket merge that adds a ``db_io.db_tables`` declaration turns the required
check red on **every open omnibase_infra PR simultaneously**, none of which
changed anything related and none of which can fix it. It happened three times
in 21 hours (OMN-17141, OMN-16930, OMN-17290).

The fix splits the two assertions the gate had fused:

* **internal consistency** — the checked-in grants are exactly the derivation of
  the contract set this repo has pinned. Deterministic, a pure function of
  committed state, and therefore still a hard per-PR gate.
* **freshness** — the pinned contract set is current with omnimarket ``dev``.
  A property of a *pair* of repos, not of any infra PR. It moves to a scheduled
  bot that opens a regeneration PR, so drift lands as an actionable PR instead of
  an org-wide ambush.

These tests pin that split down. They fail against the live-resolve
implementation.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_PIN_FILE = _ROOT / ".github" / "omnimarket-contract-pin.yaml"
_REFRESH_WORKFLOW = (
    _ROOT / ".github" / "workflows" / "omnimarket-contract-pin-refresh.yml"
)
_RESOLVER = _ROOT / "scripts" / "resolve_omnimarket_contract_pin.py"
_ADVANCE_GUARD = _ROOT / "scripts" / "ci" / "check_omnimarket_contract_pin_advance.py"

_GATE_JOB = "application-database-domain-enforcement"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _gate_job_text() -> str:
    """Return the enforcement job rendered back to text for shape assertions."""
    document = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    return yaml.safe_dump(document["jobs"][_GATE_JOB], sort_keys=True)


def _pin_document() -> dict[str, object]:
    return yaml.safe_load(_PIN_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The defect itself: a foreign repo's moving branch tip is a derivation input.
# ---------------------------------------------------------------------------


def test_grants_gate_does_not_resolve_a_foreign_moving_branch_tip() -> None:
    """OMN-17292 RED: the gate must not read omnimarket's live dev HEAD.

    This is the whole coupling. While the job resolves
    ``repos/OmniNode-ai/omnimarket/commits/dev`` at run time, the verdict of an
    omnibase_infra PR is a function of *when* it ran and of *what someone else
    merged in another repository* — so an already-green PR flips red with no
    change on this side and no action available to its author.
    """
    job = _gate_job_text()

    assert "commits/dev" not in job, (
        "the enforcement job still live-resolves omnimarket's dev tip; an "
        "omnimarket merge can red every open omnibase_infra PR (OMN-17292)"
    )
    assert "Live-resolve omnimarket dev HEAD" not in job, (
        "the OMN-15703 live-resolve step must be replaced by a committed pin"
    )


def test_grants_gate_derivation_input_is_a_committed_pin() -> None:
    """The checked-out omnimarket ref must come from committed repo state."""
    job = _gate_job_text()

    assert "scripts/resolve_omnimarket_contract_pin.py" in job, (
        "the enforcement job must resolve the omnimarket contract ref from the "
        "committed pin file, not from the network"
    )
    # The OMN-15703 trailer escape hatch (Omnimarket-Source-Ref) is preserved:
    # the pin is the default, a trailer still overrides it for cross-repo
    # co-development. OMN-17294 owns the trailer parser itself.
    assert "resolve_node_migration_source_ref.py" in job, (
        "the Omnimarket-Source-Ref trailer override must survive the pinning"
    )


# ---------------------------------------------------------------------------
# The pin is immutable, typed, and offline-resolvable.
# ---------------------------------------------------------------------------


def test_pre_setup_steps_run_on_stdlib_only() -> None:
    """The gate's pin steps run before ``Setup Python and uv``, so stdlib only.

    Found by review, not by CI: the first cut of this change ran
    ``python3 scripts/resolve_omnimarket_contract_pin.py`` (which imported
    ``yaml``) and ``uv run python .../check_omnimarket_contract_pin_advance.py``
    roughly seventy lines *above* this job's ``Setup Python and uv`` step. The
    runners' ambient ``python3`` has no PyYAML -- this same workflow's
    contract-compliance job carries an explicit
    ``python3 -m pip install --quiet --user pyyaml`` step precisely because of
    that -- and ``uv`` is not on PATH until the setup step runs. Either would
    have ``ImportError``/``command not found`` on **every** PR and taken the
    required check down repo-wide: the exact org-wide red this ticket removes,
    self-inflicted.

    So the constraint is structural, not stylistic, and is asserted here rather
    than left to reviewer memory.
    """
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    job_body = workflow.split(f"  {_GATE_JOB}:", maxsplit=1)[1]
    before_setup = job_body.split("- name: Setup Python and uv", maxsplit=1)[0]

    assert "uv run" not in before_setup, (
        "a step before `Setup Python and uv` invokes uv, which is not on PATH "
        "until that step runs"
    )

    # Both scripts reachable before setup must import stdlib only.
    third_party = ("yaml", "pydantic", "requests", "omnibase_infra", "omnibase_core")
    for script in (_RESOLVER, _ADVANCE_GUARD):
        assert script.name in before_setup, (
            f"{script.name} is expected to run before `Setup Python and uv`; if "
            "it moved after setup, relax this assertion deliberately"
        )
        source = script.read_text(encoding="utf-8")
        imports = {
            line.split()[1].split(".")[0]
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and len(line.split()) > 1
        }
        offenders = imports & set(third_party)
        assert not offenders, (
            f"{script.name} runs on the runners' bare python3 before "
            f"`Setup Python and uv` but imports {sorted(offenders)}; that "
            "ImportErrors on every PR and reds the required check repo-wide"
        )


def test_pin_file_declares_an_immutable_commit_sha() -> None:
    """A mutable ref (branch/tag) as the pin would reintroduce the coupling."""
    assert _PIN_FILE.is_file(), f"missing contract pin file: {_PIN_FILE}"
    document = _pin_document()

    ref = document["omnimarket_contract_ref"]
    assert isinstance(ref, str) and _SHA_RE.fullmatch(ref), (
        f"omnimarket_contract_ref must be a full 40-hex commit sha, got {ref!r}; "
        "a branch name would make the derivation non-deterministic again"
    )
    assert document["repository"] == "OmniNode-ai/omnimarket"


def test_pin_resolution_is_offline_and_deterministic() -> None:
    """The resolver must be a pure function of committed state.

    No token, no network, no ``GITHUB_EVENT_PATH`` dependence — running it twice
    in an empty environment must yield the pinned sha both times. This is what
    makes an unrelated PR's verdict immune to an omnimarket merge.
    """
    assert _RESOLVER.is_file(), f"missing resolver: {_RESOLVER}"
    expected = _pin_document()["omnimarket_contract_ref"]

    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, str(_RESOLVER)],
            capture_output=True,
            text=True,
            check=True,
            cwd=_ROOT,
            # Strip every credential and event-payload handle. If the resolver
            # still answers, it cannot be reaching GitHub or reading the PR
            # body -- which is precisely the property that decouples this
            # repo's verdict from omnimarket's branch tip.
            env={
                key: value
                for key, value in os.environ.items()
                if key
                not in {
                    "GH_TOKEN",
                    "GITHUB_TOKEN",
                    "GITHUB_EVENT_PATH",
                    "GITHUB_OUTPUT",
                }
            },
        )
        outputs.append(completed.stdout.strip())

    assert outputs == [expected, expected]

    source = _RESOLVER.read_text(encoding="utf-8")
    for forbidden in ("gh api", "requests", "urllib.request", "subprocess"):
        assert forbidden not in source, (
            f"resolver must not reach the network ({forbidden!r} present)"
        )


# ---------------------------------------------------------------------------
# AC3: the OMN-15701 staleness/reversion failure mode is explicitly handled.
# ---------------------------------------------------------------------------


_FIXTURES = _ROOT / "tests" / "fixtures" / "omn17292"
# The OMN-15701 pin pair, as GitHub reports it: 4637e625 (the stale pin ci.yml
# actually carried) is 61 commits behind and DIVERGED from 54356a83 (the pin
# that replaced it in omnibase_infra#2657).
COMPARE_INCIDENT = (
    _FIXTURES / "omnimarket-compare-54356a83-4637e625.gh-api.json.captured"
)
COMPARE_FORWARD = (
    _FIXTURES / "omnimarket-compare-2f123b4c-fd3e66c7.gh-api.json.captured"
)
COMPARE_BACKWARD = (
    _FIXTURES / "omnimarket-compare-fd3e66c7-2f123b4c.gh-api.json.captured"
)

COMPARE_INCIDENT_SHA256 = (
    "d63e211c0d3fc386e34a08a78a3a06ec4de1d2ced12d77da3a64bfb87f8fa041"
)
# These two compare captures are live-fetch-sanitized, not raw API bytes. A
# denylisted tenant slug appeared in commit-message text inside both payloads; it
# is replaced by an equal-length synthetic stand-in. The guard reads only url,
# status, ahead_by, and behind_by, so the replay verdict is unchanged.
COMPARE_FORWARD_SHA256 = (
    "364af3872b14d693296d8aa9946c5837dbf46aad103a8d7c65376bfa920fa211"
)
COMPARE_BACKWARD_SHA256 = (
    "942677ddf7531216a9de510f623a4f6b1334a97fa27dda8c5ea72decee3ea431"
)

STALE_PIN_OMN15701 = "4637e625c99ef17c190aa471a5e51b7f646c6dfd"
REPAIR_PIN_OMN15701 = "54356a831e3d8876c69373cac884a3df2a5653f7"
OMNIMARKET_OLDER = "2f123b4c01eabd2c51f7d703491e9cdf36f89bcd"
OMNIMARKET_NEWER = "fd3e66c71ccfd4f7383904baa19e5bd700993a05"


def _resolver_module() -> Any:
    """Import the real resolver by path, exactly as CI invokes it."""
    spec = importlib.util.spec_from_file_location(
        "resolve_omnimarket_contract_pin_omn17292", _RESOLVER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _advance_guard() -> Any:
    """Import the real guard by path, exactly as CI invokes it."""
    spec = importlib.util.spec_from_file_location(
        "check_omnimarket_contract_pin_advance_omn17292", _ADVANCE_GUARD
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_compare_captures_are_unmodified() -> None:
    """R1: these are the registered compare captures, byte for byte."""
    for path, expected in (
        (COMPARE_INCIDENT, COMPARE_INCIDENT_SHA256),
        (COMPARE_FORWARD, COMPARE_FORWARD_SHA256),
        (COMPARE_BACKWARD, COMPARE_BACKWARD_SHA256),
    ):
        assert path.is_file(), f"missing capture: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, (
            f"{path} was edited after capture; it no longer proves what happened"
        )


def test_the_omn15701_stale_pin_is_refused_by_the_real_guard() -> None:
    """OMN-15701 replay: the pin that reverted 8 live grants must be refused.

    ``4637e625`` is the omnimarket ref ``ci.yml`` pinned while
    omnibase_infra#2632 regenerated the TABLE grants and silently dropped
    #2634's ``tenant_projection_writer`` entries. GitHub's own answer for that
    pair is ``diverged`` -- 61 commits behind -- which is precisely the fact no
    guard was asking for at the time.
    """
    ok, message = _advance_guard().check_advance(
        COMPARE_INCIDENT, REPAIR_PIN_OMN15701, STALE_PIN_OMN15701
    )
    assert ok is False, "the guard accepted the OMN-15701 stale pin"
    assert "diverged" in message.lower()
    assert "61" in message


def test_a_backwards_pin_move_is_refused() -> None:
    """The other reject direction, on a real ancestor pair from omnimarket dev."""
    ok, message = _advance_guard().check_advance(
        COMPARE_BACKWARD, OMNIMARKET_NEWER, OMNIMARKET_OLDER
    )
    assert ok is False
    assert "backwards" in message.lower()


def test_a_genuine_forward_advance_is_accepted() -> None:
    """The discriminator: an accept-only or reject-only guard is not a guard.

    Without this, a blanket-reject implementation would satisfy every rejection
    test above while making the refresh bot permanently unable to advance the
    pin -- which would silently reintroduce the staleness OMN-15703 removed.
    """
    ok, message = _advance_guard().check_advance(
        COMPARE_FORWARD, OMNIMARKET_OLDER, OMNIMARKET_NEWER
    )
    assert ok is True, message
    assert "forward" in message.lower()


def test_a_capture_for_a_different_pair_is_not_accepted_as_proof() -> None:
    """A compare payload only proves the advance it actually describes."""
    ok, message = _advance_guard().check_advance(
        COMPARE_FORWARD, REPAIR_PIN_OMN15701, STALE_PIN_OMN15701
    )
    assert ok is False
    assert "different pair" in message


def test_pin_advance_guard_is_a_pure_function_of_bytes() -> None:
    """No network, no git subprocess -- that is what makes it replayable."""
    module = _ADVANCE_GUARD.read_text(encoding="utf-8")
    for forbidden in ("subprocess", "gh api", "urllib.request", "requests"):
        assert forbidden not in module, (
            f"the guard must stay a pure function of the compare payload "
            f"({forbidden!r} present)"
        )


def test_omnimarket_dev_moving_past_the_pin_does_not_move_the_derivation_input(
    tmp_path: Path,
) -> None:
    """OMN-17290 replay: upstream advanced, and this repo's input did not.

    The captured compare is GitHub's own answer that omnimarket ``dev`` moved
    **2 commits past** ``2f123b4c`` -- the very commit omnibase_infra#3061
    regenerated the ``open_obligations`` grant against. Under the OMN-15703
    live-resolve, movement of exactly this kind is what recomputed the
    derivation input underneath every open omnibase_infra PR and turned the
    required ``CI Summary`` context red on all of them at once, three times in
    21 hours.

    With the pin, that same movement changes nothing here: the resolver is a
    function of committed state, so an unrelated PR's verdict is unaffected.
    Direction is ``false_red`` -- the old behaviour failed a good input -- so
    ``test_the_resolver_refuses_a_mutable_ref`` is the required discriminator:
    an accept-only proof cannot tell a working resolver from one stuck open.
    """
    payload = json.loads(COMPARE_FORWARD.read_text(encoding="utf-8"))
    assert payload["status"] == "ahead"
    assert payload["ahead_by"] >= 1, "the capture must show upstream actually moved"

    resolver = _resolver_module()
    # Upstream moving is, to this repo, not an event at all: the answer is the
    # committed value, both times.
    assert (
        resolver.resolve_pin()
        == resolver.resolve_pin()
        == _pin_document()["omnimarket_contract_ref"]
    )

    # And the answer tracks the FILE, not omnimarket dev. Pointed at a pin file
    # naming the older commit in that same capture, the resolver returns the
    # older commit -- even though dev has demonstrably moved past it. Under
    # live-resolve there was no such file to consult and no way to hold still.
    older = _pin_document()
    older["omnimarket_contract_ref"] = payload["base_commit"]["sha"]
    older_pin = tmp_path / "older-pin.yaml"
    older_pin.write_text(yaml.safe_dump(older), encoding="utf-8")
    assert resolver.resolve_pin(older_pin) == payload["base_commit"]["sha"]


def test_the_resolver_refuses_a_mutable_ref(tmp_path: Path) -> None:
    """Discriminator for the false_red case above.

    A resolver that returned whatever it found would satisfy the accept proof
    while quietly permitting a branch name back into the derivation input,
    which is the coupling this ticket removes. It must fail closed instead.
    """
    resolver = _resolver_module()
    mutable = tmp_path / "pin.yaml"
    mutable.write_text(
        "repository: OmniNode-ai/omnimarket\nomnimarket_contract_ref: dev\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="40-hex"):
        resolver.resolve_pin(mutable)


# ---------------------------------------------------------------------------
# AC1/AC3: freshness does not disappear, it moves to an actionable surface.
# ---------------------------------------------------------------------------


def test_a_scheduled_bot_advances_the_pin_and_regenerates_in_one_commit() -> None:
    """Pinning without auto-advance would be a straight return to OMN-15701.

    The bot is the load-bearing half of this design: it is what makes the pin
    unable to sit stale *silently*. It must advance the pin AND regenerate both
    the topology instances and the rendered catalogs in the same commit, so the
    pin and the grants it derives can never disagree on ``dev``.
    """
    assert _REFRESH_WORKFLOW.is_file(), f"missing refresh bot: {_REFRESH_WORKFLOW}"
    document = yaml.safe_load(_REFRESH_WORKFLOW.read_text(encoding="utf-8"))

    # `on:` parses as the boolean True under YAML 1.1.
    triggers = document.get("on", document.get(True))
    assert "schedule" in triggers, "the pin must advance on a schedule"
    assert "workflow_dispatch" in triggers, "and be advanceable on demand"

    body = _REFRESH_WORKFLOW.read_text(encoding="utf-8")
    assert "generate_application_database_table_grants.py" in body
    assert "--write" in body
    # The rendered docker catalogs must move with the instances, or the
    # topology unit tests fail closed on the bot's own PR. OMN-17292 folded
    # that render into `--write` so it cannot be forgotten; the bot must stage
    # the catalogs it produces.
    assert "docker/catalog/database-topology" in body
    generator = (
        _ROOT / "scripts" / "generate_application_database_table_grants.py"
    ).read_text(encoding="utf-8")
    assert "write_database_projection" in generator, (
        "--write must re-render the derived catalogs itself rather than "
        "printing a reminder that a second command has to be remembered"
    )
    assert "gh pr create" in body, "drift must land as an actionable PR"


def test_refresh_bot_enforces_forward_only_advance() -> None:
    """The bot is the only thing that should routinely move the pin."""
    body = _REFRESH_WORKFLOW.read_text(encoding="utf-8")
    assert "check_omnimarket_contract_pin_advance.py" in body, (
        "the bot must run the forward-only guard before opening its PR"
    )


# ---------------------------------------------------------------------------
# AC2: replay the OMN-17290 shape end to end.
# ---------------------------------------------------------------------------

_OMNIMARKET_CONTRACTS = (
    _ROOT / ".proof-dependencies" / "omnimarket" / "src" / "omnimarket" / "nodes"
)

# A node contract of exactly the shape that produced OMN-17141 / OMN-16930 /
# OMN-17290: a new projection node declaring a db_io table that omnibase_infra
# has no derived grant for yet.
_UPSTREAM_ADDITION = """\
name: node_projection_omn17292_replay
contract_version: 1.0.0
node_type: REDUCER_GENERIC
node_version: 0.1.0
db_io:
  db_tables:
    - name: omn17292_replay_projection
      database_ref: application
      schema: tenant
      migration: 0001_create_omn17292_replay_projection.sql
      access: write
      role: instruction_eval_aggregate
"""


def _mirror_contracts(destination: Path) -> None:
    """Copy just the contract.yaml files (the derivation reads nothing else)."""
    for contract in _OMNIMARKET_CONTRACTS.rglob("contract.yaml"):
        target = destination / contract.relative_to(_OMNIMARKET_CONTRACTS)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contract.read_text(encoding="utf-8"), encoding="utf-8")


def _check_against(contracts_root: Path) -> int:
    return subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "generate_application_database_table_grants.py"),
            "--contracts-root",
            str(contracts_root),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
    ).returncode


@pytest.mark.skipif(
    not _OMNIMARKET_CONTRACTS.is_dir(),
    reason="requires the cross-repo omnimarket checkout (present in the enforcement job)",
)
def test_an_upstream_contract_addition_cannot_red_an_unrelated_infra_pr(
    tmp_path: Path,
) -> None:
    """AC2, proven rather than asserted.

    Replays OMN-17290: an omnimarket merge adds a ``db_io.db_tables``
    declaration and **no omnibase_infra file changes at all**.

    * Derived from a moved upstream tip, the gate goes red -- that is the
      defect, and it reproduced here on demand. Under the OMN-15703
      live-resolve this red landed on every open infra PR at once.
    * Derived from the committed pin, the same repo state stays green, because
      the pin did not move when omnimarket did.
    """
    pinned = tmp_path / "pinned"
    moved = tmp_path / "moved"
    _mirror_contracts(pinned)
    _mirror_contracts(moved)

    replay_node = moved / "node_projection_omn17292_replay"
    replay_node.mkdir(parents=True, exist_ok=True)
    (replay_node / "contract.yaml").write_text(_UPSTREAM_ADDITION, encoding="utf-8")

    # The upstream merge, seen live: red. (This is the failure mode, reproduced.)
    assert _check_against(moved) == 1, (
        "the replay contract did not produce drift; the AC2 proof is vacuous "
        "unless this direction actually fails"
    )

    # The same infra tree, derived from the pin: green.
    assert _check_against(pinned) == 0, (
        "the pinned contract set must keep the checked-in grants in sync; if "
        "this fails the pin needs advancing, which is the bot's job"
    )

    # And the pin is what the enforcement job actually feeds the derivation,
    # regardless of what omnimarket dev did in the meantime.
    assert (
        _pin_document()["omnimarket_contract_ref"]
        == subprocess.run(
            [sys.executable, str(_RESOLVER)],
            capture_output=True,
            text=True,
            check=True,
            cwd=_ROOT,
        ).stdout.strip()
    )


# ---------------------------------------------------------------------------
# The bootstrap case: the PR that INTRODUCES the pin has no base pin to
# compare against.
# ---------------------------------------------------------------------------


def _forward_only_guard_script() -> str:
    """Return the enforcement job's forward-only guard step, as it will run."""
    document = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    for step in document["jobs"][_GATE_JOB]["steps"]:
        if "only moves forward" in (step.get("name") or ""):
            return str(step["run"])
    raise AssertionError("the forward-only guard step is missing from the gate job")


def test_the_pin_introducing_pr_does_not_trip_its_own_forward_only_guard(
    tmp_path: Path,
) -> None:
    """OMN-17292 RED: a base revision with no pin file is a bootstrap, not a move.

    Found in CI on this change's own first PR (omnibase_infra#3075), which is
    the third instance of the self-inflicted class the other two review
    findings on this branch already fixed. The guard reads the base pin with::

        git show "${PIN_BASE_REVISION}:${PIN_FILE}"

    under ``set -euo pipefail``. On the PR that *creates* the pin file that path
    does not exist at the base revision, so git exits 128::

        fatal: path '.github/omnimarket-contract-pin.yaml' exists on disk,
               but not in '86421b5012b2bf0252fdf8e3069d6b2650bd6681'

    and the required check fails on the very PR that removes the org-wide red.
    A creation has no predecessor, so there is no backwards direction to guard:
    the step must recognise the bootstrap and pass, rather than fail closed on a
    comparison that cannot be formed.

    Executed against the real step script in a synthetic repository -- a text
    assertion would not prove the shell actually survives it.
    """
    repo = tmp_path / "repo"
    (repo / ".github").mkdir(parents=True)
    run = functools.partial(subprocess.run, cwd=repo, check=True, capture_output=True)

    run(["git", "init", "-q", "-b", "main"])
    run(["git", "config", "user.email", "ci@omninode.ai"])
    run(["git", "config", "user.name", "ci"])

    # Base revision: no pin file at all -- the state of dev before this change.
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    run(["git", "add", "-A"])
    run(["git", "commit", "-qm", "base without a pin"])
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # HEAD: the pin file is introduced.
    head_pin = "fd3e66c71ccfd4f7383904baa19e5bd700993a05"
    (repo / ".github" / "omnimarket-contract-pin.yaml").write_text(
        f"repository: OmniNode-ai/omnimarket\nomnimarket_contract_ref: {head_pin}\n",
        encoding="utf-8",
    )
    run(["git", "add", "-A"])
    run(["git", "commit", "-qm", "introduce the contract pin"])

    completed = subprocess.run(
        ["bash", "-c", _forward_only_guard_script()],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PIN_BASE_REVISION": base_sha, "HEAD_PIN": head_pin},
    )

    assert completed.returncode == 0, (
        "the PR that introduces the contract pin tripped its own forward-only "
        "guard -- a creation has no predecessor to move backwards from. "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )

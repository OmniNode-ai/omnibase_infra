# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A probe placed by an expression is resolved the way the runner resolves it (OMN-19412).

omninode_infra#1725 (OMN-17427, operator rulings 2026-09-28T01:58:06Z and
01:58:17Z, "checks must not rely on a specific machine") moved C13, C14 and C29
off the literal ``omnipc2-customer`` label onto
``${{ fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON) }}``, and omnibase_infra#4233
does the same for the lab canaries with ``vars.LAB_PROBE_RUNS_ON_JSON`` and a
``needs.<job>.outputs`` placement. The check read only literal labels, so every
``lane_source: runs_on`` entry over such a job failed with "a job is placed by
an expression" and turned the required CI Summary red on every omnibase_infra
pull request, and a scheduled job placed on the customer machine through a
variable was invisible to the unlisted-probe direction.

The check now resolves ``fromJSON(vars.NAME)`` (and ``vars.NAME``, each with an
optional ``|| '<literal>'`` fallback) from the value committed for that
repository in ``config/runner_routing_policy.yaml``. Anything it cannot resolve
fails naming the job and the expression. Never a pass.

The workflow bytes here are captured, not typed: C13 at the omninode_infra#1725
merge commit, and the C11 and dev-lane-liveness workflows at the
omnibase_infra#4233 head.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from scripts.ci import check_lab_probe_windows as plw

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests/fixtures/omn19412"
# omninode_infra 14972e07c96041933d20aec101f962710cb79fcc (#1725 merge).
C13_1725 = FIXTURES / "c13-customer-local-delegation.yml.captured"
# omnibase_infra 957640cebce0bd671ca3e3bd2cbc429303b0034e (#4233 head).
C11_4233 = FIXTURES / "chain-canary-c11-negative-paths.pr4233.yml.captured"
LIVENESS_4233 = FIXTURES / "dev-lane-liveness.pr4233.yml.captured"

C13_PATH = ".github/workflows/c13-customer-local-delegation.yml"
C11_PATH = ".github/workflows/chain-canary-c11-negative-paths.yml"
LIVENESS_PATH = ".github/workflows/dev-lane-liveness.yml"

VERIFY_POOL = '["self-hosted","omnibase-verify"]'


def _root(tmp_path: Path, repo: str, files: dict[str, Path | str]) -> Path:
    root = tmp_path / repo
    for rel, source in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, Path):
            target.write_bytes(source.read_bytes())
        else:
            target.write_text(source, encoding="utf-8")
    return root


def _window(
    probe_id: str,
    repo: str,
    workflow: str,
    cron: str,
    lane: str,
    lane_source: str,
    minutes: int,
) -> plw.ProbeWindow:
    return plw.ProbeWindow(
        probe_id, probe_id, repo, workflow, (cron,), lane, lane_source, minutes
    )


def _c13(lane: str) -> plw.ProbeWindow:
    return _window(
        "C13", "omninode_infra", C13_PATH, "29 4,16 * * *", lane, "runs_on", 45
    )


def _placements(repo: str, **values: str | None) -> plw.CommittedPlacements:
    return plw.CommittedPlacements({repo: values})


# --------------------------------------------------------------------------- #
# omninode_infra#1725's shape: fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON).
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_1725_c13_resolves_from_the_committed_repository_value(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    variables = _placements("omninode_infra", CUSTOMER_MACHINE_RUNS_ON_JSON=VERIFY_POOL)
    assert (
        plw.check([_c13("omnibase-verify")], {"omninode_infra": root}, variables) == []
    )


@pytest.mark.unit
def test_1725_c13_on_the_old_customer_label_is_a_lane_mismatch(
    tmp_path: Path,
) -> None:
    """The probe really moved: the window file must say which pool it now holds."""
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    variables = _placements("omninode_infra", CUSTOMER_MACHINE_RUNS_ON_JSON=VERIFY_POOL)
    errors = plw.check([_c13("omnipc2-customer")], {"omninode_infra": root}, variables)
    assert len(errors) == 1, errors
    assert "lane mismatch" in errors[0]
    assert "'omnipc2-customer'" in errors[0]
    assert "omnibase-verify,self-hosted" in errors[0]


@pytest.mark.unit
def test_1725_c13_with_the_variable_unset_fails_naming_the_job(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    errors = plw.check(
        [_c13("omnibase-verify")],
        {"omninode_infra": root},
        _placements("omninode_infra", CUSTOMER_MACHINE_RUNS_ON_JSON=None),
    )
    assert len(errors) == 1, errors
    assert "c13-customer-local" in errors[0]
    assert "CUSTOMER_MACHINE_RUNS_ON_JSON" in errors[0]
    assert "committed as unset" in errors[0]


@pytest.mark.unit
def test_1725_c13_with_no_committed_placements_fails_naming_the_job(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    errors = plw.check([_c13("omnibase-verify")], {"omninode_infra": root})
    assert len(errors) == 1, errors
    assert "c13-customer-local" in errors[0]
    assert "no committed placement values were given for omninode_infra" in errors[0]


@pytest.mark.unit
def test_1725_c13_with_an_undeclared_variable_fails_naming_the_job(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    errors = plw.check(
        [_c13("omnibase-verify")],
        {"omninode_infra": root},
        _placements("omninode_infra"),
    )
    assert len(errors) == 1, errors
    assert "c13-customer-local" in errors[0]
    assert "CUSTOMER_MACHINE_RUNS_ON_JSON" in errors[0]
    assert "declares no value" in errors[0]


@pytest.mark.unit
def test_a_variable_declared_null_uses_the_expression_fallback(
    tmp_path: Path,
) -> None:
    workflow = C13_1725.read_text(encoding="utf-8").replace(
        "vars.CUSTOMER_MACHINE_RUNS_ON_JSON",
        'vars.CUSTOMER_MACHINE_RUNS_ON_JSON || \'["self-hosted","omnibase-verify"]\'',
    )
    root = _root(tmp_path, "omninode_infra", {C13_PATH: workflow})
    variables = _placements("omninode_infra", CUSTOMER_MACHINE_RUNS_ON_JSON=None)
    assert (
        plw.check([_c13("omnibase-verify")], {"omninode_infra": root}, variables) == []
    )


@pytest.mark.unit
def test_a_variable_declared_null_without_a_fallback_fails_naming_the_job(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    variables = _placements("omninode_infra", CUSTOMER_MACHINE_RUNS_ON_JSON=None)
    errors = plw.check([_c13("omnibase-verify")], {"omninode_infra": root}, variables)
    assert len(errors) == 1, errors
    assert "c13-customer-local" in errors[0]
    assert "committed as unset" in errors[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "why"),
    [
        ("self-hosted", "is not JSON"),
        ('{"labels": ["a"]}', "is not a label or a list of labels"),
        ("[]", "is not a label or a list of labels"),
    ],
)
def test_a_variable_that_is_not_a_label_list_fails_loudly(
    tmp_path: Path, value: str, why: str
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    variables = _placements("omninode_infra", CUSTOMER_MACHINE_RUNS_ON_JSON=value)
    errors = plw.check([_c13("omnibase-verify")], {"omninode_infra": root}, variables)
    assert len(errors) == 1, errors
    assert "c13-customer-local" in errors[0]
    assert why in errors[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "runs_on",
    [
        '${{ fromJSON(vars.POOL_JSON || \'["self-hosted","omnibase-verify"]\') }}',
        "${{ vars.POOL_LABEL || 'omnibase-verify' }}",
    ],
)
def test_the_literal_fallback_answers_an_unset_variable(
    tmp_path: Path, runs_on: str
) -> None:
    workflow = textwrap.dedent(
        f"""\
        name: probe
        on:
          schedule:
            - cron: '29 4,16 * * *'
        jobs:
          probe:
            runs-on: {runs_on}
            timeout-minutes: 45
            steps:
              - run: echo probe
        """
    )
    root = _root(tmp_path, "omninode_infra", {C13_PATH: workflow})
    assert (
        plw.check(
            [_c13("omnibase-verify")],
            {"omninode_infra": root},
            _placements(
                "omninode_infra",
                POOL_JSON=None,
                POOL_LABEL=None,
            ),
        )
        == []
    )


# --------------------------------------------------------------------------- #
# omnibase_infra#4233's shapes: vars.LAB_PROBE_RUNS_ON_JSON and needs outputs.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_4233_c11_is_still_a_listed_dev_lane_probe(tmp_path: Path) -> None:
    root = _root(tmp_path, "omnibase_infra", {C11_PATH: C11_4233})
    c11 = _window(
        "C11", "omnibase_infra", C11_PATH, "17 2,8,14,20 * * *", "dev", "job_name", 10
    )
    variables = _placements("omnibase_infra", LAB_PROBE_RUNS_ON_JSON=VERIFY_POOL)
    assert plw.check([c11], {"omnibase_infra": root}, variables) == []
    unlisted = plw.check([], {"omnibase_infra": root}, variables)
    assert len(unlisted) == 1, unlisted
    assert unlisted[0].startswith(f"unlisted probe omnibase_infra:{C11_PATH}")


@pytest.mark.unit
def test_4233_c11_as_a_runs_on_entry_resolves_the_probe_pool(tmp_path: Path) -> None:
    root = _root(tmp_path, "omnibase_infra", {C11_PATH: C11_4233})
    variables = _placements("omnibase_infra", LAB_PROBE_RUNS_ON_JSON=VERIFY_POOL)
    entry = _window(
        "C11",
        "omnibase_infra",
        C11_PATH,
        "17 2,8,14,20 * * *",
        "omnibase-verify",
        "runs_on",
        10,
    )
    assert plw.check([entry], {"omnibase_infra": root}, variables) == []


@pytest.mark.unit
def test_4233_a_job_placed_by_a_needs_output_fails_naming_the_job(
    tmp_path: Path,
) -> None:
    """A placement only a previous job's run can decide is not resolvable before
    the run, so a runs_on entry over it is refused, never guessed."""
    root = _root(tmp_path, "omnibase_infra", {LIVENESS_PATH: LIVENESS_4233})
    variables = _placements("omnibase_infra", LAB_PROBE_RUNS_ON_JSON=VERIFY_POOL)
    entry = _window(
        "liveness",
        "omnibase_infra",
        LIVENESS_PATH,
        "7,22,37,52 * * * *",
        "omnibase-verify",
        "runs_on",
        20,
    )
    errors = plw.check([entry], {"omnibase_infra": root}, variables)
    assert len(errors) == 1, errors
    assert "'dev-lane-liveness'" in errors[0]
    assert "needs.resolve-lane.outputs.docker_runs_on" in errors[0]


# --------------------------------------------------------------------------- #
# The unlisted direction sees a customer-machine job placed by a variable.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.live_contact("tests/fixtures/omn19412/c13-probe-window-contact.json")
@pytest.mark.parametrize("pool", [VERIFY_POOL, '["self-hosted","customer-proof"]'])
def test_unlisted_1725_c13_is_refused_after_its_pool_moves(
    tmp_path: Path, pool: str, recorded_response: dict[str, object]
) -> None:
    provenance = recorded_response["_provenance"]
    assert isinstance(provenance, dict)
    source = REPO_ROOT / provenance["source_file"]
    assert source == C13_1725
    root = _root(tmp_path, "omninode_infra", {C13_PATH: source})
    variables = _placements("omninode_infra", CUSTOMER_MACHINE_RUNS_ON_JSON=pool)
    lane = next(label for label in json.loads(pool) if label != "self-hosted")
    assert plw.check([_c13(lane)], {"omninode_infra": root}, variables) == []
    errors = plw.check([], {"omninode_infra": root}, variables)
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"unlisted probe omninode_infra:{C13_PATH}")
    assert "29 4,16 * * *" in errors[0]
    assert lane in errors[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("runs_on", "value"),
    [
        ("${{ fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON) }}", VERIFY_POOL),
        ("${{ vars.CUSTOMER_MACHINE_RUNS_ON_JSON }}", "omnibase-verify"),
        (
            '${{ fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON || \'["self-hosted","omnibase-verify"]\') }}',
            None,
        ),
    ],
)
def test_unlisted_probe_pool_workflow_without_a_lane_title_is_refused(
    tmp_path: Path, runs_on: str, value: str | None
) -> None:
    workflow = NOT_A_PROBE.replace("runs-on: ubuntu-latest", f"runs-on: {runs_on}")
    root = _root(tmp_path, "omnibase_infra", {".github/workflows/probe.yml": workflow})
    variables = _placements(
        "omnibase_infra",
        CUSTOMER_MACHINE_RUNS_ON_JSON=value,
    )
    errors = plw.check([], {"omnibase_infra": root}, variables)
    assert len(errors) == 1, errors
    assert errors[0].startswith(
        "unlisted probe omnibase_infra:.github/workflows/probe.yml"
    )
    assert "omnibase-verify" in errors[0]


@pytest.mark.unit
@pytest.mark.parametrize("value", ["not JSON", None])
def test_unlisted_probe_pool_with_an_unreadable_value_fails_naming_the_job(
    tmp_path: Path, value: str | None
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    errors = plw.check(
        [],
        {"omninode_infra": root},
        _placements("omninode_infra", CUSTOMER_MACHINE_RUNS_ON_JSON=value),
    )
    assert len(errors) == 1, errors
    assert "cannot be classified" in errors[0]
    assert "job 'c13-customer-local'" in errors[0]
    assert "CUSTOMER_MACHINE_RUNS_ON_JSON" in errors[0]


@pytest.mark.unit
@pytest.mark.parametrize("name", ["LANE_CENSUS_RUNS_ON_JSON", "LAB_PROBE_RUNS_ON_JSON"])
def test_unlisted_nonprobe_variable_on_the_same_verify_pool_is_not_a_probe(
    tmp_path: Path, name: str
) -> None:
    workflow = NOT_A_PROBE.replace(
        "runs-on: ubuntu-latest", "runs-on: ${{ fromJSON(vars." + name + ") }}"
    )
    root = _root(tmp_path, "omnibase_infra", {".github/workflows/audit.yml": workflow})
    variables = _placements("omnibase_infra", **{name: VERIFY_POOL})
    assert plw.check([], {"omnibase_infra": root}, variables) == []


CUSTOMER_BY_VARIABLE = textwrap.dedent(
    """\
    name: customer runner smoke
    on:
      schedule:
        - cron: '5 3 * * *'
    jobs:
      smoke:
        runs-on: ${{ fromJSON(vars.CUSTOMER_RUNNER_SMOKE_RUNS_ON_JSON) }}
        timeout-minutes: 10
        steps:
          - run: echo smoke
    """
)


@pytest.mark.unit
def test_a_customer_machine_job_placed_by_a_variable_is_an_unlisted_probe(
    tmp_path: Path,
) -> None:
    root = _root(
        tmp_path,
        "omninode_infra",
        {".github/workflows/smoke.yml": CUSTOMER_BY_VARIABLE},
    )
    variables = _placements(
        "omninode_infra",
        CUSTOMER_RUNNER_SMOKE_RUNS_ON_JSON='["omnipc2-customer"]',
    )
    errors = plw.check([], {"omninode_infra": root}, variables)
    assert len(errors) == 1, errors
    assert "unlisted probe omninode_infra:.github/workflows/smoke.yml" in errors[0]
    assert "omnipc2-customer" in errors[0]


@pytest.mark.unit
def test_an_unlisted_scheduled_workflow_with_an_undeclared_variable_is_an_error(
    tmp_path: Path,
) -> None:
    root = _root(
        tmp_path,
        "omninode_infra",
        {".github/workflows/smoke.yml": CUSTOMER_BY_VARIABLE},
    )
    errors = plw.check(
        [],
        {"omninode_infra": root},
        _placements("omninode_infra"),
    )
    assert len(errors) == 1, errors
    assert errors[0].startswith(
        "omninode_infra:.github/workflows/smoke.yml: cannot be classified:"
    )
    assert "job 'smoke'" in errors[0]
    assert "declares no value" in errors[0]


# --------------------------------------------------------------------------- #
# The CLI reads the committed placement policy and validates its shape.
# --------------------------------------------------------------------------- #


def _policy_file(
    tmp_path: Path,
    declared: dict[str, dict[str, str | None]],
    name: str = "runner-routing-policy.yaml",
) -> Path:
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump({plw.PLACEMENT_KEY: declared}, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _valid_policy_values() -> dict[str, dict[str, str | None]]:
    return {
        "omnibase_infra": {"UNUSED": None},
        "omninode_infra": {"CUSTOMER_MACHINE_RUNS_ON_JSON": VERIFY_POOL},
        "omnimarket": {"UNUSED": None},
    }


WINDOWS = textwrap.dedent(
    """\
    schema_version: 1
    probes:
      - id: C13
        name: customer-local delegation
        repo: omninode_infra
        workflow: .github/workflows/c13-customer-local-delegation.yml
        cron: ['29 4,16 * * *']
        lane: omnibase-verify
        lane_source: runs_on
        max_duration_minutes: 45
    """
)

NOT_A_PROBE = textwrap.dedent(
    """\
    name: audit
    on:
      schedule:
        - cron: '23 */4 * * *'
    jobs:
      audit:
        runs-on: ubuntu-latest
        timeout-minutes: 5
        steps:
          - run: echo audit
    """
)


def _cli_argv(tmp_path: Path, policy: Path | None = None) -> list[str]:
    windows = tmp_path / "w.yaml"
    windows.write_text(WINDOWS, encoding="utf-8")
    roots = {
        "omnibase_infra": _root(
            tmp_path, "omnibase_infra", {".github/workflows/a.yml": NOT_A_PROBE}
        ),
        "omninode_infra": _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725}),
        "omnimarket": _root(
            tmp_path, "omnimarket", {".github/workflows/a.yml": NOT_A_PROBE}
        ),
    }
    if policy is None:
        policy = _policy_file(tmp_path, _valid_policy_values())
    argv = ["--windows", str(windows)]
    argv += [f"--root={name}={path}" for name, path in roots.items()]
    argv += ["--placement-policy", str(policy)]
    return argv


@pytest.mark.unit
def test_cli_resolves_the_1725_shape_from_the_committed_policy(tmp_path: Path) -> None:
    assert plw.main(_cli_argv(tmp_path)) == plw.EXIT_OK


@pytest.mark.unit
@pytest.mark.parametrize("repo", plw.REPOS)
def test_cli_refuses_a_policy_missing_a_probe_repository(
    tmp_path: Path, repo: str
) -> None:
    declared = _valid_policy_values()
    del declared[repo]
    policy = _policy_file(tmp_path, declared)
    assert plw.main(_cli_argv(tmp_path, policy)) == plw.EXIT_USAGE


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        "- not\n- a mapping\n",
        "some_other_key: {}\n",
        textwrap.dedent(
            """\
            probe_placement_variables:
              omnibase_infra: {}
              omninode_infra:
                X: value
              omnimarket:
                X: value
            """
        ),
        textwrap.dedent(
            """\
            probe_placement_variables:
              omnibase_infra:
                X: 3
              omninode_infra:
                X: value
              omnimarket:
                X: value
            """
        ),
    ],
)
def test_cli_refuses_a_malformed_placement_policy(tmp_path: Path, payload: str) -> None:
    bad = tmp_path / "bad-policy.yaml"
    bad.write_text(payload, encoding="utf-8")
    assert plw.main(_cli_argv(tmp_path, bad)) == plw.EXIT_USAGE


@pytest.mark.unit
def test_the_real_policy_declares_every_probe_repo_and_the_verify_pool() -> None:
    placements = plw.load_placements(REPO_ROOT / "config/runner_routing_policy.yaml")
    representative_names = {
        "omnibase_infra": "LAB_PROBE_RUNS_ON_JSON",
        "omninode_infra": "CUSTOMER_MACHINE_RUNS_ON_JSON",
        "omnimarket": "OMNI_OCC_AUTOBIND_RUNS_ON_JSON",
    }
    assert set(representative_names) == set(plw.REPOS)
    resolved = {
        repo: placements.lookup(repo, name)
        for repo, name in representative_names.items()
    }
    pool = resolved["omninode_infra"]
    assert pool is not None
    assert "omnibase-verify" in set(json.loads(pool))

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
optional ``|| '<literal>'`` fallback) from the Actions variables the runner
would read: the workflow's repository first, then the organisation. Anything it
cannot resolve fails naming the job and the expression. Never a pass.

The workflow bytes here are captured, not typed: C13 at the omninode_infra#1725
merge commit, and the C11 and dev-lane-liveness workflows at the
omnibase_infra#4233 head.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

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


def _vars(**scopes: dict[str, str]) -> plw.ActionsVariables:
    return plw.ActionsVariables(scopes)


# --------------------------------------------------------------------------- #
# omninode_infra#1725's shape: fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON).
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_1725_c13_resolves_from_the_repository_variable(tmp_path: Path) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    variables = _vars(
        org={}, omninode_infra={"CUSTOMER_MACHINE_RUNS_ON_JSON": VERIFY_POOL}
    )
    assert (
        plw.check([_c13("omnibase-verify")], {"omninode_infra": root}, variables) == []
    )


@pytest.mark.unit
def test_1725_c13_on_the_old_customer_label_is_a_lane_mismatch(
    tmp_path: Path,
) -> None:
    """The probe really moved: the window file must say which pool it now holds."""
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    variables = _vars(
        org={}, omninode_infra={"CUSTOMER_MACHINE_RUNS_ON_JSON": VERIFY_POOL}
    )
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
        _vars(org={}, omninode_infra={}),
    )
    assert len(errors) == 1, errors
    assert "c13-customer-local" in errors[0]
    assert "CUSTOMER_MACHINE_RUNS_ON_JSON" in errors[0]
    assert "not set" in errors[0]


@pytest.mark.unit
def test_1725_c13_with_no_variables_read_fails_naming_the_job(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    errors = plw.check([_c13("omnibase-verify")], {"omninode_infra": root})
    assert len(errors) == 1, errors
    assert "c13-customer-local" in errors[0]
    assert "no Actions variables were read for omninode_infra" in errors[0]


@pytest.mark.unit
def test_the_organisation_variable_answers_when_the_repository_has_none(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    variables = _vars(
        org={"CUSTOMER_MACHINE_RUNS_ON_JSON": VERIFY_POOL}, omninode_infra={}
    )
    assert (
        plw.check([_c13("omnibase-verify")], {"omninode_infra": root}, variables) == []
    )


@pytest.mark.unit
def test_the_repository_variable_overrides_the_organisation(tmp_path: Path) -> None:
    root = _root(tmp_path, "omninode_infra", {C13_PATH: C13_1725})
    variables = _vars(
        org={"CUSTOMER_MACHINE_RUNS_ON_JSON": '["omnipc2-customer"]'},
        omninode_infra={"CUSTOMER_MACHINE_RUNS_ON_JSON": VERIFY_POOL},
    )
    assert (
        plw.check([_c13("omnibase-verify")], {"omninode_infra": root}, variables) == []
    )


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
    variables = _vars(org={}, omninode_infra={"CUSTOMER_MACHINE_RUNS_ON_JSON": value})
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
            _vars(org={}, omninode_infra={}),
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
    variables = _vars(org={}, omnibase_infra={"LAB_PROBE_RUNS_ON_JSON": VERIFY_POOL})
    assert plw.check([c11], {"omnibase_infra": root}, variables) == []
    unlisted = plw.check([], {"omnibase_infra": root}, variables)
    assert len(unlisted) == 1, unlisted
    assert unlisted[0].startswith(f"unlisted probe omnibase_infra:{C11_PATH}")


@pytest.mark.unit
def test_4233_c11_as_a_runs_on_entry_resolves_the_probe_pool(tmp_path: Path) -> None:
    root = _root(tmp_path, "omnibase_infra", {C11_PATH: C11_4233})
    variables = _vars(org={}, omnibase_infra={"LAB_PROBE_RUNS_ON_JSON": VERIFY_POOL})
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
    variables = _vars(org={}, omnibase_infra={"LAB_PROBE_RUNS_ON_JSON": VERIFY_POOL})
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
    variables = _vars(
        org={},
        omninode_infra={"CUSTOMER_RUNNER_SMOKE_RUNS_ON_JSON": '["omnipc2-customer"]'},
    )
    errors = plw.check([], {"omninode_infra": root}, variables)
    assert len(errors) == 1, errors
    assert "unlisted probe omninode_infra:.github/workflows/smoke.yml" in errors[0]
    assert "omnipc2-customer" in errors[0]


# --------------------------------------------------------------------------- #
# The CLI reads the variables the CI job lists and refuses a missing scope.
# --------------------------------------------------------------------------- #


def _variables_file(tmp_path: Path, scope: str, values: dict[str, str]) -> Path:
    path = tmp_path / f"vars-{scope}.json"
    path.write_text(
        json.dumps([{"name": k, "value": v} for k, v in values.items()]),
        encoding="utf-8",
    )
    return path


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


def _cli_argv(tmp_path: Path, skip_scope: str | None = None) -> list[str]:
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
    scopes = {
        "org": {},
        "omnibase_infra": {},
        "omninode_infra": {"CUSTOMER_MACHINE_RUNS_ON_JSON": VERIFY_POOL},
        "omnimarket": {},
    }
    argv = ["--windows", str(windows)]
    argv += [f"--root={name}={path}" for name, path in roots.items()]
    argv += [
        f"--variables={scope}={_variables_file(tmp_path, scope, values)}"
        for scope, values in scopes.items()
        if scope != skip_scope
    ]
    return argv


@pytest.mark.unit
def test_cli_resolves_the_1725_shape_from_the_listed_variables(tmp_path: Path) -> None:
    assert plw.main(_cli_argv(tmp_path)) == plw.EXIT_OK


@pytest.mark.unit
@pytest.mark.parametrize(
    "scope", ["org", "omnibase_infra", "omninode_infra", "omnimarket"]
)
def test_cli_refuses_a_missing_variables_scope(tmp_path: Path, scope: str) -> None:
    assert plw.main(_cli_argv(tmp_path, skip_scope=scope)) == plw.EXIT_USAGE


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    ["not json", '{"name": "X"}', '[{"name": "X"}]', '[{"name": 1, "value": "v"}]'],
)
def test_cli_refuses_a_malformed_variables_file(tmp_path: Path, payload: str) -> None:
    argv = _cli_argv(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(payload, encoding="utf-8")
    argv = [a for a in argv if not a.startswith("--variables=org=")]
    argv.append(f"--variables=org={bad}")
    assert plw.main(argv) == plw.EXIT_USAGE

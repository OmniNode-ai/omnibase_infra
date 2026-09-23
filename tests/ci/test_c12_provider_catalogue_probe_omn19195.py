# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline grading tests for the C12 provider-catalogue producer (OMN-19195).

``tests/fixtures/omn19195/lane_healthy.json`` is the observer's output as
recorded on the lab dev lane's deployed ``onex-api`` container on 2026-09-22
(image ``onex-lab/omnicloud-core:d2d30be4-20260922T194843Z``, omnimarket
0.4.199). Every other case here is that recording with ONE observation broken,
so each test says which property of the deployed subject the verdict turns on.
Nothing here touches docker, the network or a lane.
"""

from __future__ import annotations

import copy
import json
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import c12_provider_catalogue_observe as observer
from scripts.ci import c12_provider_catalogue_probe as probe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn19195"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "chain-canary-c12-provider-catalogue.yml"
)


def _healthy() -> dict[str, Any]:
    payload = json.loads((FIXTURES / "lane_healthy.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _failed(record: probe.Record) -> set[str]:
    return {c.name for c in record.checks if not c.ok}


@pytest.mark.unit
def test_the_recorded_lab_reading_grades_pass() -> None:
    record = probe.grade(_healthy())
    assert record.verdict == "pass", record.failures
    assert record.exit_code == probe.EXIT_OK
    clauses = {c.clause for c in record.checks}
    assert clauses == {"parity", "no_claude", "no_house", "surface", "negative"}


@pytest.mark.unit
def test_an_empty_observation_cannot_pass() -> None:
    """Every check reads ABSENT as a failure, never as a vacuous pass."""
    record = probe.grade({})
    assert record.verdict == "fail"
    assert record.exit_code == probe.EXIT_FINDINGS
    # No offered provider means no per-provider intake check is generated, but
    # every fixed check still exists and fails.
    assert all(not c.ok for c in record.checks)


def _set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(obs: dict[str, Any]) -> None:
        cur = obs
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    return mutate


def _drop(path: tuple[str, ...]) -> Callable[[dict[str, Any]], None]:
    def mutate(obs: dict[str, Any]) -> None:
        cur = obs
        for key in path[:-1]:
            cur = cur[key]
        del cur[path[-1]]

    return mutate


BROKEN: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    # parity
    ("empty catalogue", _set(("shipped", "offered"), []), "catalogue_non_empty"),
    (
        "no handler-backed rungs",
        _set(("shipped", "house_keyed_slugs"), []),
        "handler_backed_set_non_empty",
    ),
    (
        "unbacked row shipped",
        _set(("shipped", "offered"), ["glm", "openai", "openrouter"]),
        "declared_equals_handler_backed",
    ),
    (
        "handler-backed provider missing",
        _set(("shipped", "not_offered"), ["gemini"]),
        "declared_equals_handler_backed",
    ),
    (
        "offered and declined at once",
        _set(("shipped", "not_offered"), ["gemini", "glm", "vertex"]),
        "offered_and_not_offered_disjoint",
    ),
    (
        "deployed parity function disagrees",
        _set(("shipped", "parity_gap", "missing_from_catalogue"), ["vertex"]),
        "deployed_parity_gap_clean",
    ),
    # no Claude
    (
        "a Claude catalogue id",
        _set(("shipped", "not_offered"), ["anthropic", "gemini", "vertex"]),
        "no_claude_catalogue_id",
    ),
    (
        "a Claude value on a row",
        _set(("shipped", "claude_hits"), ["us.anthropic.opus"]),
        "deployed_value_scan_empty",
    ),
    (
        "intake admits Claude",
        _set(("intake", "anthropic"), {"accepted": True}),
        "intake_refuses_anthropic",
    ),
    (
        "intake refuses Claude on the wrong field",
        _set(
            ("intake", "Claude-3", "errors"),
            [{"loc": ["name"], "type": "missing", "msg": "x"}],
        ),
        "intake_refuses_Claude-3",
    ),
    # no house entry
    (
        "a house finding",
        _set(
            ("shipped", "house_entry"),
            {"findings": [{"provider": "glm", "finding_class": "declared_house_ref"}]},
        ),
        "house_validator_zero_findings",
    ),
    (
        "the validator raised",
        _set(
            ("shipped", "house_entry"),
            {"raised": {"type": "HouseCatalogueError", "message": "x"}},
        ),
        "house_validator_zero_findings",
    ),
    # surface
    (
        "route bound to another model",
        _set(("route", "bound_to_catalogue_model"), False),
        "route_body_is_catalogue_model",
    ),
    (
        "the wrong route was read",
        _set(("route", "prefix"), "/v1/whoami"),
        "route_is_the_intake_route",
    ),
    (
        "intake refuses an offered provider",
        _set(("intake", "glm"), {"accepted": False, "errors": []}),
        "intake_accepts_glm",
    ),
    (
        "intake admits a declined provider",
        _set(("intake", "vertex"), {"accepted": True}),
        "intake_refuses_vertex",
    ),
    (
        "intake admits an unbacked provider",
        _set(("intake", "openai"), {"accepted": True}),
        "intake_refuses_openai",
    ),
    (
        "intake observation missing",
        _drop(("intake", "openrouter")),
        "intake_accepts_openrouter",
    ),
    # negative test: a checker that stopped biting
    (
        "parity check stops seeing an unbacked row",
        _set(("negative", "parity_unbacked", "unbacked_in_catalogue"), []),
        "unbacked_row_detected",
    ),
    (
        "parity check stops seeing a dropped row",
        _set(("negative", "parity_missing", "missing_from_catalogue"), []),
        "dropped_row_detected",
    ),
    (
        "Claude scan stops biting",
        _set(("negative", "claude_row", "claude_hits"), []),
        "claude_row_detected",
    ),
    (
        "laundered house ref passes",
        _set(("negative", "house_declared_ref", "findings"), []),
        "house_declared_ref_detected",
    ),
    (
        "house ref injection not constructible",
        _set(("negative", "house_declared_ref"), {"not_constructible": True}),
        "house_declared_ref_detected",
    ),
    (
        "rung collision passes",
        _set(("negative", "house_rung_collision", "findings"), []),
        "house_rung_collision_detected",
    ),
    (
        "un-registerable provider passes",
        _set(("negative", "no_registerable_key", "findings"), []),
        "no_registerable_key_detected",
    ),
    (
        "renamed finding class",
        _set(
            ("negative", "no_registerable_key", "findings"),
            [{"provider": "c12probe", "finding_class": "something_else"}],
        ),
        "no_registerable_key_detected",
    ),
    (
        "empty rung set passes vacuously",
        _set(("negative", "empty_rungs_fail_closed"), {"findings": []}),
        "empty_rungs_fail_closed",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "mutate", "check"), BROKEN, ids=[b[0] for b in BROKEN]
)
def test_each_broken_observation_grades_fail_naming_its_check(
    label: str, mutate: Callable[[dict[str, Any]], None], check: str
) -> None:
    obs = copy.deepcopy(_healthy())
    mutate(obs)
    record = probe.grade(obs)
    assert record.verdict == "fail", label
    assert record.exit_code == probe.EXIT_FINDINGS
    assert check in _failed(record), (label, record.failures)


@pytest.mark.unit
def test_the_finding_classes_are_declared_in_the_grader_not_read_back() -> None:
    """A renamed class in the deployed validator must red, not grade itself."""
    obs = copy.deepcopy(_healthy())
    obs["finding_classes"] = dict.fromkeys(obs["finding_classes"], "renamed")
    for key in ("house_declared_ref", "house_rung_collision", "no_registerable_key"):
        for finding in obs["negative"][key]["findings"]:
            finding["finding_class"] = "renamed"
    record = probe.grade(obs)
    assert {
        "house_declared_ref_detected",
        "house_rung_collision_detected",
        "no_registerable_key_detected",
    } <= _failed(record)


@pytest.mark.unit
def test_grader_and_observer_constants_agree() -> None:
    assert probe.UNBACKED_PROVIDER == observer.UNBACKED_PROVIDER
    assert probe.CLAUDE_PROVIDERS == observer.CLAUDE_PROVIDERS
    assert probe.SYNTHETIC_TOKEN_PROVIDER == observer.SYNTHETIC_TOKEN_PROVIDER
    assert probe.ROUTE_PREFIX == observer.ROUTE_PREFIX
    assert Path(observer.__file__).resolve() == probe.OBSERVER


@pytest.mark.unit
def test_an_observer_error_is_could_not_run_not_a_verdict() -> None:
    with pytest.raises(probe.ProbeInputError):
        probe.parse_observer_output(
            '{"error": {"type": "ImportError", "message": "x"}}\n'
        )
    with pytest.raises(probe.ProbeInputError):
        probe.parse_observer_output("")
    with pytest.raises(probe.ProbeInputError):
        probe.parse_observer_output("Traceback (most recent call last):\n")


def _fake_docker(tmp_path: Path, body: str) -> str:
    script = tmp_path / "docker"
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


@pytest.mark.unit
def test_an_unreachable_container_cannot_produce_a_pass(tmp_path: Path) -> None:
    docker = _fake_docker(
        tmp_path, 'echo "error: no such object: onex-api" >&2; exit 1'
    )
    record_path = tmp_path / "c12.json"
    rc = probe.main(["--docker-bin", docker, "--record", str(record_path)])
    assert rc == probe.EXIT_INPUT
    assert not record_path.exists()


@pytest.mark.unit
def test_a_stopped_container_cannot_produce_a_pass(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path, 'echo "img|exited|2026-09-22T00:00:00Z"')
    assert probe.main(["--docker-bin", docker]) == probe.EXIT_INPUT


@pytest.mark.unit
def test_the_live_path_grades_what_the_container_printed(tmp_path: Path) -> None:
    """The docker argv is pinned and the exec's stdout is what gets graded."""
    healthy = json.dumps(_healthy())
    argv_log = tmp_path / "argv.log"
    docker = _fake_docker(
        tmp_path,
        f'echo "$@" >> "{argv_log}"\n'
        'if [ "$1" = inspect ]; then echo "onex-lab/omnicloud-core:x|running|t"; exit 0; fi\n'
        "cat > /dev/null\n"
        f"cat <<'JSON'\n{healthy}\nJSON",
    )
    record_path = tmp_path / "c12.json"
    rc = probe.main(["--docker-bin", docker, "--record", str(record_path)])
    assert rc == probe.EXIT_OK
    calls = argv_log.read_text(encoding="utf-8").splitlines()
    assert calls[1] == "exec -i -u appuser onex-api python - /app"
    written = json.loads(record_path.read_text(encoding="utf-8"))
    assert written["criterion"] == "C12"
    assert written["verdict"] == "pass"
    assert written["target"]["image"] == "onex-lab/omnicloud-core:x"
    assert written["as_of"].endswith("Z")


@pytest.mark.unit
def test_replay_round_trips_through_main(tmp_path: Path) -> None:
    record_path = tmp_path / "c12.json"
    summary_path = tmp_path / "summary.md"
    rc = probe.main(
        [
            "--replay",
            str(FIXTURES / "lane_healthy.json"),
            "--record",
            str(record_path),
            "--summary",
            str(summary_path),
        ]
    )
    assert rc == probe.EXIT_OK
    assert "**Verdict: `pass`**" in summary_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_workflow_cannot_soften_its_verdict() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error" not in text
    assert "|| true" not in text
    assert "runs-on: [self-hosted, omnibase-verify, host-201]" in text
    # No pull_request trigger: a path-scoped PR job sits in neither required
    # surface and the advisory-job gate refuses it (the C11 finding).
    triggers = yaml.safe_load(text)[True]  # PyYAML reads the bare `on` key as True
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert "scripts/ci/c12_provider_catalogue_probe.py" in text
    assert 'exit "${status}"' in text


@pytest.mark.unit
def test_the_observer_is_standard_library_plus_the_subject() -> None:
    """The observer runs in a foreign interpreter; it may import only the
    standard library and the deployed subject it is reading."""
    import ast

    tree = ast.parse(Path(observer.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    allowed_subject = {"omnimarket", "routers"}
    stdlib = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    assert roots - stdlib - allowed_subject == set(), roots


@pytest.mark.unit
def test_the_probe_is_standard_library_only() -> None:
    import ast

    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    stdlib = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in stdlib, a.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            assert node.module.split(".")[0] in stdlib | {"__future__"}, node.module

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19410 -- the lab-desired-state.v1 contract (lab release-sync plan, T0.1).

Seam L0.1 is release or merge -> generator -> detector. The generator (T1.1)
writes a desired-state document per lab surface and the detector (T1.2) reads
it. These cases pin the contract both sides build against:

- AC1: a document missing any required field is refused, and the refusal names
  the field. Pinned over the committed missing-field fixture AND over every
  required field at every level of every valid fixture, so a field added to the
  schema later is covered without editing this file.
- AC2: ``desired_state_sha256`` is identical across two loads and across
  key-order shuffles of the same document.

The validator is stdlib-only because the census runs under the system
interpreter with no venv. One case runs it with ``-I -S`` (no site-packages) so
an accidental third-party import fails here, not on the lab host. A parity case
checks the stdlib validator against the jsonschema library on every fixture, so
the two readings of the schema cannot drift apart.
"""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "lab_sync" / "desired_state.py"
_SCHEMA = _REPO / "deploy" / "lab-sync" / "desired-state.schema.json"
_FIXTURES = _REPO / "tests" / "fixtures" / "lab_sync"
sys.path.insert(0, str(_SCRIPT.parent))

from desired_state import (
    DesiredStateRefusalError,
    compute_desired_state_sha256,
    load_desired_state,
    parse_desired_state,
)

pytestmark = pytest.mark.unit

_VALID = _FIXTURES / "desired_state_valid.json"
_VALID_FIXTURES = sorted(_FIXTURES.glob("desired_state_valid*.json"))


def _doc(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _write(tmp_path: Path, doc: object, name: str = "doc.json") -> Path:
    target = tmp_path / name
    target.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return target


def _reseal(doc: dict[str, Any]) -> dict[str, Any]:
    doc["desired_state_sha256"] = compute_desired_state_sha256(doc)
    return doc


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in node and set(node) - {"$ref", "description"} == set():
        name = node["$ref"].removeprefix("#/$defs/")
        node = schema["$defs"][name]
    if "$ref" in node:
        name = node["$ref"].removeprefix("#/$defs/")
        merged = dict(schema["$defs"][name])
        merged.update({k: v for k, v in node.items() if k != "$ref"})
        return merged
    return node


def _required_paths(
    schema: dict[str, Any], node: dict[str, Any], value: object, path: tuple[Any, ...]
) -> Iterator[tuple[Any, ...]]:
    """Every (path to a required field) present in ``value`` under ``node``."""
    node = _resolve(schema, node)
    if isinstance(value, dict):
        for name in node.get("required", []):
            yield (*path, name)
        for name, sub in node.get("properties", {}).items():
            if name in value:
                yield from _required_paths(schema, sub, value[name], (*path, name))
    elif isinstance(value, list) and "items" in node:
        for index, item in enumerate(value):
            yield from _required_paths(schema, node["items"], item, (*path, index))


def _render(path: tuple[Any, ...]) -> str:
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else part)
    return out


def _all_required_cases() -> list[tuple[Path, tuple[Any, ...]]]:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    cases: list[tuple[Path, tuple[Any, ...]]] = []
    for fixture in _VALID_FIXTURES:
        cases.extend(
            (fixture, p) for p in _required_paths(schema, schema, _doc(fixture), ())
        )
    return cases


def _shuffled(value: object, rng: random.Random) -> object:
    if isinstance(value, dict):
        keys = list(value)
        rng.shuffle(keys)
        return {k: _shuffled(value[k], rng) for k in keys}
    if isinstance(value, list):
        return [_shuffled(v, rng) for v in value]
    return value


# --------------------------------------------------------------------- AC1


def test_missing_field_fixture_is_refused_naming_the_field() -> None:
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_FIXTURES / "desired_state_missing_field.json")
    refusal = excinfo.value
    assert refusal.field == "containers[2].config_hash"
    assert "config_hash" in str(refusal)
    assert "missing" in str(refusal)


@pytest.mark.parametrize(
    ("fixture", "path"),
    _all_required_cases(),
    ids=lambda v: v.name if isinstance(v, Path) else _render(v),
)
def test_missing_any_required_field_is_refused_naming_it(
    tmp_path: Path, fixture: Path, path: tuple[Any, ...]
) -> None:
    doc = _doc(fixture)
    parent: Any = doc
    for part in path[:-1]:
        parent = parent[part]
    del parent[path[-1]]
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, doc))
    assert excinfo.value.field == _render(path)
    assert f"'{path[-1]}'" in str(excinfo.value)


def test_missing_required_field_case_list_is_not_empty() -> None:
    """Positive control: the parametrized walk above found real fields."""
    rendered = {_render(p) for _, p in _all_required_cases()}
    assert "target_ref.commit" in rendered
    assert "containers[0].config_hash" in rendered
    assert "broker.cluster_config" in rendered
    assert "runners.pools[0].workdir" in rendered
    assert "allowed_undeclared[0].owner" in rendered
    assert len(rendered) > 40


def test_extra_field_fixture_is_refused_naming_the_field() -> None:
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_FIXTURES / "desired_state_extra_field.json")
    assert excinfo.value.field == "generated_at"
    assert "not allowed" in str(excinfo.value)


# --------------------------------------------------------------------- AC2


def test_deterministic_sha_is_identical_across_two_loads() -> None:
    first = load_desired_state(_VALID)
    second = load_desired_state(_VALID)
    assert first.desired_state_sha256 == second.desired_state_sha256
    assert first.desired_state_sha256 == _doc(_VALID)["desired_state_sha256"]


@pytest.mark.parametrize("seed", range(25))
def test_deterministic_sha_is_identical_across_key_order_shuffles(
    tmp_path: Path, seed: int
) -> None:
    rng = random.Random(seed)
    for fixture in _VALID_FIXTURES:
        original = load_desired_state(fixture)
        shuffled = _shuffled(_doc(fixture), rng)
        reloaded = load_desired_state(_write(tmp_path, shuffled, fixture.name))
        assert reloaded.desired_state_sha256 == original.desired_state_sha256


def test_deterministic_sha_changes_when_the_body_changes() -> None:
    """Positive control for the determinism cases: a real change moves the hash."""
    doc = _doc(_VALID)
    before = compute_desired_state_sha256(doc)
    doc["containers"][0]["health_required"] = not doc["containers"][0][
        "health_required"
    ]
    assert compute_desired_state_sha256(doc) != before


def test_deterministic_sha_ignores_the_sha_field_itself() -> None:
    doc = _doc(_VALID)
    stamped = compute_desired_state_sha256(doc)
    doc["desired_state_sha256"] = "0" * 64
    assert compute_desired_state_sha256(doc) == stamped


# ------------------------------------------------------- fail-closed reads


def test_embedded_sha_that_does_not_match_the_body_is_refused(tmp_path: Path) -> None:
    doc = _doc(_VALID)
    doc["target_ref"]["ref"] = "v9.9.9"
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, doc))
    assert excinfo.value.field == "desired_state_sha256"


def test_a_timestamp_in_the_body_is_refused(tmp_path: Path) -> None:
    doc = _doc(_VALID)
    doc["surface"]["rendered_at"] = "2026-09-24T17:00:00Z"
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, _reseal(doc)))
    assert excinfo.value.field == "surface.rendered_at"


def test_unsorted_containers_are_refused(tmp_path: Path) -> None:
    doc = _doc(_VALID)
    doc["containers"].reverse()
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, _reseal(doc)))
    assert excinfo.value.field == "containers"
    assert "sorted" in str(excinfo.value)


def test_duplicate_container_names_are_refused(tmp_path: Path) -> None:
    doc = _doc(_VALID)
    doc["containers"].insert(1, copy.deepcopy(doc["containers"][0]))
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, _reseal(doc)))
    assert excinfo.value.field == "containers"


def test_unsorted_runner_labels_are_refused(tmp_path: Path) -> None:
    doc = _doc(_FIXTURES / "desired_state_valid_runner_fleet.json")
    doc["runners"]["pools"][0]["labels"].reverse()
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, _reseal(doc)))
    assert excinfo.value.field == "runners.pools[0].labels"


def test_a_duplicate_json_key_is_refused(tmp_path: Path) -> None:
    text = _VALID.read_text(encoding="utf-8").replace(
        '"mode": "reconcile"', '"mode": "reconcile", "mode": "detect_only"', 1
    )
    target = tmp_path / "dup.json"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(target)
    assert "duplicate key 'mode'" in str(excinfo.value)


def test_unparseable_json_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(DesiredStateRefusalError):
        load_desired_state(target)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DesiredStateRefusalError):
        load_desired_state(tmp_path / "absent.json")


def test_a_compose_lane_without_a_broker_is_refused(tmp_path: Path) -> None:
    doc = _doc(_VALID)
    doc["broker"] = None
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, _reseal(doc)))
    assert excinfo.value.field == "broker"
    assert "compose_lane" in str(excinfo.value)


def test_a_host_surface_may_not_declare_containers(tmp_path: Path) -> None:
    doc = _doc(_FIXTURES / "desired_state_valid_host.json")
    doc["containers"] = _doc(_VALID)["containers"][:1]
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, _reseal(doc)))
    assert excinfo.value.field == "containers"


def test_a_boolean_is_not_an_integer(tmp_path: Path) -> None:
    doc = _doc(_FIXTURES / "desired_state_valid_runner_fleet.json")
    doc["runners"]["pools"][0]["expected_count"] = True
    with pytest.raises(DesiredStateRefusalError) as excinfo:
        load_desired_state(_write(tmp_path, _reseal(doc)))
    assert excinfo.value.field == "runners.pools[0].expected_count"


# -------------------------------------------------- the loaded model and key


def test_the_key_is_surface_and_target_ref() -> None:
    state = load_desired_state(_VALID)
    doc = _doc(_VALID)
    assert state.key == (
        doc["surface"]["id"],
        doc["target_ref"]["repo"],
        doc["target_ref"]["commit"],
    )
    assert state.surface_kind == "compose_lane"
    assert state.document == doc


def test_every_valid_fixture_loads() -> None:
    kinds = {load_desired_state(f).surface_kind for f in _VALID_FIXTURES}
    assert kinds == {"compose_lane", "runner_fleet", "host"}


def test_parse_matches_load() -> None:
    assert (
        parse_desired_state(_VALID.read_text(encoding="utf-8")).desired_state_sha256
        == load_desired_state(_VALID).desired_state_sha256
    )


# ------------------------------------------- parity with the jsonschema reading


def _mutations() -> Iterator[tuple[str, dict[str, Any]]]:
    base = _doc(_VALID)
    mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("wrong schema_version", lambda d: d.update(schema_version="v2")),
        ("bad kind", lambda d: d["surface"].update(kind="lane")),
        ("short commit", lambda d: d["target_ref"].update(commit="abc")),
        ("uppercase hash", lambda d: d["containers"][0].update(config_hash="A" * 64)),
        ("revision int", lambda d: d["containers"][0].update(revision=7)),
        ("empty cluster_config", lambda d: d["broker"].update(cluster_config={})),
        ("bad memory", lambda d: d["broker"].update(memory="lots")),
        (
            "runners on a lane",
            lambda d: d.update(runners={"pools": [], "containers": []}),
        ),
        (
            "package bad name",
            lambda d: d["containers"][2]["packages"].update({"Bad": "1"}),
        ),
        ("empty version", lambda d: d["containers"][2]["packages"].update({"x": ""})),
    )
    for name, mutate in mutations:
        doc = copy.deepcopy(base)
        mutate(doc)
        yield name, doc


def test_the_stdlib_validator_agrees_with_jsonschema() -> None:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    reference = jsonschema.Draft202012Validator(schema)
    cases: list[tuple[str, dict[str, Any]]] = [
        (f.name, _doc(f)) for f in sorted(_FIXTURES.glob("desired_state_*.json"))
    ]
    cases.extend(_mutations())
    disagreements = []
    for name, doc in cases:
        doc = _reseal(doc) if "extra" not in name and "missing" not in name else doc
        reference_ok = reference.is_valid(doc)
        try:
            parse_desired_state(json.dumps(doc))
            ours_ok = True
        except DesiredStateRefusalError:
            ours_ok = False
        if reference_ok != ours_ok:
            disagreements.append((name, reference_ok, ours_ok))
    assert disagreements == []
    assert any(not reference.is_valid(d) for _, d in _mutations())


# ----------------------------------------------------------------- the CLI


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-S", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO,
    )


def test_self_check_passes_under_an_isolated_interpreter() -> None:
    """-I -S: no site-packages and no PYTHONPATH, as on the lab host."""
    result = _run("self-check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "self-check OK" in result.stdout


def test_self_check_refuses_a_schema_keyword_the_validator_does_not_implement(
    tmp_path: Path,
) -> None:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    schema["$defs"]["surface"]["properties"]["id"]["format"] = "uri"
    bad = _write(tmp_path, schema, "schema.json")
    result = _run("self-check", "--schema", str(bad))
    assert result.returncode == 1
    assert "format" in result.stderr


def test_self_check_refuses_a_fixture_the_schema_no_longer_accepts(
    tmp_path: Path,
) -> None:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    schema["$defs"]["container"]["required"].append("image_digest")
    bad = _write(tmp_path, schema, "schema.json")
    result = _run("self-check", "--schema", str(bad))
    assert result.returncode == 1
    assert "image_digest" in result.stderr


def test_validate_cli_prints_the_sha_and_refuses_with_exit_1() -> None:
    ok = _run("validate", str(_VALID))
    assert ok.returncode == 0, ok.stderr
    assert _doc(_VALID)["desired_state_sha256"] in ok.stdout
    bad = _run("validate", str(_FIXTURES / "desired_state_missing_field.json"))
    assert bad.returncode == 1
    assert "containers[2].config_hash" in bad.stderr

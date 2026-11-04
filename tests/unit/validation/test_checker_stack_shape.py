# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21891: every compose service is the runtime, a dependency, a one-shot or a baselined entry.

The shape checker and its contract are omnibase_core's ``ValidatorStackShape``; the rule values
are ``config/validation/stack_shape.yaml``. Its glob finds stack files and its baseline is grouped
per service. The existing architecture-validation node extends this same gate with command and
console-script rules, using T1's compose parser. A command-only source covers the external forwarder. The core
validator reads a literal compose file list per lane, so :func:`effective_data` expands the data
file against the tree into that format (one lane per compose file) before the validator runs.
This file runs the validator

* over small fixture stacks (the two services the old deny-list let through, a scaled runtime,
  a stale and a grown baseline), and
* over the real compose files of every lane, which is the gate the pre-commit hook and the CI job
  ``Stack Shape (OMN-21891)`` run.

``STACK_SHAPE_BASE_REF`` names the merge base the baseline may not grow from (CI sets it; the
hook leaves it at ``HEAD``).
"""

from __future__ import annotations

import fnmatch
import functools
import importlib.metadata
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
import yaml

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.common.model_validation_issue import ModelValidationIssue
from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates
from omnibase_core.validation.checker_stack_shape import (
    RULE_BASELINE,
    baseline_growth,
    compose_services,
)
from omnibase_core.validation.validator_stack_shape import ValidatorStackShape
from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.nodes.node_architecture_validator.handlers.handler_architecture_validation import (
    HandlerArchitectureValidation,
)
from omnibase_infra.nodes.node_architecture_validator.models.model_architecture_validation_request import (
    ModelArchitectureValidationRequest,
)
from omnibase_infra.nodes.node_architecture_validator.models.model_stack_shape_snapshot import (
    ModelStackShapeSnapshot,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO / "tests" / "fixtures" / "stack_shape"
_DATA = Path("config/validation/stack_shape.yaml")
_INFRA = Path("docker/docker-compose.infra.yml")
_OVERLAY = Path("docker/docker-compose.overlay.yml")
_PLANTED = ("projection-claude-hook-events-writer", "cost-estimator-web")
_RULES = {r.rule_id: r for r in ValidatorStackShape().contract.rules}


# -- the data file, expanded into the format the core validator reads ------------------------------


def _data(text: str) -> Mapping[str, object]:
    loaded = load_yaml_mapping_no_duplicates(text, source="stack-shape data")
    return {str(key): value for key, value in loaded.items()}


def _strings(data: Mapping[str, object], key: str) -> list[str]:
    value = data.get(key) or []
    assert isinstance(value, list) and all(isinstance(v, str) for v in value), key
    return value


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _judged(root: Path, data: Mapping[str, object]) -> list[str]:
    """The compose files the data file's glob finds, less the ones declared not a stack."""
    pattern = data.get("compose_files")
    assert isinstance(pattern, str), "compose_files must be a glob"
    found = sorted(p.relative_to(root).as_posix() for p in root.glob(pattern))
    skip = _strings(data, "not_a_stack")
    return [f for f in found if not _matches(f, skip)]


@functools.lru_cache(maxsize=128)
def _services_of(text: str) -> dict[str, dict[object, object]]:
    """The services of one compose file's text, parsed once per distinct text."""
    return {str(n): dict(b) for n, b in compose_services(text, _RULES).items()}


def _is_one_shot(body: Mapping[object, object]) -> bool:
    return body.get("restart") in ("no", False)


def _issue(path: Path, message: str) -> ModelValidationIssue:
    return ModelValidationIssue(
        severity=_RULES[RULE_BASELINE].severity,
        message=" ".join(message.split()),
        code=RULE_BASELINE,
        file_path=path,
        rule_name=RULE_BASELINE,
    )


def effective_data(
    root: Path, text: str
) -> tuple[str, tuple[ModelValidationIssue, ...]]:
    """The core-format data for ``root`` and the baseline-count issues the core format cannot hold.

    A baseline item names a service (or a ``*`` pattern over service names) and the number of
    compose files that declare it. When the files hold exactly that many, the item becomes one
    ``(file, service)`` baseline entry per declaration. When they hold fewer, or none, the item is
    stale and is reported here. When they hold more, the item admits none of them, so the core
    validator refuses every declaration by service name.
    """
    data = _data(text)
    runtime = data["runtime"]
    assert isinstance(runtime, dict)
    files = _judged(root, data)
    services: dict[str, dict[str, dict[object, object]]] = {}
    for file in files:
        try:
            services[file] = _services_of((root / file).read_text(encoding="utf-8"))
        except ModelOnexError:
            services[file] = {}  # the core validator reports the unreadable file itself
    one_shots = _strings(data, "one_shots")
    patterns = [s for s in one_shots if "*" in s]
    admitted = {s for s in one_shots if "*" not in s} | {
        name
        for declared in services.values()
        for name, body in declared.items()
        if _is_one_shot(body) and _matches(name, patterns)
    }
    allowed = set(runtime["services"]) | set(_strings(data, "dependencies")) | admitted
    entries: list[dict[str, str]] = []
    issues: list[ModelValidationIssue] = []
    baseline = data.get("baseline") or []
    assert isinstance(baseline, list)
    for item in baseline:
        service, budget = str(item["service"]), int(item["files"])
        ticket = str(item.get("ticket") or "")
        hits = [
            (file, name)
            for file, declared in services.items()
            for name in declared
            if name not in allowed and fnmatch.fnmatchcase(name, service)
        ]
        if not hits:
            issues.append(
                _issue(
                    _DATA,
                    f"baseline entry {service} names a service that no compose file declares any "
                    "more: delete the entry; the baseline only shrinks.",
                )
            )
        elif len(hits) < budget:
            issues.append(
                _issue(
                    _DATA,
                    f"baseline entry {service} counts {budget} compose files but {len(hits)} "
                    "declare it: lower the count; the baseline only shrinks.",
                )
            )
        elif len(hits) > budget:
            issues.append(
                _issue(
                    _DATA,
                    f"service {service} is declared in {len(hits)} compose files and the "
                    f"baseline allows {budget}: {', '.join(sorted(f for f, _ in hits))}.",
                )
            )
            continue
        entries.extend(
            {"file": file, "service": name, "ticket": ticket} for file, name in hits
        )
    kept = data.get("kept_outside_runtime") or []
    assert isinstance(kept, list)
    for item in kept:
        service = str(item["service"])
        ticket = str(item.get("ticket") or "")
        if not str(item.get("reason") or "").strip():
            issues.append(
                _issue(
                    _DATA,
                    f"kept service {service} carries no reason: name the second bus connection "
                    "or failure domain the runtime does not provide.",
                )
            )
        for file in item["files"]:
            declared = services.get(file)
            if (
                declared is None
            ):  # not judged as a stack (not_a_stack): read the file itself
                path = root / file
                declared = (
                    _services_of(path.read_text(encoding="utf-8"))
                    if path.is_file()
                    else {}
                )
            if service not in declared:
                issues.append(
                    _issue(
                        _DATA,
                        f"kept service {service} names {file}, which declares no service "
                        f"{service}: delete the entry or correct its files.",
                    )
                )
            elif file in services:
                entries.append({"file": file, "service": service, "ticket": ticket})
    effective = {
        "runtime": runtime,
        "dependencies": _strings(data, "dependencies"),
        "one_shots": sorted(admitted),
        "lanes": {file: {"compose_files": [file]} for file in files},
        "baseline": entries,
    }
    return yaml.safe_dump(effective, sort_keys=False), tuple(issues)


def _budgets(text: str) -> dict[str, int]:
    baseline = _data(text).get("baseline") or []
    assert isinstance(baseline, list)
    return {str(item["service"]): int(item["files"]) for item in baseline}


def _git_show(root: Path, ref: str, path: Path) -> str | None:
    shown = subprocess.run(
        ["git", "-C", str(root), "show", f"{ref}:{path.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(),
    )
    return shown.stdout if shown.returncode == 0 else None


def _growth(
    root: Path, text: str, base_ref: str | None
) -> tuple[ModelValidationIssue, ...]:
    """Baseline items the base revision does not allow; none before the data file is on the base."""
    base_text = _git_show(root, base_ref, _DATA) if base_ref else None
    if base_text is None:
        return ()
    try:
        base = _budgets(base_text)
    except (ModelOnexError, ValueError, KeyError, AssertionError):
        return ()
    return baseline_growth(_budgets(text), base, {}, _RULES)


def _judge(
    root: Path, base_ref: str | None = None, text: str | None = None
) -> tuple[ModelValidationIssue, ...]:
    text = (root / _DATA).read_text(encoding="utf-8") if text is None else text
    effective, issues = effective_data(root, text)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stack_shape.yaml"
        path.write_text(effective, encoding="utf-8")
        result = ValidatorStackShape(root=root, data_path=str(path)).validate(path)
    data = _data(text)
    files = sorted(
        set(_judged(root, data)) | set(_strings(data, "command_only_compose_files"))
    )
    source = {file: (root / file).read_text(encoding="utf-8") for file in files}
    base_source = (
        {
            file: original
            for file in files
            if (original := _git_show(root, base_ref, Path(file))) is not None
        }
        if base_ref
        else {}
    )
    scripts: dict[str, dict[str, str]] = {}
    if (root / "pyproject.toml").is_file():
        manifest = tomllib.loads((root / "pyproject.toml").read_text())
        scripts[manifest["project"]["name"]] = manifest["project"].get("scripts", {})
        inventory = data.get("console_scripts", {})
        patterns = _strings(data, "console_script_package_patterns")
        names = {str(name).replace("_", "-").lower(): str(name) for name in inventory}
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata["Name"].replace("_", "-").lower()
            if not _matches(name, patterns):
                continue
            package = names.get(name, name)
            if package not in scripts:
                scripts[package] = {
                    e.name: e.value
                    for e in distribution.entry_points
                    if e.group == "console_scripts"
                }
    command_result = HandlerArchitectureValidation().handle(
        ModelArchitectureValidationRequest(
            stack_shape=ModelStackShapeSnapshot(
                policy_yaml=text,
                compose_files=source,
                base_policy_yaml=_git_show(root, base_ref, _DATA) if base_ref else None,
                base_compose_files=base_source,
                package_scripts=scripts,
            )
        )
    )
    command_issues = tuple(
        ModelValidationIssue(
            severity=_RULES[RULE_BASELINE].severity,
            message=v.message,
            code=v.rule_id,
            file_path=Path(v.location or _DATA),
            rule_name=v.rule_id,
        )
        for v in command_result.violations
    )
    return (
        issues + tuple(result.issues) + _growth(root, text, base_ref) + command_issues
    )


def _text(issues: tuple[ModelValidationIssue, ...]) -> str:
    return "\n".join(f"{i.file_path}: {i.message}" for i in issues)


# -- fixtures ----------------------------------------------------------------------------------


def _stack(tmp_path: Path) -> Path:
    """A writable copy of the fixture stack."""
    root = tmp_path / "stack"
    shutil.copytree(_FIXTURES / "stack", root)
    return root


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _plant(path: Path, block: str) -> None:
    """Insert ``block`` (services at two-space indent) first under the file's ``services:``."""
    head, found, tail = path.read_text(encoding="utf-8").partition("\nservices:\n")
    assert found, f"{path} has no top-level services mapping"
    path.write_text(head + found + block + tail, encoding="utf-8")


def _planted_block() -> str:
    return (_FIXTURES / "planted_services.yml").read_text(encoding="utf-8")


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(),
    )


def _commit_all(root: Path) -> None:
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "fixture")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")


# -- AC1: the allowlist ---------------------------------------------------------------------------


def test_fixture_stack_is_clean(tmp_path: Path) -> None:
    """Positive control: the fixture judges clean, so a refusal below comes from the plant."""
    assert _judge(_stack(tmp_path)) == ()


def test_planted_copies_are_each_reported_by_name(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    _plant(root / _OVERLAY, _planted_block())
    refused = _text(_judge(root))
    for name in _PLANTED:
        assert f"service {name} in {_OVERLAY}" in refused, refused
    assert "legacy-consumer" not in refused


def test_planted_copies_are_refused_in_every_compose_file_of_the_real_stack(
    tmp_path: Path,
) -> None:
    """Every lane's compose files: a planted copy of each service is reported by name.

    The lanes are the compose files the data file's glob finds, so a lane added later is covered
    without naming it here. The two plants go into one file at a time, in a copy of the tree.
    """
    root = tmp_path / "tree"
    (root / "docker").mkdir(parents=True)
    (root / "config/validation").mkdir(parents=True)
    shutil.copy(_REPO / _DATA, root / _DATA)
    for compose in (_REPO / "docker").glob("docker-compose*.yml"):
        shutil.copy(compose, root / "docker" / compose.name)
    files = _judged(root, _data((root / _DATA).read_text(encoding="utf-8")))
    assert files, "the data file's glob finds no compose file"
    unreported: list[str] = []
    for file in files:
        original = (root / file).read_text(encoding="utf-8")
        declared = _services_of(original)
        # a file cannot hold a second service of the same name, so a plant goes where it is new
        names = [n for n in _PLANTED if n not in declared]
        _plant(root / file, _planted_block())
        refused = _text(_judge(root))
        (root / file).write_text(original, encoding="utf-8")
        unreported += [f"{n} planted in {file}" for n in names if n not in refused]
    assert unreported == []
    assert any(
        n not in _services_of((root / f).read_text("utf-8"))
        for f in files
        for n in _PLANTED
    )


def test_service_outside_the_allowlist_in_the_base_file_is_refused(
    tmp_path: Path,
) -> None:
    root = _stack(tmp_path)
    _edit(root / _DATA, "  - service: legacy-consumer\n", "  - service: someone-else\n")
    refused = _text(_judge(root))
    assert f"service legacy-consumer in {_INFRA}" in refused
    assert "someone-else" in refused  # the stale baseline entry, too


def test_one_shot_pattern_admits_only_a_service_that_declares_restart_no(
    tmp_path: Path,
) -> None:
    root = _stack(tmp_path)
    _edit(
        root / _INFRA,
        '  create-topics:\n    image: example/redpanda:tag\n    restart: "no"\n',
        "  create-topics:\n    image: example/redpanda:tag\n",
    )
    assert f"service create-topics in {_INFRA}" in _text(_judge(root))


# -- AC4: replicas ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "replicas",
    [
        "    replicas: 2\n",
        "    scale: 2\n",
        "    deploy:\n      replicas: 2\n",
        "    deploy:\n      replicas: ${RUNTIME_REPLICAS:-2}\n",
    ],
    ids=["replicas", "scale", "deploy.replicas", "deploy.replicas-default"],
)
def test_runtime_replicas_over_one_fail(tmp_path: Path, replicas: str) -> None:
    root = _stack(tmp_path)
    overlay = root / _OVERLAY
    _edit(overlay, "      LANE: fixture\n", "      LANE: fixture\n" + replicas)
    assert "runs 2 replicas, above the limit of 1" in _text(_judge(root))


def test_runtime_at_one_replica_passes(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    _edit(
        root / _OVERLAY,
        "      LANE: fixture\n",
        "      LANE: fixture\n    replicas: 1\n",
    )
    assert _judge(root) == ()


# -- AC2: the baseline only shrinks ---------------------------------------------------------------


def test_baseline_entry_for_a_removed_service_fails(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    infra = root / _INFRA
    infra.write_text(
        infra.read_text(encoding="utf-8").split("  legacy-consumer:")[0]
        + "  build-image:\n    image: example/runtime:tag\n",
        encoding="utf-8",
    )
    refused = _text(_judge(root))
    assert "baseline entry legacy-consumer" in refused
    assert "no compose file declares any more" in refused


def test_baseline_count_above_what_the_files_hold_fails(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    _edit(root / _DATA, "    files: 2\n", "    files: 3\n")
    assert "counts 3 compose files but 2 declare it" in _text(_judge(root))


def test_baseline_service_declared_in_one_more_file_is_refused(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    _plant(
        root / _OVERLAY,
        "  legacy-consumer:\n    image: example/runtime:tag\n",
    )
    refused = _text(_judge(root))
    assert "service legacy-consumer is declared in 2 compose files" in refused
    assert f"service legacy-consumer in {_OVERLAY}" in refused


def test_baseline_entry_without_a_ticket_fails(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    _edit(root / _DATA, "    ticket: OMN-21288\n", "")
    assert "carries no ticket" in _text(_judge(root))


def test_baseline_grown_against_the_base_revision_fails(tmp_path: Path) -> None:
    """A new service plus its baseline entry is refused once the data file is on the base."""
    root = _stack(tmp_path)
    _commit_all(root)
    _plant(root / _INFRA, "  new-worker:\n    image: example/runtime:tag\n")
    data = root / _DATA
    data.write_text(
        data.read_text(encoding="utf-8")
        + "  - service: new-worker\n    files: 1\n    ticket: OMN-21288\n",
        encoding="utf-8",
    )
    refused = _text(_judge(root, base_ref="HEAD"))
    assert "baseline entry new-worker::" not in refused
    assert "new-worker is not on the base revision" in refused
    assert "legacy-consumer" not in refused


def test_baseline_count_raised_against_the_base_revision_fails(tmp_path: Path) -> None:
    """A second declaration of a baselined service plus a raised count is refused."""
    root = _stack(tmp_path)
    _commit_all(root)
    _plant(root / _OVERLAY, "  legacy-consumer:\n    image: example/runtime:tag\n")
    _edit(
        root / _DATA,
        "  - service: legacy-consumer\n    files: 1\n",
        "  - service: legacy-consumer\n    files: 2\n",
    )
    refused = _text(_judge(root, base_ref="HEAD"))
    assert "legacy-consumer is not on the base revision" in refused


def test_baseline_shrinking_against_the_base_revision_passes(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    _commit_all(root)
    infra = root / _INFRA
    infra.write_text(
        infra.read_text(encoding="utf-8").split("  legacy-consumer:")[0]
        + "  build-image:\n    image: example/runtime:tag\n",
        encoding="utf-8",
    )
    _edit(
        root / _DATA,
        "  - service: legacy-consumer\n    files: 1\n    ticket: OMN-21288\n",
        "",
    )
    assert _judge(root, base_ref="HEAD") == ()


# -- the real compose files of every lane -----------------------------------------------------


def _real_text() -> str:
    return (_REPO / _DATA).read_text(encoding="utf-8")


def test_real_data_file_exists_and_finds_compose_files() -> None:
    """A deleted data file would make the validator judge nothing, so its presence is asserted."""
    assert (_REPO / _DATA).is_file()
    data = _data(_real_text())
    judged = _judged(_REPO, data)
    assert judged, "the data file's glob finds no compose file"
    # the compose files of the dev stack are among them
    assert {_INFRA.as_posix(), "docker/docker-compose.dev-lane.yml"} <= set(judged)


def test_every_not_a_stack_entry_matches_a_compose_file() -> None:
    """A stale not_a_stack entry cannot linger once its file is deleted."""
    data = _data(_real_text())
    on_disk = [
        p.relative_to(_REPO).as_posix() for p in _REPO.glob(str(data["compose_files"]))
    ]
    stale = [s for s in _strings(data, "not_a_stack") if not _matches_any(on_disk, s)]
    assert stale == []


def _matches_any(names: Iterable[str], pattern: str) -> bool:
    return any(fnmatch.fnmatchcase(n, pattern) for n in names)


def test_real_compose_files_hold_only_allowed_services() -> None:
    assert (_REPO / _DATA).is_file(), (
        f"{_DATA} is missing: the validator would judge nothing"
    )
    base_ref = os.environ.get("STACK_SHAPE_BASE_REF") or "HEAD"
    issues = _judge(_REPO, base_ref=base_ref)
    assert issues == (), _text(issues)


def test_real_run_reads_the_stack_it_judges() -> None:
    """Positive control: dropping a baseline item from a copy of the real data makes it fail."""
    text = _real_text()
    data = _data(text)
    baseline = data["baseline"]
    assert isinstance(baseline, list) and baseline, "the real baseline holds no entry"
    victim = next(i for i in baseline if "*" not in i["service"])
    assert victim["ticket"], "every baseline entry names a ticket"
    shrunk = yaml.safe_dump(
        {**data, "baseline": [i for i in baseline if i is not victim]}
    )
    refused = _text(_judge(_REPO, text=shrunk))
    assert f"service {victim['service']} in docker/" in refused


# -- OMN-22162: the cost estimator page on the report host is not a lane's stack ----------------

_REPORT_HOST_COMPOSE = "docker/docker-compose.cost-estimator-web.yml"
_REPORT_HOST_ENTRY = f"  - {_REPORT_HOST_COMPOSE}\n"


def test_report_host_page_file_is_declared_not_a_stack_and_is_not_judged() -> None:
    """The report host's compose file is named under not_a_stack; the dev lane's copy stays judged."""
    data = _data(_real_text())
    assert _REPORT_HOST_COMPOSE in _strings(data, "not_a_stack")
    judged = _judged(_REPO, data)
    assert _REPORT_HOST_COMPOSE not in judged
    assert "docker/docker-compose.dev-lane.yml" in judged
    assert (_REPO / _REPORT_HOST_COMPOSE).is_file()
    # the dev lane's cost-estimator-web keeps its one baseline file until it leaves that file
    assert _budgets(_real_text())["cost-estimator-web"] == 1


def test_report_host_page_file_judged_as_a_stack_is_refused() -> None:
    """RED control: without the not_a_stack entry the same tree is refused for a second declaration."""
    text = _real_text()
    assert _REPORT_HOST_ENTRY in text
    refused = _text(_judge(_REPO, text=text.replace(_REPORT_HOST_ENTRY, "", 1)))
    assert "service cost-estimator-web is declared in 2 compose files" in refused
    assert _REPORT_HOST_COMPOSE in refused
    assert _judge(_REPO, text=text) == ()


# -- OMN-22393 (T26): the lane-mirror forwarder is kept outside the runtime, with its reason -------

_FORWARDER = "gateway-forwarder"
_FORWARDER_COMPOSE = "docker/docker-compose.gateway.yml"
_KEPT_FIXTURE = (
    "kept_outside_runtime:\n"
    "  - service: fixture-bridge\n"
    "    files:\n"
    "      - docker/docker-compose.overlay.yml\n"
    "    ticket: OMN-22393\n"
    "    reason: a second bus connection the fixture runtime does not hold\n"
)
_BRIDGE = (
    "  fixture-bridge:\n    image: example/runtime:tag\n    restart: unless-stopped\n"
)


def _kept(text: str) -> dict[str, Mapping[str, object]]:
    items = _data(text).get("kept_outside_runtime") or []
    assert isinstance(items, list)
    return {str(i["service"]): i for i in items}


def test_forwarder_decision_is_recorded_with_its_reason() -> None:
    """AC1: the allowlist entry names the forwarder and the second bus connection or failure domain."""
    entry = _kept(_real_text()).get(_FORWARDER)
    assert entry is not None, f"{_DATA} records no decision for {_FORWARDER}"
    reason = str(entry.get("reason") or "").strip()
    assert reason, "the forwarder entry carries an empty reason"
    assert "second bus connection" in reason or "failure domain" in reason, reason
    assert entry.get("ticket") == "OMN-22393"
    assert entry.get("files") == [_FORWARDER_COMPOSE]


def test_every_kept_entry_names_a_service_its_files_declare() -> None:
    """The entry is about a real service: each listed file declares it."""
    kept = _kept(_real_text())
    assert kept, "the data file records no kept service"
    for service, entry in kept.items():
        files = entry["files"]
        assert isinstance(files, list) and files, service
        for file in files:
            declared = _services_of((_REPO / str(file)).read_text(encoding="utf-8"))
            assert service in declared, f"{file} declares no service {service}"


def test_forwarder_is_absent_from_every_judged_dev_compose_file() -> None:
    """AC2: the dev file set holds no forwarder service and judges clean; the edge file is the one home."""
    data = _data(_real_text())
    holders = [
        f
        for f in _judged(_REPO, data)
        if _FORWARDER in _services_of((_REPO / f).read_text(encoding="utf-8"))
    ]
    assert holders == []
    assert _FORWARDER_COMPOSE in _strings(data, "not_a_stack")
    assert _judge(_REPO) == ()


def test_kept_entry_with_an_empty_reason_fails() -> None:
    """AC1 falsifier: blanking the reason of the real forwarder entry is refused."""
    data = _data(_real_text())
    kept = data["kept_outside_runtime"]
    assert isinstance(kept, list)
    blanked = yaml.safe_dump(
        {
            **data,
            "kept_outside_runtime": [
                {**i, "reason": "  "} if i["service"] == _FORWARDER else i for i in kept
            ],
        }
    )
    refused = _text(_judge(_REPO, text=blanked))
    assert f"kept service {_FORWARDER} carries no reason" in refused


def test_kept_service_is_allowed_only_in_the_files_its_entry_names(
    tmp_path: Path,
) -> None:
    """A kept service passes in its named file and is refused by name in any other judged file."""
    root = _stack(tmp_path)
    data = root / _DATA
    data.write_text(data.read_text(encoding="utf-8") + _KEPT_FIXTURE, encoding="utf-8")
    # control: no listed file declares it yet, so the entry is stale
    assert "declares no service fixture-bridge" in _text(_judge(root))
    _plant(root / _OVERLAY, _BRIDGE)
    assert _judge(root) == ()
    _plant(root / _INFRA, _BRIDGE)
    assert f"service fixture-bridge in {_INFRA}" in _text(_judge(root))


def test_kept_service_without_an_entry_is_refused(tmp_path: Path) -> None:
    """Control: the same service with no kept entry is refused by name."""
    root = _stack(tmp_path)
    _plant(root / _OVERLAY, _BRIDGE)
    assert f"service fixture-bridge in {_OVERLAY}" in _text(_judge(root))


# -- the wiring: the hook and the unconditional CI job -------------------------------------------

_JOB = "stack-shape"
_CONTEXT = "Stack Shape (OMN-21891)"


def _hook() -> dict[str, object]:
    config = yaml.safe_load((_REPO / ".pre-commit-config.yaml").read_text())
    hooks: list[dict[str, object]] = [
        h for r in config["repos"] for h in r["hooks"] if h["id"] == _JOB
    ]
    assert len(hooks) == 1
    return hooks[0]


def test_hook_is_renamed_and_the_deny_list_is_gone() -> None:
    config = (_REPO / ".pre-commit-config.yaml").read_text()
    assert "dev-stack-runtime-only" not in config
    assert "test_dev_stack_runtime_only" not in config
    assert not (
        _REPO / "tests/unit/docker/test_dev_stack_runtime_only_omn21288.py"
    ).exists()
    assert "test_checker_stack_shape.py" in str(_hook()["entry"])


def test_hook_triggers_on_every_compose_file_and_the_data_file() -> None:
    pattern = re.compile(str(_hook()["files"]))
    compose = sorted(
        p.relative_to(_REPO).as_posix()
        for p in _REPO.glob("docker/docker-compose*.yml")
    )
    assert compose
    assert [p for p in compose if not pattern.search(p)] == []
    assert pattern.search("docker/docker-compose.a-lane-that-does-not-exist-yet.yml")
    assert pattern.search(_DATA.as_posix())


def test_ci_job_runs_unconditionally_on_every_pull_request() -> None:
    workflow = yaml.safe_load((_REPO / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"][_JOB]
    assert job["name"] == _CONTEXT
    # no `if`, no `needs`, no path filter: a baseline-only PR is judged too
    assert not {"if", "needs", "paths", "paths-ignore"} & set(job)
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert not {"paths", "paths-ignore"} & set(triggers["pull_request"])
    steps = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert "tests/unit/validation/test_checker_stack_shape.py" in steps
    assert "STACK_SHAPE_BASE_REF" in steps

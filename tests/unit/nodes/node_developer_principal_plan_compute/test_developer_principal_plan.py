# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20821: the developer principal plan compute node applies the lab issuer's rules.

The lab issuer (``scripts/lab_principal_issuer.py``) and the node must accept and
refuse the same declarations, name the same principal, digest the same bytes and
grant the same list. The node is pure: it plans from what the request carries and
touches no file, socket, process, environment, clock or random source. It declares
the local-dev profile, which no deployed runtime carries, so neither main nor
effects attaches it.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import time
from collections.abc import Callable, Iterator, MutableMapping
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers import handler_developer_principal_plan
from omnibase_infra.handlers.handler_developer_principal_plan import (
    HandlerDeveloperPrincipalPlan,
)
from omnibase_infra.models.developer_principal import (
    ModelDeveloperPrincipalPlan,
    ModelDeveloperPrincipalPlanRequest,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from scripts import lab_principal_issuer as lpi

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB = REPO_ROOT / "deploy" / "lab"
LAB_DECLARATION = LAB / "developer-principal-grants.yaml"
LOCAL_DECLARATION = REPO_ROOT / "deploy" / "local" / "developer-principal-grants.yaml"
NODE_DIR = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_developer_principal_plan_compute"
)
LOGIN = "dev.person@omninode.ai"


def _local() -> dict[str, object]:
    loaded = yaml.safe_load(LOCAL_DECLARATION.read_text())
    assert isinstance(loaded, dict)
    return loaded


def _with(**changes: object) -> dict[str, object]:
    doc = _local()
    doc.update(changes)
    return doc


def _with_grant(grant: dict[str, object]) -> dict[str, object]:
    doc = _local()
    grants = doc["grants"]
    assert isinstance(grants, list)
    return {**doc, "grants": [*grants, grant]}


def _lab_with_long_exact_principal() -> dict[str, object]:
    doc = yaml.safe_load(LAB_DECLARATION.read_text())
    assert isinstance(doc, dict)
    scoped = dict(doc["principal_grants_from"])
    scoped["dev-" + "a" * 60] = "h105-scm-pr-create-derived-grants.json"
    return {**doc, "principal_grants_from": scoped}


# The ten refusal fixtures AC1 names, each a declaration the lab issuer refuses.
REFUSED: dict[str, Callable[[], dict[str, object]]] = {
    "wrong_schema": lambda: _with(schema="developer-principal-grants.v2"),
    "empty_grants": lambda: _with(grants=[]),
    "unknown_operation": lambda: _with_grant(
        {
            "resource": "topic",
            "name": yaml.safe_load(LOCAL_DECLARATION.read_text())["grants"][0]["name"],
            "pattern": "literal",
            "operations": ["alter"],
        }
    ),
    "star_with_read": lambda: _with_grant(
        {
            "resource": "group",
            "name": "*",
            "pattern": "literal",
            "operations": ["describe", "read"],
        }
    ),
    "short_prefixed_group_read": lambda: _with_grant(
        {
            "resource": "group",
            "name": "local.short.",
            "pattern": "prefixed",
            "operations": ["read"],
        }
    ),
    "bad_prefix": lambda: _with(principal_prefix="Dev_"),
    "max_devices_zero": lambda: _with(max_devices_per_login=0),
    "max_devices_twenty_one": lambda: _with(max_devices_per_login=21),
    "principal_name_over_63": _lab_with_long_exact_principal,
}
assert len(REFUSED) == 9  # plus the lab file itself, accepted: the ten fixtures


def _derived_texts(lane: str = "dev") -> dict[str, str]:
    directory = LAB if lane == "dev" else LOCAL_DECLARATION.parent
    return {
        path.name: path.read_text() for path in directory.glob("*-derived-grants.json")
    }


def _issuer_base(path: Path, tmp_path: Path) -> Path:
    doc = yaml.safe_load(path.read_text())
    doc.pop("machine_grants_from", None)
    target = tmp_path / "developer-principal-grants.yaml"
    target.write_text(yaml.safe_dump(doc, sort_keys=False))
    return target


def _issuer_accepts(declaration: Path) -> lpi.Declaration | None:
    try:
        return lpi.load_declaration(declaration)
    except lpi.DeclarationError:
        return None


def _plan(
    text: str,
    *,
    login: str = LOGIN,
    device: str = "mac",
    existing: tuple[str, ...] = (),
) -> ModelDeveloperPrincipalPlan:
    return HandlerDeveloperPrincipalPlan().handle(
        ModelDeveloperPrincipalPlanRequest(
            declaration_yaml=text,
            derived_documents=_derived_texts(str(yaml.safe_load(text)["lane"])),
            login=login,
            device=device,
            existing_principals=existing,
        )
    )


def _issuer_grants(
    declaration: lpi.Declaration, principal: str
) -> set[tuple[object, ...]]:
    return {
        (
            g.resource,
            g.for_principal(principal).name,
            g.pattern,
            g.operations,
            g.create_topic,
        )
        for g in declaration.grants_for(principal)
    }


def _plan_grants(plan: ModelDeveloperPrincipalPlan) -> set[tuple[object, ...]]:
    return {
        (g.resource, g.name, g.pattern, g.operations, g.create_topic)
        for g in plan.grants
    }


# --- AC1: the node and the lab issuer accept and refuse the same fixtures -----


@pytest.mark.parametrize(
    "declaration", [LAB_DECLARATION, LOCAL_DECLARATION], ids=["lab", "local"]
)
def test_parity_with_lab_issuer_on_an_accepted_declaration(
    declaration: Path, tmp_path: Path
) -> None:
    # The local machine union is checked separately; lab issuer parity covers its base.
    if declaration == LOCAL_DECLARATION:
        declaration = _issuer_base(declaration, tmp_path)
    issued = _issuer_accepts(declaration)
    assert issued is not None
    plan = _plan(declaration.read_text())
    assert plan.refusal_kind == "", plan.refusal
    principal = lpi.principal_name(issued, LOGIN, "mac")
    assert plan.principal == principal
    assert plan.lane == issued.lane
    assert plan.declaration_digest == issued.digest
    assert _plan_grants(plan) == _issuer_grants(issued, principal)


@pytest.mark.parametrize("fixture", sorted(REFUSED))
def test_parity_with_lab_issuer_on_a_refused_declaration(
    tmp_path: Path, fixture: str
) -> None:
    text = yaml.safe_dump(REFUSED[fixture](), sort_keys=False)
    declaration = tmp_path / "developer-principal-grants.yaml"
    declaration.write_text(text)
    for name, derived in _derived_texts(str(yaml.safe_load(text)["lane"])).items():
        (tmp_path / name).write_text(derived)

    assert _issuer_accepts(declaration) is None, f"the lab issuer accepts {fixture}"
    plan = _plan(text)
    assert (plan.decision, plan.refusal_kind) == ("refused", "declaration"), plan


# --- AC2: the device cap, rotation and the pinned naming cases -----------------


def test_device_cap_refuses_a_new_principal_and_rotates_a_present_one() -> None:
    text = LOCAL_DECLARATION.read_text()
    cap = _local()["max_devices_per_login"]
    assert isinstance(cap, int)
    held = tuple(f"dev-dev-person-device{i}" for i in range(cap))

    new = _plan(text, device="another", existing=held)
    assert (new.decision, new.refusal_kind) == ("refused", "device_cap")

    present = _plan(text, device="device0", existing=held)
    assert (present.decision, present.principal) == ("rotate", "dev-dev-person-device0")

    under = _plan(text, device="another", existing=held[:-1])
    assert under.decision == "create"


def test_device_cap_ignores_principals_outside_the_declaration_prefix() -> None:
    held = tuple(f"lab-runtime-{i}" for i in range(20))
    assert _plan(LOCAL_DECLARATION.read_text(), existing=held).decision == "create"


def test_device_cap_collision_pair_plans_rotate_for_the_second() -> None:
    text = LOCAL_DECLARATION.read_text()
    first = _plan(text, login="a@omninode.ai", device="b-c")
    assert (first.decision, first.principal) == ("create", "dev-a-b-c")
    second = _plan(
        text, login="a-b@omninode.ai", device="c", existing=(first.principal,)
    )
    assert (second.decision, second.principal) == ("rotate", "dev-a-b-c")


def test_device_cap_seventy_character_name_is_capped_with_its_digest(
    tmp_path: Path,
) -> None:
    login, device = "a" * 40 + "@omninode.ai", "d" * 25
    plan = _plan(LOCAL_DECLARATION.read_text(), login=login, device=device)
    uncapped = f"dev-{'a' * 40}-{'d' * 25}"
    assert len(uncapped) == 70
    assert len(plan.principal) == 63
    assert plan.principal.endswith(hashlib.sha256(uncapped.encode()).hexdigest()[:8])
    declaration = lpi.load_declaration(_issuer_base(LOCAL_DECLARATION, tmp_path))
    assert plan.principal == lpi.principal_name(declaration, login, device)


# --- AC3: the handler is pure ---------------------------------------------------

ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "dataclasses",
        "hashlib",
        "re",
        "collections.abc",
        "typing",
        "pydantic",
        "yaml",
    }
)


def _raise(*args: object, **kwargs: object) -> object:
    raise AssertionError("the handler performed I/O or read ambient state")


class _RaisingEnviron(MutableMapping[str, str]):
    """An environment every read of which fails the test."""

    def __getitem__(self, key: str) -> str:
        raise AssertionError(f"the handler read the environment ({key})")

    def __setitem__(self, key: str, value: str) -> None:
        raise AssertionError(f"the handler wrote the environment ({key})")

    def __delitem__(self, key: str) -> None:
        raise AssertionError(f"the handler wrote the environment ({key})")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("the handler listed the environment")

    def __len__(self) -> int:
        raise AssertionError("the handler sized the environment")


def test_pure_handler_plans_with_io_and_ambient_state_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ModelDeveloperPrincipalPlanRequest(
        declaration_yaml=LAB_DECLARATION.read_text(),
        derived_documents=_derived_texts(),
        login=LOGIN,
        device="mac",
    )
    handler = HandlerDeveloperPrincipalPlan()
    expected = handler.handle(request)

    with monkeypatch.context() as patch:
        for target, name in (
            (builtins, "open"),
            (Path, "read_text"),
            (socket, "socket"),
            (subprocess, "Popen"),
            (time, "time"),
            (random, "random"),
        ):
            patch.setattr(target, name, _raise)
        patch.setattr(os, "environ", _RaisingEnviron())
        planned = handler.handle(request)

    assert planned == expected


def test_pure_handler_module_imports_only_the_allowed_modules() -> None:
    source = Path(handler_developer_principal_plan.__file__).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    outside = sorted(
        name
        for name in imported
        if name not in ALLOWED_IMPORTS and not name.startswith("omnibase_infra.")
    )
    assert outside == [], f"the handler imports {outside}"


# --- AC5: the same request twice returns the same plan -------------------------


@pytest.mark.parametrize(
    "declaration", [LAB_DECLARATION, LOCAL_DECLARATION], ids=["lab", "local"]
)
def test_deterministic_same_request_twice_is_byte_identical(declaration: Path) -> None:
    text = declaration.read_text()
    first = _plan(text, existing=("dev-dev-person-mac",)).model_dump_json()
    second = _plan(text, existing=("dev-dev-person-mac",)).model_dump_json()
    assert first == second
    assert json.loads(first)["decision"] == "rotate"


# --- AC6: no main or effects runtime owns the node -----------------------------


@pytest.mark.parametrize("profile", ["main", "effects"])
def test_profile_ownership_main_and_effects_own_none_of_the_node(profile: str) -> None:
    manifest = discover_contracts_from_paths([NODE_DIR / "contract.yaml"])
    assert [c.name for c in manifest.contracts] == [
        "node_developer_principal_plan_compute"
    ]
    result = filter_manifest_for_runtime_profile(manifest, profile, environ={})
    assert [c.name for c in result.manifest.contracts] == []


def test_profile_ownership_the_local_dev_profile_owns_the_node() -> None:
    """Positive control: the filter is not refusing every contract."""
    manifest = discover_contracts_from_paths([NODE_DIR / "contract.yaml"])
    result = filter_manifest_for_runtime_profile(manifest, "local-dev", environ={})
    assert [c.name for c in result.manifest.contracts] == [
        "node_developer_principal_plan_compute"
    ]


# --- AC7: the pre-commit entry runs these tests --------------------------------


def test_pre_commit_entry_runs_the_plan_and_grants_parity_tests() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("id") == "developer-principal-plan"
    ]
    assert len(hooks) == 1, "no single developer-principal-plan pre-commit entry"
    (hook,) = hooks
    assert "tests/unit/nodes/node_developer_principal_plan_compute" in hook["entry"]
    assert "tests/scripts/test_developer_principal_grants_parity.py" in hook["entry"]
    assert hook["pass_filenames"] is False
    files = re.compile(hook["files"])
    for path in (
        "src/omnibase_infra/nodes/node_developer_principal_plan_compute/contract.yaml",
        "scripts/lab_principal_issuer.py",
        "deploy/lab/developer-principal-grants.yaml",
        "deploy/local/developer-principal-grants.yaml",
    ):
        assert files.search(path), path


def test_parity_with_lab_issuer_on_the_scoped_write_topics() -> None:
    assert (
        handler_developer_principal_plan.SCOPED_WRITE_TOPICS == lpi.SCOPED_WRITE_TOPICS
    )


def test_machine_supplement_is_bound_to_digest_but_never_a_developer_plan() -> None:
    documents = _derived_texts()
    request = ModelDeveloperPrincipalPlanRequest(
        declaration_yaml=LAB_DECLARATION.read_text(),
        derived_documents=documents,
        login=LOGIN,
        device="mac",
    )
    handler = HandlerDeveloperPrincipalPlan()
    plan = handler.handle(request)
    assert plan.refusal_kind == ""
    assert not any(
        grant.name == "onex.evt.omnimarket.credential-registered.v1"
        or (
            grant.name.startswith("onex.evt.omnimarket.work-ledger-")
            and "append" not in grant.name
            and "query" not in grant.name
        )
        for grant in plan.grants
    )
    source = "machine-principal-derived-grants.json"
    changed = handler.handle(
        request.model_copy(
            update={
                "derived_documents": {**documents, source: documents[source] + "\n"},
            }
        )
    )
    assert changed.declaration_digest != plan.declaration_digest
    assert changed.grants == plan.grants
    missing = handler.handle(
        request.model_copy(
            update={
                "derived_documents": {
                    k: v for k, v in documents.items() if k != source
                },
            }
        )
    )
    assert missing.refusal_kind == "declaration"
    assert "derived document was not supplied" in missing.refusal


# OMN-20825 AC6: local grants are the complete approved union.
EXPECTED_LOCAL_BASE: set[tuple[str, str, str, tuple[str, ...], bool]] = {
    (
        "topic",
        "onex.cmd.omnimarket.delegate-skill.v1",
        "literal",
        ("describe", "write"),
        False,
    ),
    (
        "topic",
        "onex.evt.omnimarket.delegate-skill-completed.v1",
        "literal",
        ("describe", "read", "write"),
        False,
    ),
    (
        "topic",
        "onex.evt.omnimarket.delegate-skill-failed.v1",
        "literal",
        ("describe", "read", "write"),
        False,
    ),
    (
        "group",
        "local.omnibase_core.runtime_local_",
        "prefixed",
        ("describe", "read"),
        False,
    ),
    ("group", "*", "literal", ("describe",), False),
}


def _expected_local_machine_grants() -> set[
    tuple[str, str, str, tuple[str, ...], bool]
]:
    bindings = yaml.safe_load((LAB / "machine-principal-bindings.yaml").read_text())[
        "bindings"
    ]
    assert len(bindings) == 13
    assert all(b["broker"] == "dev" and b["direction"] == "produce" for b in bindings)
    return EXPECTED_LOCAL_BASE | {
        ("topic", b["physical_topic"], "literal", ("describe", "write"), False)
        for b in bindings
    }


def _local_machine_request() -> ModelDeveloperPrincipalPlanRequest:
    local = _local()
    local["machine_grants_from"] = "machine-principal-derived-grants.json"
    doc = json.loads((LAB / "machine-principal-derived-grants.json").read_text())
    for row in doc["grants"]:
        row["broker"] = "local"
    return ModelDeveloperPrincipalPlanRequest(
        declaration_yaml=yaml.safe_dump(local, sort_keys=False),
        derived_documents={"machine-principal-derived-grants.json": json.dumps(doc)},
        login=LOGIN,
        device="mac",
    )


def test_local_machine_supplement_whole_plan_matches_union() -> None:
    request = _local_machine_request()
    result = HandlerDeveloperPrincipalPlan().handle(request)
    assert result.refusal_kind == ""
    plan = result
    expected = _expected_local_machine_grants()
    assert _plan_grants(plan) == expected
    assert len(plan.grants) == len(expected)
    raw = {
        (
            g["resource"],
            g["name"],
            g["pattern"],
            tuple(g["operations"]),
            g.get("create_topic", False),
        )
        for g in _local()["grants"]
    }
    assert raw == EXPECTED_LOCAL_BASE


def test_local_machine_supplement_shipped_document() -> None:
    local = _local()
    source = local["machine_grants_from"]
    assert source == "machine-principal-derived-grants.json"
    text = (LOCAL_DECLARATION.parent / source).read_text()
    request = ModelDeveloperPrincipalPlanRequest(
        declaration_yaml=LOCAL_DECLARATION.read_text(),
        derived_documents={source: text},
        login=LOGIN,
        device="mac",
    )
    result = HandlerDeveloperPrincipalPlan().handle(request)
    assert result.refusal_kind == "", result.refusal
    assert result.decision == "create"
    expected = _expected_local_machine_grants()
    assert _plan_grants(result) == expected
    assert len(result.grants) == len(expected)


def test_local_machine_supplement_digest_binds_document() -> None:
    request = _local_machine_request()
    baseline = HandlerDeveloperPrincipalPlan().handle(request)
    assert baseline.refusal_kind == ""
    baseline_digest = baseline.declaration_digest
    baseline_grants = _plan_grants(baseline)

    doc = json.loads(request.derived_documents["machine-principal-derived-grants.json"])
    doc_bytes = json.dumps(doc).encode() + b"\n"
    request2 = request.model_copy(
        update={
            "derived_documents": {
                "machine-principal-derived-grants.json": doc_bytes.decode()
            }
        }
    )
    result2 = HandlerDeveloperPrincipalPlan().handle(request2)
    assert result2.refusal_kind == ""
    assert result2.declaration_digest != baseline_digest
    assert _plan_grants(result2) == baseline_grants


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "empty",
        "malformed",
        "wrong_broker",
        "invalid_operation",
        "invalid_principal",
    ],
)
def test_local_machine_supplement_invalid_documents(case: str) -> None:
    request = _local_machine_request()
    doc = json.loads((LAB / "machine-principal-derived-grants.json").read_text())
    for row in doc["grants"]:
        row["broker"] = "local"
    if case == "missing":
        documents: dict[str, str] = {}
    elif case == "empty":
        doc["grants"] = []
        documents = {"machine-principal-derived-grants.json": json.dumps(doc)}
    elif case == "malformed":
        documents = {"machine-principal-derived-grants.json": "{"}
    elif case == "wrong_broker":
        doc["grants"][0]["broker"] = "dev"
        documents = {"machine-principal-derived-grants.json": json.dumps(doc)}
    elif case == "invalid_operation":
        doc["grants"][0]["operation"] = "ALTER"
        documents = {"machine-principal-derived-grants.json": json.dumps(doc)}
    elif case == "invalid_principal":
        doc["principal"] = "somebody"
        documents = {"machine-principal-derived-grants.json": json.dumps(doc)}
    else:
        raise AssertionError(f"unknown case {case}")
    req = request.model_copy(update={"derived_documents": documents})
    result = HandlerDeveloperPrincipalPlan().handle(req)
    assert result.refusal_kind == "declaration"


@pytest.mark.parametrize("lane", ["prod", "stability"])
def test_local_machine_supplement_other_lane_refused(lane: str) -> None:
    request = _local_machine_request()
    local = yaml.safe_load(request.declaration_yaml)
    local["lane"] = lane
    req = request.model_copy(
        update={"declaration_yaml": yaml.safe_dump(local, sort_keys=False)}
    )
    result = HandlerDeveloperPrincipalPlan().handle(req)
    assert result.refusal_kind == "declaration"

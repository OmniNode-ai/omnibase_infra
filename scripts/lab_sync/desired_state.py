#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""desired_state.py -- the lab-desired-state.v1 contract's validator (OMN-19410).

Seam L0.1 of the lab release-sync plan: release or merge -> generator (T1.1)
-> detector (T1.2). The generator writes one desired-state document per lab
surface (a compose lane, a runner fleet, or a host's allowed-undeclared list);
the detector reads it through :func:`load_desired_state`. This module is the
reading side's contract and the writing side's stamp
(:func:`compute_desired_state_sha256`).

STDLIB ONLY, on purpose: the census runs under the lab host's system
interpreter with no venv. The schema file
``deploy/lab-sync/desired-state.schema.json`` is the one source of truth; this
module interprets it rather than restating it. It implements exactly the JSON
Schema keywords that file uses, plus two of its own (``x-sorted-by`` and
``x-sorted``), and REFUSES a schema that uses any other keyword, so a schema
edit can never be silently unenforced here (``self-check``).

Fail closed. A document that is unreadable, not JSON, carries a duplicate key,
misses a required field, carries an unknown field, is out of canonical order,
or whose embedded ``desired_state_sha256`` does not match its body raises
:class:`DesiredStateRefusalError`, naming the field. Nothing here returns a partial
or defaulted state: the detector turns a refusal into ``desired_state_unreadable``,
never into a clean reading.

Determinism. ``desired_state_sha256`` is the sha256 of the canonical body: the
document minus ``desired_state_sha256``, as JSON with sorted keys, separators
``(",", ":")``, UTF-8 and no ASCII escaping. Object key order therefore never
moves the hash; array order does, which is why every array of named entries
must be sorted and unique (``x-sorted-by``) and every label list sorted
(``x-sorted``).

CLI::

    desired_state.py validate PATH [PATH...]     exit 0 all valid, 1 any refused
    desired_state.py render --input INPUTS.json [--output DESIRED.json]
    desired_state.py render --ref REF --host HOST --lane LANE
                           --compose-file FILE [--compose-file OVERLAY]
                           [--composition REFS.json] [--broker-file PROFILE]
    desired_state.py self-check [--schema PATH] [--fixtures DIR]
                                                 exit 0 OK, 1 refused

``render`` runs in the repository's uv environment. Its source adapter reads
manifest and lock at the resolved commit, refuses Compose files differing from
that ref, and runs only ``compose config --format json`` and ``config --hash``.
Supply the accepted deploy's environment, exact file order and original project
directory: Compose includes absolute bind paths in its hash. ``--composition``
is the accepted build's list of sibling ``{repo, ref, commit}`` records. For
stability-test, the caller supplies the latest release tag (D2); for dev, the
last accepted build tuple. Resolving release/deploy authority and rollout stays
with those existing owners. ``--input`` replays the typed source-artifact bundle
for generated deployments; the bundle may contain credentials and is not an
output artifact. Only desired state is printed. ``--output`` preserves the
file's mtime when the rendered bytes already match.

``self-check`` proves the contract holds together: every keyword in the schema
is one this module enforces, every ``desired_state_valid*.json`` fixture
loads, and every ``desired_state_missing_field*`` / ``desired_state_extra_field*``
fixture is refused. ``tests/unit/scripts/test_lab_desired_state.py`` runs it
under ``-I -S`` (no site-packages, no PYTHONPATH), so a third-party import in
this file fails the required unit suite instead of the lab host. It is not a
separately wired gate: the required unit suite is selected for every path this
contract lives in, and the blocking check over live lab state is ``lane_sync``
(T1.4), which consumes :func:`load_desired_state`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "lab-desired-state.v1"
SHA_FIELD = "desired_state_sha256"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = _REPO_ROOT / "deploy" / "lab-sync" / "desired-state.schema.json"
DEFAULT_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "lab_sync"

# Keywords that only annotate. They are allowed anywhere and enforce nothing.
_ANNOTATIONS = frozenset({"$schema", "$id", "title", "description", "$comment"})
# Keywords this module enforces. A schema using anything else is refused.
_ENFORCED = frozenset(
    {
        "type",
        "const",
        "enum",
        "$ref",
        "required",
        "additionalProperties",
        "properties",
        "propertyNames",
        "minProperties",
        "items",
        "minItems",
        "maxItems",
        "x-sorted-by",
        "x-sorted",
        "pattern",
        "minLength",
        "minimum",
        "allOf",
        "if",
        "then",
    }
)
_ROOT_ONLY = frozenset({"$defs"})

_TYPE_NAMES = {
    dict: "object",
    list: "array",
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    type(None): "null",
}


class DesiredStateRefusalError(ValueError):
    """A desired-state document was refused.

    ``field`` is the path of the first offending field (``""`` for the whole
    document), for example ``containers[2].config_hash``. ``errors`` carries
    every ``(field, reason)`` found, in schema traversal order.
    """

    def __init__(self, errors: list[tuple[str, str]]) -> None:
        if not errors:
            errors = [("", "refused with no reason recorded")]
        self.errors: tuple[tuple[str, str], ...] = tuple(errors)
        self.field, self.reason = self.errors[0]
        lines = [f"{field or '<document>'}: {reason}" for field, reason in errors]
        extra = (
            f" (and {len(lines) - 1} more: {'; '.join(lines[1:])})" if lines[1:] else ""
        )
        super().__init__(f"{SCHEMA_VERSION} refused: {lines[0]}{extra}")


class SchemaUnsupportedError(ValueError):
    """The schema uses a keyword this validator does not enforce."""


@dataclass(frozen=True)
class DesiredState:
    """A validated desired-state document.

    ``key`` is the document's identity, ``(surface.id, target_ref.repo,
    target_ref.commit)``: a re-render for the same key must produce the same
    ``desired_state_sha256``.
    """

    document: dict[str, Any]
    desired_state_sha256: str

    @property
    def surface_id(self) -> str:
        return str(self.document["surface"]["id"])

    @property
    def surface_kind(self) -> str:
        return str(self.document["surface"]["kind"])

    @property
    def key(self) -> tuple[str, str, str]:
        target = self.document["target_ref"]
        return (self.surface_id, str(target["repo"]), str(target["commit"]))


# ------------------------------------------------------------------ hashing


def canonical_body_bytes(document: Mapping[str, Any]) -> bytes:
    """The bytes ``desired_state_sha256`` is computed over."""
    body = {k: v for k, v in document.items() if k != SHA_FIELD}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_desired_state_sha256(document: Mapping[str, Any]) -> str:
    """The generator's stamp and the loader's check: sha256 of the canonical body."""
    return hashlib.sha256(canonical_body_bytes(document)).hexdigest()


# ------------------------------------------------------------------ parsing


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DesiredStateRefusalError(
                [("", f"duplicate key '{key}' in one object")]
            )
        out[key] = value
    return out


def _refuse_constant(name: str) -> Any:
    raise DesiredStateRefusalError([("", f"non-JSON constant {name}")])


def _parse_json(text: str) -> Any:
    try:
        return json.loads(
            text, object_pairs_hook=_no_duplicate_keys, parse_constant=_refuse_constant
        )
    except json.JSONDecodeError as exc:
        raise DesiredStateRefusalError([("", f"not valid JSON: {exc}")]) from exc


# ---------------------------------------------------------------- the schema


def _render(path: tuple[str | int, ...]) -> str:
    out = ""
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else part
    return out


def _json_type(value: object) -> str:
    return _TYPE_NAMES.get(type(value), type(value).__name__)


def _is_type(value: object, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return _json_type(value) == name


def _json_equal(a: object, b: object) -> bool:
    return _json_type(a) == _json_type(b) and a == b


def _compile(pattern: str) -> re.Pattern[str]:
    # JSON Schema patterns are searched, as re.search does, but Python's `$`
    # also matches before a trailing newline. A trailing `$` is made strict.
    if pattern.endswith("$") and not pattern.endswith("\\$"):
        pattern = pattern[:-1] + r"\Z"
    return re.compile(pattern)


class _Schema:
    def __init__(self, schema: dict[str, Any]) -> None:
        self.root = schema
        self.defs: dict[str, Any] = schema.get("$defs", {})
        self._patterns: dict[str, re.Pattern[str]] = {}

    # ---- the keyword audit

    def unsupported(self) -> list[str]:
        """Every keyword in the schema this module would not enforce."""
        found: list[str] = []
        self._audit(self.root, "#", found, root=True)
        for name, sub in sorted(self.defs.items()):
            self._audit(sub, f"#/$defs/{name}", found, root=False)
        return found

    def _audit(self, node: object, where: str, found: list[str], *, root: bool) -> None:
        if not isinstance(node, dict):
            found.append(f"{where}: a schema must be an object, got {_json_type(node)}")
            return
        for key, value in node.items():
            if key in _ANNOTATIONS or (root and key in _ROOT_ONLY):
                continue
            if key not in _ENFORCED:
                found.append(
                    f"{where}: keyword '{key}' is not enforced by {Path(__file__).name}"
                )
                continue
            if key == "$ref" and (
                not isinstance(value, str)
                or not value.startswith("#/$defs/")
                or value.removeprefix("#/$defs/") not in self.defs
            ):
                found.append(
                    f"{where}: $ref {value!r} does not name a local $defs entry"
                )
            if key in {"properties"}:
                for name, sub in value.items():
                    self._audit(sub, f"{where}/properties/{name}", found, root=False)
            elif key == "allOf":
                for index, sub in enumerate(value):
                    self._audit(sub, f"{where}/allOf/{index}", found, root=False)
            elif key in {"items", "propertyNames", "if", "then"} or (
                key == "additionalProperties" and isinstance(value, dict)
            ):
                self._audit(value, f"{where}/{key}", found, root=False)
            elif key == "pattern":
                try:
                    _compile(value)
                except re.error as exc:
                    found.append(f"{where}: pattern {value!r} does not compile: {exc}")
        if "if" in node and "then" not in node:
            found.append(f"{where}: 'if' without 'then'")

    # ---- validation

    def _pattern(self, pattern: str) -> re.Pattern[str]:
        compiled = self._patterns.get(pattern)
        if compiled is None:
            compiled = self._patterns[pattern] = _compile(pattern)
        return compiled

    def errors(self, value: object) -> list[tuple[str, str]]:
        found: list[tuple[tuple[str | int, ...], str]] = []
        self._check(value, self.root, (), found)
        return [(_render(path), reason) for path, reason in found]

    def _check(
        self,
        value: object,
        node: dict[str, Any],
        path: tuple[str | int, ...],
        found: list[tuple[tuple[str | int, ...], str]],
    ) -> None:
        if "type" in node:
            allowed = node["type"] if isinstance(node["type"], list) else [node["type"]]
            if not any(_is_type(value, t) for t in allowed):
                found.append(
                    (path, f"expected {' or '.join(allowed)}, got {_json_type(value)}")
                )
                return
        if "const" in node and not _json_equal(value, node["const"]):
            found.append((path, f"must be {node['const']!r}, got {value!r}"))
            return
        if "enum" in node and not any(_json_equal(value, v) for v in node["enum"]):
            found.append((path, f"must be one of {node['enum']!r}, got {value!r}"))
            return
        if "$ref" in node:
            self._check(
                value, self.defs[node["$ref"].removeprefix("#/$defs/")], path, found
            )
        if isinstance(value, dict):
            self._check_object(value, node, path, found)
        elif isinstance(value, list):
            self._check_array(value, node, path, found)
        elif isinstance(value, str):
            if "minLength" in node and len(value) < node["minLength"]:
                found.append(
                    (path, f"must be at least {node['minLength']} character(s)")
                )
            if "pattern" in node and not self._pattern(node["pattern"]).search(value):
                found.append((path, f"{value!r} does not match {node['pattern']!r}"))
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and "minimum" in node
            and value < node["minimum"]
        ):
            found.append((path, f"must be at least {node['minimum']}, got {value!r}"))
        for branch in node.get("allOf", []):
            self._check(value, branch, path, found)
        if "if" in node:
            self._check_conditional(value, node, path, found)

    def _check_conditional(
        self,
        value: object,
        branch: dict[str, Any],
        path: tuple[str | int, ...],
        found: list[tuple[tuple[str | int, ...], str]],
    ) -> None:
        probe: list[tuple[tuple[str | int, ...], str]] = []
        self._check(value, branch["if"], path, probe)
        if probe:
            return
        condition = _describe_condition(branch["if"])
        conditional: list[tuple[tuple[str | int, ...], str]] = []
        self._check(value, branch["then"], path, conditional)
        found.extend((p, f"{reason} when {condition}") for p, reason in conditional)

    def _check_object(
        self,
        value: dict[str, Any],
        node: dict[str, Any],
        path: tuple[str | int, ...],
        found: list[tuple[tuple[str | int, ...], str]],
    ) -> None:
        for name in node.get("required", []):
            if name not in value:
                found.append(((*path, name), f"required field '{name}' is missing"))
        properties: dict[str, Any] = node.get("properties", {})
        extra = node.get("additionalProperties", True)
        for name in sorted(value):
            if name in properties:
                continue
            if extra is False:
                found.append(
                    ((*path, name), f"additional property '{name}' is not allowed")
                )
            elif isinstance(extra, dict):
                self._check(value[name], extra, (*path, name), found)
        if "propertyNames" in node:
            for name in sorted(value):
                probe: list[tuple[tuple[str | int, ...], str]] = []
                self._check(name, node["propertyNames"], (*path, name), probe)
                found.extend((p, f"property name {reason}") for p, reason in probe)
        if "minProperties" in node and len(value) < node["minProperties"]:
            found.append((path, f"must have at least {node['minProperties']} field(s)"))
        for name, sub in properties.items():
            if name in value:
                self._check(value[name], sub, (*path, name), found)

    def _check_array(
        self,
        value: list[Any],
        node: dict[str, Any],
        path: tuple[str | int, ...],
        found: list[tuple[tuple[str | int, ...], str]],
    ) -> None:
        if "minItems" in node and len(value) < node["minItems"]:
            found.append(
                (
                    path,
                    f"must have at least {node['minItems']} item(s), got {len(value)}",
                )
            )
        if "maxItems" in node and len(value) > node["maxItems"]:
            found.append(
                (
                    path,
                    f"must have at most {node['maxItems']} item(s), got {len(value)}",
                )
            )
        sort_field = node.get("x-sorted-by")
        if sort_field is not None and all(
            isinstance(item, dict) and isinstance(item.get(sort_field), str)
            for item in value
        ):
            _check_canonical_order(
                [item[sort_field] for item in value], sort_field, path, found
            )
        if node.get("x-sorted") is True and all(
            isinstance(item, str) for item in value
        ):
            _check_canonical_order(list(value), None, path, found)
        if "items" in node:
            for index, item in enumerate(value):
                self._check(item, node["items"], (*path, index), found)


def _check_canonical_order(
    keys: list[str],
    field: str | None,
    path: tuple[str | int, ...],
    found: list[tuple[tuple[str | int, ...], str]],
) -> None:
    on = f" on '{field}'" if field else ""
    by = f" by '{field}'" if field else ""
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            found.append((path, f"entries must be unique{on}; '{key}' appears twice"))
            return
        seen.add(key)
    if keys != sorted(keys):
        found.append(
            (
                path,
                f"entries must be sorted{by} (canonical order keeps the hash deterministic)",
            )
        )


def _describe_condition(node: dict[str, Any], prefix: str = "") -> str:
    parts: list[str] = []
    if "const" in node:
        parts.append(f"{prefix or '<document>'} is {node['const']!r}")
    for name, sub in node.get("properties", {}).items():
        described = _describe_condition(sub, f"{prefix}.{name}" if prefix else name)
        if described:
            parts.append(described)
    return " and ".join(parts)


# ------------------------------------------------------------------ loading


_SCHEMAS: dict[Path, _Schema] = {}


def _load_schema(schema_path: Path | None) -> _Schema:
    path = (schema_path or DEFAULT_SCHEMA_PATH).resolve()
    cached = _SCHEMAS.get(path)
    if cached is not None:
        return cached
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaUnsupportedError(f"schema {path} is unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise SchemaUnsupportedError(f"schema {path} is not a JSON object")
    schema = _Schema(raw)
    problems = schema.unsupported()
    if problems:
        raise SchemaUnsupportedError(f"schema {path}: " + "; ".join(problems))
    _SCHEMAS[path] = schema
    return schema


def parse_desired_state(text: str, *, schema_path: Path | None = None) -> DesiredState:
    """Validate one desired-state document given as JSON text."""
    schema = _load_schema(schema_path)
    document = _parse_json(text)
    errors = schema.errors(document)
    if errors:
        raise DesiredStateRefusalError(errors)
    assert isinstance(document, dict)  # the schema's root type is object
    actual = compute_desired_state_sha256(document)
    if document[SHA_FIELD] != actual:
        raise DesiredStateRefusalError(
            [
                (
                    SHA_FIELD,
                    f"embedded {document[SHA_FIELD]} does not match the body's {actual}",
                )
            ]
        )
    return DesiredState(document=document, desired_state_sha256=actual)


def load_desired_state(path: Path, *, schema_path: Path | None = None) -> DesiredState:
    """Read and validate one desired-state document. Raises DesiredStateRefusalError."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DesiredStateRefusalError([("", f"unreadable: {exc}")]) from exc
    return parse_desired_state(text, schema_path=schema_path)


# ---------------------------------------------------------------------- CLI


def render_desired_state(inputs: dict[str, Any]) -> bytes:
    """Render recorded target-ref artifacts, then validate against T0.1.

    Dependencies are lazy: validate/self-check remain stdlib-only under -I -S.
    Rendering needs the repository environment (uv run). Raw compose inputs may
    contain credentials; only the resulting desired state is ever printed.
    """
    import asyncio
    import importlib

    import yaml

    from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
    from omnibase_core.errors.model_onex_error import ModelOnexError
    from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
    from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
    from omnibase_infra.runtime.contract_loaders.handler_routing_loader import (
        load_and_validate_contract_yaml,
    )
    from omnibase_infra.runtime.core_runtime.routing_map_builder import (
        import_model_cls,
    )

    contract = load_and_validate_contract_yaml(
        _REPO_ROOT
        / "src/omnibase_infra/nodes/node_lab_proof_plan_compute/contract.yaml"
    )
    operation = "lab_desired_state.render"
    routing = contract.raw["handler_routing"]
    entries = [e for e in routing["handlers"] if e.get("operation") == operation]
    if routing["routing_strategy"] != "operation_match" or len(entries) != 1:
        raise ValueError("desired-state operation must resolve to exactly one route")
    entry = entries[0]
    handler_ref = ModelHandlerRef.model_validate(entry["handler"])
    request_cls = import_model_cls(ModelHandlerRef.model_validate(entry["input_model"]))
    result_cls = import_model_cls(ModelHandlerRef.model_validate(entry["output_model"]))
    request = request_cls.model_validate(inputs)
    handler = getattr(importlib.import_module(handler_ref.module), handler_ref.name)()

    async def dispatch() -> bytes:
        # This invocation owns a private, zero-infrastructure bus. The channel is
        # the contract's operation key, not a new shared-broker topic. No caller
        # can select a node implementation or override the declared model seam.
        bus = EventBusInmemory(group=operation, max_history=1)
        results: list[bytes] = []

        async def receive(message: ModelEventMessage) -> None:
            try:
                model = request_cls.model_validate_json(message.value)
                result = handler.handle(model)
                validated = result_cls.model_validate(result).model_dump(mode="json")
                results.append(validated["document_json"].encode("utf-8"))
            except (ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError):
                # Refuse at the transport boundary without logging source
                # artifacts: validation errors can contain credentials. The
                # bus propagates ModelOnexError; it cannot swallow this failure.
                raise ModelOnexError(
                    message="desired-state render refused",
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                ) from None

        await bus.start()
        try:
            unsubscribe = await bus.subscribe(
                operation, on_message=receive, group_id=operation
            )
            try:
                await bus.publish(operation, None, request.model_dump_json().encode())
            finally:
                await unsubscribe()
        finally:
            await bus.close()
        if len(results) != 1:
            raise ValueError("desired-state route did not return exactly one result")
        return results[0]

    try:
        payload = asyncio.run(dispatch())
    except ModelOnexError:
        raise ValueError("desired-state render refused") from None
    parse_desired_state(payload.decode("utf-8"))
    return payload


def _capture_render_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Read versioned sources and ask Compose for its own hashes, read-only.

    The caller supplies the deploy's exact file order, project directory and
    environment. Compose resolves relative bind paths into absolute paths, so
    moving a checkout or substituting an environment changes the config hash.
    Never approximate that hash, or use a running label as a desired value.
    """
    import yaml

    def run(argv: list[str]) -> str:
        result = subprocess.run(
            argv,
            cwd=args.source_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            # Compose stderr can include interpolated credentials. Do not echo it.
            raise ValueError(
                f"{argv[0]} source read/render failed (exit {result.returncode})"
            )
        return result.stdout

    commit = run(["git", "rev-parse", "--verify", f"{args.ref}^{{commit}}"]).strip()

    def source(path: str) -> str:
        return run(["git", "show", f"{commit}:{path}"])

    manifest = source("deploy/lane-census/lane-manifest.yaml")
    inputs: dict[str, Any] = {
        "target_ref": {
            "repo": "OmniNode-ai/omnibase_infra",
            "ref": args.ref,
            "commit": commit,
            "kind": args.kind,
            "composition": json.loads(args.composition.read_text())
            if args.composition
            else [],
        },
        "host": args.host,
        "lane": args.lane,
        "surface_kind": args.surface_kind,
        "manifest_yaml": manifest,
        "lock_toml": source("uv.lock"),
        "compose_json": '{"services": {}}',
        "compose_hashes": "",
    }
    if args.surface_kind != "host":
        if not args.compose_file:
            raise ValueError("render requires the deploy's ordered --compose-file list")
        # Refuse a checkout whose compose inputs differ from the target ref.
        # Generated overlays belong to the accepted deployment, not this source
        # adapter; those inputs are replayed explicitly with --input instead.
        for path in args.compose_file:
            if (args.source_root / path).read_text() != source(path):
                raise ValueError(f"compose file {path} differs from target ref")
        project = (
            yaml.safe_load(manifest)["lanes"][args.lane]["compose_project"]
            if args.surface_kind == "compose_lane"
            else args.project
        )
        if not project:
            raise ValueError("runner render requires --project")
        command = ["docker", "compose"]
        for path in args.compose_file:
            command.extend(["-f", path])
        command.extend(["-p", project, "--profile", "*", "config"])
        inputs["compose_json"] = run([*command, "--format", "json"])
        services = sorted(json.loads(inputs["compose_json"])["services"])
        if not services:
            raise ValueError("compose render returned no services")
        inputs["compose_hashes"] = run([*command, "--hash", ",".join(services)])
        inputs["broker_yaml"] = source(args.broker_file) if args.broker_file else "{}"
    if args.surface_kind == "runner_fleet":
        inputs.update(
            fleet_yaml=source("config/runner_fleet.yaml"),
            runner_host_address=args.runner_host,
            runner_workdir=args.runner_workdir,
        )
    return inputs


def _cmd_render(args: argparse.Namespace) -> int:
    inputs = (
        json.loads(args.input.read_text())
        if args.input
        else _capture_render_inputs(args)
    )
    payload = render_desired_state(inputs)
    if args.output:
        # A replay for the same key rewrites nothing, including the file's mtime.
        if not args.output.exists() or args.output.read_bytes() != payload:
            args.output.write_bytes(payload)
    else:
        sys.stdout.buffer.write(payload)
    return 0


def _cmd_validate(paths: list[Path], schema_path: Path | None) -> int:
    refused = 0
    for path in paths:
        try:
            state = load_desired_state(path, schema_path=schema_path)
        except DesiredStateRefusalError as exc:
            refused += 1
            sys.stderr.write(f"{path}: REFUSED {exc}\n")
            continue
        sys.stdout.write(
            f"{path}: OK {SHA_FIELD}={state.desired_state_sha256} key={'|'.join(state.key)}\n"
        )
    return 1 if refused else 0


def _cmd_self_check(schema_path: Path | None, fixture_dir: Path) -> int:
    failures: list[str] = []
    valid = sorted(fixture_dir.glob("desired_state_valid*.json"))
    invalid = sorted(
        [
            *fixture_dir.glob("desired_state_missing_field*.json"),
            *fixture_dir.glob("desired_state_extra_field*.json"),
        ]
    )
    if not valid:
        failures.append(f"no desired_state_valid*.json fixture under {fixture_dir}")
    if not invalid:
        failures.append(f"no missing-field or extra-field fixture under {fixture_dir}")
    for path in valid:
        try:
            load_desired_state(path, schema_path=schema_path)
        except DesiredStateRefusalError as exc:
            failures.append(f"{path.name} must load and was refused: {exc}")
    for path in invalid:
        try:
            load_desired_state(path, schema_path=schema_path)
        except DesiredStateRefusalError:
            continue
        failures.append(f"{path.name} must be refused and loaded")
    if failures:
        for line in failures:
            sys.stderr.write(f"self-check FAILED: {line}\n")
        return 1
    sys.stdout.write(
        f"self-check OK: schema {schema_path or DEFAULT_SCHEMA_PATH}, "
        f"{len(valid)} valid fixture(s) loaded, {len(invalid)} invalid fixture(s) refused\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate desired-state documents")
    validate.add_argument("paths", nargs="+", type=Path)
    validate.add_argument("--schema", type=Path, default=None)
    check = sub.add_parser("self-check", help="check the schema and fixtures agree")
    check.add_argument("--schema", type=Path, default=None)
    check.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_DIR)
    render = sub.add_parser(
        "render", help="render target-ref desired state (uv environment required)"
    )
    render.add_argument(
        "--input",
        type=Path,
        help="replay recorded source artifacts; may contain secrets",
    )
    render.add_argument("--output", type=Path)
    render.add_argument("--source-root", type=Path, default=_REPO_ROOT)
    render.add_argument("--ref")
    render.add_argument("--kind", choices=["release", "merge"], default="merge")
    render.add_argument(
        "--composition",
        type=Path,
        help="accepted build tuple's sibling refs as a JSON list",
    )
    render.add_argument("--host")
    render.add_argument("--lane")
    render.add_argument(
        "--surface-kind",
        choices=["compose_lane", "runner_fleet", "host"],
        default="compose_lane",
    )
    render.add_argument("--compose-file", action="append")
    render.add_argument(
        "--broker-file", help="target ref's mounted broker bootstrap profile"
    )
    render.add_argument("--project", help="runner compose project")
    render.add_argument("--runner-host")
    render.add_argument("--runner-workdir")
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            if not args.input and not all((args.ref, args.host, args.lane)):
                parser.error("render requires --input or --ref, --host and --lane")
            return _cmd_render(args)
        if args.command == "validate":
            return _cmd_validate(args.paths, args.schema)
        return _cmd_self_check(args.schema, args.fixtures)
    except SchemaUnsupportedError as exc:
        sys.stderr.write(f"schema refused: {exc}\n")
        return 1
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired):
        # Validation exceptions may include their input (including secrets).
        sys.stderr.write("render REFUSED: invalid or unreadable target-ref artifacts\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

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
    desired_state.py self-check [--schema PATH] [--fixtures DIR]
                                                 exit 0 OK, 1 refused

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
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _cmd_validate(args.paths, args.schema)
        return _cmd_self_check(args.schema, args.fixtures)
    except SchemaUnsupportedError as exc:
        sys.stderr.write(f"schema refused: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

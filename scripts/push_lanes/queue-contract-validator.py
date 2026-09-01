#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate one private push-lane contract and emit a safe JSON projection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import unquote, urlsplit

import yaml


class StrictLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(
    loader: StrictLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("contract root and nested fields must be mappings")
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("contract mapping keys must be strings")
        if key in result:
            raise ValueError(f"duplicate contract key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_REPO_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")

_CONTRACT_ROOT = Path.home() / ".omnibase" / "infra" / "push-lane-contracts"
_EXPECTED_SCHEMA = "1.0"
_EXPECTED_KIND = "omnibase.push-lane.contract"
_EXPECTED_HOST = "omninode-pc"
_EXPECTED_HOOK_NAME = "PREPUSH_201_GATE_RUNNER_HOSTNAME"
_CANONICAL_GITHUB_HOST = "github.com"
_DISALLOWED_CONTROL_BYTES = frozenset(range(0x20)) - {0x0A}

_ALLOWED_TARGETS = {
    "omnibase_core": {
        "lane": "RSD-CORE-18df22d5",
        "worktree": "/data/omninode/omni_home/omni_worktrees/rsd-core-precommit-baseline/omnibase_core",
        "branch": "codex/core-precommit-baseline-16d8",
        "head_sha": "18df22d5a80dafeeb51b0103685fd241b3fac971",
        "repository": "OmniNode-ai/omnibase_core",
        "git_dir": "/home/jonah/push-lanes/omnibase_core/.git/worktrees/omnibase_core",
        "git_common_dir": "/home/jonah/push-lanes/omnibase_core/.git",
    },
    "omnimarket": {
        "lane": "RSD-OMNIMARKET-e1bd4177",
        "worktree": "/data/omninode/omni_home/omni_worktrees/rsd-pinned-single-attempt-dev-rebuild/omnimarket",
        "branch": "codex/rsd-pinned-single-attempt-dev-rebuild",
        "head_sha": "e1bd41773e9bda6f2c56caf58a36ea56d2220d64",
        "repository": "OmniNode-ai/omnimarket",
        "git_dir": "/home/jonah/push-lanes/omnimarket-rsd.git/.git/worktrees/omnimarket",
        "git_common_dir": "/home/jonah/push-lanes/omnimarket-rsd.git/.git",
    },
}


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        _fail(f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}")
    return value


def _string(value: Any, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        _fail(f"{label} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(f"{label} has an invalid format")
    return value


def _exact_string(value: Any, label: str, expected: str) -> str:
    result = _string(value, label)
    if result != expected:
        _fail(f"{label} must equal {expected!r}")
    return result


def _canonical_remote_urls(repository: str) -> tuple[str, str]:
    """Return the only fetch/push URL forms the queue may use.

    The queue never accepts a URL supplied by a lane contract.  This helper is
    deliberately literal: an exact comparison rejects userinfo, query strings,
    fragments, alternate hosts, ports, and percent-encoded variants before git
    is permitted to consult a credential helper.
    """

    return (
        f"https://{_CANONICAL_GITHUB_HOST}/{repository}.git",
        f"git@{_CANONICAL_GITHUB_HOST}:{repository}.git",
    )


def _expected_push_argv(remote: str, branch: str) -> list[str]:
    """Build the only allowed push argv: remote alias plus exact refspec."""

    return ["git", "push", remote, f"{branch}:{branch}"]


def _is_canonical_remote_url(value: str, repository: str) -> bool:
    """Classify a remote URL without ever returning it to a caller or log."""

    expected_https, expected_ssh = _canonical_remote_urls(repository)
    if value not in {expected_https, expected_ssh}:
        return False
    # A literal comparison above is authoritative.  These parsed assertions
    # make the invariant obvious and protect future changes to the literals.
    if unquote(value) != value:
        return False
    if value == expected_https:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and parsed.hostname == _CANONICAL_GITHUB_HOST
            and parsed.username is None
            and parsed.password is None
            and parsed.port is None
            and not parsed.query
            and not parsed.fragment
            and parsed.path == f"/{repository}.git"
        )
    return value == expected_ssh


def _validate_remote_urls_from_stdin(repository: str) -> None:
    """Require exactly one canonical, tokenless configured remote URL.

    URLs enter on stdin rather than process argv so an accidental credential in
    git configuration cannot appear in a process listing.
    """

    if repository not in {target["repository"] for target in _ALLOWED_TARGETS.values()}:
        _fail("remote repository is not allowlisted")
    raw = sys.stdin.buffer.read()
    # Do not call splitlines() until all non-delimiter control bytes have been
    # rejected. In particular, CR could otherwise be silently discarded and
    # make a hostile URL appear canonical after line splitting.
    if any(byte in _DISALLOWED_CONTROL_BYTES or byte == 0x7F for byte in raw):
        _fail("remote URL contains a raw control byte")
    raw_lines = raw.split(b"\n")
    if raw.endswith(b"\n"):
        raw_lines.pop()
    if len(raw_lines) != 1:
        _fail("remote configuration must contain exactly one URL")
    try:
        value = raw_lines[0].decode("utf-8", "strict")
    except UnicodeDecodeError:
        _fail("remote URL is not valid UTF-8")
    if not _is_canonical_remote_url(value, repository):
        _fail("remote URL is not canonical and tokenless")


def _validate_lane_lines_from_stdin() -> None:
    """Validate queue bytes before decoding or splitting them.

    Queue entries are identifiers, never a general command channel. Newline is
    the sole permitted record delimiter; every other C0/DEL byte is fatal.
    """

    raw = sys.stdin.buffer.read()
    if any(byte in _DISALLOWED_CONTROL_BYTES or byte == 0x7F for byte in raw):
        _fail("queue contains a raw control byte")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        _fail("queue is not valid UTF-8")
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        if _BRANCH_RE.fullmatch(line) is None:
            _fail("queue lane identifier has an invalid format")


def _run_self_test() -> None:
    """Exercise hostile URL forms without printing a URL or token-like value."""

    repository = _ALLOWED_TARGETS["omnibase_core"]["repository"]
    https_url, ssh_url = _canonical_remote_urls(repository)
    rejected = (
        "https://user@github.com/OmniNode-ai/omnibase_core.git",
        "https://github.com/OmniNode-ai/omnibase_core.git?token=opaque",
        "https://github.com/OmniNode-ai/omnibase_core.git#fragment",
        "https://github.com%40evil.invalid/OmniNode-ai/omnibase_core.git",
        "https://github.com/OmniNode-ai/omnibase_core%2egit",
        "git@github.com:OmniNode-ai/omnibase_core.git?token=opaque",
    )
    if not _is_canonical_remote_url(https_url, repository):
        raise AssertionError("canonical HTTPS URL rejected")
    if not _is_canonical_remote_url(ssh_url, repository):
        raise AssertionError("canonical SSH URL rejected")
    if any(_is_canonical_remote_url(value, repository) for value in rejected):
        raise AssertionError("hostile remote URL accepted")
    for raw in (
        b"https://github.com/OmniNode-ai/omnibase_core.git\r\n",
        b"x\x00\n",
        b"x\x1f\n",
    ):
        try:
            previous_stdin = sys.stdin
            sys.stdin = type("Input", (), {"buffer": __import__("io").BytesIO(raw)})()
            _validate_lane_lines_from_stdin()
        except ValueError:
            pass
        else:
            raise AssertionError("raw control byte accepted")
        finally:
            sys.stdin = previous_stdin
    if _expected_push_argv("origin", "branch") != [
        "git",
        "push",
        "origin",
        "branch:branch",
    ]:
        raise AssertionError("tokenless push argv changed")
    print(
        "remote-url self-test: 2 canonical accepted; 6 hostile forms rejected; argv tokenless"
    )


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        root = _CONTRACT_ROOT.resolve(strict=True)
        root_stat = root.stat()
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid():
            _fail("contract root must be an owner-controlled directory")
        if stat.S_IMODE(root_stat.st_mode) != 0o700:
            _fail("contract root mode must be 0700")
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_uid != os.getuid():
            _fail("contract must be an owner-controlled regular file")
        if stat.S_IMODE(path_stat.st_mode) != 0o600:
            _fail("contract mode must be 0600")
        if path.resolve(strict=True).parent != root:
            _fail("contract must reside directly in the private contract root")
        raw = path.read_bytes()
    except OSError as exc:
        _fail(f"contract filesystem check failed: {exc}")
    try:
        value = yaml.load(raw, Loader=StrictLoader)
    except yaml.YAMLError as exc:
        _fail(f"contract YAML is invalid: {exc}")
    if not isinstance(value, dict):
        _fail("contract document must be a mapping")
    return value, hashlib.sha256(raw).hexdigest()


def validate(path: Path) -> dict[str, Any]:
    data, contract_sha256 = _load_contract(path)
    root = _keys(
        data,
        {
            "schema_version",
            "kind",
            "lane",
            "repo",
            "worktree",
            "branch",
            "head_sha",
            "remote",
            "command",
            "hook_identity",
            "capacity",
        },
        "contract",
    )
    _exact_string(root["schema_version"], "schema_version", _EXPECTED_SCHEMA)
    _exact_string(root["kind"], "kind", _EXPECTED_KIND)
    repo = _string(root["repo"], "repo", _REPO_RE)
    target = _ALLOWED_TARGETS.get(repo)
    if target is None:
        _fail("repo is not an allowlisted push target")
    lane = _exact_string(root["lane"], "lane", target["lane"])
    worktree = _exact_string(root["worktree"], "worktree", target["worktree"])
    branch = _exact_string(root["branch"], "branch", target["branch"])
    head_sha = _exact_string(root["head_sha"], "head_sha", target["head_sha"])
    remote = _exact_string(root["remote"], "remote", "origin")

    command = _keys(root["command"], {"argv", "cwd"}, "command")
    argv = command["argv"]
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        _fail("command.argv must be a string list")
    expected_argv = _expected_push_argv(remote, branch)
    if argv != expected_argv:
        _fail(f"command.argv must equal {expected_argv!r}")
    _exact_string(command["cwd"], "command.cwd", worktree)

    hook = _keys(root["hook_identity"], {"name", "value"}, "hook_identity")
    _exact_string(hook["name"], "hook_identity.name", _EXPECTED_HOOK_NAME)
    _exact_string(hook["value"], "hook_identity.value", _EXPECTED_HOST)

    capacity = _keys(root["capacity"], {"max_load_ratio", "poll_seconds"}, "capacity")
    ratio = capacity["max_load_ratio"]
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or ratio != 1.0:
        _fail("capacity.max_load_ratio must be exactly 1.0")
    poll = capacity["poll_seconds"]
    if not isinstance(poll, int) or isinstance(poll, bool) or not 1 <= poll <= 300:
        _fail("capacity.poll_seconds must be an integer from 1 through 300")

    return {
        "contract_sha256": contract_sha256,
        "lane": lane,
        "repo": repo,
        "repository": target["repository"],
        "git_dir": target["git_dir"],
        "git_common_dir": target["git_common_dir"],
        "worktree": worktree,
        "branch": branch,
        "head_sha": head_sha,
        "remote": remote,
        "argv": argv,
        "cwd": worktree,
        "hook_name": _EXPECTED_HOOK_NAME,
        "hook_value": _EXPECTED_HOST,
        "max_load_ratio": ratio,
        "poll_seconds": poll,
    }


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        try:
            _run_self_test()
        except (AssertionError, ValueError, TypeError) as exc:
            print(f"self-test failed: {exc}", file=sys.stderr)
            return 1
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--validate-remote-urls":
        try:
            _validate_remote_urls_from_stdin(sys.argv[2])
        except (ValueError, TypeError) as exc:
            print(f"remote rejected: {exc}", file=sys.stderr)
            return 1
        return 0
    if sys.argv[1:] == ["--validate-lane-lines"]:
        try:
            _validate_lane_lines_from_stdin()
        except (ValueError, TypeError) as exc:
            print(f"queue rejected: {exc}", file=sys.stderr)
            return 1
        return 0
    if len(sys.argv) != 2:
        print("usage: queue-contract-validator.py CONTRACT", file=sys.stderr)
        return 2
    try:
        print(json.dumps(validate(Path(sys.argv[1])), sort_keys=True))
    except (ValueError, TypeError) as exc:
        print(f"contract rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

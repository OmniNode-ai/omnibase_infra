# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Read-only GitHub Action verifier for OmniGate PR receipts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

_RECEIPT_START = "<!-- OMNIGATE_RECEIPT_START -->"
_RECEIPT_END = "<!-- OMNIGATE_RECEIPT_END -->"
_PASS_ACTION = "pass"
_FAIL_ACTION = "fail"
_ACTION_VALUES = frozenset({"pass", "fail", "label", "comment", "auto-close"})
datetime = dt.datetime


def _extract_receipt_from_pr_body(pr_body: str, *, max_bytes: int) -> str | None:
    """Extract exactly one framed inline receipt from a PR body."""
    if pr_body.count(_RECEIPT_START) != 1 or pr_body.count(_RECEIPT_END) != 1:
        return None
    start = pr_body.index(_RECEIPT_START) + len(_RECEIPT_START)
    end = pr_body.index(_RECEIPT_END)
    if end <= start:
        return None

    receipt_json = pr_body[start:end].strip()
    if len(receipt_json.encode("utf-8")) > max_bytes:
        return None
    return receipt_json


def verify_pr_receipt(
    pr_body: str,
    repo_path: Path,
    config_path: Path,
    *,
    repository_id: str,
    repository_url: str,
    base_sha: str,
    head_sha: str,
    actor: str | None = None,
) -> dict[str, object]:
    """Verify an inline PR receipt against trusted GitHub event metadata."""
    try:
        _verify_commit_object(repo_path, base_sha)
        _verify_commit_object(repo_path, head_sha)
        config = _load_omnigate_config(config_path)
        gate_policy = _get_attr(config, "gate")
        receipt_policy = _get_attr(config, "receipt")
        if actor is not None and actor in tuple(
            getattr(gate_policy, "exempt_users", ()),
        ):
            return _decision(
                ok=True,
                action=_PASS_ACTION,
                reason="Actor exempted by trusted OmniGate config",
            )

        diff_hash = _compute_pr_diff_hash(
            repo_path,
            base_sha=base_sha,
            head_sha=head_sha,
        )
        config_hash = _compute_config_hash(config_path)
        receipt_json = _extract_receipt_from_pr_body(
            pr_body,
            max_bytes=_get_int_attr(receipt_policy, "max_receipt_bytes"),
        )
        if receipt_json is None:
            return _decision(
                ok=False,
                action=_FAIL_ACTION,
                reason="No valid OmniGate receipt found in PR body",
            )

        try:
            receipt = _model_validate_receipt_json(receipt_json)
        except (TypeError, ValueError) as exc:
            return _decision(
                ok=False,
                action=_FAIL_ACTION,
                reason=f"Invalid OmniGate receipt: {exc}",
            )

        receipt_diff_hash = _get_str_attr(receipt, "diff_hash")
        failure_reason = _validate_receipt(
            receipt,
            config=config,
            repository_id=repository_id,
            repository_url=repository_url,
            base_sha=base_sha,
            head_sha=head_sha,
            diff_hash=diff_hash,
            config_hash=config_hash,
        )
        if failure_reason is not None:
            return _decision(
                ok=False,
                action=_FAIL_ACTION,
                reason=failure_reason,
                receipt_diff_hash=receipt_diff_hash,
            )

        return _decision(
            ok=True,
            action=_PASS_ACTION,
            reason="All checks passed, signature verified",
            receipt_diff_hash=receipt_diff_hash,
        )
    except Exception as exc:  # noqa: BLE001 - verifier must fail closed.
        return _decision(
            ok=False,
            action=_FAIL_ACTION,
            reason=f"OmniGate verifier failed closed: {exc}",
        )


def _validate_receipt(
    receipt: object,
    *,
    config: object,
    repository_id: str,
    repository_url: str,
    base_sha: str,
    head_sha: str,
    diff_hash: str,
    config_hash: str,
) -> str | None:
    receipt_policy = _get_attr(config, "receipt")
    if _get_str_attr(receipt, "repository_id") != repository_id:
        return "Repository id mismatch"
    if _normalize_url(_get_str_attr(receipt, "project_url")) != _normalize_url(
        repository_url,
    ):
        return "Repository URL mismatch"
    if (
        _get_str_attr(receipt, "base_sha") != base_sha
        or _get_str_attr(receipt, "head_sha") != head_sha
        or _get_str_attr(receipt, "commit_sha") != head_sha
    ):
        return "Receipt commit/base/head SHA mismatch"
    if _get_str_attr(receipt, "diff_hash") != diff_hash:
        return (
            f"Diff hash mismatch: receipt={_get_str_attr(receipt, 'diff_hash')}, "
            f"actual={diff_hash}"
        )
    if _get_str_attr(receipt, "config_hash") != config_hash:
        return "Config hash mismatch against trusted base config"

    age_reason = _receipt_age_failure_reason(receipt, config)
    if age_reason is not None:
        return age_reason

    if not _no_blocking_checks_failed(
        receipt,
        advisory_blocks=_get_bool_attr(receipt_policy, "advisory_blocks"),
    ):
        return f"Checks failed: {', '.join(_blocking_check_names(receipt, config))}"

    if _get_str_attr(receipt_policy, "signing") == "sigstore":
        identity_policy = _get_attr(receipt_policy, "identity")
        if identity_policy is None:
            return "Sigstore identity policy missing from trusted config"
        if not _verify_sigstore_identity_policy(receipt, identity_policy):
            return "Sigstore signature verification failed"
    elif not _get_bool_attr(receipt_policy, "allow_unsigned"):
        return "Unsigned receipts are disabled by trusted config"
    return None


def _verify_sigstore_identity_policy(receipt: object, identity_policy: object) -> bool:
    """Verify Sigstore signatures using trusted identity policy only."""
    expected_issuer = _get_str_attr(identity_policy, "expected_issuer")
    for identity in getattr(identity_policy, "allowed_identities", ()):
        if _verify_with_signer(
            receipt,
            expected_identity=str(identity),
            expected_issuer=expected_issuer,
        ):
            return True

    regexes = tuple(getattr(identity_policy, "allowed_identity_regexes", ()))
    if regexes:
        return _verify_sigstore_regex_identity(receipt, identity_policy, regexes)
    return False


def _verify_sigstore_regex_identity(
    receipt: object,
    identity_policy: object,
    regexes: tuple[str, ...],
) -> bool:
    """Fail closed until certificate identity extraction exists for regex policy."""
    for pattern in regexes:
        re.compile(pattern)
    _ = receipt, identity_policy
    return False


def _receipt_age_failure_reason(
    receipt: object,
    config: object,
) -> str | None:
    timestamp = _get_attr(receipt, "timestamp")
    if not isinstance(timestamp, dt.datetime):
        return "Receipt timestamp is invalid"
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.UTC)
    age_minutes = (datetime.now(tz=dt.UTC) - timestamp).total_seconds() / 60
    max_age = _get_int_attr(_get_attr(config, "receipt"), "max_age_minutes")
    if age_minutes > max_age:
        return f"Receipt expired: {age_minutes:.0f}m old, max {max_age}m"
    return None


def _no_blocking_checks_failed(
    receipt: object,
    *,
    advisory_blocks: bool,
) -> bool:
    method = getattr(receipt, "no_blocking_checks_failed", None)
    if callable(method):
        return bool(method(advisory_blocks=advisory_blocks))
    return not _blocking_check_names_from_statuses(receipt, advisory_blocks)


def _blocking_check_names(
    receipt: object,
    config: object,
) -> list[str]:
    return _blocking_check_names_from_statuses(
        receipt,
        advisory_blocks=_get_bool_attr(_get_attr(config, "receipt"), "advisory_blocks"),
    )


def _blocking_check_names_from_statuses(
    receipt: object,
    advisory_blocks: bool,
) -> list[str]:
    blocking = {"fail", "pending"}
    if advisory_blocks:
        blocking.add("advisory")

    names: list[str] = []
    for check in getattr(receipt, "checks", ()):
        status = getattr(
            getattr(check, "status", ""), "value", getattr(check, "status", "")
        )
        if str(status) in blocking:
            names.append(str(getattr(check, "name", "<unnamed>")))
    return names


def _decision(
    *,
    ok: bool,
    action: str,
    reason: str,
    receipt_diff_hash: str | None = None,
) -> dict[str, object]:
    if action not in _ACTION_VALUES:
        msg = f"Unsupported OmniGate action: {action}"
        raise ValueError(msg)
    return {
        "ok": ok,
        "action": action,
        "reason": reason,
        "receipt_diff_hash": receipt_diff_hash,
        "checked_at": datetime.now(tz=dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
    }


def _verify_commit_object(repo_path: Path, sha: str) -> None:
    subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


def _normalize_url(value: str) -> str:
    return value.rstrip("/")


def _get_attr(target: object, name: str) -> object:
    if not hasattr(target, name):
        msg = f"Object missing required attribute: {name}"
        raise ValueError(msg)
    return getattr(target, name)


def _get_str_attr(target: object, name: str) -> str:
    value = _get_attr(target, name)
    return str(value).rstrip("/") if name == "project_url" else str(value)


def _get_int_attr(target: object, name: str) -> int:
    value = _get_attr(target, name)
    if not isinstance(value, int):
        msg = f"Object attribute must be an int: {name}"
        raise ValueError(msg)
    return value


def _get_bool_attr(target: object, name: str) -> bool:
    value = _get_attr(target, name)
    if not isinstance(value, bool):
        msg = f"Object attribute must be a bool: {name}"
        raise ValueError(msg)
    return value


def _load_omnigate_config(config_path: Path) -> object:
    from omnibase_core.gate.config_loader import load_omnigate_config

    return cast("object", load_omnigate_config(config_path))


def _compute_pr_diff_hash(repo_path: Path, *, base_sha: str, head_sha: str) -> str:
    from omnibase_core.gate.diff_hash import compute_pr_diff_hash

    return str(compute_pr_diff_hash(repo_path, base_sha=base_sha, head_sha=head_sha))


def _compute_config_hash(config_path: Path) -> str:
    from omnibase_core.gate.diff_hash import compute_config_hash

    return str(compute_config_hash(config_path))


def _model_validate_receipt_json(receipt_json: str) -> object:
    from omnibase_core.models.gate.model_omnigate_receipt import ModelOmniGateReceipt

    return cast("object", ModelOmniGateReceipt.model_validate_json(receipt_json))


def _signer() -> object:
    from omnibase_infra.gate.signer import OmniGateSigner

    return cast("object", OmniGateSigner())


def _verify_with_signer(
    receipt: object,
    *,
    expected_identity: str,
    expected_issuer: str,
) -> bool:
    verify = getattr(_signer(), "verify", None)
    if not callable(verify):
        return False
    return bool(
        verify(
            receipt,
            expected_identity=expected_identity,
            expected_issuer=expected_issuer,
        ),
    )


def _read_event(event_path: Path) -> Mapping[str, object]:
    return cast(
        "Mapping[str, object]",
        json.loads(event_path.read_text(encoding="utf-8")),
    )


def _mapping_value(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        msg = f"GitHub event missing required key: {key}"
        raise ValueError(msg)
    return mapping[key]


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    msg = f"GitHub event key must be an object: {name}"
    raise ValueError(msg)


def _decision_from_event(
    *,
    event_path: Path,
    repo_path: Path,
    config_path: Path,
) -> dict[str, object]:
    event = _read_event(event_path)
    pull_request = _as_mapping(_mapping_value(event, "pull_request"), "pull_request")
    repository = _as_mapping(_mapping_value(event, "repository"), "repository")
    sender = _as_mapping(event.get("sender", {}), "sender")
    base = _as_mapping(_mapping_value(pull_request, "base"), "pull_request.base")
    head = _as_mapping(_mapping_value(pull_request, "head"), "pull_request.head")
    actor_value = sender.get("login")
    return verify_pr_receipt(
        str(pull_request.get("body") or ""),
        repo_path,
        config_path,
        repository_id=str(_mapping_value(repository, "id")),
        repository_url=str(_mapping_value(repository, "html_url")),
        base_sha=str(_mapping_value(base, "sha")),
        head_sha=str(_mapping_value(head, "sha")),
        actor=str(actor_value) if actor_value is not None else None,
    )


def _write_decision(path: Path, decision: dict[str, object]) -> None:
    path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-path", required=True, type=Path)
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--decision-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    decision = _decision_from_event(
        event_path=args.event_path,
        repo_path=args.repo_path,
        config_path=args.config,
    )
    _write_decision(args.decision_out, decision)
    return 0 if decision["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_RECEIPT_END",
    "_RECEIPT_START",
    "_extract_receipt_from_pr_body",
    "main",
    "verify_pr_receipt",
]

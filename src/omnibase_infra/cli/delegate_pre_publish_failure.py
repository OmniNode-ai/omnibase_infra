# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Report a delegation that failed before publish as itself (OMN-19131).

**The measured defect.** Between 2026-09-21 and 2026-09-22 every
``onex delegate --timeout`` run failed, 48 of 48, and none was ever published.
The CLI wrote ``requested_timeout_seconds`` into the request payload, the
installed request model forbade extra fields, and ``RuntimeLocal`` refused the
payload in ``_build_initial_payload`` before anything reached the broker. The
runtime records that refusal in the capture log only; its workflow result says
``failed`` and carries no ``wire_correlation_id``.

The CLI then looked for a delegation terminal, found none, and told the
operator the receipt "carries no resolvable delegation terminal". That sentence
is true and describes a bus-side outage, so it sent the reader to the deployed
lane, which was healthy. One lane spent four attempts and a wrong diagnosis on
it.

**The discriminator.** ``RuntimeLocal`` records ``wire_correlation_id`` as soon
as it has built the payload and resolved the correlation it is about to
publish. A run that did not complete, carries no terminal, and recorded no wire
correlation therefore never published, and a missing terminal is the expected
consequence rather than the finding. Transport failures are excluded: a broker
that never answered has its own typed refusal (OMN-18925).

**What the operator is told.** The field the model refused, the model's import
path and distribution, the payload file, and the capture log path. The field is
recovered by validating the payload file the CLI wrote against the request
model the contract names, which is the same pair the runtime validated. Nothing
here parses a log line. ``missing`` errors are not reported, because the
runtime injects its own run-identity defaults for absent required fields before
it validates, so a ``missing`` result from this re-validation is not a refusal
the runtime made. When no field-level refusal is found the message says so and
points at the capture log, rather than guessing.
"""

from __future__ import annotations

import importlib
import json
from importlib import metadata
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
)

__all__ = [
    "DelegatePrePublishFailureError",
    "describe_pre_publish_failure",
    "pre_publish_failure_from_receipt",
]

#: Receipt fields a terminal can be carried in. A pre-publish failure has
#: neither; a run that has either is not one.
_TERMINAL_CARRIER_FIELDS: tuple[str, ...] = ("terminal_payload", "handler_result")

#: The one workflow result that says the run reached its end. A completed run
#: with no terminal is OMN-18569's shape and keeps failing closed there.
_COMPLETED = "completed"


class DelegatePrePublishFailureError(DelegateTerminalUnresolvedError):
    """A delegation failed before its command was published.

    Subclasses :class:`DelegateTerminalUnresolvedError` so every existing catch
    site keeps catching it, while its message never claims a terminal was lost:
    none was ever requested.
    """


def pre_publish_failure_from_receipt(
    envelope: dict[str, object],
) -> dict[str, object] | None:
    """Return the runtime summary when this receipt records a pre-publish failure.

    ``None`` for every other shape, including a run that published and heard
    nothing back, a completed run, and a transport failure.
    """
    result = envelope.get("result")
    if not isinstance(result, dict):
        return None
    if "ModelReceiptRuntimeSummary" not in str(envelope.get("result_model") or ""):
        return None
    if str(result.get("workflow_result") or "") == _COMPLETED:
        return None
    if result.get("wire_correlation_id"):
        return None
    if result.get("runtime_error_is_transport"):
        return None
    if any(result.get(field) is not None for field in _TERMINAL_CARRIER_FIELDS):
        return None
    return result


def pre_publish_failure_error(envelope: dict[str, object]) -> str:
    """The base sentence, for a caller that has no payload or contract at hand."""
    result = envelope.get("result")
    workflow_result = (
        str(result.get("workflow_result") or "") if isinstance(result, dict) else ""
    )
    runtime_error_type = (
        str(result.get("runtime_error_type") or "") if isinstance(result, dict) else ""
    )
    cause = f" ({runtime_error_type})" if runtime_error_type else ""
    return (
        f"delegate run {envelope.get('run_id')} failed before publish{cause}: "
        f"the run ended with workflow result "
        f"'{workflow_result or 'absent'}' and the runtime recorded no wire "
        "correlation id, so no command reached the broker and "
        "there is no delegation terminal to look for. The deployed lane is not "
        "implicated."
    )


def _request_model_path(contract_path: Path) -> str | None:
    """The request model the contract names, as ``module.Class``.

    Reads the top-level ``input_model``, which is where the delegate
    orchestrator's contract declares it and the first place ``RuntimeLocal``
    looks. Accepts the dotted form and the ``{module, class}`` mapping.
    """
    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(contract, dict):
        return None
    spec = contract.get("input_model")
    if isinstance(spec, str) and "." in spec:
        return spec
    if isinstance(spec, dict):
        module = spec.get("module")
        cls = spec.get("class") or spec.get("name")
        if isinstance(module, str) and isinstance(cls, str) and module and cls:
            return f"{module}.{cls}"
    return None


def _distribution_of(module_path: str) -> str:
    """Name the installed distribution and version that provides a module."""
    top_level = module_path.split(".", 1)[0]
    distributions = metadata.packages_distributions().get(top_level) or []
    for name in distributions:
        try:
            return f"{name} {metadata.version(name)}"
        except metadata.PackageNotFoundError:
            continue
    return f"{top_level} (distribution not found)"


def _field_refusals(model_path: str, payload_path: Path) -> tuple[list[str], str]:
    """Validate the payload against the model and name every field it refuses.

    Returns the refusals and, when none could be computed, the reason why.
    """
    module_name, _, class_name = model_path.rpartition(".")
    try:
        model = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        return [], f"the request model could not be imported ({exc})"
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return [], "the request model is not a pydantic model"
    try:
        raw = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"the payload file could not be read ({exc})"
    try:
        model.model_validate(raw)
    except ValidationError as exc:
        refusals = [
            "`{loc}` {type}: {msg}".format(
                loc=".".join(str(part) for part in error.get("loc", ())) or "<root>",
                type=error.get("type", "?"),
                msg=error.get("msg", ""),
            )
            for error in exc.errors(include_url=False)
            if error.get("type") != "missing"
        ]
        if refusals:
            return refusals, ""
        return [], "the model refused no field this CLI supplied"
    return [], "the payload validates against the model"


def describe_pre_publish_failure(
    *,
    envelope: dict[str, object],
    contract_path: Path,
    payload_path: Path,
    capture_log_path: Path,
) -> str:
    """Say what refused the run before publish, and where the record is.

    Names the offending field, the rejecting model's import path and
    distribution, the payload file and the capture log, so the reader never
    has to open the capture log to learn which field was wrong.
    """
    base = pre_publish_failure_error(envelope)
    model_path = _request_model_path(contract_path)
    if model_path is None:
        cause = (
            f" The request model could not be read from {contract_path}, so the "
            "refusing field is not named here."
        )
    else:
        refusals, unresolved = _field_refusals(model_path, payload_path)
        provider = _distribution_of(model_path)
        if refusals:
            cause = (
                f" Cause: the request payload was refused by {model_path} "
                f"({provider}): " + "; ".join(refusals) + "."
            )
        else:
            cause = (
                f" Cause: not a field-level refusal by {model_path} ({provider}): "
                f"{unresolved}. The runtime's own error is in the capture log."
            )
    return (
        f"{base}{cause} Payload: {payload_path.resolve()}. "
        f"Capture log: {capture_log_path}."
    )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tenant offboarding and report evidence for the existing slot teardown."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from pathlib import Path

import yaml

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_prepr_teardown_context import (
    ModelPreprTeardownContext,
    PreprTeardownRunner,
)
from omnibase_infra.models.model_prepr_teardown_files import ModelPreprTeardownFiles
from omnibase_infra.models.model_prepr_teardown_request import ModelPreprTeardownRequest
from omnibase_infra.models.model_prepr_teardown_result import ModelPreprTeardownResult
from omnibase_infra.models.model_prepr_teardown_settings import (
    ModelPreprTeardownSettings,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

PROOF_MISSING = "slot gateway is gone without a matching saved authentication proof"
RESIDUE_AXES = frozenset(
    {
        "containers",
        "volumes",
        "images",
        "topics",
        "groups",
        "valkey_keys",
        "databases",
        "roles",
        "staging_root",
    }
)
CONTROL_AXES = frozenset(
    {
        "dev_containers",
        "other_topics",
        "other_groups",
        "valkey_index0_keys",
        "databases",
    }
)


class HandlerPreprTeardownSlot:
    """Own offboard evidence across interrupted cleanup passes.

    The host-local CLI injects its existing command adapter. Runtime callers
    can construct the handler through the ONEX container; no bus access or
    credential is retained in the request or result model.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        context = container.get_service(ModelPreprTeardownContext)
        self._run = context.runner
        self._env = context.environment
        self._files = context.files

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(self, request: ModelPreprTeardownRequest) -> ModelPreprTeardownResult:
        """Offboard one policy-derived tenant or inspect its cleanup report."""
        if request.compose_project != "omnibase-infra-prepr-" + request.slot_token[5:]:
            return ModelPreprTeardownResult(
                success=False,
                errors=("slot selectors disagree",),
                exit_code=2,
            )
        if request.operation == "inspect_report":
            return self._inspect_report(request)
        if request.operation == "write_report":
            return self._write_report(request)
        return self._revoke_tenant(request)

    def _write_report(
        self, request: ModelPreprTeardownRequest
    ) -> ModelPreprTeardownResult:
        report: dict[str, object] = {
            "schema": "urn:onex:prepr:slot-teardown:v1",
            "slot": int(request.slot_token[5:]),
            "compose_project": request.compose_project,
            "planned": request.planned,
            "errors": list(request.errors),
        }
        if request.plan_only:
            verdict = "PLAN"
            clean = not request.errors
        else:
            report.update(residue=request.residue, controls=request.controls)
            complete = (
                set(request.residue) == RESIDUE_AXES
                and set(request.controls) == CONTROL_AXES
            )
            if not complete:
                report["errors"] = [
                    *request.errors,
                    "teardown readback axes are incomplete",
                ]
            if (
                not complete
                or request.errors
                or any(value != 0 for value in request.residue.values())
            ):
                verdict = "INCOMPLETE"
            elif any(value <= 0 for value in request.controls.values()):
                verdict = "UNPROVEN"
            else:
                verdict = "CLEAN"
            clean = verdict == "CLEAN"
        report["verdict"] = verdict
        if request.authentication is not None:
            report["tenant_authentication"] = request.authentication
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if request.report_path is not None:
            self._files.write(request.report_path, text)
        self._files.output(text)
        return ModelPreprTeardownResult(success=clean, exit_code=0 if clean else 13)

    def _revoke_tenant(
        self, request: ModelPreprTeardownRequest
    ) -> ModelPreprTeardownResult:
        credential = request.staging_root / "tenant-state" / "credential.env"
        if not self._files.present(credential):
            return ModelPreprTeardownResult(success=True)
        listing = self._run(
            [
                "docker",
                "ps",
                "-a",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={request.compose_project}",
            ],
            dict(self._env),
        )
        if listing.returncode != 0:
            return ModelPreprTeardownResult(
                success=False,
                errors=("slot container listing is unreadable",),
            )
        if not listing.stdout.strip():
            return self._saved_proof(request, credential)
        env = {
            **self._env,
            "COMPOSE_PROJECT": request.compose_project,
            "LAB_TENANT_SLUG": f"onex-{request.slot_token}",
        }
        done = self._run(
            [
                "bash",
                str(request.repo_root / "scripts/smoke/smoke_delegation.sh"),
                "--target",
                "compose",
                "--tenant-lifecycle",
                "revoke",
            ],
            env,
        )
        if done.returncode != 0:
            return ModelPreprTeardownResult(
                success=False,
                errors=(
                    f"revoke slot tenant and prove authentication refusal: exit {done.returncode}: "
                    + sanitize_error_string(
                        (done.stderr or "").strip(), max_length=300
                    ),
                ),
            )
        try:
            proofs = [
                json.loads(line)
                for line in done.stdout.splitlines()
                if line.startswith("{")
            ]
        except ValueError:
            return ModelPreprTeardownResult(
                success=False,
                errors=(
                    "slot tenant offboard returned malformed authentication proof",
                ),
            )
        if (
            not proofs
            or not isinstance(proofs[-1], dict)
            or not (
                proofs[-1].get("revoked") is True
                and proofs[-1].get("auth_before") == 200
                and proofs[-1].get("auth_after") in (401, 403)
            )
        ):
            return ModelPreprTeardownResult(
                success=False,
                errors=("slot tenant offboard returned no authentication proof",),
            )
        return ModelPreprTeardownResult(
            success=True,
            authentication={
                key: proofs[-1][key]
                for key in ("tenant_id", "revoked", "auth_before", "auth_after")
            },
        )

    def _saved_proof(
        self,
        request: ModelPreprTeardownRequest,
        credential: Path,
    ) -> ModelPreprTeardownResult:
        # A previous pass removed the gateway after measuring revocation but
        # failed a later cleanup step. Absence alone is never offboard proof.
        try:
            state = dict(
                line.split("=", 1)
                for line in self._files.read(credential).splitlines()
                if "=" in line
            )
            tenant_id = str(uuid.UUID(state["TENANT_ID"]))
            key = state["ONEX_API_KEY"]
            proof = json.loads(
                self._files.read(Path(str(credential) + ".revoked.json"))
            )
            valid = (
                isinstance(proof, dict)
                and state.get("TENANT_SLUG") == f"onex-{request.slot_token}"
                and bool(key)
                and proof.get("tenant_id") == tenant_id
                and proof.get("key_sha256") == hashlib.sha256(key.encode()).hexdigest()
                and proof.get("auth_before") == 200
                and proof.get("auth_after") in (401, 403)
            )
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            return ModelPreprTeardownResult(success=False, errors=(PROOF_MISSING,))
        return ModelPreprTeardownResult(
            success=True,
            authentication={
                "tenant_id": tenant_id,
                "revoked": True,
                "auth_before": proof["auth_before"],
                "auth_after": proof["auth_after"],
                "evidence": "saved-offboard-proof-no-containers",
            },
        )

    def _inspect_report(
        self, request: ModelPreprTeardownRequest
    ) -> ModelPreprTeardownResult:
        diagnostic: dict[str, object] = {
            "report_path": str(request.report_path),
            "returncode": request.returncode,
        }
        try:
            if request.report_path is None:
                raise ValueError("missing report path")
            report = json.loads(self._files.read(request.report_path))
            if not isinstance(report, dict):
                raise ValueError("teardown report is not an object")
        except (OSError, ValueError) as exc:
            diagnostic["reason"] = f"teardown report unreadable: {type(exc).__name__}"
            return ModelPreprTeardownResult(success=False, diagnostics=diagnostic)
        verdict = report.get("verdict")
        diagnostic["verdict"] = (
            verdict
            if verdict in ("CLEAN", "INCOMPLETE", "UNPROVEN", "PLAN")
            else "UNKNOWN"
        )
        for axis in ("residue", "controls"):
            values = report.get(axis)
            if isinstance(values, dict):
                diagnostic[axis] = {
                    key: value
                    for key, value in values.items()
                    if key in RESIDUE_AXES.union(CONTROL_AXES) and type(value) is int
                }
        errors = report.get("errors", [])
        if isinstance(errors, list) and errors:
            missing_gateway = (
                "FATAL: no running onex-api container in compose project "
                + request.compose_project
            )
            if any(
                isinstance(error, str) and missing_gateway in error for error in errors
            ):
                diagnostic["reason"] = "slot gateway is not running"
            elif PROOF_MISSING in errors:
                diagnostic["reason"] = PROOF_MISSING
            else:
                diagnostic["reason"] = "teardown report contains errors"
        elif verdict != "CLEAN":
            diagnostic["reason"] = "teardown residue or positive control is unproven"
        elif request.returncode != 0:
            diagnostic["reason"] = "teardown command exited nonzero"
        return ModelPreprTeardownResult(
            success=request.returncode == 0 and verdict == "CLEAN",
            diagnostics=diagnostic,
        )


def load_settings(
    repo_root: Path, files: ModelPreprTeardownFiles
) -> ModelPreprTeardownSettings:
    """Read the node's declared installer budgets at the composition boundary."""
    contract = (
        repo_root / "src/omnibase_infra/nodes/node_prepr_teardown_slot/contract.yaml"
    )
    return ModelPreprTeardownSettings.model_validate(
        yaml.safe_load(files.read(contract))["operational"]
    )


def _unit_quote(value: str) -> str:
    if any(c in value for c in "\n\r\0"):
        raise ValueError("unit path contains a control character")
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def render_units(
    release: Path,
    helper: Path,
    interpreter: Path,
    unit: str,
    files: ModelPreprTeardownFiles,
) -> tuple[str, str]:
    """Render the existing reaper units from the immutable package snapshot."""
    templates = release / "src/omnibase_infra/nodes/node_prepr_teardown_slot/templates"
    service = files.read(templates / "reaper.service.tmpl").format(
        python=_unit_quote(str(interpreter)), helper=_unit_quote(str(helper))
    )
    timer = files.read(templates / "reaper.timer.tmpl").format(unit=unit)
    return service, timer


def create_container(
    runner: PreprTeardownRunner,
    environment: Mapping[str, str],
    files: ModelPreprTeardownFiles,
) -> ModelONEXContainer:
    """Register host dependencies and the handler in one application container."""
    container = ModelONEXContainer()
    registry = container.service_registry
    if registry is None:
        raise RuntimeError("slot teardown service registry is unavailable")
    context = ModelPreprTeardownContext(runner, dict(environment), files)
    asyncio.run(registry.register_instance(ModelPreprTeardownContext, context))
    instance = HandlerPreprTeardownSlot(container)
    asyncio.run(registry.register_instance(HandlerPreprTeardownSlot, instance))
    return container


def execute_teardown_evidence(
    runner: PreprTeardownRunner,
    environment: Mapping[str, str],
    fields: dict[str, object],
    files: ModelPreprTeardownFiles,
) -> ModelPreprTeardownResult:
    """Invoke the resolved handler with the adapter's collected request."""
    container = create_container(runner, environment, files)
    handler = container.get_service(HandlerPreprTeardownSlot)
    return handler.handle(ModelPreprTeardownRequest.model_validate(fields))

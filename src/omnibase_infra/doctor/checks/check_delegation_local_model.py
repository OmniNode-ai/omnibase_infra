# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Doctor check for an OpenAI-compatible local delegation model."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit, urlunsplit

import yaml

from omnibase_core.doctor.doctor_check_base import DoctorCheckBase
from omnibase_core.enums.enum_doctor_category import EnumDoctorCategory
from omnibase_core.models.doctor.model_doctor_check_result import (
    ModelDoctorCheckResult,
)
from omnibase_infra.doctor.delegation_doctor_support import (
    DOCTOR_HTTP_TIMEOUT_SECONDS,
    render_diagnosis,
    run_async,
    unknown_result,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)
from omnibase_infra.doctor.protocol_models_transport import ProtocolModelsTransport
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)

# The routing authority (omnimarket) binds the overlay through BIFROST_OVERLAY_PATH and
# otherwise reads the machine-local default below; the doctor reads the same two.
_OVERLAY_ENV_KEYS = ("BIFROST_OVERLAY_PATH",)


class CheckDelegationLocalModel(DoctorCheckBase):
    """Verify the declared local model is present on its models endpoint."""

    check_id: ClassVar[str] = "delegation_local_model"
    check_name: ClassVar[str] = "Delegation local model"
    category: ClassVar[EnumDoctorCategory] = EnumDoctorCategory.SERVICES

    def __init__(
        self,
        *,
        overlay_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
        transport: ProtocolModelsTransport | None = None,
    ) -> None:
        resolved_environ = os.environ if environ is None else environ
        self._overlay_path = overlay_path or self._path_from_environ(resolved_environ)
        self._transport = (
            transport
            if transport is not None
            else GatewayTransportHttpx(timeout_seconds=DOCTOR_HTTP_TIMEOUT_SECONDS)
        )

    @staticmethod
    def _path_from_environ(environ: Mapping[str, str]) -> Path:
        for key in _OVERLAY_ENV_KEYS:
            value = environ.get(key)
            if value is not None and value.strip():
                return Path(value.strip()).expanduser()
        return Path.home() / ".omninode" / "delegation" / "bifrost_overrides.yaml"

    def _load_backend(self) -> tuple[str, str | None] | None:
        raw: object = yaml.safe_load(self._overlay_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("overlay root is not a mapping")
        backends = raw.get("backends")
        if not isinstance(backends, list):
            return None
        for item in backends:
            if not isinstance(item, dict) or item.get("tier") != "local":
                continue
            endpoint = item.get("endpoint_url")
            # Only a chat rung answers a delegation prompt; the embedding backend
            # is a local-tier entry too and its served id is not the chat model.
            if not (
                isinstance(endpoint, str)
                and endpoint.strip().rstrip("/").endswith("/chat/completions")
            ):
                continue
            model = item.get("model_name")
            if not isinstance(model, str) or not model.strip():
                model = item.get("served_model_id")
            declared = (
                model.strip() if isinstance(model, str) and model.strip() else None
            )
            return (endpoint.strip(), declared)
        return None

    @staticmethod
    def _models_url(endpoint_url: str) -> tuple[str, str]:
        parsed = urlsplit(endpoint_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint is not an HTTP origin")
        marker = "/v1/"
        if marker not in parsed.path:
            raise ValueError("endpoint has no OpenAI-compatible v1 path")
        prefix, _, _ = parsed.path.partition(marker)
        models_path = f"{prefix}/v1/models"
        origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        return (urlunsplit((parsed.scheme, parsed.netloc, models_path, "", "")), origin)

    async def _served_model_ids(self, models_url: str) -> tuple[int, set[str]]:
        response = await self._transport.get(
            models_url,
            timeout=DOCTOR_HTTP_TIMEOUT_SECONDS,
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            return (response.status, set())
        payload: object = json.loads(await response.text())
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("models response has an invalid shape")
        served: set[str] = set()
        for item in payload["data"]:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    served.add(model_id.strip())
        return (response.status, served)

    def _no_local_model(self) -> tuple[ModelDelegationDiagnosis, bool]:
        return (
            ModelDelegationDiagnosis(
                fault=EnumDelegationDoctorFault.NO_LOCAL_MODEL,
                detail=f"No local chat backend is declared in {self._overlay_path}.",
                fix=(
                    f"Edit {self._overlay_path}: add a backend with `tier: local`, "
                    "a chat-completions `endpoint_url`, and optionally `model_name`."
                ),
            ),
            True,
        )

    def _evaluate(self) -> tuple[ModelDelegationDiagnosis, bool]:
        try:
            backend = self._load_backend()
        except FileNotFoundError:
            return self._no_local_model()
        except (OSError, UnicodeError, yaml.YAMLError, ValueError):
            return self._unknown("the overlay is unreadable or invalid")
        if backend is None:
            return self._no_local_model()
        endpoint_url, declared_model = backend
        try:
            models_url, origin = self._models_url(endpoint_url)
        except ValueError:
            return self._unknown("the local endpoint URL is invalid")

        try:
            status, served_models = run_async(
                lambda: self._served_model_ids(models_url)
            )
        except InfraUnavailableError:
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.LOCAL_MODEL_NOT_SERVING,
                    detail=f"The local model server at {origin} could not be reached.",
                    fix=f"Start the model server at {origin}.",
                ),
                True,
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return self._unknown("the models response was unreadable or invalid")

        if status != 200:
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.LOCAL_MODEL_NOT_SERVING,
                    detail=f"The local model server returned HTTP {status}.",
                    fix=f"Start the model server at {origin}.",
                ),
                True,
            )
        if declared_model is not None and declared_model not in served_models:
            served = ", ".join(sorted(served_models)) or "no model ids"
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.LOCAL_MODEL_ID_MISMATCH,
                    detail=(
                        f"The overlay declares '{declared_model}', but the server "
                        f"reports {served}."
                    ),
                    fix=(
                        f"Edit {self._overlay_path}: set model_name to one of the "
                        "served ids, or serve the declared model."
                    ),
                ),
                True,
            )
        return (
            ModelDelegationDiagnosis(
                fault=None,
                detail=(
                    f"Local model '{declared_model}' is serving at {origin}."
                    if declared_model is not None
                    else f"A local model server is answering at {origin}."
                ),
                fix="",
            ),
            True,
        )

    @staticmethod
    def _unknown(reason: str) -> tuple[ModelDelegationDiagnosis, bool]:
        return (
            ModelDelegationDiagnosis(
                fault=None,
                detail=f"Delegation local model was not judged because {reason}.",
                fix="",
            ),
            False,
        )

    def diagnose(self) -> ModelDelegationDiagnosis:
        """Return the local-model diagnosis without raising."""
        try:
            return self._evaluate()[0]
        except Exception:  # noqa: BLE001 - doctor diagnostics are total functions
            return self._unknown("the check failed safely")[0]

    def run(self) -> ModelDoctorCheckResult:
        """Run the local-model check without allowing an exception to escape."""
        try:
            diagnosis, judged = self._evaluate()
            return render_diagnosis(
                name=self.check_name,
                diagnosis=diagnosis,
                judged=judged,
            )
        except Exception:  # noqa: BLE001 - required total doctor boundary
            return unknown_result(name=self.check_name, dependency=self.check_id)


__all__ = ["CheckDelegationLocalModel"]

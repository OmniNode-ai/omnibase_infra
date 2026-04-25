# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from pydantic import BaseModel, ConfigDict


class ModelMonitorAlertContract(BaseModel):
    """Typed backing model for monitor_alert_contract.yaml.

    First typed wrapper — sub-field tightening (e.g. typed EventBus model)
    is follow-up work. event_bus remains dict[str, list[str]] intentionally.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    contract_version: str | dict[str, int]
    description: str = ""
    event_bus: dict[str, list[str]] = {}

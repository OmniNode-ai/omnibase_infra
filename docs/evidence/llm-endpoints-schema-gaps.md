# LLM Endpoint Topology Schema Gaps

`contracts/llm_endpoints.yaml` currently supports narrow validation for:

- endpoint role: `role`
- deployment status: `status`
- endpoint base URL: `endpoint_url`
- served model identity: `model_hf_id`
- launch/runtime identity: `launchd_unit_or_none`, plus `hardware` as operator context
- context-window budget: `context_window_budgeted`
- env alias ownership: `url_env_var`, `role_env_alias`

The current typed schema does not support these plan fields without a broader
contract update:

- served model aliases or multiple served model IDs separate from `model_hf_id`
- an explicit `base_url` field separate from `endpoint_url`
- health-check path
- cost basis or pricing metadata
- context-window source/provenance separate from `context_window_budgeted`
- structured alias policy separate from `url_env_var` and `role_env_alias`

This document therefore records that validation was added only for the fields already present in the
contract, and leaves the broader topology schema expansion to a dedicated design
change.

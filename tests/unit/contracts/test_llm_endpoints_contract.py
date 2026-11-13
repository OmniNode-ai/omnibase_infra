# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Inspection tests for ``contracts/llm_endpoints.yaml`` (plan Task 4).

Enforces stable-canonical + operator-annotation schema, required slot_ids,
reasoning-moe-35b fields, and a closed role taxonomy. Planned slots may
have null host/port/endpoint_url/model_hf_id. See OMN-9292 / OMN-9294.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONTRACT_PATH = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"

_STABLE_CANONICAL_KEYS: frozenset[str] = frozenset(
    [
        "slot_id",
        "host",
        "port",
        "endpoint_url",
        "url_env_var",
        "role_env_alias",
        "model_hf_id",
        "role",
        "status",
    ]
)
_OPERATOR_ANNOTATION_KEYS: frozenset[str] = frozenset(
    ["hardware", "context_window_budgeted", "launchd_unit_or_none", "notes"]
)
_REQUIRED_SLOT_IDS: frozenset[str] = frozenset(
    [
        "coder-5090",
        "coder-fast-4090",
        "embeddings-201",
        "reasoning-deepseek-32b",
        "reasoning-moe-35b",
        "embeddings-200",
        "second-opinion-gemma",
        "vision-planned",
        "stt-planned",
        "tts-planned",
        "reranker-planned",
    ]
)
_ROLE_TAXONOMY: frozenset[str] = frozenset(
    [
        "coder_slow",
        "coder_fast",
        "embedding",
        "reasoning",
        "reasoning_fast",
        "reasoning_lightweight",
        "reasoning_transient",
        "vision",
        "stt",
        "tts",
        "reranker",
    ]
)
_PLANNED_NULLABLE_FIELDS: frozenset[str] = frozenset(
    ["host", "port", "endpoint_url", "model_hf_id"]
)
# OMN-16442 (2026-08-28): LLM_CODER_FAST_URL REMOVED from this set. The rule
# this frozenset encodes is "this env var must be owned by a RUNNING topology
# slot". LLM_CODER_FAST_URL's slot (coder-fast-4090, .201:8001) is the RTX 4090
# physically removed from the host for RMA (OMN-16407) — live re-probe returns
# curl exit 7 "Couldn't connect to server" — so the slot is now `disabled` and,
# per the contract's own rule, owns no url_env_var. Keeping the var in this set
# would force the dead slot to keep claiming `running` to make a test pass.
#
# The ENV VAR itself still exists and is still required by the runtime
# (docker-compose.infra.yml uses ${LLM_CODER_FAST_URL:?...}); operators alias it
# onto the surviving .201:8000 coder endpoint until replacement hardware is
# registered. See config/shared_key_registry.yaml for that guidance. The
# topology contract models HARDWARE SLOTS, not consumer-facing aliases.
_RUNTIME_REQUIRED_URL_ENV_VARS: frozenset[str] = frozenset(
    [
        "LLM_CODER_URL",
        "LLM_EMBEDDING_URL",
    ]
)

# OMN-16442: slots whose backing hardware/listener is gone. Asserted `disabled`
# so a future edit cannot flip one back to `running` without re-probing.
_DECOMMISSIONED_SLOT_IDS: frozenset[str] = frozenset(
    {"coder-fast-4090", "reasoning-deepseek-32b", "reasoning-moe-35b", "embeddings-200"}
)
_SUPPORTED_TOPOLOGY_FIELDS: frozenset[str] = frozenset(
    [
        "role",
        "status",
        "endpoint_url",
        "model_hf_id",
        "url_env_var",
        "role_env_alias",
        "launchd_unit_or_none",
        "context_window_budgeted",
    ]
)


def _load_endpoints() -> list[dict[str, Any]]:
    assert _CONTRACT_PATH.exists(), f"Missing contract file: {_CONTRACT_PATH}"
    data = yaml.safe_load(_CONTRACT_PATH.read_text())
    assert isinstance(data, dict) and "endpoints" in data, "Need top-level 'endpoints'"
    endpoints = data["endpoints"]
    assert isinstance(endpoints, list) and endpoints, "'endpoints' must be non-empty"
    return endpoints


@pytest.mark.unit
class TestLlmEndpointsContract:
    """Schema inspection for the canonical LLM topology contract."""

    def test_schema_and_required_fields(self) -> None:
        expected = _STABLE_CANONICAL_KEYS | _OPERATOR_ANNOTATION_KEYS
        for ep in _load_endpoints():
            assert expected.issubset(ep.keys()), (
                f"Entry {ep.get('slot_id')!r} missing keys: {expected - ep.keys()}"
            )
            for key in ("slot_id", "role", "status"):
                assert ep.get(key), f"Entry {ep.get('slot_id')!r} has empty {key}"

    def test_running_slots_have_core_fields_non_null(self) -> None:
        """Running slots must have host/port/endpoint_url/model_hf_id non-null.

        url_env_var and role_env_alias are allowed to be null on running slots
        (Docker-internal slots have no external env-var; aliases may be unassigned).
        disabled, on_demand, and planned slots are explicitly exempt.

        Slots flagged ``endpoint_from_overlay`` keep host/port/endpoint_url null
        in the public contract (the address is read from the llm_endpoints overlay
        at run time, OMN-24017); only model_hf_id stays required.
        """
        for slot in _load_endpoints():
            if slot["status"] == "running":
                if slot.get("endpoint_from_overlay"):
                    assert slot["model_hf_id"] is not None, (
                        f"running overlay slot {slot['slot_id']!r} must have "
                        "non-null model_hf_id"
                    )
                    continue
                for key in _PLANNED_NULLABLE_FIELDS:
                    assert slot[key] is not None, (
                        f"running slot {slot['slot_id']!r} must have non-null {key}"
                    )

    def test_supported_topology_fields_are_declared_on_every_slot(self) -> None:
        """Real YAML carries the convergence fields supported by today's schema."""
        for slot in _load_endpoints():
            missing = _SUPPORTED_TOPOLOGY_FIELDS - slot.keys()
            assert not missing, (
                f"slot {slot.get('slot_id')!r} missing supported topology fields: "
                f"{sorted(missing)}"
            )

            assert slot["role"], f"slot {slot['slot_id']!r} must declare endpoint role"
            assert slot["status"], (
                f"slot {slot['slot_id']!r} must declare deployment status"
            )

            # Overlay slots keep host/port/endpoint_url null in the public
            # contract; their address is read from the llm_endpoints overlay at
            # run time (OMN-24017). The endpoint_url consistency check below
            # is skipped for them; model_hf_id is still checked below.
            if slot.get("endpoint_from_overlay"):
                for key in ("host", "port", "endpoint_url"):
                    assert slot[key] is None, (
                        f"overlay slot {slot['slot_id']!r} must keep {key} null"
                    )

            if slot["endpoint_url"] is not None:
                parsed = urlparse(slot["endpoint_url"])
                assert parsed.scheme in {"http", "https"}, (
                    f"slot {slot['slot_id']!r} endpoint_url must be http(s)"
                )
                assert parsed.hostname == slot["host"], (
                    f"slot {slot['slot_id']!r} endpoint_url host must match host"
                )
                assert parsed.port == slot["port"], (
                    f"slot {slot['slot_id']!r} endpoint_url port must match port"
                )
                assert parsed.path in {"", "/"}, (
                    f"slot {slot['slot_id']!r} endpoint_url must be a base URL"
                )
                assert not parsed.query and not parsed.fragment, (
                    f"slot {slot['slot_id']!r} endpoint_url must not carry query/fragment"
                )

            if slot["model_hf_id"] is not None:
                assert "/" in slot["model_hf_id"], (
                    f"slot {slot['slot_id']!r} model_hf_id must include namespace/model"
                )

            if slot["launchd_unit_or_none"] is not None:
                assert slot["launchd_unit_or_none"].startswith("com."), (
                    f"slot {slot['slot_id']!r} launchd unit must be a launchd label"
                )

            if slot["context_window_budgeted"] is not None:
                assert slot["context_window_budgeted"] > 0, (
                    f"slot {slot['slot_id']!r} context window budget must be positive"
                )

    def test_endpoint_alias_fields_are_canonical_when_present(self) -> None:
        for slot in _load_endpoints():
            for key in ("url_env_var", "role_env_alias"):
                value = slot.get(key)
                if value is None:
                    continue
                assert value.startswith("LLM_") and value.endswith("_URL"), (
                    f"slot {slot['slot_id']!r} {key} must use LLM_*_URL naming"
                )

    def test_required_slot_ids_present(self) -> None:
        present = {ep["slot_id"] for ep in _load_endpoints()}
        missing = _REQUIRED_SLOT_IDS - present
        assert not missing, f"Missing required slot_ids: {sorted(missing)}"

    def test_role_values_are_closed_taxonomy(self) -> None:
        for ep in _load_endpoints():
            assert ep["role"] in _ROLE_TAXONOMY, (
                f"Entry {ep['slot_id']!r} role {ep['role']!r} outside taxonomy"
            )

    def test_lane_overlay_binding_on_overlay_slots(self) -> None:
        for slot in _load_endpoints():
            if slot.get("endpoint_from_overlay"):
                binding = slot.get("lane_overlay_binding")
                if binding is not None:
                    assert isinstance(binding, dict)
                    assert binding.get("backend"), (
                        f"slot {slot['slot_id']!r} lane_overlay_binding.backend must be non-empty"
                    )
                    assert binding.get("env_var"), (
                        f"slot {slot['slot_id']!r} lane_overlay_binding.env_var must be non-empty"
                    )

    def test_embeddings_201_has_lane_overlay_binding(self) -> None:
        for slot in _load_endpoints():
            if slot["slot_id"] == "embeddings-201":
                assert slot["endpoint_from_overlay"] is True
                binding = slot.get("lane_overlay_binding")
                assert binding is not None, (
                    "embeddings-201 must have lane_overlay_binding"
                )
                assert binding["backend"] == "local-embedding"
                assert binding["env_var"] == "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL"
                assert slot["host"] is None
                assert slot["port"] is None
                assert slot["endpoint_url"] is None

    def test_reasoning_moe_35b_is_disabled_after_endpoint_loss(self) -> None:
        """OMN-16442 (supersedes the OMN-9292 running-slot field pins).

        This test used to assert reasoning-moe-35b was a RUNNING slot at
        ``http://192.168.86.200:8102`` serving
        ``mlx-community/Qwen3.6-35B-A3B-8bit`` and owning ``LLM_QWEN3_NEXT_URL``.
        That port has no listener: live probe 2026-08-28, `curl
        http://192.168.86.200:8102/v1/models` -> exit 7 "Couldn't connect to
        server". The canonical inventory
        (omni_home/docs/reference/AI_LAB_HARDWARE.md, verified 2026-08-28)
        records Mac Studio ports 8100/8102/8103 as having no model listener and
        calls references to DeepSeek-R1-Distill / Qwen2.5-72B / Qwen2-VL on
        those ports STALE.

        The slot is kept (it is in ``_REQUIRED_SLOT_IDS``) but disabled, and the
        assertions are inverted to the contract rule that actually matters: a
        disabled slot owns no runtime env var.
        """
        by_slot = {ep["slot_id"]: ep for ep in _load_endpoints()}
        ep = by_slot.get("reasoning-moe-35b")
        assert ep is not None, "Missing required slot 'reasoning-moe-35b'"
        assert ep["status"] == "disabled"
        assert ep["url_env_var"] is None
        assert ep["role_env_alias"] is None
        assert ep["endpoint_url"] is None
        assert ep["role"] == "reasoning_fast"

    def test_decommissioned_slots_are_disabled_and_own_no_env_vars(self) -> None:
        """OMN-16442: a slot whose listener is gone must not look routable.

        Re-probed 2026-08-28, all three return curl exit 7:
        coder-fast-4090 (.201:8001, RTX 4090 removed for RMA — OMN-16407),
        reasoning-moe-35b (.200:8102), embeddings-200 (.200:8100).
        """
        by_slot = {ep["slot_id"]: ep for ep in _load_endpoints()}
        for slot_id in sorted(_DECOMMISSIONED_SLOT_IDS):
            ep = by_slot.get(slot_id)
            assert ep is not None, f"Missing slot {slot_id!r}"
            assert ep["status"] == "disabled", (
                f"slot {slot_id!r} has no live listener and must not claim "
                f"status {ep['status']!r}"
            )
            assert ep["url_env_var"] is None, (
                f"disabled slot {slot_id!r} must not own a runtime env var"
            )

    def test_retired_reasoning_slot_releases_all_deployment_fields(self) -> None:
        by_slot = {ep["slot_id"]: ep for ep in _load_endpoints()}
        slot = by_slot["reasoning-deepseek-32b"]
        assert slot["status"] == "disabled"
        for field in (
            "host",
            "port",
            "endpoint_url",
            "url_env_var",
            "role_env_alias",
            "model_hf_id",
            "context_window_budgeted",
            "launchd_unit_or_none",
        ):
            assert slot[field] is None, field

    def test_live_slots_record_their_probed_served_identity(self) -> None:
        """The running slots carry their last live-probed identities.

        Pinned so a silent hardware/model swap (the exact failure mode behind
        OMN-16419 and OMN-16407) shows up as a red test rather than as silent
        mis-attribution to a model that is not running.

        coder-5090 re-pinned 2026-09-03 (OMN-14379); embeddings remain on
        their 2026-08-28 probe (OMN-16442).
        """
        by_slot = {ep["slot_id"]: ep for ep in _load_endpoints()}

        # GET .201:8000/v1/models, 2026-09-03T16:07Z -> id "Qwen3.6-35B-A3B",
        # owned_by "vllm", max_model_len 131072. The SGLang "qwen3.8" backend
        # this replaces was not running at all: :8000 had no listener until
        # vllm-gpu0.service was resized off its unstartable 0.92 GPU fraction.
        # vLLM, unlike SGLang, rejects an unknown model id -- the old pin
        # "qwen3.8" now returns HTTP 404 against this endpoint.
        coder = by_slot["coder-5090"]
        assert coder["status"] == "running"
        assert coder["endpoint_url"] == "http://192.168.86.201:8000"
        assert coder["model_hf_id"] == "Qwen/Qwen3.8-27B"
        assert coder["context_window_budgeted"] == 131072

        # OMN-24017: the embedding slot's address left the public contract and
        # moved to the llm_endpoints overlay; the entry keeps its served
        # identity (Qwen/Qwen3-Embedding-0.6B, 1024-dim) but no address.
        emb = by_slot["embeddings-201"]
        assert emb["status"] == "running"
        assert emb["model_hf_id"] == "Qwen/Qwen3-Embedding-0.6B"

    def test_embeddings_201_address_moves_to_the_llm_endpoints_overlay(self) -> None:
        """The embeddings-201 slot carries no address; the overlay owns it.

        OMN-24017: the embedding server on the first GPU shared it with the
        27B chat model and pushed it into CUDA out of memory, so its
        address is the llm_endpoints overlay value (key embeddings-201), read
        by the deploy guard. The public contract must carry no address on
        this slot.
        """
        by_slot = {ep["slot_id"]: ep for ep in _load_endpoints()}
        ep = by_slot.get("embeddings-201")
        assert ep is not None, "Missing required slot 'embeddings-201'"
        assert ep["endpoint_from_overlay"] is True
        assert ep["url_env_var"] == "LLM_EMBEDDING_URL"
        assert ep["host"] is None
        assert ep["port"] is None
        assert ep["endpoint_url"] is None
        assert ep["hardware"] is None
        assert ep["role"] == "embedding"

    def test_no_slot_names_the_removed_embedding_port(self) -> None:
        """No slot and no line of the contract file names the removed port.

        The embedding server on the first GPU is gone (OMN-24017); the
        removed port must not survive on any slot, in the notes, or in the
        file header.
        """
        text = _CONTRACT_PATH.read_text(encoding="utf-8")
        assert "8002" not in text, (
            "contracts/llm_endpoints.yaml must not name the removed "
            "embedding port on the lab model host"
        )
        for ep in _load_endpoints():
            assert ep.get("port") != 8002, (
                f"slot {ep.get('slot_id')!r} must not name the removed port"
            )

    def test_runtime_required_env_vars_are_owned_by_running_slots(self) -> None:
        """Runtime-required URL env vars must point at running canonical slots."""
        by_env = {
            ep["url_env_var"]: ep
            for ep in _load_endpoints()
            if ep.get("url_env_var") is not None
        }

        for env_var in _RUNTIME_REQUIRED_URL_ENV_VARS:
            ep = by_env.get(env_var)
            assert ep is not None, f"{env_var} is not assigned to any endpoint slot"
            assert ep["status"] == "running", (
                f"{env_var} is assigned to non-running slot {ep['slot_id']!r}"
            )

    def test_disabled_slots_do_not_own_runtime_url_env_vars(self) -> None:
        for ep in _load_endpoints():
            if ep["status"] == "disabled":
                assert ep["url_env_var"] is None, (
                    f"disabled slot {ep['slot_id']!r} must not own url_env_var"
                )

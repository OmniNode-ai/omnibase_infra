# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for HandlerSecretSeed — OMN-16897.

Every test drives the real handler with an injected store, an injected
source reader, and an injected auth probe. No network, no filesystem, no
live Infisical, and no real credential anywhere in this file.

The load-bearing tests are the two that guard the value-flow constraint:

* ``test_request_model_rejects_inline_secret_values`` — the request model
  must have no way to carry a value, because a node request is serialised
  onto the bus and into the event log.
* ``test_no_secret_value_reaches_the_receipt_when_the_store_echoes_it`` —
  the store raises an exception whose message embeds the key, the exact
  shape a chatty SDK produces. The value must appear nowhere in the
  serialised receipt.

If either of those regresses, this node leaks key material into durable
storage, which is strictly worse than not having the node at all.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_secret_seed_effect.handlers.handler_secret_seed import (
    REQUIRED_AUTH_ENV_VARS,
    HandlerSecretSeed,
    parse_source,
    render_receipt,
)
from omnibase_infra.nodes.node_secret_seed_effect.models.enum_secret_seed_verdict import (
    EnumSecretSeedVerdict,
)
from omnibase_infra.nodes.node_secret_seed_effect.models.model_secret_seed_request import (
    ModelSecretSeedRequest,
)

pytestmark = pytest.mark.unit

# A fake key with a distinctive shape so a leak is unambiguous when asserted
# against a whole serialised receipt. Not a real credential.
# Chosen so the string itself trips NONE of sanitize_error_message's
# SENSITIVE_PATTERNS. A fixture containing "key" or "secret" would be blanked
# by the sanitiser on sight, and every redaction assertion below would then
# pass without the explicit value-redaction layer existing at all.
_FAKE_VALUE = "sk-test-ZZZZ-0001112223334445556667778"
_OTHER_VALUE = "sk-test-YYYY-3334445556667778889990001"
_HOST = "https://infisical.invalid:8881"
_PROJECT = UUID("11111111-2222-3333-4444-555555555555")
_PATH = "/shared/llm/"


def _request(**overrides: object) -> ModelSecretSeedRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "source_path": "/nonexistent/seed.env",
        "infisical_host": _HOST,
        "project_id": _PROJECT,
        "environment_slug": "dev",
        "secret_path": _PATH,
    }
    fields.update(overrides)
    return ModelSecretSeedRequest(**fields)  # type: ignore[arg-type]


class _RecordingStore:
    """In-memory stand-in for ``InfisicalSecretStore``.

    Records the NAMES it was asked to write and how many writes happened, so
    a test can assert "zero writes" directly rather than inferring it.
    """

    def __init__(
        self,
        existing: list[str] | None = None,
        fail_on: set[str] | None = None,
        raise_on_list: bool = False,
        drop_on_readback: set[str] | None = None,
        error_template: str = (
            "upstream rejected secret {key} with payload {value} (HTTP 422)"
        ),
    ) -> None:
        self.existing = list(existing or [])
        self.fail_on = fail_on or set()
        self.raise_on_list = raise_on_list
        self.drop_on_readback = drop_on_readback or set()
        self.error_template = error_template
        self.writes: list[str] = []
        self.write_count = 0
        self.closed = False
        self.list_calls = 0

    async def set_secret(self, key: str, value: str) -> bool:
        self.write_count += 1
        if key in self.fail_on:
            # A chatty SDK echoing the submitted value back inside its own
            # error message. This is the leak vector the redaction exists for.
            raise RuntimeError(self.error_template.format(key=key, value=value))
        self.writes.append(key)
        if key not in self.existing:
            self.existing.append(key)
        return True

    async def list_keys(self, prefix: str | None = None) -> list[str]:
        self.list_calls += 1
        if self.raise_on_list:
            raise ConnectionError("infisical name listing unavailable")
        names = [n for n in self.existing if n not in self.drop_on_readback]
        if prefix is None:
            return names
        return [n for n in names if n.startswith(prefix)]

    async def close(self, timeout_seconds: float = 30.0) -> None:
        self.closed = True


def _handler(
    store: _RecordingStore,
    source: str,
    missing_auth: list[str] | None = None,
) -> HandlerSecretSeed:
    return HandlerSecretSeed(
        store_factory=lambda _request: store,
        source_reader=lambda _path: source,
        auth_probe=lambda: list(missing_auth or []),
    )


# --- the value-flow guards -------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    ["value", "values", "secret", "secrets", "secret_value", "key_value", "payload"],
)
def test_request_model_rejects_inline_secret_values(field_name: str) -> None:
    """A node request must have no way to carry a secret value.

    Node inputs are serialised onto the bus and into the event log, so a
    value accepted here would be durably persisted in both. The rejection
    must also EXPLAIN itself — a caller who reaches for `--value` should be
    told to use a source file, not handed a generic "extra inputs are not
    permitted".
    """
    with pytest.raises(ValidationError) as excinfo:
        _request(**{field_name: _FAKE_VALUE})

    message = str(excinfo.value)
    assert "source_path" in message, (
        "the rejection must point the caller at the supported shape"
    )
    assert _FAKE_VALUE not in message, (
        "the rejection message must not echo the value it just refused"
    )


def test_request_model_has_no_value_carrying_field() -> None:
    """Structural: no field on the request may plausibly hold a value."""
    forbidden = {"value", "values", "secret", "secrets", "secret_value", "payload"}
    assert not (set(ModelSecretSeedRequest.model_fields) & forbidden)


@pytest.mark.asyncio
async def test_no_secret_value_reaches_the_receipt_when_the_store_echoes_it() -> None:
    """The whole serialised receipt must be free of key material.

    The store raises an error containing the submitted value verbatim — the
    exact shape a chatty upstream produces. Asserting on the full JSON dump
    rather than on ``detail`` alone is deliberate: it catches a leak into
    ANY field, including one added later.
    """
    store = _RecordingStore(fail_on={"LLM_GLM_API_KEY"})
    handler = _handler(
        store,
        f"LLM_GLM_API_KEY={_FAKE_VALUE}\nLLM_GLM_MODEL_NAME=glm-4\n",
    )

    result = await handler.handle(_request(execute=True))

    receipt = render_receipt(result)
    assert _FAKE_VALUE not in receipt
    assert _FAKE_VALUE not in json.dumps(result.model_dump(mode="json"))
    assert result.verdict is EnumSecretSeedVerdict.WRITE_FAILED
    # The NAME is still reported — redaction must not cost diagnosability.
    assert result.failed_names == ["LLM_GLM_API_KEY"]
    assert any("LLM_GLM_API_KEY" in err for err in result.errors)


@pytest.mark.asyncio
async def test_value_is_redacted_even_when_the_sanitiser_does_not_fire() -> None:
    """The second redaction layer is the one that actually has to work.

    ``sanitize_error_message`` blanks messages it recognises as
    credential-SHAPED — the message in the test above says "secret" and
    "payload", so it gets blanked wholesale and the value never survives.
    That is exactly why it cannot be the only defence: an upstream that
    echoes a key inside an ordinary-looking sentence trips none of its
    patterns.

    Here the message is deliberately innocuous. If the handler relied on the
    sanitiser alone the key would pass straight through into the receipt, so
    this test is what proves the explicit value redaction exists and runs.
    """
    # The NAME must also be innocuous here: "LLM_GLM_API_KEY" contains
    # "api_key", so a message quoting it is blanked by the sanitiser and this
    # test would prove nothing.
    store = _RecordingStore(
        fail_on={"GLM_MODEL_NAME"},
        error_template="upstream returned 422 for {key} while sending {value}",
    )
    handler = _handler(store, f"GLM_MODEL_NAME={_FAKE_VALUE}\n")

    result = await handler.handle(_request(execute=True))

    joined = " ".join(result.errors)
    assert _FAKE_VALUE not in render_receipt(result)
    assert "***" in joined, (
        "the value must be replaced by the redaction marker, proving the "
        "second layer fired rather than the message merely being blanked"
    )
    assert "422" in joined, "redaction must preserve the diagnostic remainder"


def test_parse_source_never_echoes_line_content_on_a_malformed_line() -> None:
    """A malformed line in a secrets file may itself BE a secret."""
    leaked = "sk-test-bare-token-with-no-equals-sign"
    with pytest.raises(ValueError) as excinfo:
        parse_source(f"GOOD=1\n{leaked}\n")

    message = str(excinfo.value)
    assert "line 2" in message
    assert leaked not in message


def test_parse_source_wraps_values_so_repr_cannot_leak() -> None:
    parsed = parse_source(f"A={_FAKE_VALUE}\n")
    assert _FAKE_VALUE not in repr(parsed)
    assert parsed["A"].get_secret_value() == _FAKE_VALUE


# --- fail-fast on missing auth ---------------------------------------------


@pytest.mark.asyncio
async def test_missing_auth_fails_fast_without_reading_or_writing() -> None:
    """No auth material => no fallback identity, no write, and no file read.

    The auth probe runs BEFORE the source is read on purpose: if we cannot
    write, there is no reason to pull key material into memory at all.
    """
    store = _RecordingStore()
    reads: list[str] = []

    def _reader(path: str) -> str:
        reads.append(path)
        return f"LLM_GLM_API_KEY={_FAKE_VALUE}\n"

    handler = HandlerSecretSeed(
        store_factory=lambda _request: store,
        source_reader=_reader,
        auth_probe=lambda: ["INFISICAL_CLIENT_SECRET"],
    )

    result = await handler.handle(_request(execute=True))

    assert result.verdict is EnumSecretSeedVerdict.AUTH_UNAVAILABLE
    assert result.success is False
    assert store.write_count == 0
    assert reads == [], "the source must not be read when we cannot write"
    # The missing VARIABLE NAME is reported; no value ever is.
    assert "INFISICAL_CLIENT_SECRET" in result.detail


def test_required_auth_env_vars_are_the_universal_auth_pair() -> None:
    """Pinned: drift here silently changes what 'fail-fast' checks for."""
    assert REQUIRED_AUTH_ENV_VARS == (
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
    )


@pytest.mark.asyncio
async def test_store_construction_failure_is_auth_unavailable_not_a_crash() -> None:
    def _explode(_request: ModelSecretSeedRequest) -> _RecordingStore:
        raise RuntimeError(f"login failed for client with secret {_FAKE_VALUE}")

    handler = HandlerSecretSeed(
        store_factory=_explode,
        source_reader=lambda _path: f"A={_FAKE_VALUE}\n",
        auth_probe=list,
    )
    result = await handler.handle(_request(execute=True))

    assert result.verdict is EnumSecretSeedVerdict.AUTH_UNAVAILABLE
    assert _FAKE_VALUE not in render_receipt(result)


# --- dry run writes nothing ------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_writes_nothing_and_still_reports_the_plan() -> None:
    store = _RecordingStore(existing=["LLM_GLM_MODEL_NAME"])
    handler = _handler(
        store,
        f"LLM_GLM_API_KEY={_FAKE_VALUE}\nLLM_GLM_MODEL_NAME=glm-4\n",
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumSecretSeedVerdict.DRY_RUN
    assert result.success is True
    assert result.dry_run is True
    assert store.write_count == 0
    assert store.writes == []
    # The plan is still complete: create vs update is resolved from a NAME
    # listing, which is the only read this node ever performs.
    assert result.created_names == ["LLM_GLM_API_KEY"]
    assert result.updated_names == ["LLM_GLM_MODEL_NAME"]
    assert result.written_count == 0


def test_dry_run_is_the_default_mode() -> None:
    """Forgetting the flag must be the mode that writes nothing."""
    assert _request().dry_run is True
    assert _request().execute is False
    assert _request(execute=True).dry_run is False


# --- the happy path and its verification -----------------------------------


@pytest.mark.asyncio
async def test_execute_upserts_and_verifies_by_name() -> None:
    store = _RecordingStore(existing=["LLM_GLM_MODEL_NAME"])
    handler = _handler(
        store,
        f"LLM_GLM_API_KEY={_FAKE_VALUE}\nLLM_GLM_MODEL_NAME=glm-4\n",
    )

    result = await handler.handle(_request(execute=True))

    assert result.verdict is EnumSecretSeedVerdict.SEEDED
    assert result.success is True
    assert result.created_names == ["LLM_GLM_API_KEY"]
    assert result.updated_names == ["LLM_GLM_MODEL_NAME"]
    assert result.verified_names == ["LLM_GLM_API_KEY", "LLM_GLM_MODEL_NAME"]
    assert result.unverified_names == []
    assert result.written_count == 2
    assert store.closed is True
    assert _FAKE_VALUE not in render_receipt(result)


@pytest.mark.asyncio
async def test_execute_with_readback_disabled_does_not_claim_verification() -> None:
    store = _RecordingStore()
    handler = _handler(store, f"LLM_GLM_API_KEY={_FAKE_VALUE}\n")

    result = await handler.handle(_request(execute=True, verify_readback=False))

    assert result.verdict is EnumSecretSeedVerdict.SEEDED
    assert result.success is True
    assert result.created_names == ["LLM_GLM_API_KEY"]
    assert result.verified_names == []
    assert result.unverified_names == []
    assert store.list_calls == 1
    assert "name readback was skipped by request" in result.detail
    assert "confirmed present by name readback" not in result.detail


@pytest.mark.asyncio
async def test_reseeding_the_same_name_is_an_update_not_a_duplicate() -> None:
    """Idempotency: the second run of the same command is a clean update."""
    store = _RecordingStore()
    handler = _handler(store, f"LLM_GLM_API_KEY={_FAKE_VALUE}\n")

    first = await handler.handle(_request(execute=True))
    second = await handler.handle(_request(execute=True))

    assert first.created_names == ["LLM_GLM_API_KEY"]
    assert second.created_names == []
    assert second.updated_names == ["LLM_GLM_API_KEY"]
    assert second.verdict is EnumSecretSeedVerdict.SEEDED


@pytest.mark.asyncio
async def test_a_write_that_does_not_appear_on_readback_fails_closed() -> None:
    """An unconfirmed write is not a confirmed one."""
    store = _RecordingStore(drop_on_readback={"LLM_GLM_API_KEY"})
    handler = _handler(store, f"LLM_GLM_API_KEY={_FAKE_VALUE}\n")

    result = await handler.handle(_request(execute=True))

    assert result.verdict is EnumSecretSeedVerdict.VERIFY_FAILED
    assert result.success is False
    assert result.unverified_names == ["LLM_GLM_API_KEY"]


@pytest.mark.asyncio
async def test_partial_failure_reports_every_name_rather_than_aborting() -> None:
    """A half-finished seed the operator cannot see the shape of is worse."""
    store = _RecordingStore(fail_on={"B_KEY"})
    handler = _handler(
        store, f"A_KEY={_FAKE_VALUE}\nB_KEY={_OTHER_VALUE}\nC_KEY=plain\n"
    )

    result = await handler.handle(_request(execute=True))

    assert result.verdict is EnumSecretSeedVerdict.WRITE_FAILED
    assert result.failed_names == ["B_KEY"]
    assert sorted(result.created_names) == ["A_KEY", "C_KEY"]
    assert store.write_count == 3, "the run must not stop at the first failure"
    receipt = render_receipt(result)
    assert _FAKE_VALUE not in receipt
    assert _OTHER_VALUE not in receipt


# --- addressing, selection, and the failing-empty rule ----------------------


@pytest.mark.asyncio
async def test_keys_allowlist_seeds_only_the_named_key() -> None:
    store = _RecordingStore()
    handler = _handler(
        store, f"LLM_GLM_API_KEY={_FAKE_VALUE}\nUNRELATED={_OTHER_VALUE}\n"
    )

    result = await handler.handle(_request(execute=True, keys=["LLM_GLM_API_KEY"]))

    assert store.writes == ["LLM_GLM_API_KEY"]
    assert result.created_names == ["LLM_GLM_API_KEY"]


@pytest.mark.asyncio
async def test_requested_key_absent_from_source_is_a_failing_run() -> None:
    """Seeding nothing must never read green."""
    store = _RecordingStore()
    handler = _handler(store, "SOMETHING_ELSE=1\n")

    result = await handler.handle(_request(execute=True, keys=["LLM_GLM_API_KEY"]))

    assert result.verdict is EnumSecretSeedVerdict.NO_KEYS
    assert result.success is False
    assert result.missing_from_source_names == ["LLM_GLM_API_KEY"]
    assert store.write_count == 0


@pytest.mark.asyncio
async def test_empty_source_is_a_failing_run() -> None:
    store = _RecordingStore()
    handler = _handler(store, "# only a comment\n\n")

    result = await handler.handle(_request(execute=True))

    assert result.verdict is EnumSecretSeedVerdict.NO_KEYS
    assert result.success is False


@pytest.mark.asyncio
async def test_unreadable_source_is_reported_not_raised() -> None:
    def _reader(_path: str) -> str:
        raise FileNotFoundError("no such file")

    handler = HandlerSecretSeed(
        store_factory=lambda _request: _RecordingStore(),
        source_reader=_reader,
        auth_probe=list,
    )
    result = await handler.handle(_request(execute=True))

    assert result.verdict is EnumSecretSeedVerdict.SOURCE_UNREADABLE
    assert result.success is False


@pytest.mark.asyncio
async def test_name_listing_failure_is_store_unreachable_not_auth() -> None:
    """Identity and address are different pages for an operator."""
    store = _RecordingStore(raise_on_list=True)
    handler = _handler(store, f"A={_FAKE_VALUE}\n")

    result = await handler.handle(_request(execute=True))

    assert result.verdict is EnumSecretSeedVerdict.STORE_UNREACHABLE
    assert store.write_count == 0


@pytest.mark.parametrize(
    "field_name",
    ["infisical_host", "project_id", "environment_slug", "secret_path", "source_path"],
)
def test_every_addressing_field_is_required_with_no_default(field_name: str) -> None:
    """A guessed target seeds a real key somewhere nobody meant (Rule 8)."""
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "source_path": "/nonexistent/seed.env",
        "infisical_host": _HOST,
        "project_id": _PROJECT,
        "environment_slug": "dev",
        "secret_path": _PATH,
    }
    del fields[field_name]
    with pytest.raises(ValidationError):
        ModelSecretSeedRequest(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "infisical_host",
    [
        "infisical.invalid:8881",
        "ftp://infisical.invalid:8881",
        "https://user:pass@infisical.invalid:8881",
        "https://infisical.invalid:8881?token=redacted",
        "https://infisical.invalid:8881/#frag",
        "https://infisical.invalid:8881/api",
    ],
)
def test_unsafe_or_non_base_host_is_rejected(infisical_host: str) -> None:
    with pytest.raises(ValidationError):
        _request(infisical_host=infisical_host)


def test_https_host_is_normalised_by_trimming_trailing_slash() -> None:
    request = _request(infisical_host="https://infisical.invalid:8881/")

    assert request.infisical_host == "https://infisical.invalid:8881"


def test_relative_secret_path_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(secret_path="shared/llm/")


def test_parse_source_handles_export_prefix_quotes_and_comments() -> None:
    parsed = parse_source(
        "# comment\nexport A=\"quoted\"\n\nB='single'\nC=has=equals=inside\n"
    )
    assert parsed["A"].get_secret_value() == "quoted"
    assert parsed["B"].get_secret_value() == "single"
    assert parsed["C"].get_secret_value() == "has=equals=inside"


def test_parse_source_rejects_a_duplicate_name() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_source("A=1\nA=2\n")


def test_plain_http_host_is_accepted() -> None:
    """``http://`` is deliberately NOT rejected, and that is pinned here.

    All three Infisical instances this node exists to seed are plain http:
    the two ``.201`` lanes and the in-cluster service address, all three
    named in ``docs/runbooks/headless-secret-seeding.md``. An https-only
    rule would reject every address the node actually targets — including
    every invocation the runbook documents — which is a validator that
    reads as security while delivering none and that the operator must
    route around. Transport confidentiality on a LAN or in-cluster hop is a
    deployment property, not something this validator can assert.

    If TLS is ever terminated in front of those instances, tighten the rule
    and the runbook together, in that order, and delete this test then.
    """
    assert (
        _request(infisical_host="http://infisical.invalid:8881").infisical_host
        == "http://infisical.invalid:8881"
    )

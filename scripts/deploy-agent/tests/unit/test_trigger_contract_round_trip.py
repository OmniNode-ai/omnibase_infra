# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The trigger's envelope and the agent's contract cannot drift (OMN-16442).

These tests exist because the previous arrangement could not detect the failure
it was supposed to. ``deploy-agent-trigger.sh`` hand-wrote the command JSON in
an embedded Python snippet, and ``test_trigger_helper_signature.py``
*reproduced* that snippet in a local helper rather than calling it — so the
snippet and the model drifted apart (``reason`` published but forbidden,
``runtime_lane`` required but absent) while the test suite stayed green and
every operator command on the .201 dev lane was rejected.

The assertion that matters is the round trip: what the trigger publishes is
fed straight back through ``ModelRebuildRequested`` and through
``verify_command``, the same two checks the agent applies. Nothing here
re-implements either side.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest
from deploy_agent.auth import verify_command
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.lane_policy import (
    ENV_ALLOWED_LANES,
    resolve_default_runtime_lane_from_env,
)
from deploy_agent.tracking_ref import ENV_TRACKING_REF
from deploy_agent.trigger import (
    ENV_HMAC_SECRET,
    TriggerRefusedError,
    build_rebuild_command,
    command_to_signed_envelope,
    main,
    masked_envelope_json,
)

_SECRET = "test-trigger-secret-abc123"
_CID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")

_LANE_ENV = {
    ENV_TRACKING_REF: "dev",
    ENV_ALLOWED_LANES: "dev",
}


def _command(**overrides: object) -> ModelRebuildRequested:
    kwargs: dict = {
        "git_ref": "origin/dev",
        "runtime_lane": EnumRuntimeLane.DEV,
        "scope": Scope.RUNTIME,
        "build_source": BuildSource.RELEASE,
        "requested_by": "operator-manual",
        "correlation_id": _CID,
        "services": [],
    }
    kwargs.update(overrides)
    return build_rebuild_command(**kwargs)  # type: ignore[arg-type]


# ── the round trip ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_published_envelope_validates_as_the_agent_contract() -> None:
    """The exact bytes the trigger publishes reconstruct the command model.

    This is the assertion the old shell snippet could never satisfy: it emitted
    `reason` (extra_forbidden) and omitted `runtime_lane` (required).
    """
    envelope = command_to_signed_envelope(_command(), _SECRET)

    # Round-trip through the wire form the producer actually serialises.
    on_the_wire = json.loads(json.dumps(envelope, separators=(",", ":")))
    body = {k: v for k, v in on_the_wire.items() if k != "_signature"}

    reconstructed = ModelRebuildRequested.model_validate(body)
    assert reconstructed == _command()


@pytest.mark.unit
def test_envelope_carries_every_contract_field_and_no_others() -> None:
    """Field set is the model's, so adding or removing one moves both sides at once."""
    envelope = command_to_signed_envelope(_command(), _SECRET)
    published = set(envelope) - {"_signature"}
    assert published == set(ModelRebuildRequested.model_fields)


@pytest.mark.unit
def test_runtime_lane_is_published_and_reason_is_not() -> None:
    """The two specific fields the drift turned on, asserted by name."""
    envelope = command_to_signed_envelope(_command(), _SECRET)
    assert envelope["runtime_lane"] == EnumRuntimeLane.DEV.value
    assert "reason" not in envelope


@pytest.mark.unit
def test_signature_is_accepted_by_the_agents_verifier() -> None:
    envelope = command_to_signed_envelope(_command(), _SECRET)
    with patch.dict("os.environ", {ENV_HMAC_SECRET: _SECRET}):
        assert verify_command(envelope) is True


@pytest.mark.unit
def test_tampering_after_signing_is_rejected() -> None:
    envelope = command_to_signed_envelope(_command(), _SECRET)
    envelope["git_ref"] = "origin/evil-branch"
    with patch.dict("os.environ", {ENV_HMAC_SECRET: _SECRET}):
        assert verify_command(envelope) is False


@pytest.mark.unit
def test_unsigned_publish_is_refused_rather_than_silently_dropped() -> None:
    with pytest.raises(TriggerRefusedError, match=ENV_HMAC_SECRET):
        command_to_signed_envelope(_command(), "")


@pytest.mark.unit
def test_masked_envelope_never_prints_the_whole_signature() -> None:
    envelope = command_to_signed_envelope(_command(), _SECRET)
    rendered = masked_envelope_json(envelope)
    assert envelope["_signature"] not in rendered
    assert "<masked>" in rendered


# ── declared defaults, never literals ────────────────────────────────────────


@pytest.mark.unit
def test_git_ref_defaults_from_the_declared_tracking_ref() -> None:
    with patch.dict("os.environ", _LANE_ENV, clear=False):
        cmd = _command(git_ref=None)
    assert cmd.git_ref == "origin/dev"


@pytest.mark.unit
def test_runtime_lane_defaults_from_a_single_lane_fence() -> None:
    with patch.dict("os.environ", {ENV_ALLOWED_LANES: "dev"}, clear=False):
        assert resolve_default_runtime_lane_from_env() is EnumRuntimeLane.DEV


@pytest.mark.unit
def test_runtime_lane_defaults_from_the_tracking_ref_when_the_fence_is_wide() -> None:
    env = {ENV_ALLOWED_LANES: "dev,stability-test", ENV_TRACKING_REF: "dev"}
    with patch.dict("os.environ", env, clear=False):
        assert resolve_default_runtime_lane_from_env() is EnumRuntimeLane.DEV


@pytest.mark.unit
def test_runtime_lane_refuses_rather_than_guessing() -> None:
    env = {ENV_ALLOWED_LANES: "dev,prod", ENV_TRACKING_REF: "feature/x"}
    with patch.dict("os.environ", env, clear=False):
        with pytest.raises(RuntimeError, match="--runtime-lane"):
            resolve_default_runtime_lane_from_env()


@pytest.mark.unit
def test_contract_validation_happens_before_publish() -> None:
    """A prod command with no digest is refused here, not in a journal on the host."""
    with pytest.raises(ValueError, match="image_digest"):
        _command(runtime_lane=EnumRuntimeLane.PROD)


@pytest.mark.unit
def test_service_outside_scope_is_refused_before_publish() -> None:
    with pytest.raises(ValueError, match="not in scope"):
        _command(scope=Scope.CORE, services=["omninode-runtime"])


# ── CLI surface ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_dry_run_publishes_nothing_and_prints_the_lane(capsys) -> None:  # type: ignore[no-untyped-def]
    env = {**_LANE_ENV, ENV_HMAC_SECRET: _SECRET}
    with patch.dict("os.environ", env, clear=False):
        rc = main(["--git-ref", "origin/dev", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "runtime_lane:   dev" in out
    assert "dry-run" in out


@pytest.mark.unit
def test_reason_is_an_audit_note_and_never_reaches_the_wire(capsys) -> None:  # type: ignore[no-untyped-def]
    env = {**_LANE_ENV, ENV_HMAC_SECRET: _SECRET}
    with patch.dict("os.environ", env, clear=False):
        rc = main(
            [
                "--git-ref",
                "origin/dev",
                "--reason",
                "manual integration pass",
                "--dry-run",
            ]
        )
    out = capsys.readouterr().out
    assert rc == 0
    assert "reason:         manual integration pass" in out
    assert "local audit only, not in envelope" in out
    assert '"reason"' not in out


@pytest.mark.unit
def test_missing_secret_exits_nonzero_with_a_named_refusal(capsys) -> None:  # type: ignore[no-untyped-def]
    env = {**_LANE_ENV, ENV_HMAC_SECRET: ""}
    with patch.dict("os.environ", env, clear=False):
        rc = main(["--git-ref", "origin/dev", "--dry-run"])
    assert rc == 1
    assert ENV_HMAC_SECRET in capsys.readouterr().err


# ── transport is the agent's, not a second copy ──────────────────────────────


@pytest.mark.unit
def test_trigger_resolves_the_same_transport_the_agent_starts_with() -> None:
    """The prefixed lane credential names resolve, which the old script could not.

    The previous script read unprefixed KAFKA_SASL_USERNAME / KAFKA_SASL_PASSWORD
    while the .201 lane env file declares the dev SCRAM principal under DEV_-
    prefixed names, so its SASL block was skipped and it bootstrapped in the
    clear against a broker that requires SASL. There is one loader now, so this
    asserts against the module the agent itself calls.
    """
    from deploy_agent.kafka_config import load_deploy_agent_kafka_config_from_env

    env = {
        "KAFKA_BOOTSTRAP_SERVERS": "broker.invalid:19092",
        "KAFKA_SECURITY_PROTOCOL": "SASL_PLAINTEXT",
        "KAFKA_SASL_MECHANISM": "SCRAM-SHA-256",
        "KAFKA_SASL_ENV_PREFIX": "DEV_",
        "DEV_KAFKA_SASL_USERNAME": "lane-principal",
        "DEV_KAFKA_SASL_PASSWORD": "lane-secret",
    }
    with patch.dict("os.environ", env, clear=False):
        config = load_deploy_agent_kafka_config_from_env()

    assert config.security_protocol == "SASL_PLAINTEXT"
    assert config.sasl_mechanism == "SCRAM-SHA-256"
    kwargs = config.producer_kwargs()
    assert kwargs["sasl_plain_username"] == "lane-principal"
    # The two names the old script hardcoded, asserted as NOT chosen.
    assert kwargs["security_protocol"] != "SASL_SSL"
    assert kwargs["sasl_mechanism"] != "PLAIN"


@pytest.mark.unit
def test_publish_declares_no_compression() -> None:
    """rpk's snappy default is what crash-looped the agent; this producer says so."""
    from unittest.mock import Mock

    from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
    from deploy_agent.trigger import publish_signed_command

    config = ModelDeployAgentKafkaConfig(
        bootstrap_servers="broker.invalid:19092",
        security_protocol="PLAINTEXT",
    )
    envelope = command_to_signed_envelope(_command(), _SECRET)
    producer = Mock()
    with patch("kafka.KafkaProducer", return_value=producer) as factory:
        publish_signed_command(envelope, config)

    assert factory.call_args.kwargs["compression_type"] is None
    producer.send.assert_called_once()
    producer.close.assert_called_once()

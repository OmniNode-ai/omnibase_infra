# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wiring-report ratchet: failed, duplicate, quarantined and unproduced counts only fall (OMN-26335)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_wiring_report_counts import (
    HandlerWiringReportCounts,
)
from omnibase_infra.handlers.handler_wiring_report_ratchet import (
    COUNT_FIELDS,
    HandlerWiringReportRatchet,
)
from omnibase_infra.models.wiring_ratchet import (
    ModelWiringCounts,
    ModelWiringReportRatchetRequest,
)
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_from_paths,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = Path(
    "src/omnibase_infra/nodes/node_wiring_report_ratchet_compute/contract.yaml"
)
HANDLER_MODULE = "omnibase_infra.handlers.handler_wiring_report_ratchet"


def _ratchet_values(text: str) -> dict[str, dict[str, int]]:
    return yaml.safe_load(text)["metadata"]["wiring_ratchet"]


def _contract_baseline() -> ModelWiringCounts:
    values = _ratchet_values((REPO_ROOT / CONTRACT).read_text(encoding="utf-8"))
    return ModelWiringCounts(**values["baseline"])


def _base_baseline(base: str) -> ModelWiringCounts | None:
    """The baseline the contract held at ``base``, or None when it did not exist.

    A ``base`` named in the environment that does not resolve fails the check:
    a raised baseline must never pass because the base was not fetched.
    """
    resolves = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "--verify",
                "--quiet",
                f"{base}^{{commit}}",
            ],
            env=scrub_git_location_env(),
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )
    if not resolves:
        assert not os.environ.get("WIRING_RATCHET_BASE"), (
            f"base {base} does not resolve"
        )
        return None
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{base}:{CONTRACT}"],
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return ModelWiringCounts(**_ratchet_values(result.stdout)["baseline"])


def _with(counts: ModelWiringCounts, **changes: int) -> ModelWiringCounts:
    return counts.model_copy(update=changes)


def _check(
    observed: ModelWiringCounts,
    baseline: ModelWiringCounts,
    base_baseline: ModelWiringCounts | None = None,
    scope: str = "test",
):
    return HandlerWiringReportRatchet().handle(
        ModelWiringReportRatchetRequest(
            observed=observed,
            baseline=baseline,
            base_baseline=base_baseline,
            scope=scope,
        )
    )


def _rules(report) -> set[str]:
    return {f.rule_id for f in report.findings}


def _failed(report) -> bool:
    return any(f.severity == "FAIL" for f in report.findings)


def _contract_yaml(name: str, subscribe: str) -> str:
    return (
        f"name: {name}\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "node_type: COMPUTE_GENERIC\n"
        "event_bus:\n"
        f"  subscribe_topics: [onex.evt.test.{subscribe}.v1]\n"
        "handler_routing:\n"
        "  routing_strategy: operation_match\n"
        "  handlers:\n"
        "    - operation: x.check\n"
        "      handler:\n"
        "        name: HandlerWiringReportRatchet\n"
        f"        module: {HANDLER_MODULE}\n"
    )


def _counts_of(tmp_path: Path, contracts: dict[str, str]) -> ModelWiringCounts:
    paths = []
    for directory, text in contracts.items():
        target = tmp_path / directory
        target.mkdir()
        (target / "contract.yaml").write_text(text, encoding="utf-8")
        paths.append(target / "contract.yaml")
    return HandlerWiringReportCounts().handle(discover_contracts_from_paths(paths))


@pytest.mark.parametrize("name", COUNT_FIELDS)
def test_wiring_ratchet_rises(name: str) -> None:
    """A count one above a baseline that was lowered by one turns the check red."""
    observed = ModelWiringCounts(
        failed=3, duplicates=3, quarantined=3, unproduced_subscriptions=3
    )
    held = observed.model_copy()
    assert not _failed(_check(observed, held))
    lowered = _with(held, **{name: getattr(held, name) - 1})
    report = _check(observed, lowered)
    assert _failed(report)
    assert "wiring-count-above-baseline" in _rules(report)


def test_wiring_ratchet_duplicate_registration(tmp_path: Path) -> None:
    """A pull request that adds a duplicate registration raises the count and fails."""
    one = {"a": _contract_yaml("node_a", "alpha")}
    held = _counts_of(tmp_path, one)
    assert held.duplicates == 0
    assert not _failed(_check(held, held))

    shared_topic = {**one, "b": _contract_yaml("node_b", "alpha")}
    (tmp_path / "t").mkdir()
    topic_dup = _counts_of(tmp_path / "t", shared_topic)
    assert topic_dup.duplicates == 1
    report = _check(topic_dup, held)
    assert _failed(report)
    assert "wiring-count-above-baseline" in _rules(report)

    same_name = {**one, "c": _contract_yaml("node_a", "gamma")}
    (tmp_path / "n").mkdir()
    name_dup = _counts_of(tmp_path / "n", same_name)
    assert name_dup.duplicates == 1
    assert _failed(_check(name_dup, held))


def test_wiring_ratchet_unproduced_and_missing_handler(tmp_path: Path) -> None:
    """A subscription nobody publishes and a handler that does not exist are counted."""
    text = _contract_yaml("node_a", "alpha").replace(
        "HandlerWiringReportRatchet", "HandlerThatIsNotThere"
    )
    counts = _counts_of(tmp_path, {"a": text})
    assert counts.failed == 1
    assert counts.unproduced_subscriptions == 1


def test_wiring_ratchet_lowering() -> None:
    """A raised baseline is refused; a fall passes and names the new baseline."""
    base = _contract_baseline()
    raised = _with(base, duplicates=base.duplicates + 1)
    refused = _check(base, raised, base_baseline=base)
    assert _failed(refused)
    assert "wiring-baseline-raised" in _rules(refused)

    if base.duplicates:
        fallen = _with(base, duplicates=base.duplicates - 1)
        report = _check(fallen, base, base_baseline=base)
        assert not _failed(report)
        messages = [f.message for f in report.findings]
        assert any(f"the new baseline is {fallen.duplicates}" in m for m in messages)
        assert not _failed(_check(fallen, fallen, base_baseline=base))


def test_wiring_ratchet_live_catalog() -> None:
    """The boot catalog read statically stays within the contract baseline."""
    observed = HandlerWiringReportCounts().handle(discover_contracts())
    base = os.environ.get("WIRING_RATCHET_BASE") or "origin/dev"
    report = _check(
        observed,
        _contract_baseline(),
        base_baseline=_base_baseline(base),
        scope="installed boot catalog",
    )
    assert not _failed(report), [f.message for f in report.findings]
    for finding in report.findings:
        print(finding.message)


@pytest.mark.live_contact("tests/ci/fixtures/wiring_report_2026_11_21.json")
def test_wiring_report_2026_11_21(recorded_response: dict[str, object]) -> None:
    """The recorded lab report holds at its recorded counts and fails when one rises."""
    recorded = recorded_response["report"]
    baseline_values = _ratchet_values((REPO_ROOT / CONTRACT).read_text("utf-8"))[
        "lab_report_2026_11_21"
    ]
    assert baseline_values == {
        "failed": recorded["failed"],
        "duplicates": recorded["duplicates"],
        "quarantined": recorded["quarantined"],
    }
    # The recorded report carries no unproduced-subscription count; compare the
    # other three and hold that one at zero on both sides.
    observed = ModelWiringCounts(**baseline_values, unproduced_subscriptions=0)
    assert not _failed(_check(observed, observed, scope="lab report 2026-11-21"))
    for name in ("failed", "duplicates", "quarantined"):
        rose = _with(observed, **{name: getattr(observed, name) + 1})
        report = _check(rose, observed, scope="lab report 2026-11-21")
        assert _failed(report), name
        assert "wiring-count-above-baseline" in _rules(report)

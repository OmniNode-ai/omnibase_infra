# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-25912: the ticket's own falsifier selectors run the window and announce cases.

The ticket's criteria name ``uv run pytest scripts/deploy-agent/tests/unit -q -k
recreate_window`` and ``-k recreate_announce``. The cases live with the handler
(``tests/unit/handlers``); this module collects the same functions under the
deploy agent's test directory so those two commands select them, instead of
``recreate_announce`` selecting no test there (pytest exit 5).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from deploy_agent import agent as agent_mod

HANDLER_TESTS = Path(__file__).resolve().parents[4] / "tests" / "unit" / "handlers"
# The two directories both own a ``tests`` package, so the handler modules are
# loaded by path under their own file names, which is what keeps them importable.
_SELECTED = {
    "test_recreate_announce_omn25912": (
        "test_recreate_announce_a_hold_names_the_next_recreate_once",
        "test_recreate_announce_a_recreate_with_no_announcement_is_refused",
        "test_recreate_announce_agent_refuses_an_unannounced_idle_converge",
        "test_recreate_announce_precedes_the_recreate",
    ),
    "test_recreate_window_omn25912": (
        "test_recreate_window_four_recorded_starts_are_held_to_the_window",
        "test_recreate_window_install_path_idle_converge_and_restore_wait",
        "test_recreate_window_no_recreate_off_the_hour_without_a_pinned_fix",
        "test_recreate_window_two_hour_simulation_with_a_merge_every_four_minutes",
        "test_recreate_window_values_come_from_the_policy_not_the_handler",
    ),
}


# The deploy agent's own venv pins a released omnibase_infra that predates the window
# modules, so the handler cases import them the way the agent does: by file path out of
# the checkout, under their dotted names, before the case modules are collected.
_CHECKOUT_MODULES = (
    "enums/enum_dev_recreate_verdict.py",
    "enums/enum_dev_recreate_reason.py",
    "models/model_dev_recreate_policy.py",
    "models/model_dev_recreate_request.py",
    "models/model_dev_recreate_decision.py",
    "models/model_dev_recreate_announcement.py",
    "handlers/dev_recreate_window/handler_dev_recreate_window.py",
)


def _load_checkout_modules() -> None:
    roll_events = agent_mod.load_dev_boot_guard(agent_mod.DEV_ROLL_EVENTS)
    for relative in _CHECKOUT_MODULES:
        roll_events.events._load_sibling(relative)


def _load(stem: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"omn25912_handler_cases_{stem}", HANDLER_TESTS / f"{stem}.py"
    )
    assert spec is not None and spec.loader is not None
    _load_checkout_modules()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for _stem, _names in _SELECTED.items():
    _module = _load(_stem)
    for _name in _names:
        globals()[_name] = getattr(_module, _name)

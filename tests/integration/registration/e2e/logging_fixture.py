# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reversible logger configuration used only by registration E2E tests."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def configured_e2e_logging() -> Generator[None, None, None]:
    """Apply E2E log levels without leaking package state to later tests."""
    e2e_logger = logging.getLogger("tests.integration.registration.e2e")
    infra_logger = logging.getLogger("omnibase_infra")
    prior_e2e_level = e2e_logger.level
    prior_e2e_propagate = e2e_logger.propagate
    prior_infra_level = infra_logger.level
    prior_infra_propagate = infra_logger.propagate
    added_handler: logging.Handler | None = None

    try:
        e2e_logger.setLevel(logging.DEBUG)
        infra_logger.setLevel(logging.INFO)
        if not any(
            isinstance(handler, logging.StreamHandler)
            for handler in e2e_logger.handlers
        ):
            added_handler = logging.StreamHandler()
            added_handler.setLevel(logging.DEBUG)
            added_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            e2e_logger.addHandler(added_handler)
        yield
    finally:
        if added_handler is not None:
            if added_handler in e2e_logger.handlers:
                e2e_logger.removeHandler(added_handler)
            added_handler.close()
        e2e_logger.setLevel(prior_e2e_level)
        e2e_logger.propagate = prior_e2e_propagate
        infra_logger.setLevel(prior_infra_level)
        infra_logger.propagate = prior_infra_propagate

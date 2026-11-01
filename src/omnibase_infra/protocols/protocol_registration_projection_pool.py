# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Connection pool type accepted by the registration projection reader."""

import asyncpg

type RegistrationProjectionPool = asyncpg.Pool

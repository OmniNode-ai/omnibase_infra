# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import httpx


def fetch(url):
    return httpx.get(url)

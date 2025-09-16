#!/usr/bin/env python3

from enum import Enum


class ModelConsulKvStatus(Enum):
    """Enumeration of Consul KV operation status values."""

    SUCCESS = "success"
    NOT_FOUND = "not_found"
    FAILED = "failed"

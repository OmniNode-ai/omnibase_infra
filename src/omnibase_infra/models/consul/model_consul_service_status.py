#!/usr/bin/env python3

from enum import Enum
from pydantic import BaseModel


class ModelConsulServiceStatus(Enum):
    """Enumeration of Consul service status values."""
    
    PASSING = "passing"
    WARNING = "warning" 
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
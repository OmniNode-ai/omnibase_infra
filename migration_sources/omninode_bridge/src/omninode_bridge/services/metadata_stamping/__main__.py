#!/usr/bin/env python3
"""
Entry point for running the MetadataStampingService module directly.

This allows the service to be run as:
    python -m omninode_bridge.services.metadata_stamping

Or from Docker:
    python -m src.omninode_bridge.services.metadata_stamping
"""

if __name__ == "__main__":
    from .main import main

    main()

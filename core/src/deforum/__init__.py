"""
Deforum Core - CLI and Configuration

This package provides the core CLI interface and configuration
for the Deforum ecosystem. Animation logic has been moved to deforum_flux.
"""

from .config import Config
from .core import exceptions, logging_config

__version__ = "0.2.0"
__all__ = ["Config", "exceptions", "logging_config"]

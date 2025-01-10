# tests/conftest.py
import pytest

pytest_plugins = ("pytest_asyncio",)

def pytest_configure(config):
    """Set default values for asyncio fixtures."""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as requiring asyncio"
    )
import pytest

from voicebridge.providers.registry import load_builtin_providers


@pytest.fixture(scope="session", autouse=True)
def _providers():
    load_builtin_providers()


@pytest.fixture
def anyio_backend():
    return "asyncio"

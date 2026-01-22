"""Pytest configuration and custom CLI options."""

import pytest

from tests.config import DEFAULT_BATCH, DEFAULT_LOGN


def pytest_addoption(parser):
    """Register local CLI flags."""
    parser.addoption(
        "--logn",
        type=int,
        default=DEFAULT_LOGN,
        help=f"log2(N) for tests (default: {DEFAULT_LOGN})",
    )
    parser.addoption(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size for tests (default: {DEFAULT_BATCH})",
    )


@pytest.fixture(scope="session")
def logn(request):
    """Return log2(N) for tests."""
    return request.config.getoption("--logn")


@pytest.fixture(scope="session")
def batch(request):
    """Return batch size for tests."""
    return request.config.getoption("--batch")


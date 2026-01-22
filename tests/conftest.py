"""Pytest configuration and custom CLI options."""


def pytest_addoption(parser):
    """Register local CLI flags."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run fast test suite (subset of sizes and batches)",
    )

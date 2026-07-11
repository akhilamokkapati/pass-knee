"""
conftest.py
Shared pytest fixtures for the PASS knee suite.

The HuGaDB source runs Madgwick fusion at construction, which is the slow part
of the suite. A session-scoped fixture builds it once and shares the (immutable,
cached) source across every test that needs real data, keeping the suite fast.
Tests skip cleanly if the HuGaDB files are not present.
"""

from pathlib import Path

import pytest

from sources.hugadb import HuGaDBSource

HUGADB_FILE = Path(__file__).parent / "hugadb" / "HuGaDB_v2_various_01_00.csv"


@pytest.fixture(scope="session")
def hugadb_source():
    """One Madgwick-fused HuGaDB source, built once per test session."""
    if not HUGADB_FILE.exists():
        pytest.skip(f"HuGaDB data not present: {HUGADB_FILE}")
    return HuGaDBSource(HUGADB_FILE)

"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest

from database.db_manager import get_session, seed_all
from swarm.state import create_initial_state


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure database tables and initial warehouse/fleet seeds exist."""
    seed_all()
    yield


@pytest.fixture
def db_session():
    """Provide a database session for a test and ensure proper closure."""
    session = get_session()
    yield session
    session.close()


@pytest.fixture
def sample_emergency_request():
    """Sample emergency relief request fixture."""
    return {
        "destination_name": "Chennai Coastal Emergency Zone",
        "destination_lat": 13.0827,
        "destination_lon": 80.2707,
        "requested_items": {
            "medical_kits": 25,
            "water_purification": 20,
            "food_rations": 50,
            "tents_shelters": 10,
        },
    }


@pytest.fixture
def fresh_swarm_state(sample_emergency_request):
    """Provide a fresh initial SwarmState."""
    return create_initial_state(
        destination_name=sample_emergency_request["destination_name"],
        destination_lat=sample_emergency_request["destination_lat"],
        destination_lon=sample_emergency_request["destination_lon"],
        requested_items=sample_emergency_request["requested_items"],
    )


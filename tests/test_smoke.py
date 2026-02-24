from crowd_navigation.main import generate_pedestrians


def test_generate_pedestrians_count() -> None:
    assert len(generate_pedestrians(5)) == 5


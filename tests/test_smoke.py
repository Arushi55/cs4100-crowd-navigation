from main import generate_pedestrians
import numpy as np
from environment.scenarios import build_scenario, load_scenario_templates


def test_generate_pedestrians_count() -> None:
    templates = load_scenario_templates()
    rng = np.random.default_rng(123)
    scenario = build_scenario(templates["home"], rng, randomize_world=False)
    assert len(generate_pedestrians(scenario, rng, 5)) == 5


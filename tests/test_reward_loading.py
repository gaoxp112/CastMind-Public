"""Exercise the actual VeRL callback loader, including dataclass module registration."""

from verl.utils.import_utils import load_extern_object
from castmind.rewards import compute_score
from castmind.train import compose_config


def test_resolved_reward_loads_through_verl():
    config = compose_config("grpo", "model", "train.parquet", "out", "val.parquet")
    loaded = load_extern_object(config.reward.custom_reward_function.path, config.reward.custom_reward_function.name)
    assert loaded is compute_score

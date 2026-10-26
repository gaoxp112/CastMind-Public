"""Run the native VeRL GRPO trainer with a resolved launch configuration."""

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="grpo", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.load(config.launch_config))


if __name__ == "__main__":
    main()

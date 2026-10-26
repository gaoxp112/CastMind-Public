"""VeRL SFT with the paper's deterministic epoch shuffle seed."""

import hydra
from omegaconf import OmegaConf
from verl.trainer.sft_trainer import SFTTrainer


class CastMindSFTTrainer(SFTTrainer):
    def _build_dataloader(self):
        super()._build_dataloader()
        # Upstream DistributedSampler otherwise defaults to seed 0.
        self.train_sampler.seed = int(self.config.trainer.seed)


@hydra.main(config_path="configs", config_name="sft", version_base=None)
def main(config):
    from torch.distributed import destroy_process_group
    from verl.utils.distributed import initialize_global_process_group

    resolved = OmegaConf.load(config.launch_config)
    initialize_global_process_group()
    try:
        CastMindSFTTrainer(config=resolved).fit()
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()

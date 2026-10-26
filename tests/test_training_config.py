"""Ensure distributed launch defaults preserve the paper's experiment settings."""

from castmind.train import compose_config, launch_command


def test_sft_paper_settings():
    c = compose_config("sft", "base", "/tmp/sft.parquet", "/tmp/sft-out")
    assert (c.trainer.total_epochs, c.optim.lr, c.data.train_batch_size) == (3, 2e-5, 32)
    assert (c.trainer.n_gpus_per_node, c.trainer.seed) == (4, 42)
    assert c.data.truncation == "error" and c.data.pad_mode == "no_padding"
    assert "hf_model" in c.checkpoint.save_contents
    assert "--nproc_per_node=4" in launch_command("sft", "/tmp/config with spaces.yaml")


def test_grpo_paper_settings_and_five_term_reward():
    c = compose_config("grpo", "sft-hf", "/tmp/rl.parquet", "/tmp/rl-out", "/tmp/val.parquet")
    a = c.actor_rollout_ref
    assert (c.trainer.total_epochs, a.actor.optim.lr, c.data.train_batch_size) == (2, 1e-6, 16)
    assert c.algorithm.adv_estimator == "grpo"
    assert a.actor.clip_ratio == a.actor.clip_ratio_low == a.actor.clip_ratio_high == .2
    assert a.actor.kl_loss_coef == .001 and a.actor.use_kl_loss
    assert (a.rollout.n, a.rollout.temperature, c.data.max_response_length) == (8, 1, 2048)
    assert (c.trainer.n_gpus_per_node, c.data.seed) == (4, 42)
    assert a.rollout.engine_kwargs.vllm.seed == 42
    assert c.data.apply_chat_template_kwargs.enable_thinking
    assert not c.data.filter_overlong_prompts and c.data.truncation == "error"
    assert c.reward.custom_reward_function.path.endswith("castmind/rewards.py")
    assert c.reward.custom_reward_function.name == "compute_score"
    assert c.trainer.save_freq > 0

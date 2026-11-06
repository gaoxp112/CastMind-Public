"""Small launch/preflight interface for the two paper training stages."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


PACKAGE_DIR = Path(__file__).resolve().parent


def compose_config(stage, model, train_file, output_dir, val_file=None, *, profile="paper", gpus=4, max_steps=None):
    """Resolve our paper defaults against the installed VeRL configuration."""
    with initialize_config_dir(config_dir=str(PACKAGE_DIR / "configs"), version_base=None):
        config = compose(config_name=stage)
    OmegaConf.set_struct(config, False)
    config.data.train_files = str(Path(train_file).expanduser().resolve())
    config.data.custom_cls.path = str(PACKAGE_DIR / "datasets.py")
    config.trainer.default_local_dir = str(Path(output_dir).expanduser().resolve())
    config.data.val_files = str(Path(val_file).expanduser().resolve()) if val_file else None
    if stage == "sft":
        config.model.path = model
    else:
        config.actor_rollout_ref.model.path = model
        # Import as a registered package: VeRL's anonymous file loader does not
        # register modules, which breaks dataclasses with postponed annotations.
        config.reward.custom_reward_function.path = "pkg://castmind.rewards"
    if profile not in {"paper", "price-example"}:
        raise ValueError("unknown training profile")
    if gpus < 1:
        raise ValueError("gpus must be positive")
    if profile == "paper" and (gpus != 4 or max_steps is not None):
        raise ValueError("paper profile preserves four GPUs and the paper epoch budget")
    config.release_profile = profile
    if profile == "price-example":
        config.trainer.n_gpus_per_node = gpus
        batch_size = ((4 + gpus - 1) // gpus) * gpus
        config.data.train_batch_size = batch_size
        if stage == "sft":
            # Avoid packing multiple long Price windows into one GPU microbatch.
            config.data.use_dynamic_bsz = False
        else:
            config.actor_rollout_ref.actor.ppo_mini_batch_size = batch_size
            # One small-model GPU example must leave room for actor/reference.
            config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.35
            config.actor_rollout_ref.rollout.max_num_seqs = 32
            config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
            config.ray_kwargs.ray_init.num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
            config.reward.num_workers = 1
            config.actor_rollout_ref.rollout.agent.num_workers = 1
        if max_steps is not None:
            if max_steps < 1:
                raise ValueError("max_steps must be positive")
            config.trainer.total_training_steps = max_steps
    OmegaConf.resolve(config)
    return config


def _validate_file(stage, filename, tokenizer, data_config, training, profile="paper"):
    from collections import Counter
    import pandas as pd

    from castmind.datasets import PreparedRLHFDataset, PreparedSFTDataset, _plain
    from castmind.rewards import validate_metadata

    path = Path(filename).expanduser()
    if not path.is_file() or path.suffix != ".parquet":
        raise ValueError(f"supply an existing prepared .parquet file: {path}")
    rows = [_plain(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if not rows or (training and profile == "paper" and len(rows) != 2000):
        raise ValueError("training requires exactly 2000 prepared examples; validation must be nonempty")
    domains = Counter()
    has_teachers = False
    for index, row in enumerate(rows):
        source = row.get("data_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"row {index}: data_source must identify its domain")
        domains[source] += 1
        if stage == "grpo":
            try:
                has_teachers |= validate_metadata(row["reward_model"]["ground_truth"], row["extra_info"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid RL reward metadata at row {index}: {error}") from error
    if training and profile == "paper" and len(domains) != 8:
        raise ValueError("the paper mixture must contain all eight prepared domains")
    if profile == "price-example" and set(domains) != {"Price"}:
        raise ValueError("price-example profile requires only the Price domain")
    dataset_cls = PreparedSFTDataset if stage == "sft" else PreparedRLHFDataset
    dataset = dataset_cls(str(path), tokenizer, data_config)
    for index in range(len(dataset)):
        dataset[index]
    return {"rows": len(rows), "domains": dict(sorted(domains.items()))}, has_teachers


def preflight(stage, config):
    """Fail before distributed startup on malformed inputs or token boundaries."""
    from transformers import AutoTokenizer
    from castmind.rewards import _get_encoder

    for filename in (config.data.train_files, config.data.val_files):
        if filename and (not Path(filename).is_file() or Path(filename).suffix != ".parquet"):
            raise ValueError(f"supply an existing prepared .parquet file: {filename}")
    model = config.model.path if stage == "sft" else config.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    profile = config.get("release_profile", "paper")
    report, teachers = _validate_file(stage, config.data.train_files, tokenizer, config.data, True, profile)
    if report["rows"] < config.data.train_batch_size:
        raise ValueError("training file has fewer rows than train_batch_size")
    if config.data.val_files:
        validation, validation_teachers = _validate_file(
            stage, config.data.val_files, tokenizer, config.data, False, profile,
        )
        report["validation"] = validation
        teachers |= validation_teachers
    if teachers:
        _get_encoder()  # Missing dependencies or BGE weights must stop the run.
    report["stage"] = stage
    report["qualified_teachers_present"] = teachers
    report["seed"] = 42
    return report


def launch_command(stage, config_path, gpus=4):
    config_dir = str(PACKAGE_DIR / "configs")
    args = ["--config-path", config_dir, "--config-name", stage]
    # The resolved config includes environment-specific paths without shell interpolation.
    args += [f"+launch_config={json.dumps(str(config_path))}"]
    if stage == "sft":
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}", "-m", "castmind.sft", *args]
    return [sys.executable, "-m", "castmind.grpo", *args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sft", "grpo"))
    parser.add_argument("--model", required=True, help="base model for SFT; final SFT HF checkpoint for GRPO")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", help="prepared validation file (required for GRPO)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=["paper", "price-example"], default="paper")
    parser.add_argument("--gpus", type=int, help="price-example GPU count; paper defaults to four")
    parser.add_argument("--max-steps", type=int, help="optional bounded price-example smoke run")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="compose/print only; do not read models or input files")
    modes.add_argument("--check-only", action="store_true", help="validate prepared files and tokenizer without training")
    args = parser.parse_args()
    if args.stage == "grpo" and not args.val_file:
        parser.error("GRPO requires --val-file because upstream VeRL constructs a validation loader")
    gpus = args.gpus if args.gpus is not None else (1 if args.profile == "price-example" else 4)
    config = compose_config(args.stage, args.model, args.train_file, args.output_dir, args.val_file,
                            profile=args.profile, gpus=gpus, max_steps=args.max_steps)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    report = preflight(args.stage, config)
    print(json.dumps(report, indent=2))
    if args.check_only:
        return
    output_dir = Path(config.trainer.default_local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "launch.yaml"
    OmegaConf.save(config, config_path, resolve=True)
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    command = launch_command(args.stage, config_path, config.trainer.n_gpus_per_node)
    print(shlex.join(command), flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PACKAGE_DIR.parent) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()

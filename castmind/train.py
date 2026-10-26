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


def compose_config(stage, model, train_file, output_dir, val_file=None):
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
        config.reward.custom_reward_function.path = str(PACKAGE_DIR / "rewards.py")
    OmegaConf.resolve(config)
    return config


def _validate_file(stage, filename, tokenizer, data_config, training):
    from collections import Counter
    import pandas as pd

    from castmind.datasets import PreparedRLHFDataset, PreparedSFTDataset, _plain
    from castmind.rewards import validate_metadata

    path = Path(filename).expanduser()
    if not path.is_file() or path.suffix != ".parquet":
        raise ValueError(f"supply an existing prepared .parquet file: {path}")
    rows = [_plain(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if not rows or (training and len(rows) != 2000):
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
    if training and len(domains) != 8:
        raise ValueError("the paper mixture must contain all eight prepared domains")
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
    report, teachers = _validate_file(stage, config.data.train_files, tokenizer, config.data, True)
    if config.data.val_files:
        validation, validation_teachers = _validate_file(
            stage, config.data.val_files, tokenizer, config.data, False,
        )
        report["validation"] = validation
        teachers |= validation_teachers
    if teachers:
        _get_encoder()  # Missing dependencies or BGE weights must stop the run.
    report["stage"] = stage
    report["qualified_teachers_present"] = teachers
    report["seed"] = 42
    return report


def launch_command(stage, config_path):
    config_dir = str(PACKAGE_DIR / "configs")
    args = ["--config-path", config_dir, "--config-name", stage]
    # The resolved config includes environment-specific paths without shell interpolation.
    args += [f"+launch_config={json.dumps(str(config_path))}"]
    if stage == "sft":
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m", "castmind.sft", *args]
    return [sys.executable, "-m", "castmind.grpo", *args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sft", "grpo"))
    parser.add_argument("--model", required=True, help="base model for SFT; final SFT HF checkpoint for GRPO")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", help="prepared validation file (required for GRPO)")
    parser.add_argument("--output-dir", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="compose/print only; do not read models or input files")
    modes.add_argument("--check-only", action="store_true", help="validate prepared files and tokenizer without training")
    args = parser.parse_args()
    if args.stage == "grpo" and not args.val_file:
        parser.error("GRPO requires --val-file because upstream VeRL constructs a validation loader")
    config = compose_config(args.stage, args.model, args.train_file, args.output_dir, args.val_file)
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
    command = launch_command(args.stage, config_path)
    print(shlex.join(command), flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PACKAGE_DIR.parent) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()

"""Find the final checkpoint and export a VeRL actor for standalone inference."""

import argparse
from pathlib import Path
import subprocess
import sys


def latest_step(directory):
    paths = [path for path in Path(directory).glob("global_step_*")
             if path.is_dir() and path.name.rsplit("_", 1)[-1].isdigit()]
    if not paths:
        raise FileNotFoundError(f"no global_step_* checkpoints in {directory}")
    return max(paths, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def export(directory, stage):
    step = latest_step(directory)
    local = step / ("huggingface" if stage == "sft" else "actor/huggingface")
    if stage == "sft":
        if not (local / "config.json").is_file() or not list(local.glob("*.safetensors")):
            raise FileNotFoundError(f"SFT Hugging Face export missing from {local}")
    else:
        source = step / "actor"
        local = step / "actor_export"
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", str(source), "--target_dir", str(local)], check=True, stdout=sys.stderr)
    return local.resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--stage", choices=["sft", "grpo"], required=True)
    args = p.parse_args()
    print(export(args.directory, args.stage))


if __name__ == "__main__":
    main()

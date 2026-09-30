"""Training entry point: a thin CLI wrapper over ``src.runtime.Trainer``."""
import argparse
import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*torch.nn.utils.weight_norm.*is deprecated.*",
    category=FutureWarning,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runtime import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["flow_matching"],
        default="flow_matching",
        help="Model type to train: 'flow_matching'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML. If omitted, uses configs/train/<model>.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If specified, overrides config.seed and appends to wandb.run_name.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume training from a checkpoint directory (e.g. "
            "runs/<run>/epoch_0004 or runs/<run>/interrupt_epoch0002_step00001500). "
            "The directory must contain training_state.pt. Without --config the "
            "checkpoint's own config is used; without -r the run name is taken "
            "from the parent directory. Cannot be combined with --seed."
        ),
    )
    parser.add_argument(
        "--init-from",
        dest="init_from",
        type=str,
        default=None,
        help=(
            "Start a new run from the weights of a checkpoint directory (e.g. a "
            "phase-1 runs/<run>/epoch_0010 for a phase-2 fine-tune). Only the "
            "weights are loaded; optimizer, LR schedule and epoch/step counters "
            "start fresh under --config. Cannot be combined with --resume."
        ),
    )
    parser.add_argument(
        "-r",
        "--run",
        dest="run_name",
        type=str,
        default=None,
        help=(
            "Run name. Drives both the checkpoint output directory "
            "(runs/<run>) and wandb.run_name. If omitted, you are prompted "
            "to confirm the default derived from the config (auto-accepted in "
            "non-interactive shells)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    Trainer.from_args(parse_args(), project_root=PROJECT_ROOT).run()


if __name__ == "__main__":
    main()

"""Resolving and validating a ``--resume`` checkpoint for ``scripts/train.py``.

Resume granularity is a training *step*: the checkpoint records how many epochs
finished plus the global step, and the resumed run truncates its first epoch to
the steps that were left. Nothing is replayed, so the LR schedule and the
AdamW moments stay on the same timeline as an uninterrupted run.

Config keys that would silently invalidate that timeline are locked: the
``LambdaLR`` closure is rebuilt from the *current* config, so a changed
``train.epochs`` or ``optim`` block would jump the learning rate mid-run.
Dataloader-shaped keys (``num_workers``, ``prefetch_factor``, ``pin_memory``)
stay free on purpose — "the memory guard aborted, lower num_workers, resume" is
the whole point.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from src.utils.hf_checkpoint import is_hf_spec, resolve_model_dir

# Config subtrees that must match the checkpoint for the resumed run to be a
# continuation rather than a warm start.
LOCKED_CONFIG_KEYS = (
    "seed",
    "model",
    "optim",
    "scheduler",
    "encodec",
    "train.epochs",
    "train.ema_decay",
    "data.root",
    "data.split",
    "data.window_hop",
    "data.batch_size",
)


@dataclass
class ResumeState:
    """Everything the training loop needs to continue from a checkpoint."""
    ckpt_dir: str
    start_epoch: int
    global_step: int
    steps_per_epoch: int
    raw: Dict[str, Any]

    @property
    def first_epoch_offset(self) -> int:
        """Steps already done inside the (partial) epoch we restart into."""
        return self.global_step - self.start_epoch * self.steps_per_epoch

    @property
    def mid_epoch(self) -> bool:
        return self.first_epoch_offset > 0


def resolve_resume_dir(path: str, *, project_root: str) -> str:
    """Normalize a ``--resume`` / ``--init-from`` argument to an existing checkpoint directory."""
    if is_hf_spec(path):
        return resolve_model_dir(path)
    resolved = path if os.path.isabs(path) else os.path.join(project_root, path)
    resolved = os.path.normpath(resolved)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"not a directory: {resolved}")
    if not os.path.exists(os.path.join(resolved, "meta.json")):
        raise FileNotFoundError(f"meta.json not found in {resolved}")
    return resolved


def run_name_from_resume_dir(ckpt_dir: str) -> str:
    """The run name implied by ``runs/<run>/<epoch dir>`` — i.e. the parent."""
    return os.path.basename(os.path.dirname(os.path.normpath(ckpt_dir)))


def _select(cfg: Any, dotted: str) -> Any:
    node = OmegaConf.select(cfg, dotted)
    if isinstance(node, (DictConfig, type(None))) or OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True) if node is not None else None
    return node


def validate_resume_config(cfg: Any, meta: Dict[str, Any], *, model_type: str) -> None:
    """Hard-fail when the current config cannot continue the checkpoint's run."""
    saved_model_type = str(meta.get("model_type", ""))
    if saved_model_type and saved_model_type != model_type:
        raise ValueError(
            f"--resume: checkpoint was trained as '{saved_model_type}' but "
            f"--model is '{model_type}'."
        )

    saved_cfg = OmegaConf.create(meta.get("config") or {})
    mismatches = []
    for key in LOCKED_CONFIG_KEYS:
        now, before = _select(cfg, key), _select(saved_cfg, key)
        if now != before:
            mismatches.append(f"  {key}: checkpoint={before!r} -> current={now!r}")
    if mismatches:
        raise ValueError(
            "--resume: the following config values differ from the checkpoint "
            "and would break the resumed run (LR schedule, dataset split or "
            "model shape):\n" + "\n".join(mismatches)
            + "\n\nOmit --config to resume with the checkpoint's own config."
        )


def build_resume_state(
    ckpt_dir: str,
    state: Dict[str, Any],
    *,
    steps_per_epoch: int,
) -> ResumeState:
    """Validate the saved counters against the live dataloader and pack them."""
    saved_spe = int(state.get("steps_per_epoch", 0))
    if saved_spe and saved_spe != steps_per_epoch:
        raise ValueError(
            f"--resume: steps per epoch changed ({saved_spe} -> {steps_per_epoch}). "
            f"The dataset, its split or data.batch_size is not the same as when "
            f"the checkpoint was written; resuming would corrupt the schedule."
        )

    completed = int(state.get("completed_epochs", 0))
    global_step = int(state.get("global_step", completed * steps_per_epoch))
    offset = global_step - completed * steps_per_epoch
    if offset < 0 or offset >= max(1, steps_per_epoch):
        raise ValueError(
            f"--resume: inconsistent counters in training_state.pt "
            f"(completed_epochs={completed}, global_step={global_step}, "
            f"steps_per_epoch={steps_per_epoch})."
        )

    return ResumeState(
        ckpt_dir=ckpt_dir,
        start_epoch=completed,
        global_step=global_step,
        steps_per_epoch=steps_per_epoch,
        raw=state,
    )


def describe(resume: ResumeState, total_epochs: int, lr: Optional[float]) -> str:
    remaining = resume.steps_per_epoch - resume.first_epoch_offset
    head = (
        f"[resume] from {resume.ckpt_dir}: {resume.start_epoch}/{total_epochs} epochs done, "
        f"global_step={resume.global_step}"
    )
    if resume.mid_epoch:
        head += (
            f", continuing epoch {resume.start_epoch + 1} with {remaining}/"
            f"{resume.steps_per_epoch} steps left"
        )
    if lr is not None:
        head += f", lr={lr:.3e}"
    return head

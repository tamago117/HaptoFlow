"""``Trainer``: training entry point built on top of ``RuntimeEnv``.

scripts/train.py calls ``Trainer.from_args(args).run()``. The class holds the
resolved cfg + env + logger + viz; the training body lives in ``_run_training``.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import glob
import itertools
import os
import shutil
import sys
import time
from typing import Any, Callable, Optional

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.runtime.setup import RuntimeEnv, build_runtime_environment
from src.strategies import SpectralParams, StepContext
from src.utils.config.config_loader import load_config
from src.utils.evaluation.eval_loop import PreparedBatches, accumulate_eval_losses
from src.utils.hf_checkpoint import is_hf_spec
from src.utils.logging.wandb_logger import WandbLogger
from src.utils.training.checkpoint import (
    load_meta_json,
    load_safetensors_into_bundle,
    load_training_state,
    restore_training_state,
    save_epoch_checkpoint,
)
from src.utils.training.ema import ModelEMA, build_ema
from src.utils.training.interrupt import (
    InterruptHandler,
    TrainingInterrupted,
    reraise_after_cleanup,
)
from src.utils.training.memory_guard import (
    MEMORY_ABORT_EXIT_CODE,
    MemoryGuard,
    MemoryLimitExceeded,
    available_gb,
    build_memory_guard,
    describe_children,
)
from src.utils.training.resume import (
    ResumeState,
    build_resume_state,
    describe as describe_resume,
    resolve_resume_dir,
    run_name_from_resume_dir,
    validate_resume_config,
)
from src.utils.visualization import (
    VizConfig,
    configure_visualization_fonts,
    create_stft_and_spectrogram_images,
    create_waveform_images,
)


def _resolve_run_name(cli_value: Any, cfg: Any, model_type: str) -> str:
    """Pick a run name from CLI arg, or prompt for confirmation of the default.

    Non-interactive shells (no TTY on stdin) auto-accept the default so
    batch scripts and CI keep working.
    """
    if cli_value:
        return str(cli_value)

    default = None
    if hasattr(cfg, "wandb") and getattr(cfg.wandb, "run_name", None) is not None:
        default = str(cfg.wandb.run_name)
    if not default:
        default = model_type

    if not sys.stdin.isatty():
        print(f"[run] no -r/--run given; using default '{default}'")
        return default

    try:
        resp = input(
            f"[run] Use default '{default}'? [Y/n, or type a different name]: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n[run] using default '{default}'")
        return default

    if resp == "" or resp.lower() in ("y", "yes"):
        return default
    if resp.lower() in ("n", "no"):
        try:
            new_name = input("[run] Enter run name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n[run] using default '{default}'")
            return default
        return new_name or default
    return resp


def _apply_run_name(cfg: Any, run_name: str) -> None:
    """Propagate the resolved run name to wandb.run_name and train.ckpt_dir."""
    if hasattr(cfg, "wandb"):
        cfg.wandb.run_name = run_name
    if hasattr(cfg, "train"):
        cfg.train.ckpt_dir = f"runs/{run_name}"


def _restore_for_resume(
    resume_dir: str, *, env: RuntimeEnv, cfg: Any, scaler: Any
) -> ResumeState:
    """Load weights + optimizer/scheduler/scaler/RNG from ``resume_dir``.

    Runs after ``build_runtime_environment`` (which seeds torch) and before the
    first ``iter(train_dl)``, so the restored RNG is what the sampler picks up.
    """
    weights = glob.glob(os.path.join(resume_dir, "*.safetensors"))
    if len(weights) != 1:
        raise FileNotFoundError(
            f"--resume: expected exactly one *.safetensors in {resume_dir}, found {len(weights)}"
        )
    load_safetensors_into_bundle(weights[0], env.bundle, env.device)

    state = load_training_state(resume_dir)
    resume = build_resume_state(resume_dir, state, steps_per_epoch=len(env.train_dl))
    restore_training_state(state, bundle=env.bundle, scaler=scaler, device=env.device)

    lr = env.bundle.optimizer.param_groups[0]["lr"] if env.bundle.optimizer.param_groups else None
    print(describe_resume(resume, int(cfg.train.epochs), lr))
    if resume.start_epoch >= int(cfg.train.epochs):
        # train.epochs is locked on resume: the LR schedule uses it as T_max,
        # so a finished run cannot be extended.
        raise ValueError(
            f"--resume: the checkpoint already completed all "
            f"{int(cfg.train.epochs)} epochs of its run — there is nothing to resume. "
            f"To train longer, start a new run with a larger train.epochs."
        )
    return resume


def _load_init_weights(init_dir: str, *, env: RuntimeEnv, model_type: str) -> None:
    """``--init-from``: load only the weights of ``init_dir`` into the fresh bundle."""
    meta = load_meta_json(init_dir)
    if meta.get("model_type") is not None and str(meta["model_type"]) != model_type:
        raise ValueError(
            f"--init-from: {init_dir} is a {meta['model_type']} checkpoint, not {model_type}"
        )
    weights = glob.glob(os.path.join(init_dir, "*.safetensors"))
    if len(weights) != 1:
        raise FileNotFoundError(
            f"--init-from: expected exactly one *.safetensors in {init_dir}, found {len(weights)}"
        )
    load_safetensors_into_bundle(weights[0], env.bundle, env.device)
    print(f"[init-from] loaded weights from {weights[0]}")


class Trainer:
    """Holds the resolved runtime state and runs the training loop."""

    def __init__(
        self,
        cfg: Any,
        model_type: str,
        env: RuntimeEnv,
        logger: WandbLogger,
        viz: VizConfig,
        guard: MemoryGuard | None = None,
        resume: ResumeState | None = None,
        scaler: Any = None,
        ema: ModelEMA | None = None,
    ) -> None:
        self.cfg = cfg
        self.model_type = model_type
        self.env = env
        self.logger = logger
        self.viz = viz
        self.guard = guard
        self.resume = resume
        self.scaler = scaler
        self.ema = ema

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, project_root: str) -> "Trainer":
        """Resolve cfg + env + logger from CLI args, ready for ``run()``."""
        model_type = args.model
        resume_dir = getattr(args, "resume", None)
        init_dir = getattr(args, "init_from", None)
        resume_meta: dict[str, Any] | None = None

        if resume_dir is not None and init_dir is not None:
            raise ValueError("--init-from and --resume are mutually exclusive")
        if init_dir is not None:
            init_dir = resolve_resume_dir(init_dir, project_root=project_root)

        if resume_dir is not None:
            if is_hf_spec(resume_dir):
                raise ValueError("--resume needs training_state.pt, which hf:// checkpoints do not ship; use --init-from")
            if args.seed is not None:
                raise ValueError(
                    "--seed cannot be combined with --resume: the checkpoint's "
                    "seed is part of the run being continued."
                )
            resume_dir = resolve_resume_dir(resume_dir, project_root=project_root)
            resume_meta = load_meta_json(resume_dir)

        if args.config is not None:
            cfg = load_config(args.config)
        elif resume_meta is not None:
            # No --config on a resume: continue with the config the checkpoint
            # was trained under, which is the only one guaranteed to match.
            cfg = OmegaConf.create(resume_meta["config"])
            print(f"[resume] no --config given; using the config stored in {resume_dir}/meta.json")
        else:
            cfg = load_config(f"configs/train/{model_type}.yaml")

        if args.seed is not None:
            cfg.seed = args.seed
        if init_dir is not None:
            # Recorded in the saved config so a phase-2 checkpoint names its source;
            # an hf:// spec is kept as-is instead of its local cache path
            source = args.init_from if is_hf_spec(args.init_from) else init_dir
            OmegaConf.update(cfg, "train.init_from", source, force_add=True)

        if resume_meta is not None:
            validate_resume_config(cfg, resume_meta, model_type=model_type)
            run_name = str(getattr(args, "run_name", None) or run_name_from_resume_dir(resume_dir))
        else:
            run_name = _resolve_run_name(getattr(args, "run_name", None), cfg, model_type)
            if args.seed is not None:
                run_name = f"{run_name}_{args.seed}"
        _apply_run_name(cfg, run_name)

        # Arm the memory guard before the environment is built so dataset /
        # dataloader construction is covered. Only the hard-kill tier can act
        # this early — there is no model to checkpoint yet — but that is the
        # tier that keeps the machine alive.
        guard = build_memory_guard(cfg)
        if guard is not None:
            guard.start()

        env = build_runtime_environment(cfg, model_type, project_root=project_root)

        viz = VizConfig.from_cfg(cfg)
        configure_visualization_fonts(
            family="Arial",
            weight="bold",
            axis_label_size=viz.font_axis_label_size,
            title_size=viz.font_title_size,
            legend_size=viz.font_legend_size,
            tick_size=viz.font_tick_size,
        )
        logger = WandbLogger(cfg)

        # The GradScaler is created here (not inside the training loop) so the
        # resume path can restore its state and the graceful-stop path can save it.
        scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp))

        if init_dir is not None:
            _load_init_weights(init_dir, env=env, model_type=model_type)

        resume: ResumeState | None = None
        if resume_dir is not None:
            resume = _restore_for_resume(
                resume_dir, env=env, cfg=cfg, scaler=scaler,
            )

        # Built after the weights are loaded: the shadow starts from them
        ema = build_ema(cfg, env.bundle)
        if resume is not None and ema is not None:
            if resume.raw.get("ema") is None:
                raise ValueError(f"--resume: {resume_dir} has no EMA state (saved without train.ema_decay)")
            ema.load_training_state(resume.raw["ema"])

        return cls(
            cfg, model_type, env, logger, viz,
            guard=guard, resume=resume, scaler=scaler, ema=ema,
        )

    def run(self) -> None:
        guard = self.guard
        progress: dict[str, int] = {
            "epoch": self.resume.start_epoch if self.resume else 0,
            "global_step": self.resume.global_step if self.resume else 0,
            "steps_per_epoch": 0,
        }
        # (kind, reason, signum) — filled in when the run stops early.
        stop: tuple[str, str, Optional[int]] | None = None
        try:
            with InterruptHandler() as interrupts:
                def stop_check() -> None:
                    """Safe-point check shared by the loop and the eval passes."""
                    interrupts.raise_if_requested()
                    if guard is not None:
                        guard.raise_if_tripped()

                try:
                    _run_training(
                        self.cfg, self.model_type, self.env, self.logger, self.viz,
                        stop_check=stop_check, progress=progress,
                        scaler=self.scaler, resume=self.resume, ema=self.ema,
                    )
                except MemoryLimitExceeded as exc:
                    # Keep only the message: holding the exception would keep
                    # ``_run_training``'s frame (and its dataloader workers)
                    # alive. The handler runs outside the except block so they
                    # are freed before the save.
                    stop = ("memory", str(exc), None)
                except TrainingInterrupted as exc:
                    stop = ("interrupt", str(exc), exc.signum)

            if stop is not None:
                _stop_gracefully(
                    stop, self.cfg, self.model_type, self.env, self.logger,
                    progress, self.scaler, self.ema,
                )
        finally:
            if guard is not None:
                guard.stop()


def _stop_gracefully(
    stop: tuple[str, str, Optional[int]],
    cfg: Any,
    model_type: str,
    env: RuntimeEnv,
    logger: WandbLogger,
    progress: dict,
    scaler: Any,
    ema: ModelEMA | None = None,
) -> None:
    """Shared early-exit path for a memory abort and for SIGINT/SIGTERM.

    Order matters — the safetensors save stages tensors in host memory, so the
    dataloader workers have to go first. The checkpoint written here carries
    ``training_state.pt``, so ``--resume <dir>`` continues from it.

    Exit convention: a memory abort exits with ``MEMORY_ABORT_EXIT_CODE``; an
    interrupt re-raises the original signal so callers see the usual 130/143
    rather than a plain failure (so a shell harness can tell "the user stopped
    this" from "this run failed").
    """
    kind, reason, signum = stop
    tag = "memory_guard" if kind == "memory" else "interrupt"
    print(f"\n[{tag}] STOPPING TRAINING: {reason}", flush=True)

    env.train_dl = None  # type: ignore[assignment]
    env.val_dl = None  # type: ignore[assignment]
    env.test_dl = None  # type: ignore[assignment]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # A non-zero child count means the dataloader workers are still resident.
    print(
        f"[{tag}] after dataloader teardown: "
        f"{describe_children()} child process(es), "
        f"{available_gb():.2f} GB available",
        flush=True,
    )

    epoch = int(progress.get("epoch", 0))
    step = int(progress.get("global_step", 0))
    steps_per_epoch = int(progress.get("steps_per_epoch", 0))
    save_enabled = (
        bool(getattr(getattr(cfg.train, "memory_guard", None), "save_checkpoint", True))
        if kind == "memory" else True
    )
    if save_enabled:
        try:
            prefix = "abort" if kind == "memory" else "interrupt"
            ckpt_path = save_epoch_checkpoint(
                epoch=epoch, ckpt_dir=str(cfg.train.ckpt_dir),
                bundle=env.bundle, cfg=cfg, model_type=model_type,
                full_dataset=env.full_ds, control_stats=env.control_stats,
                num_classes=env.num_classes,
                dir_name=f"{prefix}_epoch{epoch:04d}_step{step:08d}",
                global_step=step, steps_per_epoch=steps_per_epoch, scaler=scaler, ema=ema,
            )
            print(
                f"[{tag}] checkpoint written to {ckpt_path}\n"
                f"[{tag}] resume with: python scripts/train.py --model {model_type} "
                f"--resume {ckpt_path}",
                flush=True,
            )
        except Exception as save_exc:  # noqa: BLE001 - never mask the stop
            print(f"[{tag}] checkpoint FAILED: {save_exc}", flush=True)

    logger.finish()
    if kind == "interrupt" and signum is not None:
        reraise_after_cleanup(signum)
    sys.exit(MEMORY_ABORT_EXIT_CODE)


def _rollout_images(batch: dict, num_samples: int, full_ds, viz: VizConfig) -> list:
    """Phase-2 batches: generated chains (each segment from the previous generation)
    against the ground truth, after the zero / true start prev."""
    if "rollout" not in batch:
        return []
    n = min(int(num_samples), int(batch["rollout"].shape[0]))
    flat = lambda x: [x[i].reshape(1, -1) for i in range(n)]
    # Width grows with the chain (start prev + K segments); height as configured
    base_w, base_h = viz.figure_size_waveform or (6.0, 2.0)
    figure_size = (max(base_w, 1.5 * (int(batch["rollout"].shape[1]) + 1)), base_h)
    _, images = create_waveform_images(
        flat(batch["rollout"]), flat(batch["rollout_truth"]), flat(batch["rollout_start"]),
        batch["rollout_label_idx"][:n], full_ds, True, 1.0,
        force_means_raw=batch.get("rollout_force_mean_raw"), velocity_means_raw=batch.get("rollout_velocity_mean_raw"),
        ylim=viz.waveform_ylim,
        show_x_axis_label=viz.show_x_axis_label, show_y_axis_label=viz.show_y_axis_label,
        show_legend=viz.show_legend, show_title=viz.show_title,
        figure_size=figure_size, dpi=viz.figure_dpi,
        waveform_alpha=viz.line_alpha, prev_waveform_alpha=viz.line_alpha,
    )
    return images


def _run_training(
    cfg: Any,
    model_type: str,
    env: RuntimeEnv,
    logger: WandbLogger,
    viz: VizConfig,
    stop_check: Optional[Callable[[], None]] = None,
    progress: dict | None = None,
    scaler: Any = None,
    resume: ResumeState | None = None,
    ema: ModelEMA | None = None,
) -> None:
    """Body of the training entry point: epoch loop + validation + test eval + wandb."""
    if progress is None:
        progress = {}
    if stop_check is None:
        def stop_check() -> None:  # noqa: E306 - trivial no-op default
            return None
    device = env.device
    strategy = env.strategy
    bundle = env.bundle
    full_ds = env.full_ds
    train_dl, val_dl, test_dl = env.train_dl, env.val_dl, env.test_dl
    encodec_wrapper = env.encodec_wrapper
    control_stats = env.control_stats
    num_classes = env.num_classes


    model = bundle.model
    optimizer = bundle.optimizer
    scheduler = bundle.scheduler

    spectral = SpectralParams.from_cfg(cfg)

    os.makedirs(str(cfg.train.ckpt_dir), exist_ok=True)
    ckpt_dir = str(cfg.train.ckpt_dir)
    if resume is not None:
        print(f"[resume] keeping existing checkpoints under {ckpt_dir}")
    elif os.path.exists(ckpt_dir):
        for item in os.listdir(ckpt_dir):
            item_path = os.path.join(ckpt_dir, item)
            if os.path.isdir(item_path) and item.startswith("epoch_"):
                try:
                    shutil.rmtree(item_path)
                    print(f"Removed old epoch directory: {item_path}")
                except OSError as e:
                    print(f"Warning: Failed to remove {item_path}: {e}")

    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp))
    steps_per_epoch = len(train_dl)
    total_epochs = int(cfg.train.epochs)
    start_epoch = resume.start_epoch if resume is not None else 0
    global_step = resume.global_step if resume is not None else 0
    # Steps already done inside the epoch we restart into; the first resumed
    # epoch is truncated to what is left so the run keeps exactly
    # ``total_epochs * steps_per_epoch`` steps and the LR schedule stays aligned.
    first_epoch_offset = resume.first_epoch_offset if resume is not None else 0
    progress["steps_per_epoch"] = steps_per_epoch

    for epoch in range(start_epoch, total_epochs):
        strategy.set_train_mode(bundle)
        iter_start_time = time.time()

        epoch_offset = first_epoch_offset if epoch == start_epoch else 0
        epoch_steps = steps_per_epoch - epoch_offset
        epoch_iterable = (
            itertools.islice(train_dl, epoch_steps) if epoch_offset > 0 else train_dl
        )
        pbar = tqdm(
            epoch_iterable,
            total=epoch_steps,
            desc=f"Epoch {epoch + 1}/{total_epochs}"
                 + (f" (resumed, {epoch_steps} steps left)" if epoch_offset > 0 else ""),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for batch in pbar:
            progress["epoch"] = epoch
            progress["global_step"] = global_step
            stop_check()
            data_time_ms = (time.time() - iter_start_time) * 1000.0

            step_ctx = StepContext(
                cfg=cfg, device=device, full_dataset=full_ds,
                encodec_wrapper=encodec_wrapper, scaler=scaler,
            )
            metrics = strategy.train_step(batch, bundle, step_ctx)
            if ema is not None:
                ema.update()

            if scheduler is not None:
                scheduler.step()

            epoch_float = global_step / steps_per_epoch if steps_per_epoch > 0 else epoch

            logger.log(
                {
                    "train/loss": metrics["loss"],
                    "train/lr": metrics["lr"],
                    "train/t_mean": metrics.get("t_mean", 0.0),
                    "train/t_std": metrics.get("t_std", 0.0),
                    "train/v_pred_norm": metrics.get("v_pred_norm", 0.0),
                    "train/step_time_ms": metrics["step_time_ms"],
                    "train/data_time_ms": data_time_ms,
                    "train/gpu_mem_mb": metrics["gpu_mem_mb"],
                    "train/history_noise_prob": metrics.get("history_noise_prob", 0.0),
                    "train/self_forcing_prob": metrics.get("self_forcing_prob", 0.0),
                    "train/self_forcing_steps": metrics.get("self_forcing_steps", 0),
                    "epoch": epoch_float,
                },
                step=global_step,
            )

            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                lr=f"{metrics['lr']:.2e}",
                step_ms=f"{metrics['step_time_ms']:.0f}",
            )

            if int(cfg.train.print_interval) > 0 and (global_step % int(cfg.train.print_interval) == 0):
                pbar.write(
                    f"step {global_step} | epoch {epoch_float:.3f} | loss {metrics['loss']:.6f} | "
                    f"lr {metrics['lr']:.2e} | t(mean/std) {metrics.get('t_mean', 0.0):.3f}/{metrics.get('t_std', 0.0):.3f} | "
                    f"v_norm {metrics.get('v_pred_norm', 0.0):.3f} | "
                    f"{metrics['step_time_ms']:.1f} ms/step | data {data_time_ms:.1f} ms/step"
                )

            iter_start_time = time.time()
            global_step += 1

        pbar.close()
        progress["epoch"] = epoch + 1
        progress["global_step"] = global_step
        stop_check()

        val_interval = float(cfg.train.val_interval)
        epoch_float_at_end = epoch + 1
        if val_interval > 0:
            quotient = epoch_float_at_end / val_interval
            step_increment = 1.0 / max(1, steps_per_epoch) if steps_per_epoch > 0 else 0.01
            should_validate = (
                abs(quotient - round(quotient)) * val_interval < step_increment
                or (epoch + 1) % max(1, int(val_interval)) == 0
            )
        else:
            should_validate = False

        if should_validate:
            # Validate with the EMA weights; the raw ones keep training
            with ema.applied() if ema is not None else contextlib.nullcontext():
                val_step = global_step
                strategy.set_eval_mode(bundle)

                val_ctx = StepContext(cfg=cfg, device=device, full_dataset=full_ds, encodec_wrapper=encodec_wrapper)
                # Phase 2: validate under the training prevs (zero start, own generations)
                val_batches = PreparedBatches(strategy, val_dl, bundle, val_ctx, seed=int(cfg.seed))
                val_summary = accumulate_eval_losses(
                    strategy=strategy, bundle=bundle, dataloader=val_batches, ctx=val_ctx,
                    desc=f"Val loss [epoch {epoch + 1}/{total_epochs}]",
                    on_batch=stop_check,
                )

                val_metric_batches = int(getattr(cfg.train, "val_metric_batches", 0))
                val_metrics_ctx = StepContext(
                    cfg=cfg, device=device, full_dataset=full_ds, encodec_wrapper=encodec_wrapper,
                    extras={
                        "progress_desc": f"Val metrics [epoch {epoch + 1}/{total_epochs}]",
                        "max_batches": val_metric_batches if val_metric_batches > 0 else None,
                    },
                )
                snr_mean, rmse_mean, gfc_mean, freq_spectrum_rmse_mean = strategy.compute_metrics(
                    val_batches, bundle, val_metrics_ctx
                )
                val_self_forcing = strategy.self_forcing_metrics(full_ds, val_dl.dataset.indices, bundle, val_metrics_ctx)

                val_img_interval = float(getattr(cfg.train, "val_img_interval", cfg.train.val_interval))
                if val_img_interval > 0:
                    quotient = epoch_float_at_end / val_img_interval
                    step_increment = 1.0 / max(1, steps_per_epoch) if steps_per_epoch > 0 else 0.01
                    save_images = (
                        abs(quotient - round(quotient)) * val_img_interval < step_increment
                        or (epoch + 1) % max(1, int(val_img_interval)) == 0
                    )
                else:
                    save_images = False

                val_generate_time_ms = 0.0
                if save_images:
                    vb = val_batches.first()
                    vlabels = vb["label_idx"].to(device)
                    val_gen_ctx = StepContext(
                        cfg=cfg, device=device, full_dataset=full_ds, encodec_wrapper=encodec_wrapper,
                        extras={"control_stats": control_stats},
                    )
                    gen_waveforms, true_waveforms, prev_waveforms, val_generate_time_ms = strategy.generate_samples(
                        vb, bundle, val_gen_ctx,
                    )

                    waveform_images, waveform_images_prev = create_waveform_images(
                        gen_waveforms, true_waveforms, prev_waveforms, vlabels, full_ds,
                        viz.render_prev_waveform, viz.prev_waveform_ratio,
                        force_means_raw=vb.get("force_mean_raw"), velocity_means_raw=vb.get("velocity_mean_raw"),
                        ylim=viz.waveform_ylim,
                        show_x_axis_label=viz.show_x_axis_label, show_y_axis_label=viz.show_y_axis_label,
                        show_legend=viz.show_legend, show_title=viz.show_title,
                        figure_size=viz.figure_size_waveform, dpi=viz.figure_dpi,
                        waveform_alpha=viz.line_alpha, prev_waveform_alpha=viz.line_alpha,
                    )
                    stft_images_norm, stft_images, spec_images, mel_images = [], [], [], []
                    if viz.enable_stft_spectrum or viz.enable_spectrogram or viz.enable_mel_spectrogram:
                        stft_images_norm, stft_images, spec_images, mel_images = create_stft_and_spectrogram_images(
                            gen_waveforms, true_waveforms, vlabels, full_ds,
                            spectral.stft_n_fft, spectral.stft_hop_length, spectral.stft_win_length,
                            force_means_raw=vb.get("force_mean_raw"), velocity_means_raw=vb.get("velocity_mean_raw"),
                            enable_mel_spectrogram=viz.enable_mel_spectrogram,
                            mel_n_mels=spectral.mel_n_mels,
                            mel_f_min=spectral.mel_f_min, mel_f_max=spectral.mel_f_max,
                            show_x_axis_label=viz.show_x_axis_label, show_y_axis_label=viz.show_y_axis_label,
                            show_legend=viz.show_legend, show_title=viz.show_title,
                            figure_size_spectrum=viz.figure_size_spectrum,
                            figure_size_spectrogram=viz.figure_size_spectrogram,
                            dpi=viz.figure_dpi, spectrum_alpha=viz.line_alpha,
                        )

                    if len(waveform_images) > 0:
                        logger.log_images("val/waveforms", waveform_images, step=val_step)
                        print(f"Logged {len(waveform_images)} waveform images to wandb at epoch {epoch+1}")
                    else:
                        print(f"Warning: No waveform images generated at epoch {epoch+1}")
                    if viz.render_prev_waveform:
                        if len(waveform_images_prev) > 0:
                            logger.log_images("val/waveforms_prev", waveform_images_prev, step=val_step)
                        else:
                            print(f"Warning: No previous waveform images generated at epoch {epoch+1}")
                        rollout_images = _rollout_images(vb, int(cfg.eval.num_samples), full_ds, viz)
                        if rollout_images:
                            logger.log_images("val/waveforms_rollout", rollout_images, step=val_step)
                    if viz.enable_stft_spectrum and len(stft_images) > 0:
                        logger.log_images("val/stft_spectrum", stft_images, step=val_step)
                    if viz.enable_stft_spectrum and len(stft_images_norm) > 0:
                        logger.log_images("val/stft_spectrum_normarize", stft_images_norm, step=val_step)
                    if viz.enable_spectrogram and len(spec_images) > 0:
                        logger.log_images("val/spectrograms", spec_images, step=val_step)
                    if viz.enable_mel_spectrogram and len(mel_images) > 0:
                        logger.log_images("val/mel_spectrograms", mel_images, step=val_step)

                val_epoch_float = val_step / steps_per_epoch if steps_per_epoch > 0 else epoch
                logger.log(
                    {
                        "val/loss": val_summary.loss_mean,
                        "val/infer_time_ms_mean": val_summary.infer_time_mean_ms,
                        "val/generate_time_ms": val_generate_time_ms,
                        "val/snr_db": snr_mean,
                        "val/rmse": rmse_mean,
                        "val/gfc": gfc_mean,
                        "val/freq_spectrum_rmse": freq_spectrum_rmse_mean,
                        **{f"val/{k}": v for k, v in val_self_forcing.items()},
                        "epoch": val_epoch_float,
                    },
                    step=val_step,
                )

        ckpt_interval = int(getattr(cfg.train, "ckpt_interval", 1))
        if (epoch + 1) % ckpt_interval == 0 or (epoch + 1) == total_epochs:
            save_epoch_checkpoint(
                epoch=epoch + 1, ckpt_dir=str(cfg.train.ckpt_dir),
                bundle=bundle, cfg=cfg, model_type=model_type, full_dataset=full_ds,
                control_stats=control_stats, num_classes=num_classes,
                global_step=global_step, steps_per_epoch=steps_per_epoch, scaler=scaler, ema=ema,
            )

    # Free the persistent train/val workers and their prefetch buffers (several
    # GB) before test_dl spins up.
    del train_dl
    del val_dl
    env.train_dl = None  # type: ignore[assignment]
    env.val_dl = None  # type: ignore[assignment]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stop_check()

    # The trained weights are the EMA from here on
    if ema is not None:
        ema.copy_to_bundle()

    print("Running test evaluation...")
    strategy.set_eval_mode(bundle)
    test_ctx = StepContext(cfg=cfg, device=device, full_dataset=full_ds, encodec_wrapper=encodec_wrapper)
    test_batches = PreparedBatches(strategy, test_dl, bundle, test_ctx, seed=int(cfg.seed))
    test_summary = accumulate_eval_losses(
        strategy=strategy, bundle=bundle, dataloader=test_batches, ctx=test_ctx,
        desc="Test loss",
        on_batch=stop_check,
    )

    test_metrics_ctx = StepContext(
        cfg=cfg, device=device, full_dataset=full_ds, encodec_wrapper=encodec_wrapper,
        extras={"return_per_sample_lists": True, "progress_desc": "Test metrics"},
    )
    snr_mean, rmse_mean, gfc_mean, freq_spectrum_rmse_mean, \
        snr_list, rmse_list, gfc_list, freq_spectrum_rmse_list = strategy.compute_metrics(
            test_batches, bundle, test_metrics_ctx,
        )

    num_batches_for_generate = min(5, len(test_dl)) if (hasattr(test_dl, "__len__") and len(test_dl) > 0) else 5
    batches_for_gen = list(itertools.islice(iter(test_batches), num_batches_for_generate))
    test_generate_time_list = []
    gen_waveforms, true_waveforms, prev_waveforms = None, None, None
    tb_for_viz = None
    tlabels_viz = None

    for batch_idx, tb in enumerate(batches_for_gen):
        tlabels = tb["label_idx"].to(device)
        if batch_idx == 0:
            tb_for_viz = tb
            tlabels_viz = tlabels
        test_gen_ctx = StepContext(
            cfg=cfg, device=device, full_dataset=full_ds, encodec_wrapper=encodec_wrapper,
            extras={"control_stats": control_stats},
        )
        gw, tw, pw, gen_time_ms = strategy.generate_samples(tb, bundle, test_gen_ctx)
        test_generate_time_list.append(gen_time_ms)
        if batch_idx == 0:
            gen_waveforms, true_waveforms, prev_waveforms = gw, tw, pw

    tb = tb_for_viz if (batches_for_gen and tb_for_viz is not None) else None
    tlabels = tlabels_viz if tb is not None else None
    test_generate_time_ms = (
        sum(test_generate_time_list) / len(test_generate_time_list) if test_generate_time_list else 0.0
    )

    waveform_images, waveform_images_prev = [], []
    if gen_waveforms is not None and tb is not None:
        waveform_images, waveform_images_prev = create_waveform_images(
            gen_waveforms, true_waveforms, prev_waveforms, tlabels, full_ds,
            viz.render_prev_waveform, viz.prev_waveform_ratio,
            force_means_raw=tb.get("force_mean_raw"), velocity_means_raw=tb.get("velocity_mean_raw"),
            ylim=viz.waveform_ylim,
            show_x_axis_label=viz.show_x_axis_label, show_y_axis_label=viz.show_y_axis_label,
            show_legend=viz.show_legend, show_title=viz.show_title,
            figure_size=viz.figure_size_waveform, dpi=viz.figure_dpi,
            waveform_alpha=viz.line_alpha, prev_waveform_alpha=viz.line_alpha,
        )
    stft_images_norm, stft_images, spec_images, mel_images = [], [], [], []
    if (viz.enable_stft_spectrum or viz.enable_spectrogram or viz.enable_mel_spectrogram) \
            and gen_waveforms is not None and tb is not None:
        stft_images_norm, stft_images, spec_images, mel_images = create_stft_and_spectrogram_images(
            gen_waveforms, true_waveforms, tlabels, full_ds,
            spectral.stft_n_fft, spectral.stft_hop_length, spectral.stft_win_length,
            force_means_raw=tb.get("force_mean_raw"), velocity_means_raw=tb.get("velocity_mean_raw"),
            enable_mel_spectrogram=viz.enable_mel_spectrogram,
            mel_n_mels=spectral.mel_n_mels,
            mel_f_min=spectral.mel_f_min, mel_f_max=spectral.mel_f_max,
            show_x_axis_label=viz.show_x_axis_label, show_y_axis_label=viz.show_y_axis_label,
            show_legend=viz.show_legend, show_title=viz.show_title,
            figure_size_spectrum=viz.figure_size_spectrum,
            figure_size_spectrogram=viz.figure_size_spectrogram,
            dpi=viz.figure_dpi, spectrum_alpha=viz.line_alpha,
        )

    test_step = global_step + 1
    test_epoch_float = test_step / steps_per_epoch if steps_per_epoch > 0 else int(cfg.train.epochs)

    if len(waveform_images) > 0:
        logger.log_images("test/waveforms", waveform_images, step=test_step)
        print(f"Logged {len(waveform_images)} test waveform images to wandb")
    else:
        print("Warning: No test waveform images generated")
    if viz.render_prev_waveform and len(waveform_images_prev) > 0:
        logger.log_images("test/waveforms_prev", waveform_images_prev, step=test_step)
    if viz.render_prev_waveform and tb is not None:
        rollout_images = _rollout_images(tb, int(cfg.eval.num_samples), full_ds, viz)
        if rollout_images:
            logger.log_images("test/waveforms_rollout", rollout_images, step=test_step)
    if viz.enable_stft_spectrum and len(stft_images) > 0:
        logger.log_images("test/stft_spectrum", stft_images, step=test_step)
    if viz.enable_stft_spectrum and len(stft_images_norm) > 0:
        logger.log_images("test/stft_spectrum_normarize", stft_images_norm, step=test_step)
    if viz.enable_spectrogram and len(spec_images) > 0:
        logger.log_images("test/spectrograms", spec_images, step=test_step)
    if viz.enable_mel_spectrogram and len(mel_images) > 0:
        logger.log_images("test/mel_spectrograms", mel_images, step=test_step)

    logger.log(
        {
            "test/loss": test_summary.loss_mean,
            "test/infer_time_ms_mean": test_summary.infer_time_mean_ms,
            "test/generate_time_ms": test_generate_time_ms,
            "test/snr_db": snr_mean,
            "test/rmse": rmse_mean,
            "test/gfc": gfc_mean,
            "test/freq_spectrum_rmse": freq_spectrum_rmse_mean,
            "epoch": test_epoch_float,
        },
        step=test_step,
    )
    if snr_list:
        logger.log_histogram("test/snr_db_hist", snr_list, step=test_step)
    if rmse_list:
        logger.log_histogram("test/rmse_hist", rmse_list, step=test_step)
    if gfc_list:
        logger.log_histogram("test/gfc_hist", gfc_list, step=test_step)
    if freq_spectrum_rmse_list:
        logger.log_histogram("test/freq_spectrum_rmse_hist", freq_spectrum_rmse_list, step=test_step)
    if test_summary.infer_time_list:
        logger.log_histogram("test/infer_time_ms_hist", test_summary.infer_time_list, step=test_step)
    if test_generate_time_list:
        logger.log_histogram("test/generate_time_ms_hist", test_generate_time_list, step=test_step)

    logger.finish()

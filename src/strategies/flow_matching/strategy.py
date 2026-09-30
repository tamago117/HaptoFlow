"""FlowMatching strategy: wraps the flow-matching training/eval loop."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from src.models.shared.embeddings import LabelEmbedding, SinusoidalTimeEmbedding
from src.strategies.shared.base import (
    BuildContext,
    DatasetParams,
    ModelStrategy,
    StepContext,
    TrainingBundle,
)
from src.strategies.shared.spectral_params import SpectralParams
from src.strategies.shared.registry import register
from src.utils.evaluation.self_forcing_eval import self_forcing_chain_metrics
from src.utils.training.self_forcing import get_self_forcing_params
from src.utils.data.data_augmentation import get_zero_prev_prob
from src.utils.training_loops.flow_matching.self_forcing import build_self_forcing_batch, make_generate_fn
from src.models.flow_matching.builder import build_models
from src.utils.training.model_builder import (
    build_control_embedding,
    build_optimizer_and_scheduler,
    build_prev_segment_encoder,
)
from src.utils.evaluation.evaluation_utils import compute_evaluation_metrics_on_dataloader
from src.strategies.shared.gen_via_inference import generate_samples_via_inference
from src.utils.training_loops.flow_matching import (
    compute_evaluation_losses,
    train_step,
)


@register
class FlowMatchingStrategy(ModelStrategy):
    """ConditionalFlowMatching over continuous EnCodec latents."""

    name = "flow_matching"

    def dataset_params(self, cfg: Any) -> DatasetParams:
        # Self-forcing needs the chain s_0..s_K before each target (K = num_unroll)
        num_unroll = get_self_forcing_params(getattr(getattr(cfg, "train", None), "self_forcing", None)).num_unroll
        return DatasetParams(
            use_encodec=True,
            context_segments=num_unroll,
        )

    def build(self, ctx: BuildContext) -> TrainingBundle:
        if ctx.encodec_wrapper is None:
            raise ValueError("FlowMatchingStrategy requires encodec_wrapper in BuildContext")
        cfg = ctx.cfg
        device = ctx.device
        full_ds = ctx.full_dataset

        num_classes = len(full_ds.classes)
        base_cond_dim = int(cfg.model.cond_dim)
        time_embed_dim = int(getattr(cfg.model, "time_embed_dim", 0))

        label_embed = LabelEmbedding(num_classes, base_cond_dim).to(device)
        time_embed = (
            SinusoidalTimeEmbedding(time_embed_dim).to(device) if time_embed_dim > 0 else None
        )

        in_ch_calc = ctx.encodec_wrapper.embed_dim
        in_ch = int(cfg.model.in_ch) if cfg.model.in_ch is not None else in_ch_calc

        use_prev_segment = bool(getattr(cfg.model, "use_prev_segment", False))
        prev_segment_encoder, encoder_feature_channels = build_prev_segment_encoder(
            use_prev_segment, in_ch_calc, cfg, device,
        )

        control_embedding, control_cond_dim = build_control_embedding(cfg, device)

        unet, model = build_models(
            in_ch,
            base_cond_dim,
            time_embed_dim,
            control_cond_dim,
            cfg,
            encoder_feature_channels,
            time_embed,
            device,
        )

        optimizer, scheduler = build_optimizer_and_scheduler(
            model, time_embed, prev_segment_encoder, control_embedding, cfg,
            steps_per_epoch=ctx.steps_per_epoch,
        )

        return TrainingBundle(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            extras={
                "label_embed": label_embed,
                "time_embed": time_embed,
                "prev_segment_encoder": prev_segment_encoder,
                "control_embedding": control_embedding,
                "unet": unet,
                "spectral_params": SpectralParams.from_cfg(cfg),
            },
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        bundle: TrainingBundle,
        ctx: StepContext,
    ) -> Dict[str, float]:
        x = bundle.extras
        cfg = ctx.cfg
        return train_step(
            batch,
            bundle.model,
            x["prev_segment_encoder"],
            x["control_embedding"],
            x["label_embed"],
            bundle.optimizer,
            ctx.scaler,
            ctx.encodec_wrapper,
            ctx.device,
            ctx.full_dataset.sample_rate,
            float(cfg.train.grad_clip),
            bool(cfg.train.amp),
            getattr(cfg.train, "data_augmentation", None),
            sampling_steps=int(cfg.sampling.steps),
            self_forcing_cfg=getattr(cfg.train, "self_forcing", None),
        )

    def eval_losses(
        self,
        batch: Dict[str, torch.Tensor],
        bundle: TrainingBundle,
        ctx: StepContext,
    ) -> Tuple[float, float]:
        x = bundle.extras
        return compute_evaluation_losses(
            batch,
            bundle.model,
            x["prev_segment_encoder"],
            x["control_embedding"],
            x["label_embed"],
            ctx.encodec_wrapper,
            ctx.device,
            ctx.full_dataset.sample_rate,
        )

    def generate_samples(
        self,
        batch: Dict[str, torch.Tensor],
        bundle: TrainingBundle,
        ctx: StepContext,
    ) -> Any:
        cfg = ctx.cfg
        viz_cfg = getattr(cfg, "visualization", None)
        num_samples = int(cfg.eval.num_samples)
        render_prev_waveform = bool(getattr(viz_cfg, "render_prev_waveform", False)) if viz_cfg is not None else False
        return generate_samples_via_inference(
            self, bundle, ctx, batch, num_samples, render_prev_waveform=render_prev_waveform,
        )

    def build_inference(self, bundle, ctx):
        from src.inference.flow_matching import FlowMatchingInference
        x = bundle.extras
        iface = FlowMatchingInference()
        iface.attach_to_bundle(
            cfg=ctx.cfg,
            device=ctx.device,
            full_dataset=ctx.full_dataset,
            waveform_target_len=int(ctx.full_dataset.waveform_target_len),
            model=bundle.model,
            label_embed=x["label_embed"],
            encodec_wrapper=ctx.encodec_wrapper,
            time_embed=x.get("time_embed"),
            prev_segment_encoder=x.get("prev_segment_encoder"),
            control_embedding=x["control_embedding"],
            use_prev_segment=bool(getattr(ctx.cfg.model, "use_prev_segment", False)),
            control_stats=ctx.extras.get("control_stats"),
        )
        return iface

    def prepare_eval_batch(self, batch, bundle, ctx, generator=None):
        # Phase 2: same prevs as training (zero start, own generations), plus rollouts for the images
        if "chain_waveform" not in batch:
            return batch
        cfg = ctx.cfg
        sf = get_self_forcing_params(getattr(cfg.train, "self_forcing", None))
        x = bundle.extras
        generate_fn = make_generate_fn(
            model=bundle.model,
            label_embed=x["label_embed"],
            control_embedding=x["control_embedding"],
            prev_segment_encoder=x["prev_segment_encoder"],
            encodec_wrapper=ctx.encodec_wrapper,
            steps=int(cfg.sampling.steps),
            sample_rate=float(ctx.full_dataset.sample_rate),
            device=ctx.device,
        )
        flat, _ = build_self_forcing_batch(
            batch, generate_fn, sf.prob,
            zero_start_prob=get_zero_prev_prob(getattr(cfg.train, "data_augmentation", None)),
            loss_pairs=sf.loss_pairs,
            generator=generator,
            with_rollout=True,
        )
        return flat

    def self_forcing_metrics(self, dataset, indices, bundle, ctx):
        fr = getattr(getattr(ctx.cfg, "eval", None), "self_forcing", None)
        if fr is None:
            return {}
        x = bundle.extras
        lp = x["spectral_params"]
        generate_fn = make_generate_fn(
            model=bundle.model,
            label_embed=x["label_embed"],
            control_embedding=x["control_embedding"],
            prev_segment_encoder=x["prev_segment_encoder"],
            encodec_wrapper=ctx.encodec_wrapper,
            steps=int(ctx.cfg.sampling.steps),
            sample_rate=float(dataset.sample_rate),
            device=ctx.device,
        )
        return self_forcing_chain_metrics(
            dataset, indices, generate_fn,
            num_unroll=int(fr.num_unroll),
            num_chains=int(fr.num_chains),
            sample_rate=int(dataset.sample_rate),
            stft_n_fft=lp.stft_n_fft,
            stft_hop_length=lp.stft_hop_length,
            stft_win_length=lp.stft_win_length,
        )

    def compute_metrics(self, dataloader, bundle, ctx):
        x = bundle.extras
        lp = x["spectral_params"]
        return compute_evaluation_metrics_on_dataloader(
            dataloader,
            bundle.model,
            x["prev_segment_encoder"],
            x["control_embedding"],
            x["label_embed"],
            ctx.encodec_wrapper,
            ctx.device,
            int(ctx.cfg.sampling.steps),
            lp.stft_n_fft,
            lp.stft_hop_length,
            lp.stft_win_length,
            ctx.full_dataset.sample_rate,
            return_per_sample_lists=bool(ctx.extras.get("return_per_sample_lists", False)),
            desc=ctx.extras.get("progress_desc"),
            max_batches=ctx.extras.get("max_batches"),
        )

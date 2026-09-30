"""CUDA-graph replay of ``ConditionalFlowMatching.generate`` for serving.

A serving segment is only a few latent frames, so the ODE loop is bound by
kernel-launch overhead; replaying one captured graph removes it.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from src.models.flow_matching.conditional_flow_matching import ConditionalFlowMatching


class CudaGraphGenerator:
    """Drop-in ``generate`` that captures the whole sampling loop once per input signature.

    The graph is re-captured when steps, shapes, solver or guidance change.
    Weights are read in place, so in-place parameter updates are picked up.
    """

    def __init__(self, model: ConditionalFlowMatching) -> None:
        self.model = model
        self._key = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._noise: Optional[torch.Tensor] = None
        self._cond: Optional[torch.Tensor] = None
        self._feats: Optional[List[torch.Tensor]] = None
        self._out: Optional[torch.Tensor] = None

    def _capture(self, steps: int, cond: torch.Tensor, shape: torch.Size,
                 encoder_feats: Optional[list]) -> None:
        self._graph = None  # release the previous graph's pool first
        self._noise = torch.zeros(shape, device=cond.device)  # no RNG draw during capture
        self._cond = cond.clone()
        self._feats = None if encoder_feats is None else [f.clone() for f in encoder_feats]

        def run() -> torch.Tensor:
            return self.model.generate(
                steps=steps, cond=self._cond, shape=shape,
                encoder_feats=self._feats, noise=self._noise,
            )

        # Warm up on a side stream (cuDNN / allocator setup must not be captured)
        side = torch.cuda.Stream(device=cond.device)
        side.wait_stream(torch.cuda.current_stream(cond.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream(cond.device).wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._out = run()
        self._graph = graph

    def generate(
        self,
        steps: int,
        cond: torch.Tensor,
        shape: torch.Size,
        encoder_feats: Optional[list] = None,
    ) -> torch.Tensor:
        if cond.device.type != "cuda":
            return self.model.generate(steps=steps, cond=cond, shape=shape, encoder_feats=encoder_feats)
        model = self.model
        key = (
            int(steps), tuple(shape), model.solver, model.guidance_scale, tuple(cond.shape),
            None if encoder_feats is None else tuple(tuple(f.shape) for f in encoder_feats),
        )
        if key != self._key:
            self._key = None
            self._capture(int(steps), cond, shape, encoder_feats)
            self._key = key
        # Sample outside the graph so the RNG stream matches the eager path
        self._noise.copy_(torch.randn(shape, device=cond.device))
        self._cond.copy_(cond)
        if encoder_feats is not None:
            for dst, src in zip(self._feats, encoder_feats):
                dst.copy_(src)
        self._graph.replay()
        return self._out.clone()

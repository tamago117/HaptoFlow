"""Inference engines: per-model classes that wrap a trained checkpoint and expose generate_from_control().

Used by the Gradio app (`scripts/serving/app_generate_waveform.py`). Also instantiable in-memory from a TrainingBundle via
``Strategy.build_inference()`` so training-time visualization runs through the
same code path as production.
"""
from src.inference.flow_matching import FlowMatchingInference

__all__ = [
    "FlowMatchingInference",
]

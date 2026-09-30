"""Checkpoint directories on the Hugging Face Hub, addressed as ``hf://<owner>/<repo>[/<subdir>][@<revision>]``.

:func:`resolve_model_dir` turns such a spec into a local directory in the shared
HF cache (only ``meta.json`` + ``*.safetensors`` are fetched); plain paths pass
through unchanged. The model repo is public; no credentials are needed.
"""
from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

HF_PREFIX = "hf://"
HF_MODEL_REPO_ID = "tamago117/HaptoFlow"
HF_MODEL_DIRS = (
    f"{HF_PREFIX}{HF_MODEL_REPO_ID}/haptoflow_accel",
    f"{HF_PREFIX}{HF_MODEL_REPO_ID}/haptoflow_audio",
)


def is_hf_spec(path: object) -> bool:
    return str(path).strip().startswith(HF_PREFIX)


def parse_hf_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    """``hf://owner/repo/sub/dir@rev`` → ``("owner/repo", "sub/dir", "rev")``."""
    body = str(spec).strip()
    if not body.startswith(HF_PREFIX):
        raise ValueError(f"not an {HF_PREFIX} spec: {spec!r}")
    body, sep, revision = body[len(HF_PREFIX):].partition("@")
    parts = [p for p in body.strip("/").split("/") if p]
    if len(parts) < 2 or (sep and not revision):
        raise ValueError(f"expected {HF_PREFIX}<owner>/<repo>[/<subdir>][@<revision>], got {spec!r}")
    return "/".join(parts[:2]), "/".join(parts[2:]), revision or None


@functools.lru_cache(maxsize=None)
def _download(spec: str) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    repo_id, subdir, revision = parse_hf_spec(spec)
    prefix = f"{subdir}/" if subdir else ""
    try:
        root = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[f"{prefix}meta.json", f"{prefix}*.safetensors"],
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise RuntimeError(
            f"Cannot access the model repo '{repo_id}'. Check the repo id and revision."
        ) from exc
    local = os.path.join(root, subdir)
    if not os.path.exists(os.path.join(local, "meta.json")):
        raise FileNotFoundError(f"meta.json not found for {spec} (looked in {local})")
    return local


def resolve_model_dir(path: str) -> str:
    """Local checkpoint directory for ``path``; ``hf://`` specs are downloaded once per process."""
    path = str(path).strip()
    return _download(path) if is_hf_spec(path) else path

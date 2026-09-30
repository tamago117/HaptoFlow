"""Locating the Cluster Haptic Texture Dataset, on disk or on the Hugging Face Hub.

The dataset lives in a public Hub dataset repo (see ``HF_DATASET_REPO_ID``);
no credentials are needed to download it.

:func:`resolve_dataset_root` is deliberately *local-path-first*: it never
downloads anything it can already find on disk.

    1. ``$HAPTOFLOW_DATASET_ROOT``  — explicit override, always wins
    2. an existing local directory  — e.g. ``datasets/texture_dataset``
    3. ``snapshot_download``        — only when the first two come up empty

The dataset is stored as ``.parquet`` sensor tables and ``.flac`` recordings;
read tables through :mod:`src.utils.data.sensor_io` so the ``time_ns`` column
is converted back to seconds for you.
"""
import os
from typing import Iterable, List, Optional, Sequence

HF_DATASET_REPO_ID = "tamago117/cluster-haptic-texture-dataset"
ENV_DATASET_ROOT = "HAPTOFLOW_DATASET_ROOT"
DEFAULT_LOCAL_DIR = os.path.join("datasets", "texture_dataset")

# Sensor directories under sensor_data/ that can be fetched independently.
SENSOR_SUBSETS = ("audio", "raw_audio", "accel", "force", "position")
# Non-sensor groups.
EXTRA_SUBSETS = ("images",)
ALL_SUBSETS = SENSOR_SUBSETS + EXTRA_SUBSETS

# Small files every consumer needs regardless of which subset they asked for.
ROOT_PATTERNS = ["README.md", "texture_list.xlsx"]

# src/utils/data/hf_dataset.py → repo root is four levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _normalize_subsets(subset: Optional[object]) -> List[str]:
    """Accept ``None`` / ``"all"`` / ``"a,b"`` / ``["a", "b"]`` uniformly."""
    if subset is None:
        return ["all"]
    if isinstance(subset, str):
        names = [part.strip() for part in subset.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in subset if str(part).strip()]
    if not names or "all" in names:
        return ["all"]
    unknown = [name for name in names if name not in ALL_SUBSETS]
    if unknown:
        raise ValueError(f"Unknown subset(s) {unknown}; choose from {('all',) + ALL_SUBSETS}")
    return names


def _normalize_textures(textures: Optional[object]) -> Optional[List[str]]:
    if textures is None:
        return None
    if isinstance(textures, str):
        ids = [part.strip() for part in textures.split(",") if part.strip()]
    else:
        ids = [str(part).strip() for part in textures if str(part).strip()]
    return ids or None


def build_allow_patterns(
    subset: Optional[object] = None,
    textures: Optional[object] = None,
) -> Optional[List[str]]:
    """Glob patterns for ``snapshot_download``; ``None`` means "the whole repo"."""
    names = _normalize_subsets(subset)
    texture_ids = _normalize_textures(textures)
    if names == ["all"] and texture_ids is None:
        return None
    if names == ["all"]:
        names = list(ALL_SUBSETS)

    patterns: List[str] = list(ROOT_PATTERNS)
    for name in names:
        if name in SENSOR_SUBSETS:
            if texture_ids is None:
                patterns.append(f"sensor_data/{name}/**")
            else:
                patterns.extend(f"sensor_data/{name}/{tid}/**" for tid in texture_ids)
        elif name == "images":
            if texture_ids is None:
                patterns.append("images/**")
            else:
                for tid in texture_ids:
                    patterns.append(f"images/crop_images/img{tid}/**")
                    patterns.append(f"images/images_scan_area/img{tid}.jpg")
    return patterns


def download_dataset(
    subset: Optional[object] = None,
    textures: Optional[object] = None,
    repo_id: str = HF_DATASET_REPO_ID,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Fetch (part of) the dataset from the Hub and return the local root.

    ``local_dir`` gets *real files*; the download bookkeeping lives in
    ``<local_dir>/.cache/huggingface/``.  Passing ``local_dir=None`` uses the
    shared HF cache instead, where the returned tree is made of **symlinks**
    into the blob store — fine for reading, but it breaks callers that write
    into the tree or move files in place.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    target = local_dir if local_dir is None else os.path.abspath(local_dir)
    try:
        return snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=target,
            allow_patterns=build_allow_patterns(subset, textures),
            token=token,
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise RuntimeError(
            f"Cannot access the dataset repo '{repo_id}'. "
            "Check the repo id and revision."
        ) from exc


def resolve_dataset_root(
    subset: Optional[object] = None,
    textures: Optional[object] = None,
    repo_id: str = HF_DATASET_REPO_ID,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    allow_download: bool = True,
) -> str:
    """Return a local directory holding the dataset.

    Resolution order: ``$HAPTOFLOW_DATASET_ROOT`` → an existing local directory
    → ``snapshot_download``.  With ``allow_download=False`` the last step is
    skipped and the expected local path is returned even if it does not exist
    yet, which keeps path *defaults* free of network side effects.
    """
    env_root = os.environ.get(ENV_DATASET_ROOT)
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))

    candidate = local_dir or DEFAULT_LOCAL_DIR
    if not os.path.isabs(candidate):
        candidate = os.path.join(PROJECT_ROOT, candidate)
    if os.path.isdir(candidate) or not allow_download:
        return os.path.abspath(candidate)

    return download_dataset(
        subset=subset,
        textures=textures,
        repo_id=repo_id,
        local_dir=candidate,
        revision=revision,
    )

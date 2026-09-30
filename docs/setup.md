# Setup

## Docker

The image (`docker/Dockerfile`) contains only the dependencies from `uv.lock`; the
repository is mounted at `/workspace` at run time, so rebuild only when
`pyproject.toml` / `uv.lock` change. The host needs the NVIDIA driver and the
NVIDIA Container Toolkit.

```bash
docker/build.sh                                        # build haptoflow:uv
docker/run.sh                                          # interactive shell
docker/run.sh python scripts/train.py --model flow_matching
PORTS=7860 docker/run.sh python scripts/serving/app_generate_waveform.py   # http://localhost:7860
```

`docker/run.sh` options (environment variables):

- `GPUS`: `all` by default; empty for CPU only, `device=0` for a single GPU.
- `PORTS`: host ports to publish on `127.0.0.1` (none by default).
- `SHM_SIZE`: `/dev/shm` size for DataLoader workers (default `8g`).

The container runs as your user, mounts the Hugging Face cache and `~/.netrc`, and
forwards `HF_TOKEN`, `WANDB_API_KEY` and `WANDB_MODE` when set.

## Windows: Installing PyTorch with GPU (CUDA) support

On Windows, `uv sync` installs the CPU build of PyTorch. Check your CUDA version
with `nvidia-smi` and reinstall from the matching index, e.g. for CUDA 12.8:

```powershell
uv pip uninstall torch torchaudio
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

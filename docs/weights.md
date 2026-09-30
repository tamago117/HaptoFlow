# Trained weights

Pre-trained checkpoints are hosted at
[tamago117/HaptoFlow](https://huggingface.co/tamago117/HaptoFlow). Each checkpoint
directory holds `meta.json` (training config and dataset metadata) and
`model.safetensors`; the app loads them via `hf://tamago117/HaptoFlow/<subdir>` or
a local path.

To download them locally (add `--revision <tag|commit>` to pin a version):

```bash
uv run hf download tamago117/HaptoFlow --repo-type model --local-dir ./checkpoints
```

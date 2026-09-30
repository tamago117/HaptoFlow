"""W&B (Weights & Biases) logging wrapper."""

from typing import Optional, Dict, Any, Union
from omegaconf import DictConfig, OmegaConf
import torch
from PIL import Image
import numpy as np


class WandbLogger:
    """Wrapper class for wandb logging with automatic disabled mode handling.
    
    This class simplifies wandb usage by:
    - Automatically handling disabled mode (no-op when disabled)
    - Simplifying initialization from config
    - Providing helper methods for common logging patterns
    """
    
    def __init__(self, config: Optional[Union[DictConfig, Dict[str, Any]]] = None):
        """Initialize W&B logger from config.
        
        Args:
            config: Configuration object. If None, logger is disabled.
                   Expected to have `wandb` attribute with:
                   - `mode`: "online", "offline", or "disabled"
                   - `project`: Project name
                   - `entity`: Optional entity name
                   - `run_name`: Optional run name
        """
        self._run = None
        self._is_enabled = False
        
        if config is None:
            return
        
        try:
            import wandb
        except ImportError:
            # wandb not installed: logging stays disabled
            return
        
        if hasattr(config, "wandb"):
            wandb_cfg = config.wandb
        elif isinstance(config, dict) and "wandb" in config:
            wandb_cfg = config["wandb"]
        else:
            return
        
        wandb_mode = str(wandb_cfg.get("mode", "disabled"))
        
        if wandb_mode == "disabled":
            return
        
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
        
        try:
            self._run = wandb.init(
                project=str(wandb_cfg.get("project", "default")),
                entity=None if wandb_cfg.get("entity") is None else str(wandb_cfg.get("entity")),
                mode=wandb_mode,
                name=None if wandb_cfg.get("run_name") is None else str(wandb_cfg.get("run_name")),
                config=config_dict,
            )
            self._is_enabled = True
        except Exception:
            self._run = None
            self._is_enabled = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self._is_enabled
    
    @property
    def run(self):
        """Get the underlying wandb run object (or None if disabled)."""
        return self._run
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """Log data to wandb.
        
        Args:
            data: Dictionary of metrics/values to log.
            step: Optional step number.
            commit: Whether to commit this log entry immediately.
        """
        if not self._is_enabled or self._run is None:
            return
        
        try:
            import wandb as _wandb
        except ImportError:
            return
        
        # wandb rejects non-increasing steps, so bump the step past the current one
        if step is not None:
            try:
                current_step = getattr(self._run, '_step', None)
                if current_step is not None:
                    if step < current_step:
                        step = current_step
                    elif step == current_step:
                        step = current_step + 1
            except Exception:
                pass
            
            self._run.log(data, step=step, commit=commit)
        else:
            self._run.log(data, commit=commit)
    
    def log_histogram(self, key: str, sequence, step: Optional[int] = None):
        """Log a histogram (distribution) to wandb. One key shows distribution with mean/std/min/max in UI.
        
        Args:
            key: Key/name for the histogram.
            sequence: Per-sample values (list or array) to build the histogram from.
            step: Optional step number.
        """
        if not self._is_enabled or self._run is None:
            return
        if not sequence:
            return
        try:
            import wandb as _wandb
        except ImportError:
            return
        hist = _wandb.Histogram(sequence)
        log_data = {key: hist}
        if step is not None:
            try:
                current_step = getattr(self._run, '_step', None)
                if current_step is not None:
                    if step < current_step:
                        step = current_step
                    elif step == current_step:
                        step = current_step + 1
            except Exception:
                pass
            self._run.log(log_data, step=step)
        else:
            self._run.log(log_data)
    
    def log_image(self, key: str, image: Union[torch.Tensor, Image.Image], caption: Optional[str] = None, step: Optional[int] = None):
        """Log an image to wandb.
        
        Args:
            key: Key/name for the image.
            image: Image as torch.Tensor or PIL.Image.
            caption: Optional caption for the image.
            step: Optional step number.
        """
        if not self._is_enabled or self._run is None:
            return
        
        try:
            import wandb as _wandb
        except ImportError:
            return
        
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                image = image.detach().cpu()
            wandb_image = _wandb.Image(image, caption=caption)
        elif isinstance(image, Image.Image):
            wandb_image = _wandb.Image(image, caption=caption)
        else:
            try:
                if hasattr(image, "numpy"):
                    import numpy as np
                    arr = image.numpy() if hasattr(image, "numpy") else np.array(image)
                    wandb_image = _wandb.Image(arr, caption=caption)
                else:
                    wandb_image = _wandb.Image(image, caption=caption)
            except Exception:
                return
        
        log_data = {key: wandb_image}
        if step is not None:
            self._run.log(log_data, step=step)
        else:
            self._run.log(log_data)
    
    def log_images(self, key: str, images: list, step: Optional[int] = None):
        """Log multiple images as a list.
        
        Args:
            key: Key/name for the images list.
            images: List of images (torch.Tensor or PIL.Image or wandb.Image).
            step: Optional step number.
        """
        if not self._is_enabled or self._run is None:
            return
        
        try:
            import wandb as _wandb
        except ImportError:
            return
        
        wandb_images = []
        for img in images:
            if isinstance(img, _wandb.Image):
                wandb_images.append(img)
            elif isinstance(img, torch.Tensor):
                if img.is_cuda:
                    img = img.detach().cpu()
                wandb_images.append(_wandb.Image(img))
            elif isinstance(img, Image.Image):
                wandb_images.append(_wandb.Image(img))
            else:
                try:
                    if hasattr(img, "numpy"):
                        import numpy as np
                        arr = img.numpy() if hasattr(img, "numpy") else np.array(img)
                        wandb_images.append(_wandb.Image(arr))
                    else:
                        wandb_images.append(_wandb.Image(img))
                except Exception:
                    continue
        
        if wandb_images:
            log_data = {key: wandb_images}
            if step is not None:
                # wandb rejects non-increasing steps, so bump the step past the current one
                try:
                    current_step = getattr(self._run, '_step', None)
                    if current_step is not None:
                        if step < current_step:
                            step = current_step
                        elif step == current_step:
                            step = current_step + 1
                except Exception:
                    pass
                self._run.log(log_data, step=step)
            else:
                self._run.log(log_data)
    
    def log_audio(self, key: str, audio: Union[np.ndarray, torch.Tensor], sample_rate: int, step: Optional[int] = None, caption: Optional[str] = None):
        """Log audio to wandb.
        
        Args:
            key: Key/name for the audio.
            audio: Audio array as numpy array or torch.Tensor (1D or 2D).
            sample_rate: Sample rate in Hz.
            step: Optional step number.
            caption: Optional caption for the audio.
        """
        if not self._is_enabled or self._run is None:
            return
        
        try:
            import wandb as _wandb
            import numpy as np
        except ImportError:
            return
        
        if isinstance(audio, torch.Tensor):
            if audio.is_cuda:
                audio = audio.detach().cpu()
            audio = audio.numpy()
        
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        wandb_audio = _wandb.Audio(audio, sample_rate=sample_rate, caption=caption)
        log_data = {key: wandb_audio}
        if step is not None:
            self._run.log(log_data, step=step)
        else:
            self._run.log(log_data)
    
    def finish(self):
        """Finish the wandb run."""
        if self._is_enabled and self._run is not None:
            self._run.finish()
            self._run = None
            self._is_enabled = False


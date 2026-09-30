"""EnCodec wrapper: waveform <-> continuous latent, decoded through the RVQ codes."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio

try:
    from encodec import EncodecModel
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False
    EncodecModel = None


class EncodecWrapper:
    """Wrapper for EnCodec: waveform <-> continuous latent, decoded through the RVQ codes.
    
    The EnCodec model is loaded once and frozen (eval mode, no gradients).
    """
    
    def __init__(
        self,
        model_name: str = "24khz",
        bandwidth: float = 6.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize EnCodec wrapper.
        
        Args:
            model_name: EnCodec model name ("24khz" or "48khz")
            bandwidth: Bandwidth in kbps (typically 6.0 or 24.0)
            device: Device to load model on (default: auto-detect)
        """
        if not ENCODEC_AVAILABLE:
            raise ImportError("encodec package is not available. Install with: pip install encodec")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_name = model_name
        self.bandwidth = bandwidth
        
        if model_name == "24khz":
            self.model = EncodecModel.encodec_model_24khz(pretrained=True)
        elif model_name == "48khz":
            self.model = EncodecModel.encodec_model_48khz(pretrained=True)
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Use '24khz' or '48khz'")
        
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        self.sample_rate = self.model.sample_rate
        # Latent channel count = codebook vector width of the RVQ layers
        self.embed_dim = self.model.quantizer.vq.layers[0].codebook.shape[1]
        # (orig_sr, new_sr) -> torchaudio Resample kept on self.device
        self._resamplers: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}

        if self.bandwidth is not None:
            try:
                self.model.set_target_bandwidth(float(self.bandwidth))
            except ValueError:
                # Unsupported bandwidth: fall back to the highest supported one.
                self.model.bandwidth = None

    def _prepare_waveform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize waveform shape to [B, 1, T] and move to device."""
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            # [B, T]
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected waveform tensor with 1, 2 or 3 dims, got shape {tuple(x.shape)}")

        if x.shape[1] > 1:
            x = x[:, :1, :]

        return x.to(self.device)

    def _resampler(self, orig_rate: float, new_rate: float) -> torchaudio.transforms.Resample:
        key = (int(round(orig_rate)), int(round(new_rate)))
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(*key).to(self.device)
            self._resamplers[key] = resampler
        return resampler

    def _resample_to_model_rate(self, x: torch.Tensor, input_sample_rate: float) -> torch.Tensor:
        """Resample a mono [B, 1, T] batch to the model rate on ``self.device``.

        Same transform as ``encodec.utils.convert_audio`` (torchaudio ``Resample``),
        applied to the whole batch at once instead of per item on the CPU.
        """
        x = self._resampler(input_sample_rate, self.sample_rate)(x)
        if self.model.channels == 2:
            x = x.expand(-1, 2, -1)
        return x

    def resample_from_model_rate(self, waveform: torch.Tensor, target_sample_rate: float) -> torch.Tensor:
        """Resample decoded [B, 1, T] audio on ``self.device`` (keeps the graph for training losses)."""
        return self._resampler(self.sample_rate, target_sample_rate)(waveform)

    def _encode_frame(
        self,
        frame: torch.Tensor,
        *,
        return_latent: bool,
        return_codes: bool,
        return_quantized_latent: bool,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Encode a single frame and optionally return latent representations."""
        outputs: Dict[str, Optional[torch.Tensor]] = {
            "latent": None,
            "codes": None,
            "quantized_latent": None,
            "scale": None,
        }

        with torch.no_grad():
            if self.model.normalize:
                mono = frame.mean(dim=1, keepdim=True)
                volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                scale = 1e-8 + volume
                frame = frame / scale
                outputs["scale"] = scale.view(-1, 1)
            else:
                outputs["scale"] = None

            latent = self.model.encoder(frame) if (return_latent or return_codes or return_quantized_latent) else None

            codes_btq: Optional[torch.Tensor] = None
            if return_codes or return_quantized_latent:
                assert latent is not None
                codes_btq = self.model.quantizer.encode(latent, self.model.frame_rate, self.model.bandwidth)
                # codes_btq shaped [K, B, T]; convert to [B, K, T] for convenience.
                codes = codes_btq.transpose(0, 1)
                if return_codes:
                    outputs["codes"] = codes

            if return_latent:
                outputs["latent"] = latent

            if return_quantized_latent:
                assert codes_btq is not None
                quantized = self.model.quantizer.decode(codes_btq)
                outputs["quantized_latent"] = quantized

        return outputs

    def encode_waveform(
        self,
        x: torch.Tensor,
        *,
        return_latent: bool = False,
        return_codes: bool = True,
        return_quantized_latent: bool = False,
        input_sample_rate: Optional[float] = None,
        move_to_cpu: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Encode waveform into optional latent and discrete representations.
        
        Args:
            x: Input waveform tensor
            return_latent: Whether to return continuous latent representation
            return_codes: Whether to return discrete codes
            return_quantized_latent: Whether to return quantized latent
            input_sample_rate: Optional input sample rate. If provided and different from
                             EnCodec model's sample rate, waveform will be resampled.
        """
        if not (return_latent or return_codes or return_quantized_latent):
            raise ValueError("At least one of return_latent/return_codes/return_quantized_latent must be True.")

        x = self._prepare_waveform(x)
        
        if input_sample_rate is not None and abs(input_sample_rate - self.sample_rate) > 1e-6:
            x = self._resample_to_model_rate(x, input_sample_rate)
        segment_length = self.model.segment_length
        if segment_length is None:
            segments = [(0, x)]
            stride = x.shape[-1]
        else:
            stride = self.model.segment_stride or segment_length
            segments = [
                (offset, x[:, :, offset: offset + segment_length])
                for offset in range(0, x.shape[-1], stride)
            ]

        latent_list = []
        codes_list = []
        quantized_list = []

        for i, (offset, frame) in enumerate(segments):
            frame_outputs = self._encode_frame(
                frame,
                return_latent=return_latent,
                return_codes=return_codes,
                return_quantized_latent=return_quantized_latent,
            )

            if return_latent and frame_outputs["latent"] is not None:
                latent_list.append(frame_outputs["latent"])
            if return_codes and frame_outputs["codes"] is not None:
                codes_list.append(frame_outputs["codes"])
            if return_quantized_latent and frame_outputs["quantized_latent"] is not None:
                quantized_list.append(frame_outputs["quantized_latent"])

        outputs: Dict[str, Optional[torch.Tensor]] = {"latent": None, "codes": None, "quantized_latent": None}

        if latent_list:
            latent = torch.cat(latent_list, dim=-1)
            outputs["latent"] = latent

        if codes_list:
            codes = torch.cat(codes_list, dim=-1)
            outputs["codes"] = codes

        if quantized_list:
            quantized = torch.cat(quantized_list, dim=-1)
            outputs["quantized_latent"] = quantized

        if move_to_cpu:
            for key in list(outputs.keys()):
                tensor = outputs[key]
                if tensor is not None:
                    outputs[key] = tensor.detach().cpu()

        return outputs
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform to continuous latent representation (pre-quantization).
        
        Args:
            x: Waveform tensor of shape [B, 1, T], [B, T], or [T]
        
        Returns:
            latent: Continuous latent tensor of shape [B, D, T'] (or [D, T'] if B==1)
        """
        outputs = self.encode_waveform(
            x,
            return_latent=True,
            return_codes=False,
            return_quantized_latent=False,
            move_to_cpu=True,
        )
        latent = outputs["latent"]
        if latent is None:
            raise RuntimeError("Failed to obtain latent representation from waveform.")
        if latent.shape[0] == 1:
            latent = latent.squeeze(0)
        return latent
    
    def quantize_latent(
        self,
        latent: torch.Tensor,
        bandwidth: Optional[float] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Quantize a continuous latent representation using EnCodec quantizer.
        
        Args:
            latent: Continuous latent of shape [B, D, T] or [D, T].
            bandwidth: Optional bandwidth override.
        
        Returns:
            codes: Quantized codes of shape [B, Q, T] (or [Q, T] if B==1)
            quantized_latent: Quantized latent tensor of shape [B, D, T] (or [D, T] if B==1)
        """
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        elif latent.dim() != 3:
            raise ValueError(f"Expected latent tensor with 2 or 3 dims, got shape {tuple(latent.shape)}")
        
        latent = latent.to(self.device)
        with torch.no_grad():
            codes_btq = self.model.quantizer.encode(latent, self.model.frame_rate, bandwidth or self.model.bandwidth)
            codes = codes_btq.transpose(0, 1)  # [B, Q, T]
            quantized_latent = self.model.quantizer.decode(codes_btq)
        
        codes = codes.detach().cpu()
        quantized_latent = quantized_latent.detach().cpu()
        if codes.shape[0] == 1:
            codes = codes.squeeze(0)
            quantized_latent = quantized_latent.squeeze(0)
        return codes.long(), quantized_latent
    
    def decode_from_latent(
        self,
        latent: torch.Tensor,
        target_length: Optional[int] = None,
        target_sample_rate: Optional[float] = None,
    ) -> torch.Tensor:
        """Decode a continuous latent representation directly into waveform space.
        
        Args:
            latent: Continuous latent tensor
            target_length: Optional target length for output waveform (for padding/trimming)
            target_sample_rate: Optional target sample rate. If provided and different from
                               EnCodec model's sample rate, waveform will be resampled.
        """
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        elif latent.dim() != 3:
            raise ValueError(f"Expected latent tensor with 2 or 3 dims, got shape {tuple(latent.shape)}")
        
        latent = latent.to(self.device)
        with torch.no_grad():
            waveform = self.model.decoder(latent)
            if target_length is not None:
                if waveform.shape[-1] > target_length:
                    waveform = waveform[..., :target_length]
                elif waveform.shape[-1] < target_length:
                    pad = target_length - waveform.shape[-1]
                    waveform = F.pad(waveform, (0, pad))
            
            if target_sample_rate is not None and abs(target_sample_rate - self.sample_rate) > 1e-6:
                waveform = self.resample_from_model_rate(waveform, target_sample_rate)
        
        waveform = waveform.detach().cpu()
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        return waveform
    
    def decode_from_codes(
        self,
        codes: torch.LongTensor,
        target_length: Optional[int] = None,
        target_sample_rate: Optional[float] = None,
    ) -> torch.Tensor:
        """Decode discrete codes to waveform.
        
        Args:
            codes: Discrete codes of shape [B, Q, T] or [Q, T]
            target_length: Optional target length for output waveform (for padding/trimming)
            target_sample_rate: Optional target sample rate. If provided and different from
                              EnCodec model's sample rate, waveform will be resampled.
        
        Returns:
            waveform: Decoded waveform of shape [B, 1, T'] or [1, T']
                     Note: T' may differ from T due to frame boundaries
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        
        codes = codes.to(self.device)
        B, Q, T = codes.shape
        
        with torch.no_grad():
            # EnCodec frames are (codes, scale) pairs; use unit scale.
            scale = torch.ones(B, 1, 1, device=self.device, dtype=torch.float32)
            frames = [(codes, scale)]
            
            waveform = self.model.decode(frames)  # [B, 1, T']
            
            if target_length is not None:
                if waveform.shape[-1] > target_length:
                    waveform = waveform[..., :target_length]
                elif waveform.shape[-1] < target_length:
                    pad = target_length - waveform.shape[-1]
                    waveform = F.pad(waveform, (0, pad))
            
            if target_sample_rate is not None and abs(target_sample_rate - self.sample_rate) > 1e-6:
                waveform = self.resample_from_model_rate(waveform, target_sample_rate)
            
            if codes.shape[0] == 1:
                waveform = waveform.squeeze(0)
        
        waveform = waveform.detach().cpu()
        return waveform  # [B, 1, T'] or [1, T']

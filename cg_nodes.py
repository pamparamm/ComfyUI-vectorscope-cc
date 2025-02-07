from torch import Tensor
import comfy.latent_formats
from comfy.model_patcher import ModelPatcher

from .utils import normalize_tensor


class DiffusionCG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "normalize_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "normalize_end_time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "recenter_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "recenter_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "recenter_end_time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "hook"
    CATEGORY = "model_patches/unet"

    PARAMS_NAME = "normalization_cg"

    def hook(
        self,
        model: ModelPatcher,
        normalize_start_time: float,
        normalize_end_time: float,
        recenter_strength: float,
        recenter_start_time: float,
        recenter_end_time: float,
    ):
        m = model.clone()
        latent_format = type(model.model.latent_format)

        m.model_options[self.PARAMS_NAME] = (
            latent_format,
            normalize_start_time,
            normalize_end_time,
            recenter_strength,
            recenter_start_time,
            recenter_end_time,
        )

        return (m,)

    @staticmethod
    def callback(params: tuple):
        def _callback(step: int, x0: Tensor, x: Tensor, total_steps: int):
            (
                latent_format,
                normalize_start_time,
                normalize_end_time,
                recenter_strength,
                recenter_start_time,
                recenter_end_time,
            ) = params

            scale_factor = comfy.latent_formats.SD15().scale_factor  # XL factor is causing noise
            dynamic_range = (1.0 / scale_factor) / 2.0
            current_time = step / (total_steps - 1)

            latent = x0
            target = latent.detach().clone()

            bs, cs, _, _ = latent.shape

            for b in range(bs):
                for c in range(cs):
                    if recenter_strength > 0 and recenter_start_time <= current_time <= recenter_end_time:
                        latent[b][c] += -target[b][c].mean() * recenter_strength

                    if normalize_start_time <= current_time <= normalize_end_time:
                        latent[b][c] = normalize_tensor(latent[b][c], dynamic_range)

        return _callback

from torch import Tensor
import comfy.latent_formats
from comfy.model_patcher import ModelPatcher

from .utils import NoiseMethods, apply_scaling, RGB_2_CbCr, normalize_tensor


class VectorscopeCC:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "alt": ("BOOLEAN", {"default": False}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 3.0, "step": 0.05}),
                "r": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "g": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "method": (
                    ["Straight", "Straight Abs.", "Cross", "Cross Abs.", "Ones", "N.Random", "U.Random", "Multi-Res", "Multi-Res Abs."],
                    {"default": "Straight Abs."},
                ),
                "scaling": (["Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"], {"default": "Flat"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "hook"
    CATEGORY = "model_patches/unet"

    PARAMS_NAME = "vectorscope_cc"

    def hook(
        self,
        model: ModelPatcher,
        alt: bool,
        brightness: float,
        contrast: float,
        saturation: float,
        r: float,
        g: float,
        b: float,
        method: str,
        scaling: str,
    ):
        m = model.clone()
        latent_format = type(model.model.latent_format)

        m.model_options[self.PARAMS_NAME] = (
            latent_format,
            alt,
            brightness,
            contrast,
            saturation,
            r,
            g,
            b,
            method,
            scaling,
        )

        return (m,)

    @staticmethod
    def callback(params: tuple):
        def _callback(step: int, x0: Tensor, x: Tensor, total_steps: int):
            (
                latent_format,
                alt,
                brightness,
                contrast,
                saturation,
                r,
                g,
                b,
                method,
                scaling,
            ) = params

            brightness /= total_steps
            contrast /= total_steps
            saturation = pow(saturation, 1.0 / total_steps)
            r /= total_steps
            g /= total_steps
            b /= total_steps

            latent, cross_latent = (x, x0) if alt else (x0, x)

            if "Straight" in method:
                target = latent.detach().clone()
            elif "Cross" in method:
                target = cross_latent.detach().clone()
            elif "Multi-Res" in method:
                target = NoiseMethods.multires_noise(latent, "Abs" in method)
            elif method == "Ones":
                target = NoiseMethods.ones(latent)
            elif method == "N.Random":
                target = NoiseMethods.normal_noise(latent)
            elif method == "U.Random":
                target = NoiseMethods.gaussian_noise(latent)
            else:
                raise ValueError

            if "Abs" in method:
                target = NoiseMethods.to_abs(target)

            brightness, contrast, saturation, r, g, b = apply_scaling(scaling, step, total_steps, brightness, contrast, saturation, r, g, b)
            bs, _, _, _ = latent.shape

            match latent_format:
                case comfy.latent_formats.SD15:
                    for b in range(bs):
                        # Brightness
                        latent[b][0] += target[b][0] * brightness
                        # Contrast
                        latent[b][0] += NoiseMethods.get_delta(latent[b][0]) * contrast

                        # RGB
                        latent[b][2] -= target[b][2] * r
                        latent[b][1] += target[b][1] * g
                        latent[b][3] -= target[b][3] * b

                        # Saturation
                        latent[b][2] *= saturation
                        latent[b][1] *= saturation
                        latent[b][3] *= saturation

                case comfy.latent_formats.SDXL:
                    cb, cr = RGB_2_CbCr(r, g, b)

                    for b in range(bs):
                        # Brightness
                        latent[b][0] += target[b][0] * brightness
                        # Contrast
                        latent[b][0] += NoiseMethods.get_delta(latent[b][0]) * contrast

                        # CbCr
                        latent[b][1] -= target[b][1] * cr
                        latent[b][2] -= target[b][2] * cb

                        # Saturation
                        latent[b][1] *= saturation
                        latent[b][2] *= saturation

        return _callback


class NormalizeLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "normalize"
    CATEGORY = "latent"

    @staticmethod
    def _normalize_latent(latent: Tensor, dynamic_range: list[int]):
        bs, _, _, _ = latent.shape
        for b in range(bs):
            for c in range(len(dynamic_range)):
                latent[b][c] = normalize_tensor(latent[b][c], dynamic_range[c])

    def normalize(self, latent, model: ModelPatcher):
        latent_image = latent["samples"].detach().clone()
        latent_format = type(model.model.latent_format)

        match latent_format:
            case comfy.latent_formats.SD15:
                NormalizeLatent._normalize_latent(latent_image, [18, 14, 14, 14])
            case comfy.latent_formats.SDXL:
                NormalizeLatent._normalize_latent(latent_image, [20, 16, 16])

        return ({"samples": latent_image},)

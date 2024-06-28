from math import cos, sin, pi
from random import random
from torch import Tensor
import torch

import comfy.model_management


def apply_scaling(
    alg: str,
    current_step: int,
    total_steps: int,
    bri: float,
    con: float,
    sat: float,
    r: float,
    g: float,
    b: float,
):

    if alg == "Flat":
        mod = 1.0

    else:
        ratio = float(current_step / total_steps)
        rad = ratio * pi / 2

        match alg:
            case "Cos":
                mod = cos(rad)
            case "Sin":
                mod = sin(rad)
            case "1 - Cos":
                mod = 1.0 - cos(rad)
            case "1 - Sin":
                mod = 1.0 - sin(rad)
            case _:
                mod = 1.0

    return (bri * mod, con * mod, (sat - 1.0) * mod + 1.0, r * mod, g * mod, b * mod)


def RGB_2_CbCr(r: float, g: float, b: float) -> tuple[float, float]:
    """Convert RGB channels into YCbCr for SDXL"""
    cb = -0.15 * r - 0.29 * g + 0.44 * b
    cr = 0.44 * r - 0.37 * g - 0.07 * b

    return cb, cr


class NoiseMethods:
    @staticmethod
    def get_delta(latent: Tensor) -> Tensor:
        mean = torch.mean(latent)
        return torch.sub(latent, mean)

    @staticmethod
    def to_abs(latent: Tensor) -> Tensor:
        return torch.abs(latent)

    @staticmethod
    def zeros(latent: Tensor) -> Tensor:
        return torch.zeros_like(latent)

    @staticmethod
    def ones(latent: Tensor) -> Tensor:
        return torch.ones_like(latent)

    @staticmethod
    def gaussian_noise(latent: Tensor) -> Tensor:
        return torch.rand_like(latent)

    @staticmethod
    def normal_noise(latent: Tensor) -> Tensor:
        return torch.randn_like(latent)

    @staticmethod
    def multires_noise(latent: Tensor, use_zero: bool, iterations: int = 8, discount: float = 0.4):
        """
        Credit: Kohya_SS
        https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L448
        """

        noise = NoiseMethods.zeros(latent) if use_zero else NoiseMethods.ones(latent)
        batchSize, c, w, h = noise.shape

        device = comfy.model_management.get_torch_device()
        upsampler = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)

        for b in range(batchSize):
            for i in range(iterations):
                r = random() * 2 + 2

                wn = max(1, int(w / (r**i)))
                hn = max(1, int(h / (r**i)))

                noise[b] += (upsampler(torch.randn(1, c, hn, wn).to(device)) * discount**i)[0]

                if wn == 1 or hn == 1:
                    break

        return noise / noise.std()

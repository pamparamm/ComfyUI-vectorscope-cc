from typing import Callable
import torch
from comfy.samplers import KSAMPLER


class CallbackManager:
    def __init__(self):
        self.callbacks: dict[str, tuple[Callable[[tuple], Callable], int]] = {}

    def hijack_samplers(self):
        sample_original = "sample_original"
        if not hasattr(KSAMPLER, sample_original):
            setattr(KSAMPLER, sample_original, KSAMPLER.sample)
            KSAMPLER.sample = self.sample_wrapper(getattr(KSAMPLER, sample_original))

    def register_callback(self, params_name: str, callback_func: Callable[[tuple], Callable], priority: int):
        self.callbacks[params_name] = callback_func, priority

    def sample_wrapper(_self, original_sample: Callable):
        def sample(
            self: KSAMPLER,
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image=None,
            denoise_mask=None,
            disable_pbar=False,
        ):
            model = model_wrap

            original_cb = callback
            original_cb_priority = 1000

            callbacks = []

            def add_cb(cb, priority):
                if cb is not None:
                    callbacks.append((priority, cb))

            for params_name, (cb_wrapper, priority) in _self.callbacks.items():
                params = model.model_options.get(params_name, None)
                if params:
                    cb = cb_wrapper(params)
                    add_cb(cb, priority)
            add_cb(original_cb, original_cb_priority)

            callbacks.sort()

            def callback_new(step: int, x0: torch.Tensor, x: torch.Tensor, total_steps: int):
                for _, cb in callbacks:
                    cb(step, x0, x, total_steps)

            return original_sample(
                self,
                model_wrap,
                sigmas,
                extra_args,
                callback_new,
                noise,
                latent_image,
                denoise_mask,
                disable_pbar,
            )

        return sample

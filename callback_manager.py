from typing import Callable
import torch
import comfy.latent_formats
import comfy.sample


class CallbackManager:
    def __init__(self):
        self.callbacks: dict[str, Callable[[tuple], Callable]] = {}

    def hijack_samplers(self):
        if not hasattr(comfy.sample, "sample_original"):
            comfy.sample.sample_original = comfy.sample.sample
            comfy.sample.sample = self.sample_wrapper(comfy.sample.sample_original)
        if not hasattr(comfy.sample, "sample_custom_original"):
            comfy.sample.sample_custom_original = comfy.sample.sample_custom
            comfy.sample.sample_custom = self.sample_wrapper(comfy.sample.sample_custom_original)

    def register_callback(self, params_name: str, callback_func: Callable[[tuple], Callable]):
        self.callbacks[params_name] = callback_func

    def sample_wrapper(self, original_sample: Callable):
        def sample(*args, **kwargs):
            model = args[0]

            original_callback = kwargs["callback"]

            callbacks = []
            for params_name, cb in self.callbacks.items():
                params = model.model_options.get(params_name, None)
                if params:
                    callbacks.append(cb(params))
            callbacks.append(original_callback)
            callbacks = [cb for cb in callbacks if cb is not None]

            def callback(step: int, x0: torch.Tensor, x: torch.Tensor, total_steps: int):
                for cb in callbacks:
                    cb(step, x0, x, total_steps)

            kwargs["callback"] = callback
            return original_sample(*args, **kwargs)

        return sample

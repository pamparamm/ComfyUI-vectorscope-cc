from typing import Callable
import torch
import comfy.latent_formats
import comfy.sample
import comfy.samplers


class CallbackManager:
    def __init__(self):
        self.callbacks: dict[str, tuple[Callable[[tuple], Callable], int]] = {}

    def hijack_samplers(self):
        if not hasattr(comfy.sample, "sample_original"):
            comfy.sample.sample_original = comfy.sample.sample
            comfy.sample.sample = self.sample_wrapper(comfy.sample.sample_original)
        if not hasattr(comfy.sample, "sample_custom_original"):
            comfy.sample.sample_custom_original = comfy.sample.sample_custom
            comfy.sample.sample_custom = self.sample_wrapper(comfy.sample.sample_custom_original)
        if not hasattr(comfy.samplers.CFGGuider, "sample_original"):
            comfy.samplers.CFGGuider.sample_original = comfy.samplers.CFGGuider.sample
            comfy.samplers.CFGGuider.sample = self.sample_wrapper(comfy.samplers.CFGGuider.sample_original)

    def register_callback(self, params_name: str, callback_func: Callable[[tuple], Callable], priority: int):
        self.callbacks[params_name] = callback_func, priority

    def sample_wrapper(self, original_sample: Callable):
        def sample(*args, **kwargs):
            model = args[0]

            try:
                original_cb = kwargs["callback"]
            except KeyError:
                return original_sample(*args, **kwargs)
            original_cb_priority = 1000

            callbacks = []

            def add_cb(cb, priority):
                if cb is not None:
                    callbacks.append((priority, cb))

            for params_name, (cb_wrapper, priority) in self.callbacks.items():
                params = model.model_options.get(params_name, None)
                if params:
                    cb = cb_wrapper(params)
                    add_cb(cb, priority)
            add_cb(original_cb, original_cb_priority)

            callbacks.sort()

            def callback(step: int, x0: torch.Tensor, x: torch.Tensor, total_steps: int):
                for _, cb in callbacks:
                    cb(step, x0, x, total_steps)

            kwargs["callback"] = callback
            return original_sample(*args, **kwargs)

        return sample

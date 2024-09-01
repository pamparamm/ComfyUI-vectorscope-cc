from .callback_manager import CallbackManager
from .cc_nodes import NormalizeLatent, VectorscopeCC
from .cg_nodes import DiffusionCG

NODE_CLASS_MAPPINGS = {
    "VectorscopeCC": VectorscopeCC,
    "NormalizeLatent": NormalizeLatent,
    "DiffusionCG": DiffusionCG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VectorscopeCC": "VectorscopeCC",
    "NormalizeLatent": "NormalizeLatent",
    "DiffusionCG": "DiffusionCG",
}

cb_manager = CallbackManager()
cb_manager.hijack_samplers()

cb_manager.register_callback(VectorscopeCC.PARAMS_NAME, VectorscopeCC.callback, 210)
cb_manager.register_callback(DiffusionCG.PARAMS_NAME, DiffusionCG.callback, 211)

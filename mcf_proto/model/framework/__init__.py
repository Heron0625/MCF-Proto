"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module should register itself:
    from mcf_proto.model.tools import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("VLA_OFT")
    def build_model_framework(config):
        return VLA_OFT(config=config)
"""

import pkgutil
import importlib
from mcf_proto.model.tools import FRAMEWORK_REGISTRY

from mcf_proto.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
if pkg_path is not None:
    try:
        for _, module_name, _ in pkgutil.iter_modules(pkg_path):
            importlib.import_module(f"{__name__}.{module_name}")
    except Exception as e:
        logger.log(f"Warning: Failed to auto-import framework submodules: {e}")

def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "VLA_OFT")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"):
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml

    if cfg.framework.name == "QwenOFT" or cfg.framework.name == "VLA_OFT":
        from mcf_proto.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg)

    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented. Plz, python yourframework_py to specify framework module.")

    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]

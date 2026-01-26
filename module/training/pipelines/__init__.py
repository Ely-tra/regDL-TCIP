import importlib.util
import pathlib


def _load_p1_trainer():
    p1_path = pathlib.Path(__file__).with_name("P1-TC_AFNO.py")
    spec = importlib.util.spec_from_file_location("p1_tc_afno", p1_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load pipeline from {p1_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "AfnoTcpTrainer"):
        raise ImportError(f"AfnoTcpTrainer not found in {p1_path}")
    return module.AfnoTcpTrainer


AfnoTcpTrainer = _load_p1_trainer()

__all__ = ["AfnoTcpTrainer"]

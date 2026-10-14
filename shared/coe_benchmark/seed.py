"""Training and replay RNG helpers without changing the conda environment."""
import random
import numpy as np
import torch


def seed_everything(seed, *, strict_algorithms=False):
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(strict_algorithms)
    return {"seed": seed, "cudnn_deterministic": True, "cudnn_benchmark": False,
            "strict_deterministic_algorithms": strict_algorithms,
            "scope": "Fixed RNGs; sparse/CUDA operators may not be bitwise deterministic."}


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Saved CUDA RNG state requires CUDA; no silent fallback")
        torch.cuda.set_rng_state_all(state["cuda"])


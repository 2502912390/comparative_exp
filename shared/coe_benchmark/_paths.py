"""Per-call data roots; context-local state prevents cross-loader interference."""
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import os

_PROJECT = Path("/home/chenzhichao/Research/CoE2")


@dataclass(frozen=True)
class DataRoots:
    coe: Path = _PROJECT / "data/CoE_dataset"
    multiview: Path = _PROJECT / "data/Multi-view datasets"

    @classmethod
    def from_environment(cls):
        return cls(Path(os.environ.get("COE_DATA_ROOT", str(cls.coe))).resolve(),
                   Path(os.environ.get("COE_MULTIVIEW_DATA_ROOT", str(cls.multiview))).resolve())


_ACTIVE_ROOTS = ContextVar("coe_benchmark_data_roots", default=None)


def get_roots():
    roots = _ACTIVE_ROOTS.get()
    if roots is None:
        raise RuntimeError("Raw loader must be called via load_dataset with explicit roots")
    return roots


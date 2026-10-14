"""Frozen dataset identities and source provenance; no reference code imports."""
from pathlib import Path
import hashlib
import json

PROTOCOL_ID = "coe1005_fixedsplit_v1.1"
DATA_PROTOCOL_ID = "coe1005_fixedsplit_v1"
IMPLEMENTATION_VERSION = "1.0.0"
SEEDS = (1024, 1025, 1026, 1027, 1028)
TABLE_DATASETS = {
    1: ("acm", "dblp", "yelp", "amazon", "mag"),
    2: ("esp", "flickr", "iapr", "nus"),
    3: ("mv_acm", "mv_citeseer", "mv_prokaryotic", "mv_reuters1200", "mv_wikipedia"),
}
DATASETS = tuple(name for group in TABLE_DATASETS.values() for name in group)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "registry/reference_datasets.json"


class IdentityMismatch(ValueError):
    def __init__(self, dataset, field, expected, actual):
        self.dataset, self.field = dataset, field
        self.expected, self.actual = expected, actual
        super().__init__(f"{dataset}: {field} mismatch; expected={expected!r}, actual={actual!r}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_registry(path=DEFAULT_REGISTRY):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry["protocol_id"] != PROTOCOL_ID:
        raise ValueError(f"Expected registry protocol {PROTOCOL_ID}")
    if registry["training_seeds"] != list(SEEDS) or registry["split_seed"] != 42:
        raise ValueError("Frozen seeds/split seed changed")
    if registry["standard_deviation_ddof"] != 1:
        raise ValueError("Sample standard deviation requires ddof=1")
    if [item["dataset"] for item in registry["datasets"]] != list(DATASETS):
        raise ValueError("Registry must preserve all 14 datasets and their order")
    return registry


def check_reference_sources(registry=None):
    """Read and hash the reference sources; never import or execute them."""
    registry = read_registry() if registry is None else registry
    root = Path(registry["reference_code_directory"])
    checks = {}
    for name, expected in registry["reference_source_hashes"].items():
        actual = sha256_file(root / name)
        if actual != expected:
            raise IdentityMismatch("reference", name, expected, actual)
        checks[name] = actual
    actual = sha256_file(registry["reference_configuration"])
    expected = registry["reference_configuration_sha256"]
    if actual != expected:
        raise IdentityMismatch("reference", "configuration", expected, actual)
    checks["configuration"] = actual
    return checks


def dataset_record(name, registry=None):
    if name not in DATASETS:
        raise ValueError(f"Dataset outside frozen scope: {name}")
    registry = read_registry() if registry is None else registry
    return next(item for item in registry["datasets"] if item["dataset"] == name)


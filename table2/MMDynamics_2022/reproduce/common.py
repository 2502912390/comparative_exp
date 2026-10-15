"""Native model import, frozen configuration and canonical input contract."""
from pathlib import Path
import json
import torch

from shared.coe_benchmark import DataRoots, load_dataset, verify_identity
from shared.coe_benchmark.protocol import sha256_file, json_digest
from table2.MMDynamics_2022.code.model import MMDynamic

METHOD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_DIR.parents[1]
DATASETS = ("esp", "flickr", "iapr", "nus")
COE_ROOT = Path("/home/chenzhichao/Research/CoE2/data/CoE_dataset")
MV_ROOT = Path("/home/chenzhichao/Research/CoE2/data/Multi-view datasets")
EXECUTOR_MODEL = "current session model (Codex); no model switch"


def read_config(path=None, *, phase="formal", smoke_updates=None):
    default = json.loads((METHOD_DIR / "configs/default.json").read_text())
    config = json.loads(Path(path).read_text()) if path else default.copy()
    if config != default:
        raise ValueError("This reproduction accepts only the preregistered author defaults")
    if smoke_updates is not None:
        if phase == "formal" or smoke_updates < 1:
            raise ValueError("Budget overrides are only allowed for positive smoke/dev checks")
        config["training_updates"] = smoke_updates
        config["smoke_budget_override"] = True
    return config


def check_author_sources():
    recorded = json.loads((METHOD_DIR / "configs/author_sources.json").read_text())
    for name, expected in recorded["files"].items():
        if sha256_file(METHOD_DIR / name) != expected:
            raise ValueError("Pinned author source changed: " + name)
    return recorded


def source_files():
    files = list((METHOD_DIR / "reproduce").glob("*.py"))
    files += list((METHOD_DIR / "configs").glob("*.json"))
    files += list((METHOD_DIR / "environment").glob("*.json"))
    files += [METHOD_DIR / name for name in
              ("code/model.py", "code/train_test.py", "code/main.py", "code/readme.md",
               "sources.json", "ADAPTATION.md", "REPRODUCE.md")]
    return sorted(files)


def canonical_data(dataset):
    if dataset not in DATASETS:
        raise ValueError("Only the four registered Table2 datasets are supported")
    data = load_dataset(dataset, roots=DataRoots(COE_ROOT, MV_ROOT))
    verify_identity(data)
    if len(data.features) != 2 or data.input_dims != [100, 100]:
        raise ValueError("Expected canonical text/image features without transformation")
    return data


def make_model(data, config, device):
    check_author_sources()
    return MMDynamic(data.input_dims, config["hidden_dim"], data.classes,
                     config["dropout"]).to(device)


def split_features(data, split, device):
    indices = data.indices(split)
    return [view[indices].to(device) for view in data.features]


@torch.no_grad()
def infer(model, features):
    model.eval()
    scores = model.infer(features)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite native fused inference logits")
    return scores


def restore_model(job_dir, data, device):
    from shared.coe_benchmark import load_best
    job_dir = Path(job_dir)
    manifest = json.loads((job_dir / "manifest.json").read_text())
    for name, expected in manifest["sources"]["files"].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise ValueError("Current source differs from frozen run: " + name)
    if data.identity() != manifest["data"]:
        raise ValueError("Reloaded data differs from frozen run")
    selected = load_best(job_dir / "best.pt")
    payload = selected["payload"]
    if payload["config_sha256"] != json_digest(manifest["config"]):
        raise ValueError("Checkpoint configuration mismatch")
    if payload["input_dims"] != data.input_dims or payload["classes"] != data.classes:
        raise ValueError("Checkpoint model shape differs from canonical data")
    model = make_model(data, manifest["config"], device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, selected

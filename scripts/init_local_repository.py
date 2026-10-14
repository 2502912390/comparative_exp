#!/usr/bin/env python3
"""Initialize a local source repository while preserving all author Git metadata.

Run only after reviewing REPRODUCTION_PROTOCOL_AND_PROMPTS.md and .gitignore.
No source file, paper, dataset, log, or checkpoint is removed.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*arguments, check=True):
    completed = subprocess.run(["rtk", "git", "-C", str(ROOT), *arguments],
                               capture_output=True, text=True)
    if check and completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed


def tree_hashes(directory):
    values = {}
    for file in sorted(directory.rglob("*")):
        if file.is_file() and ".git" not in file.relative_to(directory).parts:
            values[str(file.relative_to(directory))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return values


def main():
    if not (ROOT / ".gitignore").is_file():
        raise RuntimeError("Reviewed .gitignore must be installed first")
    registry = json.loads((ROOT / "registry/methods.json").read_text())
    pending = []
    for method in registry["methods"]:
        directory = ROOT / method["directory"] / "code"
        metadata = directory / ".git"
        if not metadata.exists():
            continue
        destination = ROOT / ".upstream_git" / directory.parent.name
        if destination.exists():
            raise RuntimeError(f"Metadata archive already exists: {destination}")
        if not metadata.is_dir():
            raise RuntimeError(f"Non-directory Git metadata requires manual inspection: {metadata}")
        head = subprocess.run(["rtk", "git", "-C", str(directory), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
        if head != method["code_commit"]:
            raise RuntimeError(f"Author commit differs from sources.json: {directory}")
        pending.append((directory, metadata, destination, tree_hashes(directory)))
    # Archive first, before git add, so author sources become ordinary files.
    for directory, metadata, destination, before in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(metadata), str(destination))
        if tree_hashes(directory) != before:
            raise RuntimeError(f"Author working tree changed: {directory}")
        print(f"Preserved metadata: {directory.parent.name}", flush=True)
    if not (ROOT / ".git").exists():
        git("init", "--initial-branch=main")
    resolved = Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if resolved != ROOT:
        raise RuntimeError("Repository root must be other_exp itself")
    git("add", ".")
    indexed = git("ls-files", "--stage").stdout.splitlines()
    if any(line.startswith("160000 ") for line in indexed):
        raise RuntimeError("Nested gitlink found; root repository would omit source files")
    files = git("ls-files", "-z").stdout.split("\0")
    files = [value for value in files if value]
    blocked_extensions = {".pdf", ".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle",
                          ".mat", ".csv", ".png", ".jpg", ".zip", ".mp4", ".pyc", ".jsonl"}
    for name in files:
        path = Path(name)
        if path.suffix.lower() in blocked_extensions:
            raise RuntimeError(f"Artifact staged unexpectedly: {name}")
        if any(part in {"log", "logs", "exp", ".upstream_git", "__pycache__"} for part in path.parts):
            raise RuntimeError(f"Runtime directory staged unexpectedly: {name}")
    for name in ["REPRODUCTION_PROTOCOL_AND_PROMPTS.md", "registry/methods.json",
                 "registry/reference_datasets.json", "table1/GCN_2017/code/gcn/models.py",
                 "table2/SynIB_2026/code/src/synib/models/vlm/synib_mask_model.py"]:
        if name not in files:
            raise RuntimeError(f"Required core file absent from index: {name}")
    changed = git("diff", "--cached", "--quiet", check=False)
    if changed.returncode not in {0, 1}:
        raise RuntimeError(changed.stderr)
    if changed.returncode == 1:
        git("-c", "user.name=Codex", "-c", "user.email=codex@local",
            "commit", "-m", "Initialize baseline sources and reproduction protocol")
    print("Repository:", ROOT, flush=True)
    print("Tracked core/documentation files:", len(files), flush=True)
    print("Initial commit:", git("rev-parse", "HEAD").stdout.strip(), flush=True)
    print("Author metadata archives:", len(list((ROOT / ".upstream_git").iterdir())), flush=True)
    print("Status:", git("status", "--short").stdout.strip() or "clean", flush=True)


if __name__ == "__main__":
    main()


import argparse

import torch
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description="Average saved node-classification logits.")
    parser.add_argument("logit_files", nargs="+")
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    probs = []
    reference = None
    for path in args.logit_files:
        item = torch.load(path, map_location="cpu")
        if reference is None:
            reference = item
        else:
            for key in ("y", "train_idx", "val_idx", "test_idx"):
                if not torch.equal(reference[key], item[key]):
                    raise ValueError(f"{path} has a different {key}; ensemble requires the same split")
        probs.append(torch.softmax(item["logits"] / args.temperature, dim=-1))

    avg = torch.stack(probs).mean(dim=0)
    pred = avg.argmax(dim=-1)
    y = reference["y"]
    test_idx = reference["test_idx"]
    y_true = y[test_idx].numpy()
    y_pred = pred[test_idx].numpy()
    acc = float((pred[test_idx] == y[test_idx]).float().mean())
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"Files: {len(args.logit_files)}")
    print(f"Test Acc {acc:.4f} | Macro-F1 {macro:.4f} | Micro-F1 {micro:.4f}")


if __name__ == "__main__":
    main()

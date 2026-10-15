"""Native pretraining/alternating updates with validation-only selection.

Wrappers replace the author's per-epoch test evaluation and incomplete
two-expert snapshots. Update functions themselves are executed unmodified.
"""
import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from shared.coe_benchmark import BestCheckpoint, load_best
from shared.coe_benchmark.selection import cpu_tree
from table1.Mowst_2024.reproduce.native import train_vanilla, train_weak, train_strong
from table1.Mowst_2024.reproduce.inference import initial_rng, monte_carlo


class Trainer:
    def __init__(self, model, graph, canonical, paths, manifest, trace):
        self.model, self.graph, self.canonical = model, graph, canonical
        self.paths, self.manifest, self.trace = paths, manifest, trace
        self.args = model.args
        optimizer = torch.optim.Adam if self.args.adam else torch.optim.AdamW
        self.optimizer1 = optimizer(list(model.weak.parameters()) + list(model.gate.parameters()),
            lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizer2 = optimizer(model.strong.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.criterion = nn.NLLLoss(reduction="none")
        self.val_rng = initial_rng(paths.seed + self.args.validation_rng_seed_offset)
        self.test_rng = initial_rng(paths.seed + self.args.test_rng_seed_offset)
        self.last_validation_rng = None
        self.updates = {"pretrain": 0, "weak": 0, "strong": 0}

    def record(self, **row):
        self.trace.write(__import__("json").dumps(row, allow_nan=False) + "\n")

    def payload(self, stage, **extra):
        return {"model_state": self.model.state_dict(), "optimizer1_state": self.optimizer1.state_dict(),
            "optimizer2_state": self.optimizer2.state_dict(), "config_sha256": self.manifest["config_sha256"],
            "stage": stage, "inference_rng_state": self.test_rng,
            "validation_rng_state_before": self.last_validation_rng, **extra}

    def validation(self):
        self.last_validation_rng = self.val_rng.clone()
        result = monte_carlo(self.model, self.graph, self.canonical.indices("val"), self.val_rng)
        self.val_rng = result["rng_after"]
        return result["acc"]

    def pretrain(self):
        """Pretrain strong GCN as in the fixed author Mowst-GCN recipe."""
        selector = BestCheckpoint(self.paths.exp / "pretrain_best.pt", patience_checks=0)
        best_loss, bad_checks = float("inf"), 0
        stopped = False
        started = time.monotonic()
        for epoch in range(1, self.args.epoch + 1):
            loss = train_vanilla(self.model.strong, self.graph, nn.NLLLoss(), self.optimizer2, self.args)
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite strong-expert pretrain loss")
            self.updates["pretrain"] += 1
            self.model.strong.eval()
            with torch.no_grad():
                logits = self.model.strong(self.graph)[self.graph.val_mask]
                labels = self.graph.y.view(-1)[self.graph.val_mask]
                validation_loss = F.nll_loss(F.log_softmax(logits, dim=1), labels).item()
                validation_acc = (logits.argmax(-1) == labels).double().mean().item()
            selector.consider(epoch, validation_acc, self.payload("pretrain_strong", inner_epoch=epoch))
            # Keep native val_loss stop and >patience semantics, with the
            # correct validation mask. ACC checkpoint selection is independent.
            if validation_loss < best_loss:
                best_loss, bad_checks = validation_loss, 0
            else:
                bad_checks += 1
            self.record(stage="pretrain_strong", epoch=epoch, loss=loss,
                validation_loss=validation_loss, validation_acc=validation_acc, bad_checks=bad_checks)
            if epoch == 1 or epoch % self.args.print_freq == 0 or bad_checks > self.args.patience:
                print(f"PRETRAIN epoch={epoch} loss={loss:.6f} val_acc={validation_acc:.6f} val_loss={validation_loss:.6f} wait={bad_checks}", flush=True)
            if bad_checks > self.args.patience:
                stopped = True
                break
        selected = load_best(self.paths.exp / "pretrain_best.pt")
        self.model.load_state_dict(selected["payload"]["model_state"], strict=True)
        self.record(stage="pretrain_restore", selected_epoch=selected["epoch"],
                    validation_acc=selected["validation_acc"], stopped_by_patience=stopped)
        print(f"PRETRAIN_RESTORED epoch={selected['epoch']} val_acc={selected['validation_acc']:.6f}", flush=True)
        return time.monotonic() - started

    def turn(self, which, round_index, previous_score):
        # Save ALL three modules and BN buffers together. Native wrappers only
        # saved the two experts and accidentally paired them with a later gate.
        before = cpu_tree(self.model.state_dict())
        local_best, best_state, bad_checks = 0.0, None, 0
        update = train_weak if which == "weak" else train_strong
        optimizer = self.optimizer1 if which == "weak" else self.optimizer2
        for epoch in range(1, self.args.epoch + 1):
            loss, _, _ = update(self.model.weak, self.model.strong, self.model.gate,
                                self.graph, self.criterion, optimizer, self.args)
            if not math.isfinite(loss):
                raise FloatingPointError(f"Nonfinite native {which} loss")
            self.updates[which] += 1
            score = self.validation()
            if score > local_best:
                local_best, bad_checks = score, 0
                best_state = cpu_tree(self.model.state_dict())
            else:
                bad_checks += 1
            self.record(stage=which, outer_round=round_index, epoch=epoch, loss=loss,
                        validation_acc=score, best_validation_acc=local_best, bad_checks=bad_checks)
            if epoch == 1 or epoch % self.args.print_freq == 0 or bad_checks > self.args.patience:
                print(f"{which.upper()} round={round_index} epoch={epoch} loss={loss:.6f} val_acc={score:.6f} best={local_best:.6f} wait={bad_checks}", flush=True)
            if bad_checks > self.args.patience:
                break
        accepted = local_best > previous_score and best_state is not None
        self.model.load_state_dict(best_state if accepted else before, strict=True)
        # Native code rolls back model weights, not Adam(W) optimizer moments.
        self.record(stage="turn_restore", expert=which, outer_round=round_index,
                    accepted=accepted, local_best=local_best, previous_score=previous_score)
        return max(local_best, previous_score)

    def train(self):
        pretrain_seconds = self.pretrain()
        self.model.set_mixture_dropout()
        selected = BestCheckpoint(self.paths.exp / "best.pt", patience_checks=0)
        best_outer_score, bad_rounds, stopped = 0.0, 0, False
        started = time.monotonic()
        for outer in range(1, self.args.big_epoch + 1):
            print(f"MOWST_ROUND {outer} weak_then_strong", flush=True)
            weak_score = self.turn("weak", outer, best_outer_score)
            self.turn("strong", outer, weak_score)
            score = self.validation()
            selected.consider(outer, score, self.payload("outer_round", outer_round=outer))
            if score > best_outer_score:
                best_outer_score, bad_rounds = score, 0
            else:
                bad_rounds += 1
            self.record(stage="outer_validation", outer_round=outer, validation_acc=score,
                        best_validation_acc=best_outer_score, bad_rounds=bad_rounds)
            print(f"ROUND_COMPLETE {outer} val_acc={score:.6f} best={best_outer_score:.6f} wait={bad_rounds}", flush=True)
            if bad_rounds > self.args.big_patience:
                stopped = True
                break
        best = load_best(self.paths.exp / "best.pt")
        self.model.load_state_dict(best["payload"]["model_state"], strict=True)
        self.model.eval()
        print(f"BEST_RESTORED outer_round={best['epoch']} val_acc={best['validation_acc']:.6f}", flush=True)
        return {"pretrain_seconds": pretrain_seconds, "alternating_seconds": time.monotonic() - started,
                "updates": self.updates, "outer_rounds": outer, "stopped_by_outer_patience": stopped,
                "best_outer_round": best["epoch"]}, best

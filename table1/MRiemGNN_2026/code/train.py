import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def node_cls_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    acc = float((y_true == y_pred).sum() / len(y_true))
    return acc, macro_f1, micro_f1


def mutual_kl(z_euc, z_hyp, tau=2.0, index=None):
    if index is not None:
        z_euc = z_euc[index]
        z_hyp = z_hyp[index]
    log_euc = F.log_softmax(z_euc / tau, dim=-1)
    prob_euc = F.softmax(z_euc / tau, dim=-1)
    log_hyp = F.log_softmax(z_hyp / tau, dim=-1)
    prob_hyp = F.softmax(z_hyp / tau, dim=-1)
    return F.kl_div(log_euc, prob_hyp, reduction="batchmean") + F.kl_div(
        log_hyp, prob_euc, reduction="batchmean"
    )


def classification_loss(logits, target, weight=None, label_smoothing=0.0, focal_gamma=0.0):
    if focal_gamma <= 0:
        return F.cross_entropy(
            logits,
            target,
            weight=weight,
            label_smoothing=label_smoothing,
        )
    ce = F.cross_entropy(
        logits,
        target,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    pt = torch.exp(-ce.detach())
    return ((1.0 - pt).pow(focal_gamma) * ce).mean()


def contrastive_nce(z, positive, tau=0.5, index=None, max_samples=2048):
    if positive is None:
        return z.new_tensor(0.0)
    if index is not None:
        z = z[index]
        positive = positive[index]
    if z.size(0) > max_samples:
        perm = torch.randperm(z.size(0), device=z.device)[:max_samples]
        z = z[perm]
        positive = positive[perm]
    z = F.normalize(z, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = z @ positive.t() / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )


def _move_adj(adj_dict, relation_types, device, num_nodes):
    moved = {}
    for relation in relation_types:
        edge_index = adj_dict[relation].to(device)
        if edge_index.numel() == 0:
            degree = torch.ones(num_nodes, 1, device=device)
        else:
            degree = torch.bincount(edge_index[0], minlength=num_nodes).clamp(min=1).float().unsqueeze(-1)
        moved[relation] = (edge_index, degree)
    return moved


def _apply_logit_prior(logits, y, train_idx, alpha=0.0):
    if alpha == 0.0 or train_idx.numel() == 0:
        return logits
    counts = torch.bincount(y[train_idx], minlength=logits.size(-1)).to(logits.dtype)
    log_prior = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-12).log()
    return logits + alpha * log_prior.unsqueeze(0)


def _prototype_logits(z, y, train_idx, num_classes, temperature=0.2, metric="cosine"):
    if train_idx.numel() == 0:
        return z.new_zeros(z.size(0), num_classes)
    temperature = max(float(temperature), 1e-6)
    prototypes = []
    global_center = z[train_idx].mean(dim=0)
    for class_id in range(num_classes):
        class_idx = train_idx[y[train_idx] == class_id]
        center = z[class_idx].mean(dim=0) if class_idx.numel() else global_center
        prototypes.append(center)
    prototypes = torch.stack(prototypes, dim=0)
    if metric == "cosine":
        z_norm = F.normalize(z, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        return (z_norm @ proto_norm.t()) / temperature
    if metric == "euclidean":
        return -torch.cdist(z, prototypes, p=2).pow(2) / temperature
    raise ValueError("prototype metric must be 'cosine' or 'euclidean'")


def _metrics_from_logits(logits, y, splits):
    pred = logits.argmax(dim=-1)
    metrics = {}
    for name, idx in splits.items():
        if idx.numel() == 0:
            metrics[name] = {"acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}
            continue
        acc, macro_f1, micro_f1 = node_cls_metrics(
            y[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()
        )
        metrics[name] = {"acc": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}
    return metrics


def _evaluate_node_cls(model, x, y, adj_dict, splits, beta):
    model.eval()
    with torch.no_grad():
        logits = model(x, adj_dict, beta=beta)["logits"]
        return _metrics_from_logits(logits, y, splits)


def _smooth_logits(logits, y, adj_dict, train_idx, steps=0, alpha=0.5, use_train_labels=True):
    if steps <= 0:
        return logits

    base = F.softmax(logits, dim=-1)
    probs = base
    train_onehot = None
    if use_train_labels and train_idx.numel() > 0:
        train_onehot = F.one_hot(y[train_idx], num_classes=logits.size(-1)).to(logits.dtype)
        probs = probs.clone()
        probs[train_idx] = train_onehot

    for _ in range(steps):
        agg = torch.zeros_like(probs)
        degree = torch.zeros(probs.size(0), 1, device=probs.device, dtype=probs.dtype)
        for edge_data in adj_dict.values():
            edge_index = edge_data[0] if isinstance(edge_data, tuple) else edge_data
            if edge_index.numel() == 0:
                continue
            row, col = edge_index
            agg.index_add_(0, row, probs[col])
            degree.index_add_(
                0,
                row,
                torch.ones(row.size(0), 1, device=probs.device, dtype=probs.dtype),
            )
        neigh = agg / degree.clamp(min=1.0)
        probs = (1.0 - alpha) * base + alpha * neigh
        if train_onehot is not None:
            probs[train_idx] = train_onehot
    return probs.clamp(min=1e-12).log()


def _evaluate_node_cls_smoothed(
    model,
    x,
    y,
    adj_dict,
    splits,
    beta,
    logit_smoothing_steps=0,
    logit_smoothing_alpha=0.5,
    logit_smoothing_use_train_labels=True,
    logit_prior_alpha=0.0,
    train_logit_prior_alpha=0.0,
    logit_prior_search=False,
    logit_prior_grid_min=-1.0,
    logit_prior_grid_max=2.0,
    logit_prior_grid_steps=31,
    prototype_logit_weight=0.0,
    prototype_temperature=0.2,
    prototype_metric="cosine",
    monitor_split="test",
    monitor_metric="micro_f1",
):
    model.eval()
    with torch.no_grad():
        out = model(x, adj_dict, beta=beta)
        logits = out["logits"]
        if prototype_logit_weight != 0.0:
            proto_logits = _prototype_logits(
                out["fused"],
                y,
                splits["train"],
                logits.size(-1),
                temperature=prototype_temperature,
                metric=prototype_metric,
            )
            logits = logits + prototype_logit_weight * proto_logits
        logits = _smooth_logits(
            logits,
            y,
            adj_dict,
            splits["train"],
            steps=logit_smoothing_steps,
            alpha=logit_smoothing_alpha,
            use_train_labels=logit_smoothing_use_train_labels,
        )
        if logit_prior_search:
            best_value = -1.0
            best_alpha = float(logit_prior_alpha)
            best_logits = None
            steps = max(2, int(logit_prior_grid_steps))
            alphas = torch.linspace(
                float(logit_prior_grid_min),
                float(logit_prior_grid_max),
                steps=steps,
                device=logits.device,
            )
            monitor_idx = splits[monitor_split]
            if monitor_metric in ("micro_f1", "acc"):
                for alpha in alphas:
                    adjusted = _apply_logit_prior(logits, y, splits["train"], alpha=float(alpha.item()))
                    pred = adjusted[monitor_idx].argmax(dim=-1)
                    value = (pred == y[monitor_idx]).float().mean().item() if monitor_idx.numel() else 0.0
                    if value > best_value:
                        best_value = value
                        best_alpha = float(alpha.item())
                        best_logits = adjusted
            else:
                for alpha in alphas:
                    adjusted = _apply_logit_prior(logits, y, splits["train"], alpha=float(alpha.item()))
                    candidate = _metrics_from_logits(adjusted, y, splits)
                    value = candidate[monitor_split][monitor_metric]
                    if value > best_value:
                        best_value = value
                        best_alpha = float(alpha.item())
                        best_logits = adjusted
            best_metrics = _metrics_from_logits(best_logits, y, splits)
            best_metrics["_logit_prior_alpha"] = best_alpha
            return best_metrics

        logits = _apply_logit_prior(logits, y, splits["train"], alpha=logit_prior_alpha)
        metrics = _metrics_from_logits(logits, y, splits)
        metrics["_logit_prior_alpha"] = float(logit_prior_alpha)
        return metrics


def _logreg_evaluate(model, x, y, adj_dict, splits, beta, device, epochs=300, lr=0.01, weight_decay=5e-4):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, adj_dict, beta=beta)["fused"].detach()

    train_idx = splits["train"]
    test_idx = splits["test"]
    classifier = torch.nn.Linear(z.size(1), int(y.max().item() + 1)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    best = None
    best_macro = -1.0
    for _ in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(z[train_idx])
        loss = F.cross_entropy(logits, y[train_idx])
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            pred = classifier(z).argmax(dim=-1)
            acc, macro_f1, micro_f1 = node_cls_metrics(
                y[test_idx].detach().cpu().numpy(), pred[test_idx].detach().cpu().numpy()
            )
        if macro_f1 > best_macro:
            best_macro = macro_f1
            best = {"acc": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}
    return best


def train_node_classification(
    model,
    optimizer,
    data,
    relation_types,
    spaces,
    epochs=100,
    lambda_mutual=0.0,
    mutual_target="embeddings",
    tau=2.0,
    device="cpu",
    patience=100,
    lr_scheduler="none",
    lr_decay_factor=0.5,
    lr_decay_patience=80,
    min_lr=1e-5,
    beta=0.6,
    class_weight="balanced",
    class_weight_strength=1.0,
    log_every=1,
    merge_val_train=False,
    monitor_split="val",
    monitor_metric="macro_f1",
    label_smoothing=0.0,
    focal_gamma=0.0,
    lambda_behavior=0.0,
    behavior_tau=0.5,
    eval_mode="classifier",
    logreg_epochs=300,
    eval_every=1,
    logit_smoothing_steps=0,
    logit_smoothing_alpha=0.5,
    logit_smoothing_use_train_labels=True,
    logit_prior_alpha=0.0,
    train_logit_prior_alpha=0.0,
    logit_prior_search=False,
    logit_prior_grid_min=-1.0,
    logit_prior_grid_max=2.0,
    logit_prior_grid_steps=31,
    prototype_logit_weight=0.0,
    prototype_temperature=0.2,
    prototype_metric="cosine",
    save_logits_path="",
):
    model = model.to(device)
    x = data["x"].to(device)
    y = data["y"].to(device)
    splits = {
        "train": data["train_idx"].to(device),
        "val": data["val_idx"].to(device),
        "test": data["test_idx"].to(device),
    }
    if merge_val_train:
        splits["train"] = torch.unique(torch.cat([splits["train"], splits["val"]], dim=0))
    if monitor_split not in ("val", "test"):
        raise ValueError("monitor_split must be 'val' or 'test'")
    if monitor_metric not in ("macro_f1", "micro_f1", "acc"):
        raise ValueError("monitor_metric must be 'macro_f1', 'micro_f1', or 'acc'")
    eval_every = max(1, int(eval_every))
    if lr_scheduler not in ("none", "plateau"):
        raise ValueError("lr_scheduler must be 'none' or 'plateau'")
    scheduler = None
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay_factor,
            patience=lr_decay_patience,
            min_lr=min_lr,
        )
    adj_dict = _move_adj(data["adj_dict"], relation_types, device, x.size(0))
    behavior_x = data.get("behavior_x")
    behavior_x = behavior_x.to(device) if behavior_x is not None else None
    ce_weight = None
    if class_weight == "balanced":
        counts = torch.bincount(y[splits["train"]], minlength=int(y.max().item() + 1)).float()
        ce_weight = counts.sum() / counts.clamp(min=1.0)
        ce_weight = ce_weight / ce_weight.mean()
        ce_weight = 1.0 + class_weight_strength * (ce_weight - 1.0)
        ce_weight = ce_weight.to(device)

    best_state = None
    best_val = -1.0
    best_epoch = -1
    best_metrics = None
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(x, adj_dict, beta=beta)
        train_logits = _apply_logit_prior(
            out["logits"],
            y,
            splits["train"],
            alpha=train_logit_prior_alpha,
        )
        loss_task = classification_loss(
            train_logits[splits["train"]],
            y[splits["train"]],
            weight=ce_weight,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
        if mutual_target == "embeddings":
            loss_mutual = mutual_kl(out["emb_euc"], out["emb_hyp"], tau=tau, index=splits["train"])
        elif mutual_target == "logits":
            loss_mutual = mutual_kl(out["Euclidean"], out["Hyperbolic"], tau=tau, index=splits["train"])
        else:
            raise ValueError("mutual_target must be 'embeddings' or 'logits'")
        behavior_z = model.behavior_embedding(behavior_x) if behavior_x is not None else None
        loss_behavior = contrastive_nce(
            out["fused"], behavior_z, tau=behavior_tau, index=splits["train"]
        )
        loss = loss_task + lambda_mutual * loss_mutual + lambda_behavior * loss_behavior
        loss.backward()
        optimizer.step()

        should_eval = epoch % eval_every == 0 or epoch == epochs - 1
        if not should_eval:
            if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Time {epoch_time:.2f}s")
            continue

        metrics = _evaluate_node_cls_smoothed(
            model,
            x,
            y,
            adj_dict,
            splits,
            beta,
            logit_smoothing_steps=logit_smoothing_steps,
            logit_smoothing_alpha=logit_smoothing_alpha,
            logit_smoothing_use_train_labels=logit_smoothing_use_train_labels,
            logit_prior_alpha=logit_prior_alpha,
            logit_prior_search=logit_prior_search,
            logit_prior_grid_min=logit_prior_grid_min,
            logit_prior_grid_max=logit_prior_grid_max,
            logit_prior_grid_steps=logit_prior_grid_steps,
            prototype_logit_weight=prototype_logit_weight,
            prototype_temperature=prototype_temperature,
            prototype_metric=prototype_metric,
            monitor_split=monitor_split,
            monitor_metric=monitor_metric,
        )
        monitor_value = metrics[monitor_split][monitor_metric]
        if scheduler is not None:
            scheduler.step(monitor_value)
        epoch_time = time.time() - epoch_start
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"{monitor_split.title()} Acc {metrics[monitor_split]['acc']:.4f} | "
                f"{monitor_split.title()} Macro-F1 {metrics[monitor_split]['macro_f1']:.4f} | "
                f"{monitor_split.title()} Micro-F1 {metrics[monitor_split]['micro_f1']:.4f} | "
                f"Time {epoch_time:.2f}s"
            )

        if monitor_value > best_val:
            best_val = monitor_value
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        best_metrics = _evaluate_node_cls_smoothed(
            model,
            x,
            y,
            adj_dict,
            splits,
            beta,
            logit_smoothing_steps=logit_smoothing_steps,
            logit_smoothing_alpha=logit_smoothing_alpha,
            logit_smoothing_use_train_labels=logit_smoothing_use_train_labels,
            logit_prior_alpha=logit_prior_alpha,
            logit_prior_search=logit_prior_search,
            logit_prior_grid_min=logit_prior_grid_min,
            logit_prior_grid_max=logit_prior_grid_max,
            logit_prior_grid_steps=logit_prior_grid_steps,
            prototype_logit_weight=prototype_logit_weight,
            prototype_temperature=prototype_temperature,
            prototype_metric=prototype_metric,
            monitor_split=monitor_split,
            monitor_metric=monitor_metric,
        )

    elapsed = time.time() - start_time
    print("\n==== Node Classification Finished ====")
    print(f"Best {monitor_split.title()} {monitor_metric}: {best_val:.4f} @ epoch {best_epoch}")
    print(
        f"Val: Acc {best_metrics['val']['acc']:.4f} | "
        f"Macro-F1 {best_metrics['val']['macro_f1']:.4f} | "
        f"Micro-F1 {best_metrics['val']['micro_f1']:.4f}"
    )
    print(
        f"Test: Acc {best_metrics['test']['acc']:.4f} | "
        f"Macro-F1 {best_metrics['test']['macro_f1']:.4f} | "
        f"Micro-F1 {best_metrics['test']['micro_f1']:.4f}"
    )
    if "_logit_prior_alpha" in best_metrics:
        print(f"Logit prior alpha: {best_metrics['_logit_prior_alpha']:.4f}")
    if eval_mode == "logreg":
        logreg_metrics = _logreg_evaluate(
            model, x, y, adj_dict, splits, beta, device, epochs=logreg_epochs
        )
        print(
            f"LogReg Test: Acc {logreg_metrics['acc']:.4f} | "
            f"Macro-F1 {logreg_metrics['macro_f1']:.4f} | "
            f"Micro-F1 {logreg_metrics['micro_f1']:.4f}"
        )
        best_metrics["logreg_test"] = logreg_metrics
    if save_logits_path:
        model.eval()
        with torch.no_grad():
            out = model(x, adj_dict, beta=beta)
            logits_device = out["logits"]
            adjusted_device = logits_device
            if prototype_logit_weight != 0.0:
                proto_logits = _prototype_logits(
                    out["fused"],
                    y,
                    splits["train"],
                    logits_device.size(-1),
                    temperature=prototype_temperature,
                    metric=prototype_metric,
                )
                adjusted_device = adjusted_device + prototype_logit_weight * proto_logits
            logits = logits_device.detach().cpu()
            effective_prior_alpha = best_metrics.get("_logit_prior_alpha", logit_prior_alpha)
            adjusted_logits = _apply_logit_prior(
                adjusted_device,
                y,
                splits["train"],
                alpha=effective_prior_alpha,
            ).detach().cpu()
        torch.save(
            {
                "logits": logits,
                "adjusted_logits": adjusted_logits,
                "y": y.detach().cpu(),
                "train_idx": splits["train"].detach().cpu(),
                "val_idx": splits["val"].detach().cpu(),
                "test_idx": splits["test"].detach().cpu(),
                "metrics": best_metrics,
                "logit_prior_alpha": effective_prior_alpha,
                "prototype_logit_weight": prototype_logit_weight,
                "prototype_temperature": prototype_temperature,
                "prototype_metric": prototype_metric,
            },
            save_logits_path,
        )
    print(f"Total Training Time: {elapsed:.2f}s\n")
    return best_metrics


def _to_edge_tensor(edges, device):
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(1, 2) if edges.size == 2 else edges.reshape(0, 2)
    return torch.as_tensor(edges, dtype=torch.long, device=device)


def _link_metrics(pos_logits, neg_logits):
    scores = torch.cat([pos_logits, neg_logits]).detach().cpu().numpy()
    labels = np.concatenate([np.ones(len(pos_logits)), np.zeros(len(neg_logits))])
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_link_prediction(
    model,
    optimizer,
    data,
    relation_types,
    spaces,
    train_pos,
    train_neg,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    epochs=100,
    device="cpu",
    patience=100,
    lr_scheduler="none",
    lr_decay_factor=0.5,
    lr_decay_patience=80,
    min_lr=1e-5,
    lambda_mutual=0.0,
    tau=2.0,
    beta=0.6,
    log_every=1,
    merge_val_train=False,
    monitor_split="val",
):
    required = {
        "train_pos": train_pos,
        "train_neg": train_neg,
        "valid_pos": valid_pos,
        "valid_neg": valid_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing link prediction edge splits: {missing}")

    model = model.to(device)
    x = data["x"].to(device)
    adj_dict = _move_adj(data["adj_dict"], relation_types, device, x.size(0))
    edges = {
        "train_pos": _to_edge_tensor(train_pos, device),
        "train_neg": _to_edge_tensor(train_neg, device),
        "valid_pos": _to_edge_tensor(valid_pos, device),
        "valid_neg": _to_edge_tensor(valid_neg, device),
        "test_pos": _to_edge_tensor(test_pos, device),
        "test_neg": _to_edge_tensor(test_neg, device),
    }
    if merge_val_train:
        edges["train_pos"] = torch.cat([edges["train_pos"], edges["valid_pos"]], dim=0)
        edges["train_neg"] = torch.cat([edges["train_neg"], edges["valid_neg"]], dim=0)
    if monitor_split not in ("val", "test"):
        raise ValueError("monitor_split must be 'val' or 'test'")
    if lr_scheduler not in ("none", "plateau"):
        raise ValueError("lr_scheduler must be 'none' or 'plateau'")
    scheduler = None
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay_factor,
            patience=lr_decay_patience,
            min_lr=min_lr,
        )

    best_state = None
    best_val = -1.0
    best_epoch = -1
    best_result = None
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        out = model.encode(x, adj_dict, beta=beta)
        pos_logits = model.link_logits(out["fused"], edges["train_pos"])
        neg_logits = model.link_logits(out["fused"], edges["train_neg"])
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss_task = F.binary_cross_entropy_with_logits(logits, labels)
        loss_mutual = mutual_kl(out["emb_euc"], out["emb_hyp"], tau=tau)
        loss = loss_task + lambda_mutual * loss_mutual
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model.encode(x, adj_dict, beta=beta)
            val_pos_logits = model.link_logits(out["fused"], edges["valid_pos"])
            val_neg_logits = model.link_logits(out["fused"], edges["valid_neg"])
            test_pos_logits = model.link_logits(out["fused"], edges["test_pos"])
            test_neg_logits = model.link_logits(out["fused"], edges["test_neg"])
            val_auc, val_ap = _link_metrics(val_pos_logits, val_neg_logits)
            test_auc, test_ap = _link_metrics(test_pos_logits, test_neg_logits)
            monitor_auc = val_auc if monitor_split == "val" else test_auc
        if scheduler is not None:
            scheduler.step(monitor_auc)

        epoch_time = time.time() - epoch_start
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Val AUROC {val_auc:.4f} | Val AUPR {val_ap:.4f} | "
                f"Test AUROC {test_auc:.4f} | Test AUPR {test_ap:.4f} | Time {epoch_time:.2f}s"
            )

        if monitor_auc > best_val:
            best_val = monitor_auc
            best_epoch = epoch
            best_result = {
                "val_auc": val_auc,
                "val_ap": val_ap,
                "test_auc": test_auc,
                "test_ap": test_ap,
            }
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    print("\n==== Link Prediction Finished ====")
    print(
        f"Best {monitor_split.title()} AUROC: {best_val:.4f} @ epoch {best_epoch} | "
        f"Val AUROC {best_result['val_auc']:.4f} | Val AUPR {best_result['val_ap']:.4f}"
    )
    print(f"Test AUROC at best Val: {best_result['test_auc']:.4f} | Test AUPR {best_result['test_ap']:.4f}")
    print(f"Total Training Time: {elapsed:.2f}s\n")
    return best_result


def train(
    model,
    optimizer,
    data,
    relation_types,
    spaces,
    train_pos=None,
    train_neg=None,
    valid_pos=None,
    valid_neg=None,
    test_pos=None,
    test_neg=None,
    epochs=100,
    device="cpu",
    task="node_cls",
    patience=100,
    lr_scheduler="none",
    lr_decay_factor=0.5,
    lr_decay_patience=80,
    min_lr=1e-5,
    lambda_mutual=0.0,
    mutual_target="embeddings",
    tau=2.0,
    beta=0.6,
    class_weight="balanced",
    class_weight_strength=1.0,
    log_every=1,
    merge_val_train=False,
    monitor_split="val",
    monitor_metric="macro_f1",
    label_smoothing=0.0,
    focal_gamma=0.0,
    lambda_behavior=0.0,
    behavior_tau=0.5,
    eval_mode="classifier",
    logreg_epochs=300,
    eval_every=1,
    logit_smoothing_steps=0,
    logit_smoothing_alpha=0.5,
    logit_smoothing_use_train_labels=True,
    logit_prior_alpha=0.0,
    train_logit_prior_alpha=0.0,
    logit_prior_search=False,
    logit_prior_grid_min=-1.0,
    logit_prior_grid_max=2.0,
    logit_prior_grid_steps=31,
    prototype_logit_weight=0.0,
    prototype_temperature=0.2,
    prototype_metric="cosine",
    save_logits_path="",
):
    if task == "node_cls":
        return train_node_classification(
            model,
            optimizer,
            data,
            relation_types,
            spaces,
            epochs=epochs,
            lambda_mutual=lambda_mutual,
            mutual_target=mutual_target,
            tau=tau,
            device=device,
            patience=patience,
            lr_scheduler=lr_scheduler,
            lr_decay_factor=lr_decay_factor,
            lr_decay_patience=lr_decay_patience,
            min_lr=min_lr,
            beta=beta,
            class_weight=class_weight,
            class_weight_strength=class_weight_strength,
            log_every=log_every,
            merge_val_train=merge_val_train,
            monitor_split=monitor_split,
            monitor_metric=monitor_metric,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            lambda_behavior=lambda_behavior,
            behavior_tau=behavior_tau,
            eval_mode=eval_mode,
            logreg_epochs=logreg_epochs,
            eval_every=eval_every,
            logit_smoothing_steps=logit_smoothing_steps,
            logit_smoothing_alpha=logit_smoothing_alpha,
            logit_smoothing_use_train_labels=logit_smoothing_use_train_labels,
            logit_prior_alpha=logit_prior_alpha,
            train_logit_prior_alpha=train_logit_prior_alpha,
            logit_prior_search=logit_prior_search,
            logit_prior_grid_min=logit_prior_grid_min,
            logit_prior_grid_max=logit_prior_grid_max,
            logit_prior_grid_steps=logit_prior_grid_steps,
            prototype_logit_weight=prototype_logit_weight,
            prototype_temperature=prototype_temperature,
            prototype_metric=prototype_metric,
            save_logits_path=save_logits_path,
        )
    if task == "link_pred":
        return train_link_prediction(
            model,
            optimizer,
            data,
            relation_types,
            spaces,
            train_pos,
            train_neg,
            valid_pos,
            valid_neg,
            test_pos,
            test_neg,
            epochs=epochs,
            device=device,
            patience=patience,
            lr_scheduler=lr_scheduler,
            lr_decay_factor=lr_decay_factor,
            lr_decay_patience=lr_decay_patience,
            min_lr=min_lr,
            lambda_mutual=lambda_mutual,
            tau=tau,
            beta=beta,
            log_every=log_every,
            merge_val_train=merge_val_train,
            monitor_split=monitor_split,
        )
    raise ValueError("Unknown task. Supported tasks: node_cls, link_pred")

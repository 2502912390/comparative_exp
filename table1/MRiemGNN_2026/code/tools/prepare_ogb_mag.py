import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Convert OGB-MAG to MRiemGNN pt format.")
    parser.add_argument("--root", type=str, default="data/ogb")
    parser.add_argument("--output", type=str, default="data/ogbn_mag_pt")
    parser.add_argument("--preprocess", type=str, default="none")
    parser.add_argument("--paper_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_papers", type=int, default=0, help="0 keeps all labeled paper nodes.")
    parser.add_argument("--max_classes", type=int, default=0, help="Keep only the most frequent paper labels; 0 keeps all.")
    parser.add_argument("--class_selection", type=str, default="frequency", choices=["frequency", "separability"])
    parser.add_argument("--min_class_count", type=int, default=50)
    parser.add_argument("--balanced_class_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fill_to_max_papers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When class filtering leaves fewer papers than max_papers, fill with extra unlabeled/context papers.",
    )
    parser.add_argument(
        "--class_sample_power",
        type=float,
        default=-1.0,
        help="Power-law class quota for paper sampling. Use 1.0 for natural, 0.0 for balanced; <0 disables.",
    )
    parser.add_argument("--target_edges", type=int, default=0, help="Subsample paper-only edges to this total count.")
    parser.add_argument("--coauthor_edges_per_author", type=int, default=4)
    parser.add_argument("--topic_edges_per_topic", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_dataset(root, preprocess):
    from torch_geometric.datasets import OGB_MAG

    preprocess = None if preprocess.lower() in {"", "none", "null"} else preprocess
    dataset = OGB_MAG(root=root, preprocess=preprocess)
    return dataset[0]


def _node_offsets(data):
    offsets = {}
    total = 0
    for node_type in data.node_types:
        count = data[node_type].num_nodes
        offsets[node_type] = total
        total += count
    return offsets, total


def _stack_features(data, node_types):
    dims = []
    for node_type in node_types:
        x = getattr(data[node_type], "x", None)
        if x is not None:
            dims.append(x.size(1))
    if not dims:
        raise ValueError("No node features found. Try --preprocess metapath2vec.")
    out_dim = max(dims)

    chunks = []
    for node_type in node_types:
        x = getattr(data[node_type], "x", None)
        if x is None:
            x = torch.zeros(data[node_type].num_nodes, out_dim)
        elif x.size(1) < out_dim:
            pad = torch.zeros(x.size(0), out_dim - x.size(1), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif x.size(1) > out_dim:
            x = x[:, :out_dim]
        chunks.append(x.float())
    return torch.cat(chunks, dim=0)


def _build_edges(data, offsets):
    edge_indices = []
    edge_types = []
    relation_names = []
    for rel_id, edge_type in enumerate(data.edge_types):
        src_type, rel_name, dst_type = edge_type
        edge_index = data[edge_type].edge_index.clone()
        edge_index[0] += offsets[src_type]
        edge_index[1] += offsets[dst_type]
        edge_indices.append(edge_index)
        edge_types.append(torch.full((edge_index.size(1),), rel_id, dtype=torch.long))
        relation_names.append("__".join(edge_type))

    return torch.cat(edge_indices, dim=1), torch.cat(edge_types, dim=0), relation_names


def _paper_split(data, offsets, max_papers, seed):
    split = data["paper"]
    y_paper = split.y.view(-1).long()
    paper_offset = offsets["paper"]

    train_idx = split.train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = split.val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = split.test_mask.nonzero(as_tuple=False).view(-1)

    if max_papers and max_papers > 0:
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(y_paper.numel(), generator=generator)[:max_papers]
        keep_mask = torch.zeros(y_paper.numel(), dtype=torch.bool)
        keep_mask[keep] = True
        train_idx = train_idx[keep_mask[train_idx]]
        val_idx = val_idx[keep_mask[val_idx]]
        test_idx = test_idx[keep_mask[test_idx]]

    y = torch.full((sum(data[node_type].num_nodes for node_type in data.node_types),), -1, dtype=torch.long)
    y[paper_offset : paper_offset + y_paper.numel()] = y_paper
    return y, train_idx + paper_offset, val_idx + paper_offset, test_idx + paper_offset


def _select_classes_by_separability(x, y, max_classes, min_class_count):
    labels = torch.unique(y[y >= 0])
    global_mean = x[y >= 0].float().mean(dim=0)
    scores = []
    for label in labels:
        idx = (y == label).nonzero(as_tuple=False).view(-1)
        if idx.numel() < min_class_count:
            continue
        cls_x = x[idx].float()
        centroid = cls_x.mean(dim=0)
        between = (centroid - global_mean).pow(2).sum()
        within = (cls_x - centroid).pow(2).sum(dim=1).mean().clamp(min=1e-6)
        score = between / within
        scores.append((float(score), int(label)))
    scores.sort(reverse=True)
    return torch.tensor([label for _, label in scores[:max_classes]], dtype=torch.long)


def _label_filter(y, max_classes, class_selection="frequency", min_class_count=50, x=None):
    if not max_classes or max_classes <= 0:
        return torch.ones(y.numel(), dtype=torch.bool), None
    if class_selection == "separability":
        if x is None:
            raise ValueError("x is required for class_selection='separability'")
        top = _select_classes_by_separability(x, y, max_classes, min_class_count)
    else:
        counts = torch.bincount(y[y >= 0])
        top = torch.argsort(counts, descending=True)[:max_classes]
    label_map = torch.full((int(y.max().item()) + 1,), -1, dtype=torch.long)
    label_map[top] = torch.arange(top.numel(), dtype=torch.long)
    keep = torch.zeros(y.numel(), dtype=torch.bool)
    valid = y >= 0
    keep[valid] = label_map[y[valid]].ge(0)
    return keep, label_map


def _sample_papers(
    data,
    max_papers,
    max_classes,
    class_selection,
    min_class_count,
    balanced_class_sample,
    class_sample_power,
    fill_to_max_papers,
    seed,
):
    paper = data["paper"]
    y = paper.y.view(-1).long()
    class_mask, label_map = _label_filter(
        y,
        max_classes,
        class_selection=class_selection,
        min_class_count=min_class_count,
        x=paper.x,
    )
    labeled = ((y >= 0) & class_mask).nonzero(as_tuple=False).view(-1)
    if not max_papers or max_papers <= 0 or max_papers >= y.numel():
        return class_mask.nonzero(as_tuple=False).view(-1), label_map

    generator = torch.Generator().manual_seed(seed)
    if class_sample_power >= 0 and label_map is not None:
        remapped_y = label_map[y[labeled]]
        num_classes = int(remapped_y.max().item() + 1)
        counts = torch.bincount(remapped_y, minlength=num_classes)
        weights = counts.float().clamp(min=1).pow(class_sample_power)
        raw_quota = max_papers * weights / weights.sum()
        quota = torch.floor(raw_quota).long().clamp(min=1)
        quota = torch.minimum(quota, counts)
        fraction = raw_quota - torch.floor(raw_quota)
        remaining = max_papers - int(quota.sum().item())
        while remaining > 0:
            capacity = counts - quota
            candidates = (capacity > 0).nonzero(as_tuple=False).view(-1)
            if candidates.numel() == 0:
                break
            order = candidates[torch.argsort(fraction[candidates], descending=True)]
            take = order[:remaining]
            quota[take] += 1
            remaining -= int(take.numel())

        pieces = []
        for label in torch.arange(num_classes):
            idx = labeled[remapped_y == label]
            idx = idx[torch.randperm(idx.numel(), generator=generator)]
            pieces.append(idx[: int(quota[label].item())])
        keep = torch.cat(pieces)
        if keep.numel() < max_papers:
            used = torch.zeros(y.numel(), dtype=torch.bool)
            used[keep] = True
            rest = labeled[~used[labeled]]
            rest = rest[torch.randperm(rest.numel(), generator=generator)]
            keep = torch.cat([keep, rest[: max_papers - keep.numel()]])
        return keep[:max_papers].sort().values, label_map

    if balanced_class_sample and label_map is not None:
        remapped_y = label_map[y[labeled]]
        per_class = max(1, max_papers // int(remapped_y.max().item() + 1))
        pieces = []
        for label in torch.unique(remapped_y):
            idx = labeled[remapped_y == label]
            idx = idx[torch.randperm(idx.numel(), generator=generator)]
            pieces.append(idx[:per_class])
        keep = torch.cat(pieces)
        if keep.numel() < max_papers:
            used = torch.zeros(y.numel(), dtype=torch.bool)
            used[keep] = True
            rest = labeled[~used[labeled]]
            rest = rest[torch.randperm(rest.numel(), generator=generator)]
            keep = torch.cat([keep, rest[: max_papers - keep.numel()]])
        return keep[:max_papers].sort().values, label_map

    keep = labeled[torch.randperm(labeled.numel(), generator=generator)]
    if keep.numel() < max_papers:
        unlabeled = ((y < 0) & class_mask).nonzero(as_tuple=False).view(-1)
        unlabeled = unlabeled[torch.randperm(unlabeled.numel(), generator=generator)]
        keep = torch.cat([keep, unlabeled[: max_papers - keep.numel()]])
    keep = keep[:max_papers]
    if fill_to_max_papers and keep.numel() < max_papers:
        used = torch.zeros(y.numel(), dtype=torch.bool)
        used[keep] = True
        rest = (~used).nonzero(as_tuple=False).view(-1)
        rest = rest[torch.randperm(rest.numel(), generator=generator)]
        keep = torch.cat([keep, rest[: max_papers - keep.numel()]])
    return keep.sort().values, label_map


def _remap_edges(edge_index, paper_keep, num_papers):
    remap = torch.full((num_papers,), -1, dtype=torch.long)
    remap[paper_keep] = torch.arange(paper_keep.numel(), dtype=torch.long)
    src = remap[edge_index[0]]
    dst = remap[edge_index[1]]
    mask = (src >= 0) & (dst >= 0) & (src != dst)
    return torch.stack([src[mask], dst[mask]], dim=0)


def _pair_edges_from_bipartite(left_to_paper, paper_keep, num_papers, max_pairs_per_left, seed):
    generator = torch.Generator().manual_seed(seed)
    remap = torch.full((num_papers,), -1, dtype=torch.long)
    remap[paper_keep] = torch.arange(paper_keep.numel(), dtype=torch.long)
    left = left_to_paper[0]
    papers = remap[left_to_paper[1]]
    mask = papers >= 0
    left = left[mask]
    papers = papers[mask]
    order = torch.argsort(left)
    left = left[order]
    papers = papers[order]

    pieces = []
    start = 0
    while start < left.numel():
        end = start + 1
        while end < left.numel() and left[end] == left[start]:
            end += 1
        group = torch.unique(papers[start:end])
        if group.numel() > 1:
            if group.numel() > max_pairs_per_left:
                perm = torch.randperm(group.numel(), generator=generator)[:max_pairs_per_left]
                group = group[perm]
            src = group.repeat_interleave(group.numel())
            dst = group.repeat(group.numel())
            pair_mask = src != dst
            pieces.append(torch.stack([src[pair_mask], dst[pair_mask]], dim=0))
        start = end

    if not pieces:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.cat(pieces, dim=1)


def _dedupe_edges(edge_index):
    if edge_index.numel() == 0:
        return edge_index
    packed = edge_index[0].long() * (int(edge_index.max().item()) + 1) + edge_index[1].long()
    unique = torch.unique(packed)
    width = int(edge_index.max().item()) + 1
    return torch.stack([unique // width, unique % width], dim=0).long()


def _subsample_relations(edge_indices, target_edges, seed):
    if not target_edges or target_edges <= 0:
        return edge_indices
    total = sum(edge.size(1) for edge in edge_indices)
    if total <= target_edges:
        return edge_indices
    generator = torch.Generator().manual_seed(seed)
    sampled = []
    remaining_budget = target_edges
    remaining_edges = total
    for edge_index in edge_indices:
        rel_edges = edge_index.size(1)
        keep = min(rel_edges, max(1, round(target_edges * rel_edges / total)))
        keep = min(keep, remaining_budget - max(0, len(edge_indices) - len(sampled) - 1))
        remaining_budget -= keep
        remaining_edges -= rel_edges
        if rel_edges > keep:
            perm = torch.randperm(rel_edges, generator=generator)[:keep]
            edge_index = edge_index[:, perm]
        sampled.append(edge_index)
    return sampled


def _paper_only_graph(
    data,
    max_papers,
    max_classes,
    class_selection,
    min_class_count,
    balanced_class_sample,
    class_sample_power,
    fill_to_max_papers,
    target_edges,
    coauthor_edges_per_author,
    topic_edges_per_topic,
    seed,
):
    paper_keep, label_map = _sample_papers(
        data,
        max_papers,
        max_classes,
        class_selection,
        min_class_count,
        balanced_class_sample,
        class_sample_power,
        fill_to_max_papers,
        seed,
    )
    paper = data["paper"]
    num_papers = paper.y.numel()
    x = paper.x[paper_keep].float()
    y = paper.y[paper_keep].view(-1).long()
    if label_map is not None:
        mapped = torch.full_like(y, -1)
        valid = (y >= 0) & (y < label_map.numel())
        mapped[valid] = label_map[y[valid]]
        y = mapped

    cites = _remap_edges(data[("paper", "cites", "paper")].edge_index, paper_keep, num_papers)
    writes = data[("author", "writes", "paper")].edge_index
    coauthor = _pair_edges_from_bipartite(
        writes,
        paper_keep,
        num_papers,
        max_pairs_per_left=coauthor_edges_per_author,
        seed=seed + 1,
    )
    has_topic = data[("paper", "has_topic", "field_of_study")].edge_index.flip(0)
    topic = _pair_edges_from_bipartite(
        has_topic,
        paper_keep,
        num_papers,
        max_pairs_per_left=topic_edges_per_topic,
        seed=seed + 2,
    )
    edge_indices = [_dedupe_edges(edge) for edge in (cites, coauthor, topic)]
    edge_indices = _subsample_relations(edge_indices, target_edges, seed)
    edge_type = torch.cat(
        [torch.full((edge.size(1),), rel_id, dtype=torch.long) for rel_id, edge in enumerate(edge_indices)]
    )
    edge_index = torch.cat(edge_indices, dim=1)
    train_idx = (paper.train_mask[paper_keep] & y.ge(0)).nonzero(as_tuple=False).view(-1)
    val_idx = (paper.val_mask[paper_keep] & y.ge(0)).nonzero(as_tuple=False).view(-1)
    test_idx = (paper.test_mask[paper_keep] & y.ge(0)).nonzero(as_tuple=False).view(-1)
    return x, y, edge_index, edge_type, train_idx, val_idx, test_idx, ["paper_cites_paper", "paper_coauthor_paper", "paper_same_topic_paper"]


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = _load_dataset(args.root, args.preprocess)
    if args.paper_only:
        x, y, edge_index, edge_type, train_idx, val_idx, test_idx, relation_names = _paper_only_graph(
            data,
            max_papers=args.max_papers,
            max_classes=args.max_classes,
            class_selection=args.class_selection,
            min_class_count=args.min_class_count,
            balanced_class_sample=args.balanced_class_sample,
            class_sample_power=args.class_sample_power,
            fill_to_max_papers=args.fill_to_max_papers,
            target_edges=args.target_edges,
            coauthor_edges_per_author=args.coauthor_edges_per_author,
            topic_edges_per_topic=args.topic_edges_per_topic,
            seed=args.seed,
        )
    else:
        offsets, _ = _node_offsets(data)
        x = _stack_features(data, data.node_types)
        edge_index, edge_type, relation_names = _build_edges(data, offsets)
        y, train_idx, val_idx, test_idx = _paper_split(data, offsets, args.max_papers, args.seed)

    torch.save(x, output / "node_features.pt")
    torch.save(y, output / "labels.pt")
    torch.save(edge_index.long(), output / "edge_index.pt")
    torch.save(edge_type.long(), output / "edge_type.pt")
    torch.save(train_idx.long(), output / "train_idx.pt")
    torch.save(val_idx.long(), output / "val_idx.pt")
    torch.save(test_idx.long(), output / "test_idx.pt")
    (output / "relations.txt").write_text("\n".join(relation_names), encoding="utf-8")

    print(f"Saved OGB-MAG pt dataset to {output}")
    print(f"nodes={x.size(0)} features={x.size(1)} edges={edge_index.size(1)} relations={len(relation_names)}")
    print(f"train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()}")


if __name__ == "__main__":
    main()

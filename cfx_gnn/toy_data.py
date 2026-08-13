"""Deterministic synthetic graph for testing the complete public pipeline."""

from pathlib import Path

import dgl
import numpy as np
import torch


def generate_toy_dataset(output_dir, num_nodes=600, seed=2027):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "toy_graph.bin"
    index_path = output_dir / "toy_index.bin"
    if graph_path.exists() and index_path.exists():
        return graph_path, index_path

    rng = np.random.default_rng(seed)
    sensitive = rng.integers(0, 2, size=num_nodes)
    utility = rng.normal(size=num_nodes)
    label_score = utility + 0.30 * sensitive + rng.normal(scale=0.65, size=num_nodes)
    labels = (label_score > np.median(label_score)).astype(np.int64)

    features = np.column_stack([
        sensitive,
        utility + rng.normal(scale=0.15, size=num_nodes),
        label_score + rng.normal(scale=0.20, size=num_nodes),
        0.55 * sensitive + rng.normal(scale=0.65, size=num_nodes),
        utility * sensitive + rng.normal(scale=0.50, size=num_nodes),
        rng.normal(size=(num_nodes, 5)),
    ]).astype(np.float32)
    features[:, 1:] = (features[:, 1:] - features[:, 1:].mean(0)) / (features[:, 1:].std(0) + 1e-6)

    sources, targets = [], []
    all_nodes = np.arange(num_nodes)
    for node in range(num_nodes):
        same_label = all_nodes[(labels == labels[node]) & (all_nodes != node)]
        same_sensitive = all_nodes[(sensitive == sensitive[node]) & (all_nodes != node)]
        random_nodes = all_nodes[all_nodes != node]
        neighbors = np.unique(np.concatenate([
            rng.choice(same_label, size=3, replace=False),
            rng.choice(same_sensitive, size=2, replace=False),
            rng.choice(random_nodes, size=2, replace=False),
            np.asarray([(node + 1) % num_nodes]),
        ]))
        sources.extend([node] * len(neighbors)); targets.extend(neighbors.tolist())
    graph = dgl.graph((sources + targets, targets + sources), num_nodes=num_nodes)
    graph = dgl.to_simple(graph)
    graph = dgl.add_self_loop(graph)
    graph.ndata["nfeat"] = torch.from_numpy(features)

    # Stratified round-robin assignment keeps every split populated by all (y, s) groups.
    train, valid, test = [], [], []
    for y in (0, 1):
        for s in (0, 1):
            group = np.where((labels == y) & (sensitive == s))[0]
            rng.shuffle(group)
            n_train, n_valid = int(.60 * len(group)), int(.20 * len(group))
            train.extend(group[:n_train]); valid.extend(group[n_train:n_train+n_valid]); test.extend(group[n_train+n_valid:])
    rng.shuffle(train); rng.shuffle(valid); rng.shuffle(test)
    info = {
        "label": torch.from_numpy(labels), "sens": torch.from_numpy(sensitive.astype(np.int64)),
        "header": ["sensitive"] + [f"feature_{i}" for i in range(1, features.shape[1])],
        "train_index": torch.tensor(train, dtype=torch.long),
        "valid_index": torch.tensor(valid, dtype=torch.long),
        "test_index": torch.tensor(test, dtype=torch.long), "sens_index": 0,
        "generator": {"name": "cfx_gnn_toy", "seed": seed, "num_nodes": num_nodes},
    }
    dgl.save_graphs(str(graph_path), graph)
    torch.save(info, str(index_path))
    return graph_path, index_path


if __name__ == "__main__":
    generate_toy_dataset(Path(__file__).resolve().parent / "dataset" / "toy")

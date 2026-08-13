# Dataset files

## Bundled toy graph

`toy/toy_graph.bin` and `toy/toy_index.bin` form a deterministic synthetic
600-node graph for an end-to-end functional demonstration. If either file is
absent, `python run.py --dataset toy` regenerates both with the fixed generator
seed in `cfx_gnn/toy_data.py`. It is not used for reported paper results.

## Paper datasets

The Bail and Pokec datasets are third-party research datasets and are not
redistributed in this repository. Obtain them in accordance with their
original terms, preprocess them into DGL graph caches, and place the files as:

```text
cfx_gnn/dataset/bail/bail_graph.bin
cfx_gnn/dataset/bail/bail_index.bin
cfx_gnn/dataset/pokec/pokec_n_graph.bin
cfx_gnn/dataset/pokec/pokec_n_index.bin
cfx_gnn/dataset/pokec/pokec_z_graph.bin
cfx_gnn/dataset/pokec/pokec_z_index.bin
```

The graph cache must contain node features, labels, and sensitive attributes;
the index cache must contain the fixed train/validation/test split expected by
`cfx_gnn/dataloading.py`. Dataset references are listed in the project README.

# CFX-GNN

Official research implementation of **CFX-GNN: Counterfactual Fairness
Explanation for Graph Neural Networks**.

CFX-GNN learns an invertible utility–bias decomposition, constructs
counterfactual embedding references through local quantile matching, identifies
bias-related feature and structure components, and predicts from their clean
complement.

> Release status: pre-publication research code. The default YAML files contain
> the best five-seed configurations available at the time this repository was
> prepared. They may be updated after the validation-only search is finalized.

## Environment

The reference environment uses Python 3.10, PyTorch 2.1, CUDA 11.8, and DGL
2.1. Install a PyTorch wheel matching your CUDA runtime, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The repository includes a deterministic 600-node synthetic `toy` dataset so
the complete training and explanation-evaluation pipeline can be exercised
without downloading third-party data. Run it with:

```bash
python run.py --dataset toy
```

This takes roughly half a minute on the reference GPU and prints the per-seed
metrics and aggregate structured summary to the terminal. No result JSON,
JSONL, or CSV files are created. The toy result demonstrates output semantics
and code integrity; it is not a paper
benchmark and must not be compared with the Bail/Pokec tables.

The experiments use Bail and two Pokec regional graphs. These are third-party
datasets and are not redistributed here. Obtain them under their original
terms and prepare the six cache files described in
[`cfx_gnn/dataset/README.md`](cfx_gnn/dataset/README.md).

Dataset references:

- Bail: Jordan and Bowman, *The Effect of Race on Incarceration* (2015).
- Pokec: Takac and Zabovsky, *Data Analysis in Public Social Networks* (2012).

For local development alongside the private experiment project, prepared
caches can be imported without making them Git-tracked:

```bash
bash scripts/import_local_caches.sh ../CFX-GNN/cfx_gnn/dataset
```

## Run

Check the pipeline with one seed and one epoch per stage:

```bash
python run.py --dataset bail --smoke-test
```

Run five consecutive rounds with seeds `1,2,3,4,5`:

```bash
python run.py --dataset bail
python run.py --dataset pokec_n
python run.py --dataset pokec_z
```

Run all datasets across two GPUs:

```bash
GPUS=0,1 bash scripts/reproduce_five_seeds.sh
```

Metrics are printed to standard output and are not recorded in result files.
Training checkpoints are isolated below `checkpoints/<dataset>/` and ignored
by Git.

## Result semantics

The learned mask represents the **bias-related graph**, while its complement is
the **clean explanation graph** used for final prediction:

| Name | Meaning |
|---|---|
| `full` | Full input graph |
| `bias` | Learned bias-related mask |
| `clean` | Clean complementary explanation |

A near-random Bias AUC is not a failed final explanation; it means the isolated
bias graph has little task utility. Clean is the final model path used for the
main utility and fairness evaluation.

## Default configurations

The release freezes one numerical configuration per dataset under `configs/`
using the fixed consecutive seeds `1,2,3,4,5`. Bail uses GCN; Pokec-n and Pokec-z use APPNP. Method
choices are not configurable: local quantile matching, orthogonal separation,
the bias/clean mask semantics, representation reduction, and clean-head-only
refinement are fixed in code. YAML files contain only dataset-specific
hyperparameters; no older method variant or checkpoint is bundled.

## Repository layout

```text
configs/              frozen dataset configurations
cfx_gnn/              models, training, evaluation, and data loading
cfx_gnn/dataset/      bundled toy data and third-party cache locations
scripts/              multi-GPU five-seed reproduction
tests/                lightweight implementation tests
run.py                public experiment entry point
```

## Reproducibility notes

- Each run prints per-seed metrics and an aggregate structured summary.
- DGL/CUDA graph kernels may not be bitwise deterministic on every platform.
- Every regular invocation starts at seed 1 and increments it once per round
  for five rounds; arbitrary seed lists are not accepted.
- `mask_feature_budget` and `mask_structure_budget` denote retained ratios of
  the bias mask; Clean uses the complement.

## Citation

Citation metadata will be updated with the paper's final bibliographic record.
Until then, use the metadata in [`CITATION.cff`](CITATION.cff).

## License

The source code is released under the [MIT License](LICENSE). Dataset licenses
and terms remain with their respective owners.

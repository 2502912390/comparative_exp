# MRiemGNN

Implementation and experimental code for:

> Multiplex Heterogeneous Graph Neural Networks with Euclidean-Riemannian Mutual Space Synergy

This repository contains a PyTorch implementation of MRiemGNN for multiplex heterogeneous graph node classification and link prediction. The current codebase includes the core model, dataset converters, split utilities, and reproducibility commands used in the local experiment log.

## Highlights

- Relation-aware kernel message passing over multiplex relation channels.
- Euclidean and hyperbolic dual-space encoders.
- Euclidean-Riemannian mutual-space synergy loss.
- Dual-space embedding aggregation with task-adaptive decoders.
- Node classification and link prediction training loops.
- Utilities for BPHGNN/SR-RSC IMDB and OGB-MAG subset conversion.
- Deterministic rerun support for final reported node-classification results.

## Repository Layout

```text
.
|-- main.py                    # CLI entry point
|-- model_mriemgnn.py          # MRiemGNN model
|-- layer_mriemgnn.py          # relation-aware message passing layer
|-- train.py                   # node classification and link prediction training
|-- data.py                    # .mat/.pt graph loading and relation construction
|-- manifolds/                 # Euclidean and hyperbolic helpers
|-- tools/
|   |-- prepare_ogb_mag.py     # OGB-MAG paper-only subset converter
|   |-- prepare_sr_rsc.py      # SR-RSC IMDB converter
|   `-- ensemble_logits.py     # saved-logit diagnostic ensembling
|-- scripts/                   # PowerShell helpers for dataset prep and final runs
`-- README.md                  # usage, dataset preparation, and best commands
```

Large local artifacts such as `data/`, `external/`, and `runs/` are intentionally ignored by Git. Rebuild datasets from the public upstream sources and conversion commands below.

## Installation

Create an environment with PyTorch, PyG, and the scientific Python stack. Example:

```bash
conda create -n mriemgnn python=3.11 -y
conda activate mriemgnn
pip install -r requirements.txt
```

Install the PyTorch/PyG builds that match your CUDA runtime when using GPU acceleration. The local experiments were run with the `scaleanygraph` environment on an RTX 5060 Ti.

## Quick Start

For the most common reproduction commands, use the PowerShell helpers in `scripts/`:

```powershell
.\scripts\prepare_mag_paper_scale.ps1
.\scripts\run_best_nodecls.ps1 -Dataset mag
```

Available names for `run_best_nodecls.ps1` are `taobao`, `mag`, `imdb`, and `alibaba`.

Node classification on a prepared `.mat` dataset:

```bash
python main.py --task node_cls --mat_path data/small_alibaba_1_10/small_alibaba_1_10.mat --log_every 999
```

Node classification on a prepared `.pt` dataset:

```bash
python main.py --task node_cls --data_type pt --pt_dir data/imdb_sr_rsc_pt --log_every 999
```

Link prediction:

```bash
python main.py --task link_pred --link_split_path data/small_alibaba_1_10/small_alibaba_1_10_linkpred_split.mat --log_every 999
```

## Citation

```bibtex
@inproceedings{li2026multiplex,
  title={Multiplex Heterogeneous Graph Neural Networks with Euclidean-Riemannian Mutual Space Synergy},
  author={Li, Xiang and Cao, Yuan and Zhao, Zhongying and Chao, Guoqing and Yu, Yanwei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={18},
  pages={15126--15134},
  year={2026}
}
```

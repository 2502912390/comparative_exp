# Actual execution checks

- Existing `/home/chenzhichao/miniconda3/envs/coe` reused read-only; no clone or package changes.
- CPU ACM short training: `exp/smoke_infomgf_cpu_acm_20261015_a/acm/seed1024/`.
- CUDA asymmetric direction, dense/sparse graph forward and input gradient, and exact blocked InfoNCE gradient oracles: `log/dev_final_inputs_20261015_b/gpu_formulas.log`.
- Full MAG 113919 nodes, native hidden256/three layers/contrast batch2560; complete one-update execution, selected head and independent strict model/score replay: `exp/smoke_infomgf_mag_cuda_20261015_a/mag/seed1024/`.
- Full canonical Flickr 12154 nodes and six classes, two own-feature graph channels, one complete encoder/head update and independent replay: `exp/smoke_infomgf_flickr_fixed_inputs_20261015_b/flickr/seed1024/`.
- Independent-process seed1024/1025 fixed-input and adaptation hashes match: `log/dev_final_inputs_20261015_b/check_report.json`.
- These are necessary implementation checks, with no hyperparameter search or extra formal probes.
- Executor: inherited current session model; exact identifier not exposed. Independent reviewer/Astra: not_run.

Formal jobs freeze the final core/config/environment. Development smoke snapshots predate final environment/documentation records.

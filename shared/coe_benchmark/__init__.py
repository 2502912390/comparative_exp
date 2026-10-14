"""Namespaced, label-budget-preserving benchmark tools for Tables 1/2/3."""
from .protocol import (PROTOCOL_ID, DATA_PROTOCOL_ID, IMPLEMENTATION_VERSION,
                       DATASETS, TABLE_DATASETS, SEEDS, IdentityMismatch,
                       check_reference_sources, read_registry)
from .data import BenchmarkData, DataRoots, load_dataset, verify_identity
from .adaptations import relation_union, graph_to_feature_context_v1, adaptation_metadata
from .seed import seed_everything, capture_rng, restore_rng
from .selection import ValidationSelector, BestCheckpoint, load_best, write_selection_record
from .io import RunPaths, write_json
from .artifacts import build_manifest, export_result, recount_result, verify_model_replay
from .statistics import five_seed_statistics, summarize_run, write_summary
from .planning import plan_method, save_plan


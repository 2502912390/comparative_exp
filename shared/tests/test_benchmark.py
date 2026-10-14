"""Meaningful protocol/identity/checkpoint tests. Fixtures are not paper results."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import scipy.sparse as sp
from scipy.io import savemat
import torch

from shared.coe_benchmark import (
    PROTOCOL_ID, SEEDS, DataRoots, load_dataset, verify_identity, IdentityMismatch,
    relation_union, graph_to_feature_context_v1, seed_everything, capture_rng, restore_rng,
    BestCheckpoint, ValidationSelector, load_best, RunPaths, build_manifest,
    export_result, recount_result, verify_model_replay, five_seed_statistics,
    summarize_run, plan_method)
from shared.coe_benchmark._paths import _ACTIVE_ROOTS
from shared.coe_benchmark._reference_loader import _relation_adjacency, create_wikipedia_masks
from shared.coe_benchmark.fingerprints import tensor_digest, dense_graph_digest
from shared.coe_benchmark.protocol import REPO_ROOT


def fixture_dir(prefix):
    base = Path(os.environ.get("COE_BENCHMARK_TEST_OUTPUT", REPO_ROOT / "shared/exp/unit_tests"))
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


class SparseAndIdentityTests(unittest.TestCase):
    def test_dense_historical_hash_is_preserved_by_sparse_chunking(self):
        dense = torch.tensor([[1., 0., 2.], [0., 0., 0.], [4., 0., 1.]])
        a, b = hashlib.sha256(), hashlib.sha256()
        tensor_digest(a, dense)
        dense_graph_digest(b, dense.to_sparse_coo(), chunk_rows=1)
        self.assertEqual(a.hexdigest(), b.hexdigest())

    def test_union_is_binary_with_exactly_one_loop_and_no_reverse_edges_added(self):
        a = torch.tensor([[9., 2., 0.], [0., 0., 0.], [0., 0., 0.]]).to_sparse_coo()
        b = torch.tensor([[0., 7., 0.], [0., 0., 5.], [0., 0., 0.]]).to_sparse_coo()
        actual = relation_union([a, b])
        expected = torch.tensor([[1., 1., 0.], [0., 1., 1.], [0., 0., 1.]])
        self.assertTrue(torch.equal(actual.to_dense(), expected))
        self.assertEqual(actual.layout, torch.sparse_coo)

    def test_relation_bridge_preserves_content_and_all_five_views(self):
        x = torch.tensor([[2., 0.], [0., 3.]])
        a = torch.tensor([[0., 1.], [1., 0.]]).to_sparse_coo()
        empty = torch.zeros(2, 2).to_sparse_coo()
        data = SimpleNamespace(dataset="mv_acm", features=(x,) * 5,
                               provided_adjs=(empty, a, a, a, a))
        effective = graph_to_feature_context_v1(data)
        self.assertEqual(len(effective), 5)
        self.assertIs(effective[0], x)
        for view in effective[1:]:
            self.assertTrue(torch.equal(view[:, :2], x))
            self.assertTrue(torch.equal(view[:, 2:], torch.eye(2)))
        data.dataset = "mv_prokaryotic"
        self.assertTrue(all(v is x for v in graph_to_feature_context_v1(data)))

    def test_rectangular_relation_requires_verified_ids(self):
        raw = sp.csr_matrix(np.array([[2.], [1.]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "verified column_node_ids"):
            _relation_adjacency(raw, 2)
        explicit = _relation_adjacency(raw, 2, np.array([1], dtype=np.int64))
        self.assertEqual(explicit.shape, (2, 2))
        self.assertTrue(torch.allclose(explicit.to_dense(), torch.tensor([[1/3, 2/3], [0., 1.]])))

    def test_wikipedia_duplicate_matching_and_fixed_test_membership(self):
        directory = fixture_dir("wikipedia_")
        first = np.array([[10., 1.], [11., 1.], [11., 1.], [13., 1.],
                          [20., 1.], [21., 1.], [22., 1.], [23., 1.]], dtype=np.float32)
        second = np.array([[i, i+1, i+2] for i in range(8)], dtype=np.float32)
        second[2] = second[1]
        labels = np.array([1]*4 + [2]*4)
        cell = np.empty((1, 2), dtype=object)
        cell[0, 0], cell[0, 1] = first[[2, 6]], second[[2, 6]]
        savemat(directory / "Wikipedia-test.mat", {"X": cell, "y": labels[[2, 6]][:, None]})
        token = _ACTIVE_ROOTS.set(DataRoots(directory, directory))
        try:
            train, val, test = create_wikipedia_masks([first, second], labels)
        finally:
            _ACTIVE_ROOTS.reset(token)
        self.assertEqual(np.flatnonzero(test).tolist(), [2, 6])
        self.assertTrue(np.all(train | val | test))
        self.assertFalse(np.any((train & val) | (train & test) | (val & test)))

    def test_namespace_isolated_from_author_module_names(self):
        code = """import sys,types
for name in ('utils','model','data_loader','joint_train','joint_model'):
    sys.modules[name]=types.ModuleType(name)
from shared.coe_benchmark import read_registry,check_reference_sources
assert len(read_registry()['datasets'])==14
assert len(check_reference_sources())==8
print('namespace isolated')
"""
        completed = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class SeedSelectionStatisticsTests(unittest.TestCase):
    def test_rng_seed_and_checkpoint_restore(self):
        seed_everything(1024)
        state = capture_rng()
        first = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(state)
        second = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_validation_tie_keeps_earliest_weights_and_patience(self):
        directory = fixture_dir("selection_")
        selector = BestCheckpoint(directory / "best.pt", patience_checks=2)
        selector.consider(1, .6, {"weight": torch.tensor(1.)})
        digest = hashlib.sha256(selector.path.read_bytes()).hexdigest()
        self.assertFalse(selector.consider(2, .6, {"weight": torch.tensor(99.)}))
        self.assertFalse(selector.consider(3, .5, {"weight": torch.tensor(98.)}))
        self.assertTrue(selector.should_stop)
        self.assertEqual(hashlib.sha256(selector.path.read_bytes()).hexdigest(), digest)
        checkpoint = load_best(selector.path)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["payload"]["weight"].item(), 1.)

    def test_invalid_validation_and_epoch_rejected(self):
        selector = ValidationSelector()
        with self.assertRaises(ValueError):
            selector.consider(1, float("nan"))
        selector.consider(1, .2)
        with self.assertRaises(ValueError):
            selector.consider(1, .3)

    def test_five_seeds_use_sample_std_and_reject_incomplete(self):
        values = dict(zip(SEEDS, [0., .25, .5, .75, 1.]))
        result = five_seed_statistics(values)
        self.assertEqual(result["mean_percent"], 50.)
        self.assertAlmostEqual(result["sample_std_percent"], math.sqrt(1562.5))
        result = five_seed_statistics({1024: .5})
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["sample_std_percent"])
        with self.assertRaises(ValueError):
            five_seed_statistics({1: .5})

    def test_run_paths_cannot_overwrite_or_accept_wrong_formal_seed(self):
        directory = fixture_dir("paths_")
        RunPaths.create(directory, "first", "acm", 1024)
        with self.assertRaises(FileExistsError):
            RunPaths.create(directory, "first", "acm", 1024)
        with self.assertRaises(ValueError):
            RunPaths.create(directory, "../escape", "acm", 1024)
        with self.assertRaises(ValueError):
            RunPaths.create(directory, "other", "acm", 1)

    def test_cross_table_plan_is_complete_and_does_not_execute(self):
        plan = plan_method("InfoMGF", "unit_fixture")
        self.assertEqual(plan["status"], "plan_only_not_executed")
        self.assertEqual(len(plan["jobs"]), 45)
        self.assertEqual(len({(job["dataset"], job["seed"]) for job in plan["jobs"]}), 45)
        self.assertIn("mag", plan["datasets"])
        self.assertIn("flickr", plan["datasets"])


class LiveDataAndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = load_dataset("mv_prokaryotic")

    def test_identity_mutation_is_rejected_without_relaxing_golden_hash(self):
        data = self.data
        altered = data.features[0].clone()
        altered[0, 0] += .1
        with self.assertRaises(IdentityMismatch):
            verify_identity(replace(data, features=(altered, *data.features[1:])))
        metadata = dict(data.fingerprints)
        metadata["source"] = "forged"
        with self.assertRaises(IdentityMismatch):
            verify_identity(replace(data, fingerprints=metadata))

    def test_two_step_fixture_checkpoint_export_replay_and_tamper_rejection(self):
        data = self.data
        seed_everything(1024)
        run_id = "smoke_api_fixture_" + str(os.getpid())
        paths = RunPaths.create(REPO_ROOT / "shared", run_id, data.dataset, 1024, "smoke")
        x = torch.cat(data.features, dim=1)
        model = torch.nn.Linear(x.shape[1], data.classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        manifest = build_manifest(
            data, paths, method="shared_api_fixture",
            config={"purpose": "functional_check_only_not_paper_result", "steps": 2},
            source_files=[Path(__file__)], adaptation={"id": "canonical_fixture_only"},
            parameter_count=sum(p.numel() for p in model.parameters()), device="cpu")
        selector = BestCheckpoint(paths.exp / "best.pt")
        for epoch in (1, 2):
            model.train()
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[data.masks["train"]]), data.labels[data.masks["train"]])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(x[data.masks["val"]]).argmax(-1)
                val_acc = float(val_pred.eq(data.labels[data.masks["val"]]).float().mean())
            selector.consider(epoch, val_acc, {"model_state": model.state_dict(),
                                              "config_sha256": manifest["config_sha256"]})
        model.load_state_dict(load_best(paths.exp / "best.pt")["payload"]["model_state"])
        model.eval()
        with torch.no_grad():
            scores = model(x)
        exported = export_result(paths, data, checkpoint=paths.exp / "best.pt",
                                 best_epoch=selector.best_epoch, best_validation_acc=selector.best_acc,
                                 node_ids=data.node_ids, predictions=scores.argmax(-1),
                                 scores=scores, score_kind="logits")
        checked = recount_result(paths.exp)
        self.assertEqual(checked["correct"], exported["test"]["correct"])
        with self.assertRaises(ValueError):
            verify_model_replay(paths.exp, node_ids=data.node_ids,
                                predictions=(scores.argmax(-1) + 1) % data.classes)
        fresh_model = torch.nn.Linear(x.shape[1], data.classes)
        fresh_model.load_state_dict(load_best(paths.exp / "best.pt")["payload"]["model_state"])
        fresh_model.eval()
        with torch.no_grad():
            fresh = fresh_model(x)
        replay = verify_model_replay(paths.exp, node_ids=data.node_ids,
                                    predictions=fresh.argmax(-1), scores=fresh)
        self.assertEqual(replay["status"], "verified")
        summary = summarize_run(paths.exp.parents[1], expected_datasets=[data.dataset])
        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["rows"][0]["mean_percent"])
        with self.assertRaises(FileExistsError):
            export_result(paths, data, checkpoint=paths.exp / "best.pt",
                          best_epoch=selector.best_epoch, best_validation_acc=selector.best_acc,
                          node_ids=data.node_ids, predictions=fresh.argmax(-1))
        source = paths.exp / "source_snapshot/shared/tests/test_benchmark.py"
        original = source.read_bytes()
        try:
            source.write_bytes(original + b"\n# unit tamper fixture\n")
            with self.assertRaisesRegex(ValueError, "Source snapshot differs"):
                recount_result(paths.exp)
        finally:
            source.write_bytes(original)
        recount_result(paths.exp)
        (paths.log / "console.log").write_text(
            "Two-step CPU API fixture only; no baseline/method result and no paper-eligible statistic.\n")


if __name__ == "__main__":
    unittest.main()


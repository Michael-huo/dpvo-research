"""CPU contracts for final Exp4 models, losses, metrics, and retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...analysis.baseline import BaselineMetricsAccumulator, streaming_mean_fmap
from ...analysis.retrieval import random_temporal_baseline, temporal_retrieval_metrics
from ...schema import DEFAULT_CONFIG_PATH, load_config
from ..evaluate import EvaluationAccumulator
from ..loss import feature_adapter_loss
from ..model import (
    JepaFMapAdapter,
    LowRankLinearBaseline,
    RandomPredictionBaseline,
    count_parameters,
    model_config_for_kind,
    model_config_sha256,
)
from ..train import atomic_torch_save, load_checkpoint, model_identity_metadata


class AdapterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, _, _ = load_config(DEFAULT_CONFIG_PATH)

    def test_adapter_forward_shape_and_parameter_limit(self) -> None:
        model = JepaFMapAdapter(**model_config_for_kind(self.config, "adapter", "small")).eval()
        self.assertLess(count_parameters(model), 20_000_000)
        with torch.no_grad():
            output = model(torch.zeros(1, 576, 768))
        self.assertEqual(tuple(output.shape), (1, 128, 120, 188))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_capacity_configs_and_hashes(self) -> None:
        counts: list[int] = []
        hashes: set[str] = set()
        for name, hidden in (("small", 256), ("medium", 512), ("large", 768)):
            model_config = model_config_for_kind(self.config, "adapter", name)
            self.assertEqual(model_config["hidden_dim"], hidden)
            hashes.add(model_config_sha256(model_config))
            model = JepaFMapAdapter(**model_config)
            counts.append(count_parameters(model))
            self.assertLess(counts[-1], 20_000_000)
            del model
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(hashes), 3)

    def test_lowrank_rank16_formula_and_small_forward(self) -> None:
        configured = model_config_for_kind(self.config, "low_rank_linear")
        self.assertEqual(configured["rank"], 16)
        count = LowRankLinearBaseline.parameter_count_for_shapes(
            configured["input_shape"], configured["output_shape"], configured["rank"],
        )
        self.assertEqual(count, 56_168_448)
        self.assertGreater(count, 20_000_000)
        model = LowRankLinearBaseline(input_shape=(4, 3), output_shape=(2, 3, 5), rank=2)
        self.assertEqual(tuple(model(torch.randn(2, 4, 3)).shape), (2, 2, 3, 5))

    def test_loss_and_metric_accumulators(self) -> None:
        teacher = torch.ones(2, 4, 2, 3)
        prediction = teacher * 2.0
        losses = feature_adapter_loss(prediction, teacher)
        self.assertTrue(all(torch.isfinite(value).item() for value in losses.values()))
        self.assertTrue(torch.allclose(losses["total"], losses["cosine"] + 0.1 * losses["l2"]))
        with self.assertRaises(ValueError):
            feature_adapter_loss(prediction, teacher[..., :-1])

        for accumulator in (EvaluationAccumulator(), BaselineMetricsAccumulator()):
            accumulator.update(prediction, teacher)
            metrics = accumulator.finalize()
            self.assertAlmostEqual(metrics["mean_cosine_similarity"], 1.0, places=6)
            self.assertAlmostEqual(metrics["mse"], 1.0, places=6)
            self.assertAlmostEqual(metrics["norm_ratio"], 2.0, places=6)
            self.assertEqual(metrics["sample_count"], 2)

    def test_random_seed_and_streaming_mean(self) -> None:
        model = RandomPredictionBaseline((2, 3, 5))
        tokens = torch.zeros(2, 4, 3)
        torch.manual_seed(1234)
        first = model(tokens)
        torch.manual_seed(1234)
        second = model(tokens)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all().item())
        loader = DataLoader([
            {"fmap_teacher": torch.tensor([[[1.0, 3.0]]])},
            {"fmap_teacher": torch.tensor([[[5.0, 7.0]]])},
        ], batch_size=1)
        mean, count = streaming_mean_fmap(loader)
        self.assertEqual(count, 2)
        self.assertTrue(torch.equal(mean, torch.tensor([[[3.0, 5.0]]])))

    def test_checkpoint_roundtrip_and_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            model = torch.nn.Linear(3, 2)
            model_config = {"hidden_dim": 256}
            identity = model_identity_metadata(
                model_kind="adapter", model_config=model_config,
                capacity="small", parameter_count=sum(p.numel() for p in model.parameters()),
            )
            atomic_torch_save(path, {
                "model_state_dict": model.state_dict(),
                "metadata": {**identity, "dataset_fingerprint": {"sha256": "dataset"}},
            })
            loaded = load_checkpoint(path, expected_dataset_sha256="dataset")
            self.assertEqual(loaded["metadata"]["capacity"], "small")
            self.assertEqual(loaded["metadata"]["model_config_sha256"], model_config_sha256(model_config))
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_dataset_sha256="other")

    def test_temporal_retrieval_and_random_baseline(self) -> None:
        angles = torch.arange(8, dtype=torch.float32) * 0.08
        embeddings = torch.stack((angles.cos(), angles.sin()), dim=1)
        identities = [("euroc", "machine_hall", "MH_01_easy")] * 8
        metrics = temporal_retrieval_metrics(embeddings, identities, query_chunk_size=3, top_k=5)
        self.assertAlmostEqual(metrics["self_similarity_mean"], 1.0, places=6)
        self.assertEqual(metrics["excluding_self_top1_within_1"], 1.0)
        first = random_temporal_baseline(identities, seed=1234)
        second = random_temporal_baseline(identities, seed=1234)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

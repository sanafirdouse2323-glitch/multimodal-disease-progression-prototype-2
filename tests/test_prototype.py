import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from prototype import (
    FEATURES,
    STAGE_LABELS,
    ForecastDataset,
    MultimodalNet,
    classification_metrics,
    create_forecast_examples,
    fit_preprocessor,
    load_model_checkpoint,
    make_synthetic_cohort,
    patient_split,
)


class PrototypeTests(unittest.TestCase):
    def test_synthetic_cohort_contains_expected_modalities(self):
        frame = make_synthetic_cohort(20, seed=4)

        self.assertIn("img_0", frame.columns)
        self.assertIn("wear_0", frame.columns)
        self.assertIn("lab_a", frame.columns)
        self.assertEqual(len(FEATURES), 20)
        self.assertTrue((frame.groupby("patient_id").size() >= 2).all())
        self.assertTrue(frame.groupby("patient_id")["time_months"].apply(
            lambda values: values.is_monotonic_increasing
        ).all())
        self.assertTrue(set(frame["stage"].unique()).issubset(set(STAGE_LABELS)))

    def test_forecast_target_is_the_next_observed_visit(self):
        frame = make_synthetic_cohort(20, seed=5)
        examples = create_forecast_examples(frame)
        row = examples[0]

        visits = frame[frame.patient_id == row["patient_id"]].sort_values("time_months")
        current_positions = np.flatnonzero(
            np.isclose(visits["time_months"].to_numpy(), row["time_months"])
        )
        self.assertEqual(len(current_positions), 1)
        current_index = int(current_positions[0])

        self.assertLess(current_index + 1, len(visits))
        self.assertEqual(row["target_stage"], int(visits.iloc[current_index + 1].stage))
        np.testing.assert_allclose(
            row["features"],
            visits.iloc[current_index][FEATURES].to_numpy(dtype=np.float32),
            equal_nan=True,
        )
        self.assertNotIn("stage", FEATURES)
        self.assertEqual(len(row["features"]), 20)

    def test_patient_ids_do_not_cross_splits_and_all_examples_are_kept(self):
        examples = create_forecast_examples(make_synthetic_cohort(40, seed=6))
        train, validation, test = patient_split(examples, seed=6)

        patient_groups = [
            {example["patient_id"] for example in part}
            for part in (train, validation, test)
        ]
        self.assertTrue(all(part for part in (train, validation, test)))
        self.assertFalse(patient_groups[0] & patient_groups[1])
        self.assertFalse(patient_groups[0] & patient_groups[2])
        self.assertFalse(patient_groups[1] & patient_groups[2])
        self.assertEqual(set.union(*patient_groups), {e["patient_id"] for e in examples})
        self.assertEqual(len(train) + len(validation) + len(test), len(examples))

    def test_model_output_shapes_match_four_stages(self):
        model = MultimodalNet()
        model.eval()
        future_logits, current_logits = model(torch.zeros(3, len(FEATURES)))

        self.assertEqual(tuple(future_logits.shape), (3, len(STAGE_LABELS)))
        self.assertEqual(tuple(current_logits.shape), (3, len(STAGE_LABELS)))

    def test_model_rejects_an_incorrect_number_of_features(self):
        model = MultimodalNet()
        with self.assertRaisesRegex(ValueError, "20"):
            model(torch.zeros(3, 24))

    def test_preprocessor_uses_training_values_and_handles_all_missing_columns(self):
        first = np.zeros(len(FEATURES), dtype=np.float32)
        second = np.full(len(FEATURES), 2.0, dtype=np.float32)
        first[0], second[0] = 1.0, 3.0
        first[1] = np.nan
        second[1] = np.nan

        mean, scale = fit_preprocessor([
            {"features": first},
            {"features": second},
        ])

        self.assertAlmostEqual(float(mean[0]), 2.0)
        self.assertAlmostEqual(float(scale[0]), 1.0)
        self.assertEqual(float(mean[1]), 0.0)
        self.assertEqual(float(scale[1]), 1.0)
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(scale).all())
        self.assertTrue((scale > 0).all())

    def test_dataset_imputes_missing_values_and_returns_finite_tensors(self):
        features = np.arange(len(FEATURES), dtype=np.float32)
        features[0] = np.nan
        features[1] = np.inf
        dataset = ForecastDataset(
            [{"features": features, "target_stage": 2, "current_stage": 1}],
            mean=np.zeros(len(FEATURES), dtype=np.float32),
            scale=np.ones(len(FEATURES), dtype=np.float32),
        )

        x, target, current = dataset[0]
        self.assertEqual(tuple(x.shape), (len(FEATURES),))
        self.assertTrue(torch.isfinite(x).all())
        self.assertEqual(float(x[0]), 0.0)
        self.assertEqual(float(x[1]), 0.0)
        self.assertEqual(int(target), 2)
        self.assertEqual(int(current), 1)

    def test_macro_f1_includes_stages_absent_from_the_test_targets(self):
        metrics = classification_metrics(
            targets=[0, 0, 1, 1],
            predictions=[0, 1, 1, 1],
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 4)
        self.assertEqual(metrics["class_support"], [2, 2, 0, 0])

    def test_checkpoint_can_be_saved_and_reloaded(self):
        model = MultimodalNet()
        model.eval()
        sample = torch.randn(2, len(FEATURES))
        with torch.no_grad():
            expected_future, expected_current = model(sample)

        checkpoint = {
            "state_dict": model.state_dict(),
            "feature_order": FEATURES,
            "stage_labels": list(STAGE_LABELS),
            "synthetic_only": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            loaded_model, metadata = load_model_checkpoint(checkpoint_path)

        with torch.no_grad():
            actual_future, actual_current = loaded_model(sample)

        torch.testing.assert_close(actual_future, expected_future)
        torch.testing.assert_close(actual_current, expected_current)
        self.assertTrue(metadata["synthetic_only"])

    def test_checkpoint_with_a_different_feature_order_is_rejected(self):
        checkpoint = {
            "state_dict": MultimodalNet().state_dict(),
            "feature_order": list(reversed(FEATURES)),
            "stage_labels": list(STAGE_LABELS),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "wrong-schema.pt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "feature order"):
                load_model_checkpoint(checkpoint_path)


if __name__ == "__main__":
    unittest.main()

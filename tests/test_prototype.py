import unittest

import torch

from prototype import MultimodalNet, create_forecast_examples, make_synthetic_cohort, patient_split


class PrototypeTests(unittest.TestCase):
    def test_synthetic_cohort_contains_modalities(self):
        frame = make_synthetic_cohort(20, seed=4)
        self.assertIn("img_0", frame)
        self.assertIn("wear_0", frame)
        self.assertIn("lab_a", frame)
        self.assertTrue((frame.groupby("patient_id").size() >= 2).all())

    def test_forecast_target_is_later_visit(self):
        frame = make_synthetic_cohort(20, seed=5)
        examples = create_forecast_examples(frame)
        row = examples[0]
        visits = frame[frame.patient_id == row["patient_id"]].sort_values("time_months")
        idx = visits.index.get_loc(visits[visits.time_months == row["time_months"]].index[0])
        self.assertEqual(row["target_stage"], int(visits.iloc[idx + 1].stage))
        self.assertEqual(len(row["features"]), 24)

    def test_patient_ids_do_not_cross_splits(self):
        examples = create_forecast_examples(make_synthetic_cohort(40, seed=6))
        train, val, test = patient_split(examples, seed=6)
        groups = [{e["patient_id"] for e in part} for part in (train, val, test)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_model_output_shapes(self):
        model = MultimodalNet()
        future, current = model(torch.zeros(3, 24))
        self.assertEqual(tuple(future.shape), (3, 4))
        self.assertEqual(tuple(current.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()

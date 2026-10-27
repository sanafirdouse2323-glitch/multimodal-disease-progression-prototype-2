"""Small synthetic multimodal longitudinal disease-progression prototype.

Educational software baseline only; not a clinical prediction system.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

N_STAGES = 4


@dataclass
class Config:
    patients: int = 240
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    output_dir: str = "results"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_cohort(n_patients: int = 240, seed: int = 42) -> pd.DataFrame:
    """Generate toy irregular visits, image-derived features, wearable data and labs."""
    if n_patients < 12:
        raise ValueError("n_patients must be at least 12 for patient-level splits")
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        patient_id = f"SYNTH_{p:05d}"
        latent = rng.normal()
        age = float(np.clip(rng.normal(61, 11), 25, 90))
        sex = int(rng.integers(0, 2))
        visits = int(rng.integers(2, 5))
        month = 0.0
        stage = int(np.clip(np.round(1 + latent * 0.45), 0, 3))
        for visit in range(visits):
            if visit:
                month += float(rng.choice([3, 6, 9, 12]))
                stage = int(np.clip(stage + int(rng.random() < 0.35) - int(rng.random() < 0.08), 0, 3))
            signal = stage / 3.0
            image = rng.normal(loc=signal + 0.1 * latent, scale=0.35, size=8)
            wearable = rng.normal(loc=1.0 - 0.55 * signal, scale=0.25, size=6)
            clinical = np.array([
                age / 90.0, float(sex),
                rng.normal(0.5 + 0.35 * signal, 0.15),
                rng.normal(0.4 + 0.4 * signal, 0.18),
            ])
            image_missing = bool(rng.random() < 0.12)
            wearable_missing = bool(rng.random() < 0.18)
            clinical[rng.random(4) < 0.08] = np.nan
            if image_missing:
                image[:] = np.nan
            if wearable_missing:
                wearable[:] = np.nan
            row = {
                "patient_id": patient_id, "visit_idx": visit, "time_months": month,
                "stage": stage, "image_missing": int(image_missing),
                "wearable_missing": int(wearable_missing),
            }
            row.update({f"img_{i}": float(v) for i, v in enumerate(image)})
            row.update({f"wear_{i}": float(v) for i, v in enumerate(wearable)})
            row.update({name: float(v) for name, v in zip(
                ["age_scaled", "sex_binary", "lab_a", "lab_b"], clinical
            )})
            rows.append(row)
    return pd.DataFrame(rows)


FEATURES = (
    [f"img_{i}" for i in range(8)]
    + [f"wear_{i}" for i in range(6)]
    + ["age_scaled", "sex_binary", "lab_a", "lab_b", "image_missing", "wearable_missing"]
)


def create_forecast_examples(frame: pd.DataFrame) -> list[dict]:
    """Predict the next observed visit's stage from current-visit features only."""
    examples = []
    for patient_id, visits in frame.groupby("patient_id"):
        visits = visits.sort_values("time_months")
        for i in range(len(visits) - 1):
            current = visits.iloc[i]
            future = visits.iloc[i + 1]
            examples.append({
                "patient_id": patient_id,
                "time_months": float(current.time_months),
                "target_stage": int(future.stage),
                "current_stage": int(current.stage),
                "features": current[FEATURES].to_numpy(dtype=np.float32),
            })
    if not examples:
        raise ValueError("No forecast examples could be created")
    return examples


def patient_split(examples: list[dict], seed: int):
    """Split by patient so visits from one patient cannot cross data partitions."""
    patients = np.array(sorted({e["patient_id"] for e in examples}))
    train_val_ids, test_ids = train_test_split(patients, test_size=0.2, random_state=seed)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.2, random_state=seed)
    id_sets = [set(train_ids), set(val_ids), set(test_ids)]
    return tuple([[e for e in examples if e["patient_id"] in ids] for ids in id_sets])


class ForecastDataset(Dataset):
    def __init__(self, examples, mean: np.ndarray, scale: np.ndarray):
        raw = np.stack([e["features"] for e in examples])
        self.x = np.nan_to_num(raw, nan=mean)
        self.x = np.clip((self.x - mean) / scale, -8, 8).astype(np.float32)
        self.y = np.asarray([e["target_stage"] for e in examples], dtype=np.int64)
        self.current_y = np.asarray([e["current_stage"] for e in examples], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.tensor(self.y[index]), torch.tensor(self.current_y[index])


class MultimodalNet(nn.Module):
    """Separate feature-group encoders and fused future/current-stage heads."""
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Linear(8, 24), nn.ReLU(), nn.Linear(24, 16), nn.ReLU())
        self.wearable_encoder = nn.Sequential(nn.Linear(6, 20), nn.ReLU(), nn.Linear(20, 16), nn.ReLU())
        self.tabular_encoder = nn.Sequential(nn.Linear(10, 24), nn.ReLU(), nn.Linear(24, 16), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.15))
        self.forecast_head = nn.Linear(32, N_STAGES)
        self.current_head = nn.Linear(32, N_STAGES)

    def forward(self, x):
        image = self.image_encoder(x[:, 0:8])
        wearable = self.wearable_encoder(x[:, 8:14])
        tabular = self.tabular_encoder(x[:, 14:24])
        fused = self.fusion(torch.cat([image, wearable, tabular], dim=1))
        return self.forecast_head(fused), self.current_head(fused)


def evaluate(model, loader, device):
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits, _ = model(x.to(device))
            targets.extend(y.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        "n_examples": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def run(config: Config):
    seed_everything(config.seed)
    frame = make_synthetic_cohort(config.patients, config.seed)
    examples = create_forecast_examples(frame)
    train_ex, val_ex, test_ex = patient_split(examples, config.seed)

    # Missing-value and scaling statistics are estimated only from training patients.
    train_matrix = np.stack([e["features"] for e in train_ex])
    mean = np.nanmean(train_matrix, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.nanstd(train_matrix, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

    train_ds = ForecastDataset(train_ex, mean, scale)
    val_ds = ForecastDataset(val_ex, mean, scale)
    test_ds = ForecastDataset(test_ex, mean, scale)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history, best_state, best_val = [], None, float("inf")

    for epoch in range(config.epochs):
        model.train()
        losses = []
        for x, y, current_y in train_loader:
            x, y, current_y = x.to(device), y.to(device), current_y.to(device)
            future_logits, current_logits = model(x)
            loss = loss_fn(future_logits, y) + 0.25 * loss_fn(current_logits, current_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **val_metrics})
        score = 1.0 - val_metrics["macro_f1"]
        if score < best_val:
            best_val, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(), "feature_order": FEATURES,
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "stage_labels": list(range(N_STAGES)), "synthetic_only": True,
        "config": asdict(config),
    }, out / "model.pt")

    rows = []
    model.eval()
    with torch.no_grad():
        for x, y, _ in test_loader:
            logits, _ = model(x.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for target, probs in zip(y.tolist(), probabilities):
                rows.append({
                    "target_stage": target, "predicted_stage": int(np.argmax(probs)),
                    **{f"prob_stage_{i}": float(v) for i, v in enumerate(probs)},
                })
    pd.DataFrame(rows).to_csv(out / "test_predictions.csv", index=False)
    summary = {
        "task": "next-observed-visit stage classification",
        "synthetic_only": True, "split_unit": "patient",
        "split_sizes": {
            "train_windows": len(train_ex), "validation_windows": len(val_ex),
            "test_windows": len(test_ex),
        },
        "test": test_metrics, "history": history,
        "interpretation": "Synthetic data only; no clinical validity claim. Follow-up times are irregular.",
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    return {"test": test_metrics, "outputs": str(out), "device": str(device)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    if args.patients < 12 or args.epochs < 1 or args.batch_size < 1:
        parser.error("--patients must be >= 12 and epochs/batch-size must be positive")
    return Config(args.patients, args.epochs, args.batch_size, args.learning_rate, args.seed, args.output_dir)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

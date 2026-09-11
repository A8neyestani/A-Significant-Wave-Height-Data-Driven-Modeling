"""Load the published GRU artifacts and make one-step VHM0 predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_LOOK_BACK = 16

# The persisted scaler uses VDMR. Public OBSEA exports commonly use VMDR.
FEATURE_ALIASES = {
    "VDMR": "VDMR",
    "VMDR": "VDMR",
    "VTPK": "VTPK",
    "VZMX": "VZMX",
    "VTZA": "VTZA",
}


def _load_artifacts(model_dir: Path):
    """Load TensorFlow and preprocessing artifacts lazily."""

    try:
        import joblib
        from tensorflow.keras.models import load_model
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "Prediction requires the runtime dependencies. Install them with "
            "`pip install -r requirements.txt`."
        ) from exc

    model_dir = Path(model_dir)
    model = load_model(model_dir / "gru_model.h5")
    feature_scaler = joblib.load(model_dir / "feature_scaler.pkl")
    target_scaler = joblib.load(model_dir / "target_scaler.pkl")
    return model, feature_scaler, target_scaler


def _as_window(values: Sequence[Sequence[float]] | np.ndarray, look_back: int, n_features: int) -> np.ndarray:
    """Validate and normalize a feature window for model inference."""

    window = np.asarray(values, dtype=np.float32)
    expected_shape = (look_back, n_features)
    if window.shape != expected_shape:
        raise ValueError(f"Expected input shape {expected_shape}, got {window.shape}.")
    if not np.isfinite(window).all():
        raise ValueError("Input data contains NaN or infinite values.")
    return window


def predict_next(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    look_back: int = DEFAULT_LOOK_BACK,
    verbose: int = 0,
) -> float:
    """Predict the next VHM0 value from a `(look_back, 4)` feature window.

    Columns must be ordered as `VDMR`, `VTPK`, `VZMX`, `VTZA`. The returned
    value is in metres and is transformed back to the original target scale.
    """

    model, feature_scaler, target_scaler = _load_artifacts(Path(model_dir))
    n_features = int(feature_scaler.n_features_in_)
    window = _as_window(values, look_back, n_features)
    scaled = feature_scaler.transform(window)
    prediction_scaled = model.predict(scaled.reshape(1, look_back, n_features), verbose=verbose)
    prediction = target_scaler.inverse_transform(np.asarray(prediction_scaled).reshape(-1, 1))
    return float(prediction[0, 0])


def _canonical_columns(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("The CSV file has no header row.")
    columns: dict[str, str] = {}
    for field in fieldnames:
        key = field.strip().upper()
        if key in FEATURE_ALIASES:
            columns[FEATURE_ALIASES[key]] = field
    required = {"VDMR", "VTPK", "VZMX", "VTZA"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {sorted(missing)}")
    return columns


def _read_feature_window(csv_path: str | Path, look_back: int) -> np.ndarray:
    """Read the last valid feature window from a CSV file."""

    rows: list[list[float]] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = _canonical_columns(reader.fieldnames)
        for row in reader:
            try:
                values = [float(row[columns[name]]) for name in ("VDMR", "VTPK", "VZMX", "VTZA")]
            except (TypeError, ValueError):
                continue
            if np.isfinite(values).all():
                rows.append(values)
    if len(rows) < look_back:
        raise ValueError(f"Need at least {look_back} valid rows; found {len(rows)}.")
    return np.asarray(rows[-look_back:], dtype=np.float32)


def predict_from_csv(
    csv_path: str | Path,
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    look_back: int = DEFAULT_LOOK_BACK,
    verbose: int = 0,
) -> float:
    """Read the latest valid rows in a CSV and predict the next VHM0 value."""

    window = _read_feature_window(csv_path, look_back)
    return predict_next(window, model_dir=model_dir, look_back=look_back, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing VDMR/VMDR, VTPK, VZMX and VTZA columns")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing the pretrained artifacts")
    parser.add_argument("--look-back", type=int, default=DEFAULT_LOOK_BACK)
    args = parser.parse_args()
    prediction = predict_from_csv(args.input, model_dir=args.model_dir, look_back=args.look_back)
    print(f"Predicted next VHM0: {prediction:.6f} m")


if __name__ == "__main__":  # pragma: no cover
    main()

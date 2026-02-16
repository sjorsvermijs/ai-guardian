"""
Train a classifier on HeAR embeddings extracted from SPRSound.

Supports two tasks:
  - Record-level: Normal vs CAS vs DAS vs CAS&DAS vs Poor Quality
  - Binary: Normal vs Adventitious (CAS/DAS/CAS&DAS grouped)

Uses sklearn MLP and Logistic Regression for quick iteration.

Usage:
  python -m src.pipelines.hear.train_classifier
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "sprsound_embeddings" / "sprsound_hear_embeddings.npz"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "sprsound_classifier"


def load_embeddings():
    """Load pre-extracted HeAR embeddings."""
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    record_labels = data["record_labels"]
    record_label_names = list(data["record_label_names"])
    splits = data["splits"]

    print(f"Loaded {len(embeddings)} samples, {len(record_label_names)} classes")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Classes: {record_label_names}")

    return embeddings, record_labels, record_label_names, splits


def make_binary_labels(record_labels, record_label_names):
    """Convert multi-class to binary: Normal (0) vs Adventitious (1).

    Drops 'Poor Quality' and 'Unknown' samples.
    """
    binary_labels = []
    keep_mask = []

    for label_idx in record_labels:
        label_name = record_label_names[label_idx]
        if label_name in ("Poor Quality", "Unknown"):
            keep_mask.append(False)
            binary_labels.append(-1)
        elif label_name == "Normal":
            keep_mask.append(True)
            binary_labels.append(0)
        else:
            # CAS, DAS, CAS & DAS -> Adventitious
            keep_mask.append(True)
            binary_labels.append(1)

    return np.array(binary_labels), np.array(keep_mask)


def train_model(X_train, X_test, y_train, y_test, label_names, task_name):
    """Train and evaluate both Logistic Regression and MLP."""
    print(f"\n{'=' * 60}")
    print(f"Task: {task_name}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Classes: {label_names}")
    print(f"{'=' * 60}")

    results = {}

    # --- Logistic Regression (baseline) ---
    print(f"\n--- Logistic Regression ---")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average="macro")
    print(f"Accuracy: {lr_acc:.4f}")
    print(f"F1 (macro): {lr_f1:.4f}")
    print(classification_report(y_test, lr_pred, target_names=label_names))

    results["logistic_regression"] = {"accuracy": lr_acc, "f1_macro": lr_f1}

    # --- MLP ---
    print(f"--- MLP Classifier ---")
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )),
    ])
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)

    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred, average="macro")
    print(f"Accuracy: {mlp_acc:.4f}")
    print(f"F1 (macro): {mlp_f1:.4f}")
    print(classification_report(y_test, mlp_pred, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, mlp_pred)
    header = "".join(f"{n[:10]:>12}" for n in label_names)
    print(f"Confusion Matrix:\n{'':>12}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"{label_names[i][:10]:>12}{row_str}")

    results["mlp"] = {"accuracy": mlp_acc, "f1_macro": mlp_f1}

    return mlp, results


def main():
    print("=" * 60)
    print("SPRSound Classifier Training (HeAR Embeddings)")
    print("=" * 60)

    embeddings, record_labels, record_label_names, splits = load_embeddings()

    # Split data using dataset splits
    train_mask = splits == "train"
    test_mask = splits == "test"

    if train_mask.sum() > 0 and test_mask.sum() > 0:
        print(f"\nUsing dataset splits: {train_mask.sum()} train, {test_mask.sum()} test")
        X_train_all, X_test_all = embeddings[train_mask], embeddings[test_mask]
        y_train_all, y_test_all = record_labels[train_mask], record_labels[test_mask]
    else:
        print("\nNo predefined splits found, using 80/20 stratified split")
        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
            embeddings, record_labels, test_size=0.2, random_state=42, stratify=record_labels,
        )

    all_results = {}

    # ============================================================
    # TASK 1: Multi-class record-level classification
    # Normal vs CAS vs DAS vs CAS&DAS vs Poor Quality
    # ============================================================
    mlp_multiclass, multiclass_results = train_model(
        X_train_all, X_test_all, y_train_all, y_test_all,
        record_label_names, "Multi-class (Record-level)",
    )
    all_results["multiclass"] = multiclass_results

    # ============================================================
    # TASK 2: Binary classification
    # Normal vs Adventitious
    # ============================================================
    binary_labels, keep_mask = make_binary_labels(record_labels, record_label_names)
    binary_label_names = ["Normal", "Adventitious"]

    # Apply mask to filter out Poor Quality / Unknown
    bin_train_mask = train_mask & keep_mask
    bin_test_mask = test_mask & keep_mask

    if bin_train_mask.sum() > 0 and bin_test_mask.sum() > 0:
        X_train_bin = embeddings[bin_train_mask]
        X_test_bin = embeddings[bin_test_mask]
        y_train_bin = binary_labels[bin_train_mask]
        y_test_bin = binary_labels[bin_test_mask]
    else:
        valid = keep_mask
        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
            embeddings[valid], binary_labels[valid],
            test_size=0.2, random_state=42, stratify=binary_labels[valid],
        )

    mlp_binary, binary_results = train_model(
        X_train_bin, X_test_bin, y_train_bin, y_test_bin,
        binary_label_names, "Binary (Normal vs Adventitious)",
    )
    all_results["binary"] = binary_results

    # ============================================================
    # Save models
    # ============================================================
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save multi-class model
    with open(MODEL_OUTPUT_DIR / "mlp_multiclass.pkl", "wb") as f:
        pickle.dump(mlp_multiclass, f)

    # Save binary model
    with open(MODEL_OUTPUT_DIR / "mlp_binary.pkl", "wb") as f:
        pickle.dump(mlp_binary, f)

    # Save metadata
    metadata = {
        "embedding_dim": int(embeddings.shape[1]),
        "feature_extractor": "HeAR (google/hear-pytorch)",
        "dataset": "SPRSound BioCAS2022",
        "multiclass": {
            "label_names": record_label_names,
            "model_file": "mlp_multiclass.pkl",
            **multiclass_results["mlp"],
        },
        "binary": {
            "label_names": binary_label_names,
            "model_file": "mlp_binary.pkl",
            **binary_results["mlp"],
        },
    }
    with open(MODEL_OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels saved to {MODEL_OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

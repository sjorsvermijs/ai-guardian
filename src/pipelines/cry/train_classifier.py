"""
Train classifiers on cry embeddings.

Compares multiple classifiers (LR, SVM, RF, kNN, MLP) with and without SMOTE.
Supports multiple embedding backends (ast, hubert).

Usage:
  python -m src.pipelines.cry.train_classifier                     # all backends
  python -m src.pipelines.cry.train_classifier --backend hubert    # specific
  python -m src.pipelines.cry.train_classifier --merge-classes     # merge rare classes
"""

import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "donate_a_cry_embeddings"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "cry_classifier"


def load_embeddings(backend: str):
    path = EMBEDDINGS_DIR / f"cry_{backend}_embeddings.npz"
    if not path.exists():
        return None, None, None
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"], list(data["label_names"])


def merge_rare_classes(embeddings, labels, label_names, min_samples=15):
    """Merge classes with fewer than min_samples into neighbors or drop them."""
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # Classes to keep as-is vs merge
    keep_classes = {idx for idx, cnt in class_counts.items() if cnt >= min_samples}
    merge_classes = {idx for idx, cnt in class_counts.items() if cnt < min_samples}

    if not merge_classes:
        return embeddings, labels, label_names

    # Strategy: merge small classes into "other"
    new_label_names = [label_names[i] for i in sorted(keep_classes)]
    new_label_names.append("other")
    other_idx = len(new_label_names) - 1

    remap = {}
    for idx in sorted(keep_classes):
        remap[idx] = new_label_names.index(label_names[idx])
    for idx in merge_classes:
        remap[idx] = other_idx

    new_labels = np.array([remap[l] for l in labels])

    # Drop "other" if it's still too small
    other_count = np.sum(new_labels == other_idx)
    if other_count < min_samples:
        mask = new_labels != other_idx
        embeddings = embeddings[mask]
        new_labels = new_labels[mask]
        new_label_names = new_label_names[:-1]
        # Re-index
        unique_remaining = sorted(set(new_labels))
        remap2 = {old: new for new, old in enumerate(unique_remaining)}
        new_labels = np.array([remap2[l] for l in new_labels])

    return embeddings, new_labels, new_label_names


def build_classifiers():
    """Return dict of classifier pipelines to evaluate."""
    return {
        "LogReg (balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, random_state=42, class_weight="balanced",
                C=1.0, solver="lbfgs",
            )),
        ]),
        "SVM-RBF (balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", class_weight="balanced",
                random_state=42, C=10.0, gamma="scale",
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=42, max_depth=None,
            )),
        ]),
        "kNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="cosine",
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42,
                learning_rate=0.1,
            )),
        ]),
        "MLP (64)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu", max_iter=1000,
                early_stopping=True, validation_fraction=0.15,
                random_state=42,
            )),
        ]),
    }


def evaluate_classifiers(X, y, label_names, use_smote=False):
    """5-fold cross-validation for all classifiers."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifiers = build_classifiers()
    results = {}

    smote_label = " + SMOTE" if use_smote else ""
    print(f"\n{'=' * 60}")
    print(f"5-Fold CV Comparison{smote_label}")
    print(f"{'=' * 60}")
    print(f"{'Classifier':<25} {'F1-macro':>10} {'Accuracy':>10}  {'Std':>8}")
    print("-" * 60)

    for name, clf in classifiers.items():
        if use_smote:
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.over_sampling import SMOTE
            # Rebuild with SMOTE inserted after scaler
            steps = list(clf.steps)
            smote_clf = ImbPipeline([
                steps[0],  # scaler
                ("smote", SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y)) - 1))),
                steps[1],  # clf
            ])
            clf_to_eval = smote_clf
        else:
            clf_to_eval = clf

        try:
            f1_scores = cross_val_score(
                clf_to_eval, X, y, cv=cv, scoring="f1_macro",
            )
            acc_scores = cross_val_score(
                clf_to_eval, X, y, cv=cv, scoring="accuracy",
            )
            results[name] = {
                "f1_mean": f1_scores.mean(),
                "f1_std": f1_scores.std(),
                "acc_mean": acc_scores.mean(),
                "acc_std": acc_scores.std(),
            }
            print(f"  {name:<23} {f1_scores.mean():>8.4f}   {acc_scores.mean():>8.4f}  +/-{f1_scores.std():.4f}")
        except Exception as e:
            print(f"  {name:<23} FAILED: {e}")
            results[name] = None

    return results


def train_best_and_save(X, y, label_names, backend, use_smote=False):
    """Train the best classifier on full data and save."""
    # Train SVM (typically best for small datasets) on full data
    if use_smote:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        k = min(3, min(np.bincount(y)) - 1)
        best_clf = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42, k_neighbors=k)),
            ("clf", SVC(
                kernel="rbf", class_weight="balanced", probability=True,
                random_state=42, C=10.0, gamma="scale",
            )),
        ])
    else:
        best_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", class_weight="balanced", probability=True,
                random_state=42, C=10.0, gamma="scale",
            )),
        ])

    best_clf.fit(X, y)

    # Save
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_file = f"svm_{backend}.pkl"
    with open(MODEL_OUTPUT_DIR / model_file, "wb") as f:
        pickle.dump(best_clf, f)

    # Full CV score for metadata
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(best_clf, X, y, cv=cv, scoring="f1_macro")
    acc_scores = cross_val_score(best_clf, X, y, cv=cv, scoring="accuracy")

    metadata = {
        "embedding_backend": backend,
        "embedding_dim": int(X.shape[1]),
        "dataset": "Donate a Cry",
        "label_names": label_names,
        "model_file": model_file,
        "model_type": "SVM-RBF (balanced)" + (" + SMOTE" if use_smote else ""),
        "cv_f1_mean": float(f1_scores.mean()),
        "cv_f1_std": float(f1_scores.std()),
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std": float(acc_scores.std()),
        "total_samples": int(len(X)),
        "class_distribution": {
            label_names[i]: int(c) for i, c in enumerate(np.bincount(y))
        },
    }
    with open(MODEL_OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model to {MODEL_OUTPUT_DIR / model_file}")
    return metadata


def run_for_backend(backend, merge_classes=False):
    print(f"\n{'#' * 60}")
    print(f"# Backend: {backend.upper()}")
    print(f"{'#' * 60}")

    embeddings, labels, label_names = load_embeddings(backend)
    if embeddings is None:
        print(f"  No embeddings found for backend '{backend}'.")
        print(f"  Run: python -m src.pipelines.cry.extract_embeddings --backend {backend}")
        return None

    print(f"Loaded {len(embeddings)} samples, {len(label_names)} classes")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Classes: {label_names}")
    for i, name in enumerate(label_names):
        count = np.sum(labels == i)
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    if merge_classes:
        print(f"\nMerging classes with < 15 samples...")
        embeddings, labels, label_names = merge_rare_classes(
            embeddings, labels, label_names, min_samples=15,
        )
        print(f"After merge: {len(label_names)} classes: {label_names}")
        for i, name in enumerate(label_names):
            count = np.sum(labels == i)
            print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    # Without SMOTE
    results = evaluate_classifiers(embeddings, labels, label_names, use_smote=False)

    # With SMOTE (if we have enough samples per class for k_neighbors)
    min_class_count = min(np.bincount(labels))
    if min_class_count >= 2:
        results_smote = evaluate_classifiers(
            embeddings, labels, label_names, use_smote=True,
        )
    else:
        print(f"\nSkipping SMOTE: min class has only {min_class_count} sample(s)")
        results_smote = None

    # Pick the best approach and save
    use_smote_final = False
    if results_smote:
        best_no_smote = max(
            (r["f1_mean"] for r in results.values() if r), default=0,
        )
        best_smote = max(
            (r["f1_mean"] for r in results_smote.values() if r), default=0,
        )
        if best_smote > best_no_smote:
            use_smote_final = True

    metadata = train_best_and_save(
        embeddings, labels, label_names, backend, use_smote=use_smote_final,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", default=None, choices=["ast", "hubert"],
        help="Run for specific backend only (default: run all available)",
    )
    parser.add_argument(
        "--merge-classes", action="store_true",
        help="Merge rare classes (< 15 samples) into 'other' or drop them",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Cry Classifier Training — Multi-model Comparison")
    print("=" * 60)

    backends = [args.backend] if args.backend else ["ast", "hubert"]
    all_results = {}

    for backend in backends:
        meta = run_for_backend(backend, merge_classes=args.merge_classes)
        if meta:
            all_results[backend] = meta

    # Summary
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY: Best CV F1-macro per backend")
        print(f"{'=' * 60}")
        for backend, meta in all_results.items():
            print(f"  {backend:>8}: F1={meta['cv_f1_mean']:.4f} +/- {meta['cv_f1_std']:.4f}  "
                  f"Acc={meta['cv_accuracy_mean']:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()

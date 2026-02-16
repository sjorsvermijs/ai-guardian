"""
Quick experiment: try different class groupings and both embedding models.

Tests:
  1. Binary: hungry vs not_hungry
  2. 3-class: hungry vs discomfort_pain vs tired_other
  3. Full 6-class (baseline)

For each, compares AST vs HuBERT embeddings with top classifiers.

Usage:
  python -m src.pipelines.cry.experiment
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "donate_a_cry_embeddings"


def load(backend):
    path = EMBEDDINGS_DIR / f"cry_{backend}_embeddings.npz"
    if not path.exists():
        return None, None, None
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"], list(data["label_names"])


def make_binary(labels, label_names):
    """hungry=0, not_hungry=1"""
    hungry_idx = label_names.index("hungry")
    new_labels = np.where(labels == hungry_idx, 0, 1)
    return new_labels, ["hungry", "not_hungry"]


def make_3class(embeddings, labels, label_names):
    """hungry=0, discomfort_pain=1 (belly+discomfort), tired_other=2 (tired+burping+cold)"""
    idx_map = {}
    for i, name in enumerate(label_names):
        if name == "hungry":
            idx_map[i] = 0
        elif name in ("belly", "discomfort"):
            idx_map[i] = 1
        else:  # tired, burping, cold
            idx_map[i] = 2

    new_labels = np.array([idx_map[l] for l in labels])
    return new_labels, ["hungry", "discomfort_pain", "tired_other"]


def classifiers():
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, random_state=42, class_weight="balanced",
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", class_weight="balanced",
                random_state=42, C=10.0,
            )),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42,
            )),
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="cosine",
            )),
        ]),
    }


def run_experiment(X, y, label_names, task_name):
    print(f"\n  --- {task_name} ---")
    print(f"  Classes: {label_names}")
    for i, name in enumerate(label_names):
        count = np.sum(y == i)
        print(f"    {name}: {count} ({count/len(y)*100:.0f}%)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"  {'Classifier':<12} {'F1-macro':>10} {'Accuracy':>10}")

    best_f1 = 0
    best_name = ""
    for name, clf in classifiers().items():
        try:
            f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
            acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            print(f"    {name:<10} {f1.mean():>8.4f}   {acc.mean():>8.4f}")
            if f1.mean() > best_f1:
                best_f1 = f1.mean()
                best_name = name
        except Exception as e:
            print(f"    {name:<10} FAILED: {e}")

    return best_f1, best_name


def main():
    print("=" * 60)
    print("CRY CLASSIFICATION EXPERIMENT")
    print("=" * 60)

    results = {}

    for backend in ["ast", "hubert", "mfcc"]:
        embeddings, labels, label_names = load(backend)
        if embeddings is None:
            print(f"\n  Skipping {backend}: no embeddings found")
            continue

        print(f"\n{'#' * 60}")
        print(f"# {backend.upper()} embeddings ({embeddings.shape})")
        print(f"{'#' * 60}")

        # 1. Full 6-class
        f1, clf_name = run_experiment(
            embeddings, labels, label_names, "6-class (original)"
        )
        results[f"{backend}_6class"] = f1

        # 2. Binary
        bin_labels, bin_names = make_binary(labels, label_names)
        f1, clf_name = run_experiment(
            embeddings, bin_labels, bin_names, "Binary (hungry vs not_hungry)"
        )
        results[f"{backend}_binary"] = f1

        # 3. 3-class
        tri_labels, tri_names = make_3class(embeddings, labels, label_names)
        f1, clf_name = run_experiment(
            embeddings, tri_labels, tri_names, "3-class (hungry / discomfort_pain / tired_other)"
        )
        results[f"{backend}_3class"] = f1

    # Feature concatenation: AST + MFCC
    ast_emb, ast_lab, ast_names = load("ast")
    mfcc_emb, _, _ = load("mfcc")
    if ast_emb is not None and mfcc_emb is not None:
        concat = np.hstack([ast_emb, mfcc_emb])
        print(f"\n{'#' * 60}")
        print(f"# AST+MFCC concatenated ({concat.shape})")
        print(f"{'#' * 60}")

        bin_labels, bin_names = make_binary(ast_lab, ast_names)
        f1, _ = run_experiment(concat, bin_labels, bin_names, "Binary (hungry vs not_hungry)")
        results["ast+mfcc_binary"] = f1

        tri_labels, tri_names = make_3class(concat, ast_lab, ast_names)
        f1, _ = run_experiment(concat, tri_labels, tri_names, "3-class")
        results["ast+mfcc_3class"] = f1

    # SMOTE experiments on best config so far
    print(f"\n{'#' * 60}")
    print("# SMOTE experiments (on AST binary)")
    print(f"{'#' * 60}")
    if ast_emb is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.combine import SMOTETomek

        bin_labels, bin_names = make_binary(ast_lab, ast_names)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        smote_clfs = {
            "SVM+SMOTE": ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=42, C=10.0)),
            ]),
            "LogReg+SMOTE": ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
            ]),
            "RF+SMOTE": ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
            ]),
            "SVM+SMOTETomek": ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTETomek(random_state=42)),
                ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=42, C=10.0)),
            ]),
        }

        print(f"\n  --- Binary + SMOTE variants ---")
        print(f"  {'Classifier':<18} {'F1-macro':>10} {'Accuracy':>10}")
        for name, clf in smote_clfs.items():
            try:
                f1 = cross_val_score(clf, ast_emb, bin_labels, cv=cv, scoring="f1_macro")
                acc = cross_val_score(clf, ast_emb, bin_labels, cv=cv, scoring="accuracy")
                print(f"    {name:<16} {f1.mean():>8.4f}   {acc.mean():>8.4f}")
                results[f"ast_binary_{name}"] = f1.mean()
            except Exception as e:
                print(f"    {name:<16} FAILED: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (best F1-macro per config)")
    print(f"{'=' * 60}")
    for key, f1 in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {key:<30} F1={f1:.4f}")


if __name__ == "__main__":
    main()

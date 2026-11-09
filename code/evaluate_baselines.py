#!/usr/bin/env python3
"""Reproducible, leakage-safe baseline comparison for the dengue paper.

The proposed method is an isotonic-calibrated GradientBoostingClassifier on
CLR-transformed leukocyte differentials.  All legacy baselines use exactly the
same cleaned records, fixed split, and train-fitted scaling.  The script writes
the comparison table and its AUC confidence-interval figure used in the paper.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RNG = 42
CSV = (
    "dengue_data/Dengue Fever Hematological Dataset Clinical Insights for "
    "Improved Diagnosis and Patient Management/Dengue Fever Hematological Dataset.csv"
)
DIFF = ["Neutrophils(%)", "Lymphocytes(%)", "Monocytes(%)", "Eosinophils(%)"]


def prepare_data():
    """Audit-informed cleaning and CLR feature engineering, prior to splitting."""
    df = pd.read_csv(CSV)
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["y"] = (df["Result"] == "positive").astype(int)

    features_before_cleaning = df.drop(columns=["Result", "y"])
    key = features_before_cleaning.astype(str).agg("|".join, axis=1)
    conflict = key.groupby(key).transform(
        lambda indices: df.loc[indices.index, "y"].nunique() > 1
    )
    df = df[~conflict].drop_duplicates(
        subset=list(features_before_cleaning.columns) + ["y"], keep="first"
    )
    # PCT is expressed as a percentage.  Its one impossible outlier is capped
    # at a pre-specified physiologic bound instead of a data-derived quantile.
    df["PCT(%)"] = df["PCT(%)"].clip(upper=1.0)

    composition = df[DIFF].astype(float).values + 0.5
    composition /= composition.sum(axis=1, keepdims=True)
    log_composition = np.log(composition)
    clr = log_composition - log_composition.mean(axis=1, keepdims=True)

    X = df.drop(columns=DIFF + ["Result", "y"]).copy()
    for i, column in enumerate(DIFF):
        X[f"clr_{column.split('(')[0].strip()}"] = clr[:, i]
    return X, df["y"]


def bootstrap_auc_ci(y_true, probabilities, n_bootstrap=2000):
    rng = np.random.RandomState(RNG)
    indices = np.arange(len(y_true))
    aucs = []
    y_values = np.asarray(y_true)
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, len(indices), replace=True)
        if np.unique(y_values[sample]).size == 2:
            aucs.append(roc_auc_score(y_values[sample], probabilities[sample]))
    return np.percentile(aucs, [2.5, 97.5])


def metrics(name, y_true, probabilities):
    prediction = (probabilities >= 0.5).astype(int)
    auc_lo, auc_hi = bootstrap_auc_ci(y_true, probabilities)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction),
        "f1": f1_score(y_true, prediction),
        "auc": roc_auc_score(y_true, probabilities),
        "auc_lo": auc_lo,
        "auc_hi": auc_hi,
        "brier": brier_score_loss(y_true, probabilities),
    }


def main():
    X, y = prepare_data()
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RNG
    )
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(
        scaler.transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )
    # Feature selection is performed after the split to avoid peeking at test labels.
    top3 = sorted(
        feature_names, key=lambda col: -abs(pointbiserialr(y_train, X_train[col])[0])
    )[:3]

    baselines = {
        "Majority class": (DummyClassifier(strategy="most_frequent"), feature_names),
        "Parsimonious LR (3 features)": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG),
            top3,
        ),
        "Full LR": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG),
            feature_names,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=500, class_weight="balanced", random_state=RNG, n_jobs=-1
            ),
            feature_names,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RNG,
                n_jobs=1,
            ),
            feature_names,
        ),
        "RBF SVM": (
            SVC(C=1, kernel="rbf", probability=True, class_weight="balanced", random_state=RNG),
            feature_names,
        ),
        "MLP": (
            MLPClassifier(
                hidden_layer_sizes=(100, 50),
                alpha=0.001,
                max_iter=2000,
                early_stopping=True,
                random_state=RNG,
            ),
            feature_names,
        ),
    }

    results = []
    for name, (model, columns) in baselines.items():
        model.fit(X_train[columns], y_train)
        probability = model.predict_proba(X_test[columns])[:, 1]
        results.append(metrics(name, y_test, probability))

    # Proposed model: calibration is learned by five folds inside the training split only.
    proposed = CalibratedClassifierCV(
        GradientBoostingClassifier(random_state=RNG), method="isotonic", cv=5
    ).fit(X_train, y_train)
    proposed_probability = proposed.predict_proba(X_test)[:, 1]
    results.append(metrics("Proposed: calibrated GB + CLR", y_test, proposed_probability))

    table = pd.DataFrame(results)
    table.to_csv("baseline_comparison.csv", index=False)
    with open("baseline_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"top3": top3, "rows": results}, handle, indent=2, default=float
        )

    # Compact forest plot: intervals and point estimates are easier to compare
    # than bars, while the proposed model remains visually distinct.
    plt.rcParams.update({
        "figure.dpi": 160,
        "font.size": 10,
        "axes.grid": False,
        "axes.facecolor": "white",
    })
    plot_table = table[table["model"] != "Majority class"].copy()
    plot_table = plot_table.iloc[::-1].reset_index(drop=True)
    labels = ["Proposed CGB-CLR", "MLP", "RBF SVM", "XGBoost", "Random forest", "Full LR", "3-feature LR"]
    y = np.arange(len(plot_table))
    fig, ax = plt.subplots(figsize=(7.6, 4.25))

    for i in range(len(plot_table)):
        if i % 2 == 0:
            ax.axhspan(i - 0.48, i + 0.48, color="#f4f7f9", zorder=0)
    ax.axvline(0.5, color="#8b98a1", linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)

    for i, row in plot_table.iterrows():
        proposed_row = row["model"].startswith("Proposed")
        color = "#d55e00" if proposed_row else "#294c60"
        ax.hlines(i, row["auc_lo"], row["auc_hi"], color=color,
                  linewidth=3.0 if proposed_row else 2.0, zorder=2)
        ax.vlines([row["auc_lo"], row["auc_hi"]], i - 0.12, i + 0.12,
                  color=color, linewidth=1.4, zorder=2)
        ax.scatter(row["auc"], i, s=82 if proposed_row else 48,
                   marker="D" if proposed_row else "o", color=color,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(0.787, i, f"{row['auc']:.3f}", ha="right", va="center",
                fontsize=9, fontweight="bold" if proposed_row else "normal",
                color=color)

    ax.set_yticks(y, labels)
    ax.set_xlim(0.49, 0.79)
    ax.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    ax.set_ylim(-0.6, len(plot_table) - 0.4)
    ax.set_xlabel("AUC-ROC (95% bootstrap CI)")
    ax.set_title("Held-out discrimination", loc="left", weight="bold", pad=8)
    ax.text(0.79, len(plot_table) - 0.17, "AUC", ha="right", va="bottom",
            fontsize=8.5, weight="bold", color="#52636d")
    ax.grid(axis="x", color="#d9e0e4", linewidth=0.7, alpha=0.75, zorder=0)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#73838c")
    fig.tight_layout()
    fig.savefig("figs/eval_baseline_auc_ci.png")
    plt.close(fig)

    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nThree transparent LR features: {top3}")
    print("Wrote baseline_comparison.csv, baseline_comparison.json, and figs/eval_baseline_auc_ci.png")


if __name__ == "__main__":
    main()

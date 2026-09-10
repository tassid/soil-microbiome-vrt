"""
model.py

Treinamento do classificador de estresse hídrico (Random Forest) com
validação cruzada aninhada, reproduzindo o protocolo descrito em
Hagen et al. (2024): 5-fold CV externa para avaliação de desempenho,
com busca de hiperparâmetros na malha interna de cada fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8],
    "min_samples_leaf": [1, 3],
}


@dataclass
class FoldResult:
    fold: int
    model: RandomForestClassifier
    test_index: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict = field(default_factory=dict)


@dataclass
class NestedCVResult:
    folds: list[FoldResult]
    metrics_mean: dict
    metrics_std: dict
    feature_names: list[str]


def _compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "auc": roc_auc_score(y_true, y_proba),
    }


def train_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    random_state: int = 42,
) -> NestedCVResult:
    """Treina um RandomForestClassifier com CV aninhada.

    Parameters
    ----------
    X:
        Tabela de features (abundância relativa por táxon), uma linha por
        amostra.
    y:
        Rótulo binário (1 = "Seca", 0 = "Controle").
    n_outer_folds:
        Número de folds da validação cruzada externa, usada para
        avaliação de desempenho (padrão: 5, igual ao artigo de
        referência).
    n_inner_folds:
        Número de folds da validação cruzada interna, usada apenas para
        otimização de hiperparâmetros dentro de cada fold externo.

    Returns
    -------
    NestedCVResult
        Contém o modelo, as predições e as métricas de cada fold, além
        da média e do desvio padrão agregados.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    fold_results: list[FoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            PARAM_GRID,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_proba)

        fold_results.append(
            FoldResult(
                fold=fold_idx,
                model=best_model,
                test_index=test_idx,
                y_test=y_test.to_numpy(),
                y_pred=y_pred,
                y_proba=y_proba,
                metrics=metrics,
            )
        )

    metric_keys = fold_results[0].metrics.keys()
    metrics_mean = {k: float(np.mean([f.metrics[k] for f in fold_results])) for k in metric_keys}
    metrics_std = {k: float(np.std([f.metrics[k] for f in fold_results])) for k in metric_keys}

    return NestedCVResult(
        folds=fold_results,
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
        feature_names=list(X.columns),
    )


if __name__ == "__main__":
    from data_simulation import simulate_grass_drought_dataset

    df = simulate_grass_drought_dataset()
    feature_cols = [c for c in df.columns if c not in ("sample_id", "label", "x", "y")]
    X = df[feature_cols]
    y = (df["label"] == "Seca").astype(int)

    result = train_nested_cv(X, y)
    print("Média das métricas (5-fold nested CV):")
    for k, v in result.metrics_mean.items():
        print(f"  {k}: {v:.3f} +/- {result.metrics_std[k]:.3f}")

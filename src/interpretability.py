"""
interpretability.py

Extração de valores SHAP para cada fold do modelo treinado em
`model.py`, seguindo o protocolo de Hagen et al. (2024): os táxons são
considerados marcadores de estresse hídrico quando o sinal de
enriquecimento (Controle x Seca) é consistente em pelo menos 4 dos 5
folds da validação cruzada externa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from model import NestedCVResult


@dataclass
class MarkerTaxaResult:
    mean_abs_shap: pd.Series
    fold_enrichment: pd.DataFrame  # linhas = táxons, colunas = folds, valores em {"Controle", "Seca", "Indefinido"}
    consensus_enrichment: pd.Series
    top_markers: pd.DataFrame


def compute_shap_per_fold(cv_result: NestedCVResult, X: pd.DataFrame) -> list[np.ndarray]:
    """Calcula os valores SHAP do modelo de cada fold sobre seu próprio
    conjunto de teste, retornando uma lista de matrizes (n_amostras_teste,
    n_features)."""
    shap_values_per_fold = []
    for fold in cv_result.folds:
        X_test_fold = X.iloc[fold.test_index]
        explainer = shap.TreeExplainer(fold.model)
        raw = explainer.shap_values(X_test_fold)
        # Compatibilidade entre versões do shap: pode retornar lista
        # [classe0, classe1] ou array 3D (amostras, features, classes).
        if isinstance(raw, list):
            sv = raw[1]
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            sv = raw[:, :, 1]
        else:
            sv = raw
        shap_values_per_fold.append(sv)
    return shap_values_per_fold


def identify_marker_taxa(
    cv_result: NestedCVResult,
    X: pd.DataFrame,
    shap_values_per_fold: list[np.ndarray],
    min_fold_agreement: int = 4,
    top_n: int = 10,
) -> MarkerTaxaResult:
    """Consolida os valores SHAP de todos os folds em uma tabela de
    táxons marcadores, com consenso de enriquecimento (Controle/Seca).
    """
    feature_names = cv_result.feature_names
    n_folds = len(cv_result.folds)

    abs_shap_per_fold = []
    enrichment_per_fold = []

    for fold, sv in zip(cv_result.folds, shap_values_per_fold):
        abs_mean = np.abs(sv).mean(axis=0)
        abs_shap_per_fold.append(abs_mean)

        # Direção do enriquecimento: correlação, feature a feature, entre
        # o valor bruto da abundância no conjunto de teste do fold e o
        # respectivo valor SHAP para a classe "Seca". Correlação positiva
        # significa que abundâncias mais altas do táxon empurram a
        # predição para "Seca" (enriquecimento sob seca); correlação
        # negativa indica enriquecimento sob "Controle". Esse é o mesmo
        # princípio usado nos gráficos de sumário SHAP (cor = valor da
        # feature, eixo x = impacto).
        X_test_fold = X.iloc[fold.test_index].to_numpy()
        n_features = sv.shape[1]
        corr = np.zeros(n_features)
        for j in range(n_features):
            feat_col = X_test_fold[:, j]
            shap_col = sv[:, j]
            if np.std(feat_col) == 0 or np.std(shap_col) == 0:
                corr[j] = 0.0
            else:
                corr[j] = np.corrcoef(feat_col, shap_col)[0, 1]
        enrichment = np.where(corr > 0, "Seca", "Controle")
        enrichment_per_fold.append(enrichment)

    abs_shap_matrix = np.vstack(abs_shap_per_fold)  # (n_folds, n_features)
    mean_abs_shap = pd.Series(abs_shap_matrix.mean(axis=0), index=feature_names).sort_values(ascending=False)

    enrichment_matrix = np.vstack(enrichment_per_fold)  # (n_folds, n_features)
    fold_enrichment = pd.DataFrame(
        enrichment_matrix.T, index=feature_names, columns=[f"fold_{i+1}" for i in range(n_folds)]
    )

    def _consensus(row: pd.Series) -> str:
        counts = row.value_counts()
        top_label, top_count = counts.index[0], counts.iloc[0]
        return top_label if top_count >= min_fold_agreement else "Indefinido"

    consensus_enrichment = fold_enrichment.apply(_consensus, axis=1)

    top_markers = pd.DataFrame(
        {
            "taxon": mean_abs_shap.index,
            "mean_abs_shap": mean_abs_shap.values,
            "consensus_enrichment": consensus_enrichment.loc[mean_abs_shap.index].values,
        }
    ).head(top_n)

    return MarkerTaxaResult(
        mean_abs_shap=mean_abs_shap,
        fold_enrichment=fold_enrichment,
        consensus_enrichment=consensus_enrichment,
        top_markers=top_markers,
    )


if __name__ == "__main__":
    from data_simulation import simulate_grass_drought_dataset
    from model import train_nested_cv

    df = simulate_grass_drought_dataset()
    feature_cols = [c for c in df.columns if c not in ("sample_id", "label", "x", "y")]
    X = df[feature_cols]
    y = (df["label"] == "Seca").astype(int)

    cv_result = train_nested_cv(X, y)
    shap_per_fold = compute_shap_per_fold(cv_result, X)
    markers = identify_marker_taxa(cv_result, X, shap_per_fold)

    print("Top táxons marcadores (consenso de enriquecimento entre folds):")
    print(markers.top_markers.to_string(index=False))

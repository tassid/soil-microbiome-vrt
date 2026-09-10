"""
prescription_map.py

Formaliza a função f: P(Seca) -> (classe, dose, composição), que converte
a saída probabilística do classificador de estresse hídrico em um mapa
de prescrição compatível com um controlador de taxa variável (VRT) de
pulverizador, conforme especificado no pré-projeto (seção 5.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DoseConfig:
    """Parâmetros operacionais da função de mapeamento probabilidade -> dose.

    prob_threshold_low:
        Abaixo deste limiar, a zona é tratada como "Controle" e recebe a
        dose mínima (ou nenhuma aplicação de bioinsumo corretivo).
    prob_threshold_high:
        Acima deste limiar, a zona recebe a dose máxima.
    dose_min_l_ha, dose_max_l_ha:
        Faixa operacional de dose do bioinsumo, em litros por hectare.
        Entre os dois limiares de probabilidade, a dose escala
        linearmente.
    """

    prob_threshold_low: float = 0.35
    prob_threshold_high: float = 0.75
    dose_min_l_ha: float = 0.0
    dose_max_l_ha: float = 40.0


def probability_to_dose(p_drought: np.ndarray, config: DoseConfig) -> np.ndarray:
    """Converte a probabilidade de classe 'Seca' em dose de bioinsumo
    (L/ha), com rampa linear entre os limiares de decisão."""
    p = np.clip(p_drought, 0.0, 1.0)
    low, high = config.prob_threshold_low, config.prob_threshold_high
    dose = np.where(
        p <= low,
        config.dose_min_l_ha,
        np.where(
            p >= high,
            config.dose_max_l_ha,
            config.dose_min_l_ha
            + (p - low) / (high - low) * (config.dose_max_l_ha - config.dose_min_l_ha),
        ),
    )
    return dose


def suggest_inoculant_composition(
    marker_top_markers: pd.DataFrame,
    max_components: int = 3,
) -> str:
    """Sugere a composição do inoculante microbiano com base nos táxons
    identificados como depletados sob "Seca" (isto é, enriquecidos sob
    "Controle"), que são candidatos a reposição via bioinoculação.

    Esta é uma regra de decisão simples e ilustrativa: em um pipeline de
    produção, a escolha do inoculante dependeria de disponibilidade
    comercial e de evidência funcional (capacidade de promoção de
    crescimento) de cada táxon, não apenas da direção do enriquecimento.
    """
    depleted = marker_top_markers[marker_top_markers["consensus_enrichment"] == "Controle"]
    top = depleted.sort_values("mean_abs_shap", ascending=False).head(max_components)
    if top.empty:
        return "Sem táxon depletado identificado com consenso suficiente"
    return " + ".join(top["taxon"].tolist())


def build_prescription_map(
    df: pd.DataFrame,
    p_drought: np.ndarray,
    marker_top_markers: pd.DataFrame,
    config: DoseConfig | None = None,
) -> pd.DataFrame:
    """Monta o mapa de prescrição final, uma linha por amostra
    georreferenciada, pronto para ser consumido por um controlador VRT.

    Parameters
    ----------
    df:
        Tabela original de amostras, deve conter as colunas
        'sample_id', 'x', 'y' (coordenadas dentro do talhão).
    p_drought:
        Vetor de probabilidades P(Seca) por amostra, na mesma ordem de
        `df`.
    marker_top_markers:
        Tabela de táxons marcadores (ver `interpretability.py`), usada
        para sugerir a composição do inoculante em zonas de estresse.
    config:
        Parâmetros de dose. Usa os valores padrão de `DoseConfig` se
        omitido.
    """
    config = config or DoseConfig()
    dose = probability_to_dose(p_drought, config)
    predicted_class = np.where(p_drought >= config.prob_threshold_low, "Seca", "Controle")
    composition = suggest_inoculant_composition(marker_top_markers)

    prescription = pd.DataFrame(
        {
            "sample_id": df["sample_id"].values,
            "x": df["x"].values,
            "y": df["y"].values,
            "p_drought": p_drought,
            "classe_prevista": predicted_class,
            "dose_bioinsumo_l_ha": dose,
            "composicao_inoculante": np.where(dose > 0, composition, "Nenhuma (dose zero)"),
        }
    )
    return prescription


if __name__ == "__main__":
    from data_simulation import simulate_grass_drought_dataset
    from model import train_nested_cv
    from interpretability import compute_shap_per_fold, identify_marker_taxa

    df = simulate_grass_drought_dataset()
    feature_cols = [c for c in df.columns if c not in ("sample_id", "label", "x", "y")]
    X = df[feature_cols]
    y = (df["label"] == "Seca").astype(int)

    cv_result = train_nested_cv(X, y)
    shap_per_fold = compute_shap_per_fold(cv_result, X)
    markers = identify_marker_taxa(cv_result, X, shap_per_fold)

    # Usa o modelo do último fold treinado para gerar P(Seca) em todas as
    # amostras, apenas para fins de demonstração do mapa de prescrição.
    final_model = cv_result.folds[-1].model
    p_drought_all = final_model.predict_proba(X)[:, 1]

    prescription = build_prescription_map(df, p_drought_all, markers.top_markers)
    print(prescription.head(10).to_string(index=False))
    print(f"\nProporção de zonas com dose > 0: {(prescription['dose_bioinsumo_l_ha'] > 0).mean():.1%}")

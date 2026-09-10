"""Testes de sanidade do pipeline completo.

Não validam acurácia científica (os dados são sintéticos), apenas
garantem que cada etapa produz saídas com a forma e os tipos esperados,
e que o pipeline executa de ponta a ponta sem erros.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_simulation import simulate_grass_drought_dataset
from model import train_nested_cv
from interpretability import compute_shap_per_fold, identify_marker_taxa
from prescription_map import DoseConfig, build_prescription_map, probability_to_dose
from vrt_simulation import simulate_field_application


@pytest.fixture(scope="module")
def dataset():
    return simulate_grass_drought_dataset(n_control=60, n_drought=60, field_size=(50, 50), seed=1)


@pytest.fixture(scope="module")
def feature_columns(dataset):
    return [c for c in dataset.columns if c not in ("sample_id", "label", "x", "y")]


def test_simulated_dataset_shape(dataset, feature_columns):
    assert len(dataset) == 120
    assert set(dataset["label"].unique()) == {"Controle", "Seca"}
    # abundâncias relativas devem somar ~1 por amostra
    sums = dataset[feature_columns].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


@pytest.fixture(scope="module")
def cv_result(dataset, feature_columns):
    X = dataset[feature_columns]
    y = (dataset["label"] == "Seca").astype(int)
    return train_nested_cv(X, y, n_outer_folds=3, n_inner_folds=2)


def test_nested_cv_metrics_in_valid_range(cv_result):
    for key, value in cv_result.metrics_mean.items():
        assert 0.0 <= value <= 1.0, f"Métrica {key} fora do intervalo [0, 1]: {value}"
    assert len(cv_result.folds) == 3


def test_shap_and_marker_consensus(dataset, feature_columns, cv_result):
    X = dataset[feature_columns]
    shap_per_fold = compute_shap_per_fold(cv_result, X)
    assert len(shap_per_fold) == len(cv_result.folds)
    for sv in shap_per_fold:
        assert sv.shape[1] == len(feature_columns)

    markers = identify_marker_taxa(cv_result, X, shap_per_fold, min_fold_agreement=2, top_n=5)
    assert len(markers.top_markers) == 5
    assert set(markers.consensus_enrichment.unique()) <= {"Controle", "Seca", "Indefinido"}


def test_probability_to_dose_monotonic():
    config = DoseConfig(prob_threshold_low=0.3, prob_threshold_high=0.7, dose_min_l_ha=0, dose_max_l_ha=50)
    probs = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    doses = probability_to_dose(probs, config)
    assert doses[0] == 0
    assert doses[-1] == 50
    assert np.all(np.diff(doses) >= 0), "A dose deve ser monotonicamente não decrescente com a probabilidade"


def test_prescription_map_and_vrt_simulation(dataset, feature_columns, cv_result):
    X = dataset[feature_columns]
    shap_per_fold = compute_shap_per_fold(cv_result, X)
    markers = identify_marker_taxa(cv_result, X, shap_per_fold, min_fold_agreement=2)

    final_model = cv_result.folds[-1].model
    p_drought = final_model.predict_proba(X)[:, 1]
    prescription = build_prescription_map(dataset, p_drought, markers.top_markers)

    assert len(prescription) == len(dataset)
    assert prescription["dose_bioinsumo_l_ha"].between(0, 40).all()

    result = simulate_field_application(prescription, field_size=(50.0, 50.0), grid_shape=(20, 20))
    assert result.volume_vrt_l >= 0
    assert result.volume_uniform_l >= 0
    assert -1e-6 <= result.savings_fraction <= 1.0 + 1e-6

"""
vrt_simulation.py

Simula a operação de um pulverizador de taxa variável (VRT) sobre um
talhão hipotético, a partir do mapa de prescrição gerado em
`prescription_map.py`. Compara o volume de bioinsumo aplicado no modo
convencional (dose uniforme) versus o modo de taxa variável guiado pelo
classificador de estresse hídrico.

Esta simulação não representa hardware real (controlador CAN/ISOBUS);
seu objetivo é quantificar, em nível de especificação, o ganho potencial
de eficiência de insumo discutido no pré-projeto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FieldGridResult:
    grid_shape: tuple[int, int]
    dose_grid: np.ndarray            # dose por célula, modo VRT (L/ha)
    class_grid: np.ndarray           # classe prevista por célula ("Seca"/"Controle")
    uniform_dose_l_ha: float
    area_ha: float
    volume_uniform_l: float
    volume_vrt_l: float
    savings_fraction: float


def interpolate_to_grid(
    prescription: pd.DataFrame,
    field_size: tuple[float, float],
    grid_shape: tuple[int, int] = (40, 40),
) -> tuple[np.ndarray, np.ndarray]:
    """Interpola o mapa de prescrição (amostras pontuais) para uma grade
    regular, usando o vizinho mais próximo — suficiente para esta
    simulação de especificação, sem exigir dependências de geoprocessamento
    completas (ex.: krigagem)."""
    from scipy.spatial import cKDTree

    nx, ny = grid_shape
    xs = np.linspace(0, field_size[0], nx)
    ys = np.linspace(0, field_size[1], ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    sample_points = prescription[["x", "y"]].to_numpy()
    tree = cKDTree(sample_points)
    _, nearest_idx = tree.query(grid_points, k=1)

    dose_grid = prescription["dose_bioinsumo_l_ha"].to_numpy()[nearest_idx].reshape(ny, nx)
    class_grid = prescription["classe_prevista"].to_numpy()[nearest_idx].reshape(ny, nx)
    return dose_grid, class_grid


def simulate_field_application(
    prescription: pd.DataFrame,
    field_size: tuple[float, float] = (40.0, 40.0),
    grid_shape: tuple[int, int] = (40, 40),
    uniform_dose_l_ha: float | None = None,
) -> FieldGridResult:
    """Executa a simulação de aplicação uniforme versus taxa variável.

    Parameters
    ----------
    prescription:
        Mapa de prescrição gerado por `build_prescription_map`.
    field_size:
        Dimensões do talhão hipotético, em metros (largura, comprimento).
    grid_shape:
        Resolução da grade de simulação (número de células em x e y).
    uniform_dose_l_ha:
        Dose fixa usada no modo de aplicação uniforme (convencional). Se
        omitida, usa a dose máxima definida no mapa de prescrição — ou
        seja, o cenário conservador em que toda a área receberia a dose
        que hoje só é aplicada nas zonas mais críticas.
    """
    dose_grid, class_grid = interpolate_to_grid(prescription, field_size, grid_shape)

    if uniform_dose_l_ha is None:
        uniform_dose_l_ha = float(prescription["dose_bioinsumo_l_ha"].max())

    area_ha = (field_size[0] * field_size[1]) / 10_000.0
    n_cells = grid_shape[0] * grid_shape[1]
    cell_area_ha = area_ha / n_cells

    volume_uniform_l = uniform_dose_l_ha * area_ha
    volume_vrt_l = float(dose_grid.sum() * cell_area_ha)
    savings_fraction = 1.0 - (volume_vrt_l / volume_uniform_l) if volume_uniform_l > 0 else 0.0

    return FieldGridResult(
        grid_shape=grid_shape,
        dose_grid=dose_grid,
        class_grid=class_grid,
        uniform_dose_l_ha=uniform_dose_l_ha,
        area_ha=area_ha,
        volume_uniform_l=volume_uniform_l,
        volume_vrt_l=volume_vrt_l,
        savings_fraction=savings_fraction,
    )


def plot_field_result(result: FieldGridResult, output_path: str) -> None:
    """Gera um heatmap do mapa de prescrição e salva como PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(result.dose_grid, origin="lower", cmap="YlOrRd", aspect="auto")
    ax.set_title("Mapa de prescrição — dose de bioinsumo (L/ha)")
    ax.set_xlabel("Posição X na grade do talhão")
    ax.set_ylabel("Posição Y na grade do talhão")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Dose (L/ha)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from data_simulation import simulate_grass_drought_dataset
    from model import train_nested_cv
    from interpretability import compute_shap_per_fold, identify_marker_taxa
    from prescription_map import build_prescription_map

    df = simulate_grass_drought_dataset()
    feature_cols = [c for c in df.columns if c not in ("sample_id", "label", "x", "y")]
    X = df[feature_cols]
    y = (df["label"] == "Seca").astype(int)

    cv_result = train_nested_cv(X, y)
    shap_per_fold = compute_shap_per_fold(cv_result, X)
    markers = identify_marker_taxa(cv_result, X, shap_per_fold)

    final_model = cv_result.folds[-1].model
    p_drought_all = final_model.predict_proba(X)[:, 1]
    prescription = build_prescription_map(df, p_drought_all, markers.top_markers)

    result = simulate_field_application(prescription, field_size=(40.0, 40.0))
    print(f"Área do talhão simulado: {result.area_ha:.2f} ha")
    print(f"Volume (aplicação uniforme): {result.volume_uniform_l:.1f} L")
    print(f"Volume (taxa variável, VRT):  {result.volume_vrt_l:.1f} L")
    print(f"Economia estimada de insumo: {result.savings_fraction:.1%}")

    plot_field_result(result, "/tmp/prescription_map_demo.png")
    print("Mapa salvo em /tmp/prescription_map_demo.png")

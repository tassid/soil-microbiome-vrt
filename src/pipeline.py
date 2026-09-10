"""
pipeline.py

Orquestra o fluxo completo do projeto:

    1. dados (reais ou simulados)
    2. treinamento do Random Forest com CV aninhada
    3. interpretabilidade (SHAP) e identificação de táxons marcadores
    4. mapa de prescrição
    5. simulação de aplicação em taxa variável (VRT)
    6. relatório e artefatos de saída

Uso:
    python src/pipeline.py --output-dir outputs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_simulation import simulate_grass_drought_dataset
from model import train_nested_cv
from interpretability import compute_shap_per_fold, identify_marker_taxa
from prescription_map import DoseConfig, build_prescription_map
from vrt_simulation import simulate_field_application, plot_field_result


def run_pipeline(output_dir: Path, field_size: tuple[float, float] = (500.0, 500.0)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dados
    df = simulate_grass_drought_dataset(field_size=field_size)
    feature_cols = [c for c in df.columns if c not in ("sample_id", "label", "x", "y")]
    X = df[feature_cols]
    y = (df["label"] == "Seca").astype(int)
    df.to_csv(output_dir / "feature_table.csv", index=False)

    # 2. Modelo
    cv_result = train_nested_cv(X, y)
    metrics_report = {
        "metrics_mean": cv_result.metrics_mean,
        "metrics_std": cv_result.metrics_std,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    # 3. Interpretabilidade
    shap_per_fold = compute_shap_per_fold(cv_result, X)
    markers = identify_marker_taxa(cv_result, X, shap_per_fold)
    markers.top_markers.to_csv(output_dir / "marker_taxa.csv", index=False)

    # 4. Mapa de prescrição
    final_model = cv_result.folds[-1].model
    p_drought_all = final_model.predict_proba(X)[:, 1]
    prescription = build_prescription_map(df, p_drought_all, markers.top_markers, config=DoseConfig())
    prescription.to_csv(output_dir / "prescription_map.csv", index=False)

    # 5. Simulação VRT
    field_result = simulate_field_application(prescription, field_size=field_size)
    plot_field_result(field_result, str(output_dir / "prescription_map.png"))

    vrt_report = {
        "area_ha": field_result.area_ha,
        "uniform_dose_l_ha": field_result.uniform_dose_l_ha,
        "volume_uniform_l": field_result.volume_uniform_l,
        "volume_vrt_l": field_result.volume_vrt_l,
        "savings_fraction": field_result.savings_fraction,
    }
    with open(output_dir / "vrt_simulation.json", "w", encoding="utf-8") as f:
        json.dump(vrt_report, f, indent=2, ensure_ascii=False)

    # 6. Relatório resumo em Markdown
    summary_lines = [
        "# Resumo da execução do pipeline\n",
        "## Desempenho do classificador (5-fold nested CV)\n",
        "| Métrica | Média | Desvio padrão |",
        "|---|---|---|",
    ]
    for k in cv_result.metrics_mean:
        summary_lines.append(
            f"| {k} | {cv_result.metrics_mean[k]:.3f} | {cv_result.metrics_std[k]:.3f} |"
        )
    summary_lines.append("\n## Táxons marcadores (top 10, consenso entre folds)\n")
    summary_lines.append(markers.top_markers.to_markdown(index=False))
    summary_lines.append("\n## Simulação de aplicação em taxa variável (VRT)\n")
    summary_lines.append(f"- Área do talhão simulado: {field_result.area_ha:.2f} ha")
    summary_lines.append(f"- Volume aplicado (uniforme): {field_result.volume_uniform_l:.1f} L")
    summary_lines.append(f"- Volume aplicado (taxa variável): {field_result.volume_vrt_l:.1f} L")
    summary_lines.append(f"- Economia estimada de insumo: {field_result.savings_fraction:.1%}")

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Pipeline concluído. Artefatos salvos em: {output_dir.resolve()}")
    return {
        "metrics": metrics_report,
        "vrt": vrt_report,
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo: microbioma do solo -> VRT")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()
    run_pipeline(Path(args.output_dir))


if __name__ == "__main__":
    main()

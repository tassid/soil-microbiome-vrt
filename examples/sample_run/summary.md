# Resumo da execução do pipeline

## Desempenho do classificador (5-fold nested CV)

| Métrica | Média | Desvio padrão |
|---|---|---|
| accuracy | 0.862 | 0.014 |
| f1 | 0.858 | 0.014 |
| precision | 0.861 | 0.025 |
| recall | 0.855 | 0.024 |
| auc | 0.928 | 0.020 |

## Táxons marcadores (top 10, consenso entre folds)

| taxon           |   mean_abs_shap | consensus_enrichment   |
|:----------------|----------------:|:-----------------------|
| Bacillus        |      0.0974857  | Seca                   |
| Kribbella       |      0.0861631  | Controle               |
| Occallatibacter |      0.080688   | Controle               |
| Streptomyces    |      0.0794382  | Seca                   |
| Genus_035       |      0.00663373 | Seca                   |
| Genus_004       |      0.00659582 | Controle               |
| Genus_036       |      0.00604539 | Seca                   |
| Genus_021       |      0.00519998 | Seca                   |
| Genus_038       |      0.00512131 | Controle               |
| Genus_028       |      0.0049601  | Seca                   |

## Simulação de aplicação em taxa variável (VRT)

- Área do talhão simulado: 25.00 ha
- Volume aplicado (uniforme): 1000.0 L
- Volume aplicado (taxa variável): 480.5 L
- Economia estimada de insumo: 51.9%
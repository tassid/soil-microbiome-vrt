# Soil Microbiome VRT

Classificação interpretável de estresse hídrico do solo via microbioma
bacteriano (Random Forest + SHAP), integrada a um mapa de prescrição
para pulverização agrícola de taxa variável (VRT).

Projeto desenvolvido para a disciplina de **Mecanização Agrícola** —
PPGTCA / UTFPR, Campus Medianeira.

> Pré-projeto de pesquisa completo (fundamentação teórica, referências,
> cronograma): `docs/pre-projeto-mecanizacao.docx`.

## Ideia do projeto

1. Um classificador treinado sobre dados de microbioma do solo (16S
   rRNA) prevê a probabilidade de uma amostra estar sob estresse
   hídrico ("Seca") ou não ("Controle"), seguindo o método de
   [Hagen et al. (2024)](https://doi.org/10.1186/s40793-024-00578-1).
2. Valores SHAP identificam quais táxons bacterianos mais contribuem
   para essa classificação (táxons marcadores), com direção de
   enriquecimento (Controle/Seca).
3. Essa saída é convertida em um **mapa de prescrição**: por zona da
   lavoura, uma dose de bioinsumo (inoculante microbiano) e uma
   composição sugerida, com base nos táxons depletados sob seca.
4. Uma simulação de campo compara a aplicação **uniforme** (dose fixa
   em toda a área) com a aplicação em **taxa variável (VRT)** guiada
   pelo mapa de prescrição, estimando a economia de insumo.

Esta é a mesma arquitetura de decisão→atuação de um projeto anterior da
autora (detecção de insetos-praga via CNN+ViT acoplada a um
pulverizador), aplicada a um novo domínio de dado (microbioma) e um
novo modelo (Random Forest interpretável).

## Estrutura do repositório

```
soil-microbiome-vrt/
├── src/
│   ├── data_simulation.py     # dados sintéticos (ver docs/methodology.md)
│   ├── model.py                # Random Forest + validação cruzada aninhada
│   ├── interpretability.py     # SHAP e consenso de táxons marcadores
│   ├── prescription_map.py     # função probabilidade -> dose/composição
│   ├── vrt_simulation.py       # simulação de campo e comparação uniforme x VRT
│   └── pipeline.py             # orquestra as etapas acima
├── tests/
│   └── test_pipeline.py        # testes automatizados (pytest)
├── docs/
│   ├── methodology.md          # detalhamento técnico de cada etapa
│   └── pre-projeto-mecanizacao.docx
├── examples/
│   └── sample_run/             # saída de exemplo já gerada (ver abaixo)
└── requirements.txt
```

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/pipeline.py --output-dir outputs/
```

Isso gera em `outputs/`:

| Arquivo | Conteúdo |
|---|---|
| `feature_table.csv` | Tabela de abundância relativa simulada, com rótulo e coordenadas |
| `metrics.json` | Métricas do classificador (média e desvio padrão, 5-fold nested CV) |
| `marker_taxa.csv` | Top 10 táxons marcadores, com importância SHAP e direção de enriquecimento |
| `prescription_map.csv` | Mapa de prescrição por amostra (classe, dose, composição do inoculante) |
| `prescription_map.png` | Heatmap do mapa de prescrição interpolado em grade |
| `vrt_simulation.json` | Comparação de volume aplicado: uniforme x taxa variável |
| `summary.md` | Resumo legível de toda a execução |

Um exemplo já executado está em `examples/sample_run/` para inspeção
sem precisar rodar o pipeline.

### Rodando os testes

```bash
pytest tests/ -v
```

## Resultado de exemplo

No run de exemplo incluído (`examples/sample_run/`), com um talhão
sintético de 25 ha:

* Acurácia média do classificador: **~86%** (5-fold nested CV)
* AUC média: **~0,93**
* Economia estimada de bioinsumo com taxa variável: **~52%** frente à
  aplicação uniforme

Esses números refletem os dados **sintéticos** gerados por
`data_simulation.py`, não o dataset real do artigo — servem para
validar a lógica do pipeline de ponta a ponta. Veja
`docs/methodology.md` para o caminho de substituição pelos dados reais
do BioProject `PRJNA369551`.

## Limitações

Descritas em detalhe em `docs/methodology.md` e no pré-projeto:
tempo de resposta do sequenciamento 16S frente à decisão em campo em
tempo real, e queda de desempenho do classificador observada pelo
próprio artigo de referência ao generalizar para outro dataset
(Sorghum-Drought).

## Referência

Hagen, M., Dass, R., Westhues, C., Blom, J., Schultheiss, S. J., & Patz, S. (2024).
Interpretable machine learning decodes soil microbiome's response to drought stress.
*Environmental Microbiome*, 19, 35. https://doi.org/10.1186/s40793-024-00578-1

## Licença

MIT — ver `LICENSE`.

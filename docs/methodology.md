# Metodologia

Este documento detalha as decisões técnicas de cada etapa do pipeline,
sua correspondência com a literatura de referência e o caminho para
substituir os dados sintéticos por dados reais.

## 1. Referência científica

O desenho do classificador segue:

> Hagen, M., Dass, R., Westhues, C., Blom, J., Schultheiss, S. J., & Patz, S. (2024).
> Interpretable machine learning decodes soil microbiome's response to drought stress.
> *Environmental Microbiome*, 19, 35. https://doi.org/10.1186/s40793-024-00578-1

Esse artigo demonstra que um Random Forest Classifier (RFC) treinado
sobre abundância relativa de táxons bacterianos do solo (16S rRNA)
discrimina amostras em regime de seca versus controle com acurácia de
92,3% no nível de gênero, e usa valores SHAP para identificar táxons
marcadores (ex.: *Kribbella*, *Streptomyces*, *Occallatibacter*).

A camada de mecanização (mapa de prescrição e simulação de taxa
variável) é uma contribuição original deste projeto, não descrita no
artigo — ela integra o classificador a um pulverizador de taxa variável
(VRT), conforme especificado no pré-projeto da disciplina de
Mecanização Agrícola (`docs/pre-projeto-mecanizacao.docx` no
repositório).

## 2. Por que dados sintéticos, e como substituí-los

O dataset real ("Grass-Drought", Naylor et al., 2017) está disponível
publicamente no NCBI Sequence Read Archive sob o BioProject
`PRJNA369551`, mas seu uso requer:

1. Download dos arquivos FASTQ brutos via SRA Toolkit (`prefetch` /
   `fasterq-dump`).
2. Processamento via DADA2 (R), incluindo remoção de erros de
   sequenciamento, atribuição taxonômica pela base SILVA com
   classificador RDP, filtragem de prevalência e rarefação.
3. Exportação de uma tabela de features (abundância relativa por
   táxon, uma linha por amostra) com metadados de regime hídrico e,
   idealmente, coordenadas geográficas de coleta.

Esse processamento depende de ferramentas de bioinformática (R,
Bioconductor, SRA Toolkit) e de acesso à rede do NCBI, fora do escopo
de execução deste repositório em ambiente local simplificado. Para
manter o projeto executável de ponta a ponta, `src/data_simulation.py`
gera uma tabela sintética com a mesma estrutura estatística: amostras
rotuladas "Controle"/"Seca", abundâncias relativas de um conjunto de
gêneros (alguns desenhados para responder à seca, simulando os
marcadores relatados no artigo) e coordenadas espaciais sintéticas.

**Para usar dados reais**, substitua a chamada a
`simulate_grass_drought_dataset()` em `src/pipeline.py` por uma função
equivalente que carregue sua tabela de features processada (por
exemplo, a partir de um `.biom`/`.csv` exportado do QIIME2), mantendo
as colunas obrigatórias: `sample_id`, `label` (`"Controle"`/`"Seca"`),
`x`, `y` (coordenadas) e uma coluna por táxon.

## 3. Modelagem (`src/model.py`)

* Random Forest Classifier, com validação cruzada aninhada:
  5 folds externos (avaliação), 3 folds internos (busca de
  hiperparâmetros via `GridSearchCV`: `n_estimators`, `max_depth`,
  `min_samples_leaf`).
* Métricas: acurácia, F1, precisão, recall e AUC — as mesmas usadas
  pelo artigo de referência.

## 4. Interpretabilidade (`src/interpretability.py`)

* Valores SHAP (`shap.TreeExplainer`) calculados no conjunto de teste
  de cada fold.
* Direção do enriquecimento por táxon: correlação entre o valor bruto
  da feature e seu respectivo valor SHAP no fold (mesmo princípio dos
  gráficos de sumário SHAP). Correlação positiva → enriquecido sob
  "Seca"; negativa → enriquecido sob "Controle".
* Consenso: um táxon só recebe rótulo de enriquecimento definitivo se a
  direção for consistente em pelo menos 4 dos 5 folds (parâmetro
  `min_fold_agreement`), replicando o critério do artigo.

## 5. Mapa de prescrição (`src/prescription_map.py`)

Define a função `f: P(Seca) → (classe, dose, composição)`:

* **Classe**: `"Seca"` se `P(Seca) ≥ prob_threshold_low`, senão
  `"Controle"`.
* **Dose**: rampa linear entre `prob_threshold_low` e
  `prob_threshold_high`, de `dose_min_l_ha` a `dose_max_l_ha`
  (parâmetros em `DoseConfig`, ajustáveis conforme o produto
  comercial usado).
* **Composição do inoculante**: os táxons com consenso "Controle"
  (isto é, depletados sob seca) e maior importância SHAP são sugeridos
  como componentes a repor — uma regra ilustrativa; em uso real, a
  composição também dependeria de disponibilidade comercial e de
  evidência funcional de cada táxon.

## 6. Simulação de campo e VRT (`src/vrt_simulation.py`)

* O mapa de prescrição (pontos amostrais) é interpolado para uma grade
  regular via vizinho mais próximo (`scipy.spatial.cKDTree`) — escolha
  deliberadamente simples, adequada ao nível de especificação deste
  projeto (métodos como krigagem poderiam refinar a interpolação em
  trabalho futuro).
* Compara o volume total de bioinsumo em dois cenários: aplicação
  uniforme (dose fixa em toda a área) versus aplicação em taxa
  variável (dose por célula, conforme o mapa de prescrição).
* Reporta a fração de economia de insumo estimada.

## 7. Limitações (ver também o pré-projeto)

1. **Tempo de resposta do sequenciamento 16S** é incompatível com
   decisão em tempo real durante a operação de pulverização; o sistema,
   tal como especificado, serve a um ciclo de diagnóstico prévio à
   safra ou a reamostragens periódicas.
2. **Generalização entre datasets**: o próprio artigo de referência
   mostra queda de desempenho ao testar o classificador em um dataset
   independente (Sorghum-Drought), o que deve ser considerado antes de
   qualquer uso operacional do mapa de prescrição.
3. **Interpolação espacial simplificada**: o vizinho mais próximo é
   adequado para demonstração, mas não substitui uma interpolação
   geoestatística adequada à densidade real de amostragem de solo em
   campo.

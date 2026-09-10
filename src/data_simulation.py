"""
data_simulation.py

Gera uma tabela de features sintética que reproduz a ESTRUTURA do dataset
"Grass-Drought" (Naylor et al., 2017; BioProject PRJNA369551), usado como
base em Hagen et al. (2024) para classificação de estresse hídrico via
microbioma do solo.

Por que dados sintéticos?
--------------------------
O pipeline original depende de sequenciamento 16S rRNA bruto (FASTQ)
disponível no NCBI Sequence Read Archive, processado via DADA2/QIIME2.
Esse download e processamento (SRA Toolkit, DADA2 em R) não é executável
neste ambiente de desenvolvimento por não haver acesso de rede ao NCBI.

Este módulo gera dados sintéticos com a mesma FORMA e o mesmo TIPO de
sinal estatístico (mudança de abundância relativa de um subconjunto de
táxons entre os regimes "Controle" e "Seca", conforme relatado no
artigo), permitindo validar toda a lógica de modelagem, interpretação e
integração mecanizada de ponta a ponta. Para uso com dados reais, veja
`docs/methodology.md`, seção "Substituindo os dados sintéticos".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Nomes de táxons inspirados nos gêneros citados como marcadores de seca
# em Hagen et al. (2024) — usados aqui apenas como rótulos plausíveis para
# os dados sintéticos, não como reprodução de valores do artigo.
MARKER_GENERA = [
    "Kribbella",
    "Streptomyces",
    "Occallatibacter",
    "Bacillus",
    "Rhodoplanes",
]

BACKGROUND_GENERA = [f"Genus_{i:03d}" for i in range(1, 41)]


def simulate_grass_drought_dataset(
    n_control: int = 320,
    n_drought: int = 303,
    field_size: tuple[int, int] = (40, 40),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Gera uma tabela de abundância relativa sintética com metadados.

    Parameters
    ----------
    n_control, n_drought:
        Número de amostras por regime hídrico. Os valores padrão
        reproduzem o tamanho do dataset Grass-Drought original.
    field_size:
        Dimensões (x, y) da grade usada para gerar coordenadas
        georreferenciadas sintéticas das amostras dentro de um talhão
        hipotético.
    seed:
        Semente do gerador de números aleatórios, para reprodutibilidade.

    Returns
    -------
    pandas.DataFrame
        Uma linha por amostra, com colunas:
        - sample_id
        - label ("Controle" / "Seca")
        - x, y (coordenadas dentro do talhão sintético, em metros)
        - uma coluna por gênero (abundância relativa, soma 1 por linha)
    """
    rng = np.random.default_rng(seed)
    n_total = n_control + n_drought
    genera = MARKER_GENERA + BACKGROUND_GENERA

    # Coordenadas sorteadas primeiro, para permitir um leve padrão
    # espacial de estresse hídrico (ex.: um gradiente de textura/relevo
    # que torna uma porção do talhão mais suscetível à seca), tornando o
    # mapa de prescrição resultante mais realista do que um padrão
    # puramente aleatório.
    xs = rng.uniform(0, field_size[0], n_total)
    ys = rng.uniform(0, field_size[1], n_total)
    spatial_trend = (xs / field_size[0]) * 0.6 + (ys / field_size[1]) * 0.2
    drought_score = spatial_trend + rng.normal(0, 0.25, n_total)

    n_drought_target = n_drought
    drought_idx = np.argsort(drought_score)[-n_drought_target:]
    labels = np.array(["Controle"] * n_total)
    labels[drought_idx] = "Seca"

    # Concentração de Dirichlet "base": todos os táxons com abundância
    # esperada semelhante, exceto os marcadores, que recebem parâmetros
    # diferentes conforme a classe (simulando enriquecimento/depleção).
    alpha_base = np.ones(len(genera)) * 2.0

    rows = []
    for i in range(n_total):
        alpha = alpha_base.copy()
        is_drought = labels[i] == "Seca"
        for j, genus in enumerate(genera[: len(MARKER_GENERA)]):
            if genus in ("Kribbella", "Occallatibacter"):
                # depletados sob seca (efeito moderado, com sobreposição
                # entre classes, para evitar separação artificialmente
                # perfeita)
                alpha[j] = 1.1 if is_drought else 2.6
            elif genus in ("Streptomyces", "Bacillus"):
                # enriquecidos sob seca
                alpha[j] = 2.6 if is_drought else 1.1
            else:
                alpha[j] = 2.2 if is_drought else 2.0

        abundances = rng.dirichlet(alpha)
        # Ruído multiplicativo (simula variação técnica de sequenciamento)
        # seguido de renormalização, para reduzir a separabilidade
        # artificial entre classes.
        noise = rng.lognormal(mean=0.0, sigma=0.35, size=abundances.shape)
        abundances = abundances * noise
        abundances = abundances / abundances.sum()
        row = {"sample_id": f"S{i+1:04d}", "label": labels[i], "x": xs[i], "y": ys[i]}
        row.update({genus: abundances[j] for j, genus in enumerate(genera)})
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = simulate_grass_drought_dataset()
    print(df.shape)
    print(df.head())

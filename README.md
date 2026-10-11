# Caça-Alucinações — BRACIS 2026 × Jusbrasil

Encontra as citações de jurisprudência e legislação em peças jurídicas e classifica
cada uma como `real` (com o `id_canonico` da base), `inventada` ou `incompleta`,
consultando a base canônica `data/desafio1_bracis.db`.

O pipeline é determinístico, usa só a biblioteca padrão do Python (≥ 3.10) e roda
offline, sem GPU: os 26 documentos do dev levam cerca de 2,5 s, incluindo a
indexação da base.

## Como rodar

### Com Python

```bash
python3 -m src.main --input data/txt --output resultados
```

Grava um `<documento_id>.json` por `.txt` de `--input`, no formato do Contrato de
Entrada e Saída. A base padrão é `data/desafio1_bracis.db`; outra pode ser passada
com `--db`.

### Com Docker

```bash
docker build -t caca-alucinacoes .
docker run --rm --network none \
  -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out" \
  caca-alucinacoes --input /data/in --output /data/out
```

A base canônica está dentro da imagem. Para usar outra, monte-a
(`-v /caminho/base.db:/data/base.db:ro`) e acrescente `--db /data/base.db`.

### Submissão

```bash
python3 json_to_submission.py resultados submission.csv   # JSONs -> submission.csv do Kaggle
```

`json_to_submission.py` é o conversor oficial da competição; o `submission.csv` é o
arquivo enviado ao Kaggle.

## Como funciona

```
.txt ─► limpeza com mapa de offsets ─► regex (formas catalogadas) ─┐
                                   └─► âncoras gerais (formas novas) ─► resolvedor ─► JSON
                                                                       (índice da base)
```

| Módulo | Papel |
|---|---|
| `src/spans.py` | **Onde estão as citações.** (1) Limpeza com mapa de offsets: desfaz mojibake ("Ã§"→"ç"), tira numeração de linha da margem e invisíveis, junta hifenização de fim de linha e palavras partidas ("Suspe nsão"), separa palavras coladas ("doart.", "REsp1.664", "443do") e devolve os spans em codepoints do texto **original**. (2) Regex das formas conhecidas, tolerantes a OCR (`ocr("súmula")` aceita "5úmula", "Súmnla", "Súmulo"; números aceitam "21737l8", "1.664.Bq3", "20Z3"). (3) Âncoras gerais, que partem do que toda citação tem, e não de uma lista de formatos: número + frase jurídica antes dele; súmula + número; artigo + enumeração + diploma, nas duas ordens; tribunal + ano + relator na mesma oração. |
| `src/classify.py` | **O que cada citação é.** Índice da base (número **próprio** de cada acórdão, lido do cabeçalho; súmulas; dispositivos) e resolvedor: único componente que pode dizer `real`, e só por consulta à base. |
| `src/normalize.py` | Compartilhado: OCR em números, chaves CNJ, diplomas (nome, sigla, número da lei, radicais), cadeia recursal, classe processual, nomes. |
| `src/main.py` | Pipeline por documento e CLI do contrato `--input/--output`. |

Regras do resolvedor:

- **processo numerado**: busca pelo número (CNJ ou curto, depois da correção de OCR), filtra por
  tribunal, UF e classe citados e desempata pela cadeia recursal (AgInt, EDcl…). Achou → `real`;
  não achou → `inventada`.
- **súmula / artigo de lei**: busca por (tribunal, número, vinculante) e por (diploma, artigo).
- **julgado descritivo** (tribunal + ano + relator, sem número): um único registro → `real`;
  vários ou nenhum → `incompleta`.
- **Tema**: `inventada`, porque a base não cobre temas.

A correção de OCR só troca **letras por dígitos** (`l`→1, `O`→0, `S`→5). Números nunca são
aproximados: no gabarito, várias citações inventadas estão a um dígito de um número real, e
chamar uma inventada de real é o erro com penalidade própria na métrica (τ).

A confiança de cada citação vem da taxa de acerto da regra que a decidiu, medida durante o
desenvolvimento em conjuntos sintéticos rotulados pela base (tabela `CONFIDENCE` em
`src/classify.py`).

## Ferramentas de desenvolvimento

O repositório contém só o que a entrega usa. Os geradores de conjuntos sintéticos e de ruído,
os benchmarks, a calibração, a métrica local, os testes e as soluções de referência
(`baseline/`) saíram da árvore e continuam no histórico do git (commit `ad4a921`).

## Resultados

RESULTADOS_AQUI

## Premissas (não confirmadas pelo dev)

- Descritiva que identifica **um único** registro vira `real`. Nenhum registro vira `incompleta`,
  não `inventada`: nos sintéticos, isso vinha de OCR no nome do relator, e o gabarito não tem
  descritiva inventada.
- Lei citada sem artigo ("Lei nº 13.467/2017") e referências vagas ("reiterados precedentes do
  STJ") não são citações: o goldenset atual não as anota.
- Número real com relator citado divergente continua `real` (resolve a um registro), mas com
  confiança 0,60.
- `data/txt/gen_n1_003`, `gen_n1_006` e `gen_n1_010` foram ajustados ao goldenset atualizado
  (inserção de "AgInt no " e "ED no AgR no "). Com isso, os 192 spans batem exatamente.
  Substitua pelos textos oficiais quando a organização os distribuir.

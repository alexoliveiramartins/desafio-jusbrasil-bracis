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

### Submissão e nota local

```bash
python3 json_to_submission.py resultados submission.csv   # JSONs -> CSV do Kaggle
pip install -r requirements.txt                            # pandas/numpy, só para a métrica
python3 -m tools.evaluate --submission submission.csv      # métrica oficial contra data/goldenset.csv
python3 -m tools.evaluate                                  # atalho: roda o pipeline e pontua o dev
python3 -m tools.verify_official_data                      # base, gabarito e métrica batem com o snapshot oficial?
python3 tools/audit_exact_spans.py --gold data/goldenset.csv --submission submission.csv --out relatorios/spans.csv
```

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

A confiança de cada citação vem da taxa de acerto da regra que a decidiu, medida nos
conjuntos de iteração (`python3 -m tools.calibrate`).

## Avaliação sem overfitting

O dev tem só 26 documentos. Ajustar regras até acertá-los não diz nada sobre o conjunto cego.
Por isso o desenvolvimento usa conjuntos sintéticos rotulados pela base, e os rótulos não
dependem do pipeline:

- `real` numerado: só registros cujo número é único na base;
- `inventada`: número ausente de todos os cabeçalhos da base;
- descritiva: contagem de relatores cujo nome contém os tokens citados.

Cada conjunto tem um papel, e a regra é: **diagnosticar só nos de iteração; holdout só se
mede**. Holdout usado para decidir algo vira iteração, e é preciso gerar outro.

| Conjunto | Papel | O que testa |
|---|---|---|
| `dev` (`data/goldenset.csv`) | referência | goldenset oficial |
| `val`, `armadilhas`, `denso`, `poluido` | iteração | formatos conhecidos; número a 1 dígito de um real; várias citações por parágrafo; digitalização ruim |
| `ineditos`, `ineditos_poluido` | iteração | catálogo inédito v1 (já foi holdout) |
| `ineditos_v3`, `ineditos_v3_poluido`, `ineditos_v3_extremo` | iteração | gramática ampla: siglas, abreviações, marcadores de número, rabos de citação, ordens livres |
| `test`, `ocr_forte`, `formatos_ineditos`, `poluido_extremo` | holdout | formatos conhecidos + inéditos, OCR pesado |
| `ineditos_v2`, `ineditos_v2_poluido` | holdout | abreviação por truncamento aleatório, descritivas em ordem aleatória, armadilhas |
| `ineditos_v4`, `ineditos_v4_poluido` | holdout | rabos de citação, abreviação por sílaba, datas, nomes abreviados, ruído de PDF novo |
| `llm_iter` / `llm_holdout` | iteração / holdout | peças e citações **escritas por um LLM** (ver abaixo) |

Os catálogos v1 a v4 foram escritos por quem escreveu o pipeline, então medem generalização de
forma otimista. Os conjuntos `llm_*` corrigem isso. Os fatos rotulados saem da base, mas quem
decide como cada citação é escrita, e o texto da peça, é o Qwen2.5-7B-Instruct local, com
temperatura > 0 e semente fixa. A validação, feita sem o pipeline, descarta citações em que o
modelo mudou dígitos, tribunal, UF, classe ou diploma. Esses conjuntos precisam de um servidor
Ollama local e só servem para avaliação.

```bash
pip install -r requirements.txt
python3 -m tools.synth                         # gera os conjuntos de iteração em data/synthetic/
python3 -m tools.evaluate dev val ineditos_v3  # métrica oficial por conjunto
python3 -m tools.evaluate ineditos_v3 --diagnose   # cada divergência (só em iteração!)
python3 -m tools.battery                       # todos os perfis × 5 sementes -> relatorios/bateria.json
python3 -m tools.synth_llm --split llm_holdout --docs 60 --seed 5000   # requer Ollama
python3 -m unittest discover -s tests -t .
```

### Robustez a ruído (`tools.stress`)

Os sintéticos acima escrevem peças novas. Os conjuntos de estresse partem dos **próprios 26 textos
do dev** e só os degradam, então medem quanto a solução perde exclusivamente por causa do ruído.
Cada caractere original guarda o que virou e o que foi inserido antes dele, e o gabarito é
remapeado para o texto novo com offsets exatos. Os rótulos continuam válidos porque dígito nunca vira
outro dígito, só letra parecida (0→O, 1→l, 5→S), e lixo de página só entra fora das citações.

| Família | O que faz |
|---|---|
| `ocr` | confusões de OCR: 0→O, 1→l/I/\|, 5→S, 8→B, rn↔m, cl↔d, e↔c, ç→c, º→° |
| `acentos` | perda de acentos por palavra; às vezes o documento inteiro em NFD (acento combinante) |
| `espacos` | espaço duplo, NBSP, espaço fino, tab, palavras coladas ("doSTJ") e partidas ("Recur so"), "5. 230.808" |
| `quebras` | reflow de PDF, parágrafo espúrio no meio da frase, hifenização ("juris-\\n", "¬\\n", soft hyphen) |
| `fim_de_linha` | CRLF (arquivo salvo no Windows), total ou misto |
| `pontuacao` | números sem pontos ou com vírgula/espaço, "nº"→"n°/no/n.", "/SP"→"-SP", pontuação de frase |
| `typos` | erros de digitação: troca, omissão, duplicação e tecla vizinha (ABNT2); siglas ficam intactas |
| `caixa` | linhas em caixa alta, palavras em caixa alta/baixa/invertida |
| `lixo` | cabeçalho/rodapé, "Página 3 de 90", carimbo de assinatura com OAB, "Autos nº <CNJ>", CPF/CNPJ, CEP, valor da causa, numeração de linha na margem |
| `mojibake` | UTF-8 lido como cp1252 ("ção"→"Ã§Ã£o", "nº"→"nÂº"), por palavra ou no documento inteiro |
| `invisiveis` | zero-width space/joiner, BOM, word joiner e soft hyphen em qualquer lugar, inclusive dentro de números |

Perfis: `limpo` (controle, reproduz o dev byte a byte), `so_<família>` (ablação: uma família na
intensidade "pesado") e `leve`, `moderado`, `pesado`, `extremo` (todas juntas, intensidade
crescente). Assim como os holdouts, é instrumento de **medida**: não ajuste regras olhando estes erros.

```bash
python3 -m tools.stress                      # gera data/stress/<perfil>_s<semente>/ (3 sementes)
python3 -m tools.stress.bench                # solução atual × baseline final_robust -> relatorios/stress/benchmark.{json,md}
python3 -m tools.evaluate data/stress/extremo_s1000   # um conjunto pelo avaliador de sempre
```

Medição de 26/09/2026, **antes** das melhorias de robustez abaixo (3 sementes, 192 citações por
conjunto; "edição" = caracteres alterados ou inseridos sobre o total):

| Perfil | Edição | Solução atual: score | spans | τ | `final_robust`: score | spans | τ |
|---|---:|---:|---:|---:|---:|---:|---:|
| limpo | 0% | 1,100 | 1,000 | 0 | 1,100 | 1,000 | 0 |
| leve | 8% | 1,016 ± 0,011 | 0,891 | 0 | 0,791 ± 0,022 | 0,694 | 0 |
| moderado | 16% | 0,938 ± 0,006 | 0,799 | 0 | 0,510 ± 0,051 | 0,444 | 0,031 |
| pesado | 32% | 0,772 ± 0,014 | 0,590 | 0 | 0,195 ± 0,059 | 0,181 | 0,031 |
| extremo | 50% | 0,656 ± 0,011 | 0,469 | 0 | 0,074 ± 0,026 | 0,059 | 0,031 |

A solução atual nunca chama uma inventada de real (τ = 0 em todos os perfis). Quase toda a perda
vem de citações não extraídas, não de classe errada. Na ablação, o que mais custa é o OCR de
dígitos fora do vocabulário do pipeline (Z→2, D→0, s→5, q→9, b→6, |→1: 120 das 141 perdas em
`so_ocr`). Depois vêm espaços estranhos, mojibake e pontuação. Numeração de linha na margem gera
espúrias ("RECURSO ESPECIAL\n  8"). CRLF, caracteres invisíveis e acentos não custam nada.

Depois dessa medição, o pipeline ganhou tolerância geral a esses defeitos (conjunto de letras de OCR
único para extração e classificação, confusões a↔o, u↔n, t↔f, g↔q, cl↔d nas palavras-chave, "riº"/"uº"
como "nº", palavras coladas e partidas, mojibake e numeração de margem), sem mudar nada no dev
(1,0999, 192/192 spans exatos) nem piorar os sintéticos de iteração. Os ajustes foram decididos só em
conjuntos de iteração; a avaliação final segue o protocolo registrado antes deles em
[`tools/stress/PROTOCOLO.md`](tools/stress/PROTOCOLO.md), com conjuntos inéditos gerados depois do
congelamento do código.

`baseline/` guarda referências que não fazem parte da solução: `poc/` (prova de conceito original)
e `final_robust/` (solução da branch `feature/final-robust-solution`, só regex, com tabelas fixas
de súmulas e artigos). A `final_robust` também acerta todo o dev, mas não generaliza: fica entre
0,05 e 0,64 nos sintéticos, contra 0,88 a 1,10 da solução atual.

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

# Caça-Alucinações — BRACIS 2026 × Jusbrasil

Encontra as citações de jurisprudência e legislação em peças jurídicas e classifica
cada uma como `real` (com o `id_canonico` da base), `inventada` ou `incompleta`,
consultando a base canônica `data/desafio1_bracis.db`.

Há duas versões, com as mesmas regras de extração e resolução:

| Versão | O que faz | Requisitos | Imagem |
|---|---|---|---|
| **regras + NLP** (submissão) | regex + âncoras gerais + resolvedor por índice da base, e um LLM aberto do Hugging Face que revisa o documento: acha citações que as regras perderam e normaliza as que ficaram sem registro por ruído | GPU (ou CPU, mais lento), Ollama 0.33.2 (dentro da imagem) e os pesos de [`model_manifest.json`](model_manifest.json) | `Dockerfile` |
| **só regras** (alternativa e fallback) | regex + âncoras gerais + resolvedor por índice da base | Python ≥ 3.10, só biblioteca padrão; sem GPU | `Dockerfile.regras` |

Nas duas, **quem decide a classe é a consulta determinística à base**; o modelo só lê campos
(tribunal, classe, número, diploma, artigo, ano, relator). A versão só com regras roda os 26
documentos do dev em ~3 s; a camada de NLP acrescenta ~8–10 s por documento numa GPU de 16 GB.
Se a camada não puder rodar (sem pesos, sem servidor, sem tempo), a imagem da submissão cai
sozinha para a versão só com regras.

## Como submeter

**Leaderboard (Kaggle):** envie o `submission.csv` (formato de `data/sample_submission.csv`: uma
linha por documento, `documento_id,citacoes`). No dev, as duas versões geram o mesmo arquivo.

```bash
python3 -m src.main --input data/txt --output resultados            # um JSON completo por documento
python3 -m tools.validar_saida --input data/txt --output resultados # confere o contrato (0 problemas)
python3 json_to_submission.py resultados submission.csv             # JSONs -> CSV do Kaggle
python3 -m tools.empacotar_saidas --output resultados --zip saidas.zip   # .zip só com os JSONs, se pedirem
```

**Pacote de verificação (reexecução pela organização):** este repositório, com `Dockerfile`
(imagem da submissão), `model_manifest.json` (HF id + revisão + sha256), `tools/baixar_modelo.py`
(baixa os pesos declarados) e este README. Antes de entregar, rode a verificação de ponta a ponta,
que constrói as imagens e executa o contrato como a organização vai executar:

```bash
bash tools/verificar_submissao.sh      # build, sem dados na imagem, --network none, contrato, nota, determinismo, fallback
```

## Como rodar

### Com Python

```bash
python3 -m src.main --input data/txt --output resultados --nlp      # regras + NLP (Ollama local com o modelo)
python3 -m src.main --input data/txt --output resultados            # só regras
```

Grava um `<documento_id>.json` por `.txt` de `--input`, no formato do Contrato de
Entrada e Saída. A base padrão é `data/desafio1_bracis.db`; outra pode ser passada
com `--db` (ou `CACA_DB`). Com `--nlp`, o modelo padrão é o do manifesto, servido por um
Ollama em `127.0.0.1:11434` (`--nlp-model`, `--nlp-url`; servidor que não seja local é recusado);
sem servidor, a execução segue só com as regras.

### Com Docker (contrato de execução da competição)

Base e pesos **não vão dentro das imagens**: são montados em runtime. Nenhuma das imagens faz
chamada de rede; ambas rodam com `--network none`.

```bash
# submissão: regras + NLP
python3 -m tools.baixar_modelo --dest modelos      # com rede, ANTES: revisão fixa do HF + sha256
docker build -t caca-alucinacoes .
docker run --rm --network none --gpus all \
  -v "$PWD/data/desafio1_bracis.db:/data/ref/desafio1_bracis.db:ro" -v "$PWD/modelos:/models:ro" \
  -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out" \
  caca-alucinacoes --input /data/in --output /data/out

# alternativa: só regras
docker build -f Dockerfile.regras -t caca-alucinacoes-regras .
docker run --rm --network none \
  -v "$PWD/data/desafio1_bracis.db:/data/ref/desafio1_bracis.db:ro" \
  -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out" \
  caca-alucinacoes-regras --input /data/in --output /data/out
```

Os pesos podem estar em qualquer layout dentro de `/models`: arquivo solto, `huggingface-cli download
--revision <rev> --local-dir` ou cache do HF (`snapshots/<revisão>/`); o sha256 é conferido na
partida. Em GPU AMD, construa com `--build-arg OLLAMA_IMAGE=<imagem_rocm do manifesto>` e troque
`--gpus all` por `--device /dev/kfd --device /dev/dri`.

### Conformidade com as regras de execução

| Regra (e-mail da organização, 28/08/2026) | Como é cumprida |
|---|---|
| Só pesos abertos e públicos, com link HF + revisão fixa | [`model_manifest.json`](model_manifest.json): `unsloth/Qwen3-4B-Instruct-2507-GGUF` na revisão `a06e946…`, arquivo e sha256 fixos; base `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0). Sem fine-tune, sem modelo gated, sem API |
| 1 GPU 24 GB, 8 vCPUs, 32 GB; média ≤ 60 s/doc; 4 h no total | modelo de 4B quantizado (~4 GB de VRAM); ~8–10 s/doc medidos numa GPU de 16 GB. A camada tem orçamento próprio: se a média passar de 40 s/doc ou o total de 3 h (ex.: sem GPU), ela se desliga e o resto sai só com as regras |
| Contêiner sem rede; nada externo em runtime | o Ollama sobe dentro do contêiner, só em `127.0.0.1`; os pesos vêm de `/models`, baixados antes por `tools/baixar_modelo.py` e conferidos pelo sha256 no início da execução |
| Pesos e dados fora da imagem | `.dockerignore` só deixa entrar `src/`, o manifesto e o Modelfile; base em `/data/ref` e pesos em `/models`, montados |
| Dockerfile com dependências fixas | imagens-base fixadas por digest (`python:3.12.3-slim`, `ollama/ollama:0.33.2`); o pipeline não tem dependência fora da biblioteca padrão |
| Entrypoint no contrato padrão | `docker run <img> --input /data/in --output /data/out` nas duas imagens |
| Seed e decodificação determinística | `temperature 0` (gulosa), `seed 42`, um pedido por vez no servidor (`OLLAMA_NUM_PARALLEL=1`), saída JSON restrita por schema, teto de 4096 tokens |
| Score reproduzido não pode cair > 5% | qualquer falha da camada (sem pesos, hash divergente, servidor fora do ar, tempo) cai para a versão só com regras, que é determinística e reproduz o score dela exatamente |

### Nota local

```bash
pip install -r requirements.txt                            # pandas/numpy, só para a métrica
python3 -m tools.evaluate --submission submission.csv      # métrica oficial contra data/goldenset.csv
python3 -m tools.evaluate                                  # atalho: roda o pipeline e pontua o dev
python3 -m tools.verify_official_data                      # base, gabarito e métrica batem com o snapshot oficial?
python3 tools/audit_exact_spans.py --gold data/goldenset.csv --submission submission.csv --out relatorios/spans.csv
```

## Como funciona

```
.txt ─► limpeza com mapa de offsets ─► regex (formas catalogadas) ─┐
                                   └─► âncoras gerais (formas novas) ─► resolvedor ─► [camada de NLP] ─► JSON
                                                                       (índice da base)   (opcional, --nlp)
```

| Módulo | Papel |
|---|---|
| `src/spans.py` | **Onde estão as citações.** (1) Limpeza com mapa de offsets: desfaz mojibake ("Ã§"→"ç"), tira numeração de linha da margem e invisíveis, junta hifenização de fim de linha e palavras partidas ("Suspe nsão"), separa palavras coladas ("doart.", "REsp1.664", "443do") e devolve os spans em codepoints do texto **original**. (2) Regex das formas conhecidas, tolerantes a OCR (`ocr("súmula")` aceita "5úmula", "Súmnla", "Súmulo"; números aceitam "21737l8", "1.664.Bq3", "20Z3"). (3) Âncoras gerais, que partem do que toda citação tem, e não de uma lista de formatos: número + frase jurídica antes dele; súmula + número; artigo + enumeração + diploma, nas duas ordens; tribunal + ano + relator na mesma oração. |
| `src/classify.py` | **O que cada citação é.** Índice da base (número **próprio** de cada acórdão, lido do cabeçalho; súmulas; dispositivos) e resolvedor: único componente que pode dizer `real`, e só por consulta à base. |
| `src/normalize.py` | Compartilhado: OCR em números, chaves CNJ, diplomas (nome, sigla, número da lei, radicais), cadeia recursal, classe processual, nomes. |
| `src/main.py` | Pipeline por documento e CLI do contrato `--input/--output`. |
| `src/nlp.py` | **Camada de NLP (opcional).** Um LLM aberto lê a peça e devolve, em JSON restrito por schema, cada citação que vê (trecho literal + campos lidos). Serve a duas coisas: *normalização* (citação que as regras deixaram `inventada`/`incompleta` é reconsultada com os campos do modelo) e *recall* (citação que as regras não extraíram entra como candidata). |
| `src/launcher.py` | Entrypoint da imagem com NLP: confere o sha256 dos pesos montados, sobe o Ollama em `127.0.0.1` e registra o modelo; qualquer falha cai para a versão só com regras. |

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

### Camada de NLP: o que o modelo pode e o que não pode decidir

O modelo **nunca decide a classe**. Os campos que ele lê viram uma forma canônica ("REsp nº
1.741.784/PR (STJ)") que passa pelo mesmo resolvedor das regras. Guardas contra o erro grave
(inventada → real) e contra espúrias, todas decididas em conjuntos de iteração:

- **número**: o do modelo precisa ser um dos números do trecho, inteiro, só com troca letra→dígito
  (`normalize.number_groups`, o mesmo leitor que as regras usam como reserva, entende OCR colado:
  "AgInt7S57430-50.2018…", "88.86OPE", "4O,2023", "RESP6 .q89.q16", UF colada, invisíveis no meio). Se o modelo
  leu mal, vale o único número do próprio trecho; dígito nunca vem do modelo;
- **tribunal, classe e diploma**: valem os que têm evidência no trecho (sigla, nome, apelido, número da lei),
  tolerando OCR ("Reccurso em Hqbcas Corpus" é RHC; "Rcel." é Rcl; "apelo nobre" não é nada). Sem evidência para
  a leitura do modelo nem para a das regras, as duas têm de levar ao mesmo registro. Lei citada pelo número é
  lida do trecho (o modelo confunde as leis);
- **citação nova**: precisa de evidência de processo (classe, tribunal ou CNJ completo), não pode estar em linha
  de distrator nem ser "Autos nº …" (o número do próprio processo); frase ou parágrafo inteiro devolvido pelo
  modelo é aparado até o núcleo da citação (processo, artigo ou súmula), sobre o texto limpo pelas regras
  (acento decomposto, mojibake, invisíveis) com os offsets devolvidos ao original;
- **nada que as regras resolveram como `real` muda**.

Decodificação determinística (temperature 0, seed 42, um pedido por vez) e orçamento de tempo: média
acima de 40 s/doc ou total acima de 3 h desliga a camada, e o resto sai só com as regras.

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
| `v5_iter`, `v5_iter_ruido`, `v5_iter_denso` | iteração | catálogo v5 (`tools.synth.official_v5`): peça no molde da amostra oficial (cabeçalho com distratores, seções, prosa, quebra dura de linha), nível 1 canônico e nível 2 com o ruído descrito no PDF |
| `v5_holdout`, `v5_holdout_ruido` | holdout | o mesmo, com formas exclusivas escritas antes da camada de NLP (apelidos de diplomas, lei pelo número, súmula por extenso, descritivas em ordem nova, tribunal junto do número) |
| `llm_llama_iter` | iteração | peças escritas pelo Llama-3.1-8B (autor diferente do modelo da camada de NLP) |
| `data/stress/*_s3000` / `*_s7700` | iteração / holdout | estresse sobre o dev com sementes novas |

Os catálogos v1 a v4 foram escritos por quem escreveu o pipeline, então medem generalização de
forma otimista. Os conjuntos `llm_*` corrigem isso. Os fatos rotulados saem da base, mas quem
decide como cada citação é escrita, e o texto da peça, é o Qwen2.5-7B-Instruct local, com
temperatura > 0 e semente fixa. A validação, feita sem o pipeline, descarta citações em que o
modelo mudou dígitos, tribunal, UF, classe ou diploma. Esses conjuntos precisam de um servidor
Ollama local e só servem para avaliação.

```bash
pip install -r requirements.txt
python3 -m tools.synth                         # gera os conjuntos de iteração em data/synthetic/
python3 -m tools.synth.official_v5 --split v5_iter --seed 1                    # catálogo v5 (iteração)
python3 -m tools.synth.official_v5 --split v5_holdout --seed 101 --holdout     # catálogo v5 (holdout)
python3 -m tools.nlp_bench --sets dev v5_iter --models <modelo> [--docs 20]     # só regras × NLP
python3 -m tools.synth_llm --split llm_llama_iter --docs 40 --seed 3100 --model llama3.1:8b   # requer Ollama
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
0,05 e 0,64 nos sintéticos, contra 0,88 a 1,10 da solução atual. `alex_regex/` é o extrator só de
spans da branch `alex` (`python3 -m baseline.alex_regex.benchmark_extractor`): acha 181 das 192
citações do dev, e em 12 conjuntos (~6.000 citações) só 1 citação que a solução atual não acha.

## Resultados

Métrica oficial (`kaggle_metric.py`; máximo 1,10 com o bônus de calibração), τ = fração das
inventadas do gabarito preditas como `real` (o erro grave). Modelo: Qwen3-4B-Instruct-2507 (GGUF
Q4_K_M), ~9 s/documento numa GPU de 16 GB.

**Dev (goldenset oficial, 26 documentos):** 1,0999 nas duas versões, τ = 0, 192/192 spans exatos. No
dev a camada de NLP não altera nenhuma citação.

**Holdout, rodada 1** (gerado depois do congelamento de 28/09, medido uma vez; protocolo em
[`tools/PROTOCOLO_NLP.md`](tools/PROTOCOLO_NLP.md)):

| Conjunto | O que testa | só regras | regras + NLP | τ |
|---|---|---:|---:|:---:|
| `v5_holdout` | formato oficial, formas inéditas | 1,0716 | **1,0957** | 0 |
| `v5_holdout_ruido` | o mesmo com ruído 1,8× | 1,0714 | **1,0841** | 0 |
| estresse leve | dev degradado (8% dos caracteres) | 1,0771 | **1,0960** | 0 |
| estresse moderado | dev degradado (16%) | 1,0645 | **1,0882** | 0 |
| estresse pesado | dev degradado (32%) | 0,9952 | **1,0482** | 0 |
| estresse extremo | dev degradado (49%) | 0,9859 | **1,0257** | 0 |
| `llm_holdout` | peças escritas por LLM (Qwen2.5) | 0,9999 | **1,0821** | 0 |
| **média** | | 1,0379 | **1,0743** | 0 |

Depois da rodada 1, as regras ganharam leitura numérica robusta a OCR colado, descritivas de qualquer
classe ("RR de 2016, Rel. Min. …"), preposição com OCR ("cla CLT"), número de lei com hífen e os
apelidos "CR/88" e "NCPC"; a camada ganhou guardas mais tolerantes a OCR. A rodada 2 do holdout
(conjuntos novos, inclusive peças escritas por Llama e Gemma-Gaia) mede a versão final.

RODADA_2_AQUI

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

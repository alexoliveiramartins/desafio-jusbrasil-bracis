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
documentos do dev em ~3 s; a camada de NLP acrescenta ~18 s por documento com os dois modelos (~9 s só
com o principal) numa GPU de 16 GB.
Se a camada não puder rodar (sem pesos, sem servidor, sem tempo), a imagem da submissão cai
sozinha para a versão só com regras.

Documentação: [`docs/funcionamento_fim_a_fim.pdf`](docs/funcionamento_fim_a_fim.pdf) (como a solução
funciona, do comando à saída, com um caso real) e [`docs/benchmarks/`](docs/benchmarks) (resultados e resumo em PDF).

## Avaliação final: ponto de entrada único

```bash
bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>
# ex.: bash run.sh data/desafio1_bracis.db data/txt saida/submission.csv
```

Recebe a base canônica (qualquer `.db` no formato original), a pasta com os `.txt` e grava
`<arquivo_saida>` no formato de submissão (`documento_id,citacoes`, o mesmo do `json_to_submission.py`
oficial). Precisa só de Docker e `python3` (biblioteca padrão). Em máquina limpa, o `run.sh`:

1. **prepara** o que faltar, e só essa etapa usa internet: baixa os pesos declarados em
   [`model_manifest.json`](model_manifest.json) (Hugging Face, revisão fixa, sha256 conferido) para
   `modelos/` e constrói a imagem Docker da GPU encontrada (NVIDIA, AMD ou nenhuma);
2. **executa sem rede** (`--network none`), com a base e os textos montados só para leitura. O índice
   da base é montado a partir do `.db` recebido a cada execução; nada é pré-calculado sobre a base de
   dev, então um `.db` novo funciona sem nenhum passo extra;
3. **converte** os JSONs para o CSV de submissão e confere o contrato (`tools/validar_saida.py`).

Com GPU, roda regras + o conjunto Qwen3-4B + Qwen3-8B (~12 GB de VRAM, ~18 s por documento, mais
~10 min de partida para conferir e importar os pesos; ~30 GB de disco no total). Sem GPU, ou se a
camada de NLP não subir, segue sozinha só com as regras. Opcionais: `GPU=nvidia|amd|cpu` força o
fabricante; `CACA_ENSEMBLE=0` usa só o modelo principal.

## Como submeter

**Leaderboard (Kaggle, fase referencial):** envie o `submission.csv` (formato de
`data/sample_submission.csv`: uma linha por documento, `documento_id,citacoes`). No dev, as duas versões
geram o mesmo arquivo, e o `run.sh` acima gera exatamente esse formato.

```bash
python3 -m src.main --input data/txt --output resultados            # um JSON completo por documento
python3 -m tools.validar_saida --input data/txt --output resultados # confere o contrato (0 problemas)
python3 json_to_submission.py resultados submission.csv             # JSONs -> CSV do Kaggle
```

**Entrega final (e-mail para desafio-bracis@jusbrasil.com.br até 01/10, 23h59):** nome da equipe e
integrantes, link deste repositório e o hash do commit da versão final. A organização executa o
`run.sh` sobre o `.db` e os documentos do conjunto final. Antes de entregar, confira de ponta a ponta
(build, nenhum dado na imagem, `--network none`, contrato, nota, estabilidade entre execuções, fallback):

```bash
bash run.sh data/desafio1_bracis.db data/txt /tmp/saida.csv   # o ponto de entrada, como a organização vai rodar
bash tools/verificar_submissao.sh                             # verificação completa das imagens
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
# submissão: regras + NLP (conjunto Qwen3-4B + Qwen3-8B)
python3 -m tools.baixar_modelo --dest modelos      # com rede, ANTES: revisões fixas do HF + sha256
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

O mesmo `Dockerfile` serve às duas GPUs; o fabricante é escolhido no build:

| GPU | build | run (no lugar de `--gpus all`) |
|---|---|---|
| NVIDIA (CUDA, padrão) | `docker build -t caca-alucinacoes .` | `--gpus all` |
| AMD (ROCm) | `docker build --build-arg GPU=amd -t caca-alucinacoes .` | `--device /dev/kfd --device /dev/dri` |
| sem GPU | `docker build --build-arg GPU=cpu -t caca-alucinacoes .` | (nada; lento, o orçamento desliga a camada) |

Variáveis opcionais do contêiner (`-e NOME=valor`): `CACA_ENSEMBLE=0` usa só o modelo principal (para GPUs
com menos de ~12 GB livres); `CACA_MAX_LOADED_MODELS` (padrão 2) mantém os dois modelos carregados juntos,
~10 GB de VRAM. Evite `CACA_MAX_LOADED_MODELS=1` com o conjunto: ele troca de modelo a cada documento, é mais
lento, e numa RX 9070 XT (ROCm) essa troca contínua travou a máquina. `CACA_VERIFY_SHA=0` pula a conferência
do sha256 na partida.

Na partida, o contêiner informa o acelerador que o Ollama encontrou (`[launcher] acelerador: cuda NVIDIA …`,
`rocm AMD Radeon …`) ou avisa que está em CPU. Os pesos podem estar em qualquer layout dentro de `/models`:
arquivo solto, `huggingface-cli download --revision <rev> --local-dir` ou cache do HF (`snapshots/<revisão>/`);
o sha256 é conferido na partida.

### Conformidade com as regras de execução

| Regra (e-mails da organização de 28/08 e 29/09/2026) | Como é cumprida |
|---|---|
| Ponto de entrada único: recebe o `.db` e a pasta dos `.txt` e gera a saída no formato da submissão | `bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>` (seção acima); por dentro, a imagem segue o contrato `docker run <img> --input /data/in --output /data/out` |
| Avaliação com um `.db` novo e documentos novos, no formato do dev | o índice (números próprios dos acórdãos, súmulas, dispositivos, relatores) é montado a partir do `.db` recebido a cada execução; nada é pré-calculado sobre a base de dev |
| Do zero, em máquina limpa; sem caminhos absolutos, passos manuais ou arquivos só da equipe | o `run.sh` baixa os pesos declarados e constrói a imagem se faltarem; todos os caminhos são relativos ao repositório ou vêm dos argumentos |
| Pesos incluídos ou referenciados em revisão fixa, baixáveis antes da execução | [`model_manifest.json`](model_manifest.json): principal `unsloth/Qwen3-4B-Instruct-2507-GGUF` (revisão `a06e946…`; base `Qwen/Qwen3-4B-Instruct-2507`) e, no conjunto, `Qwen/Qwen3-8B-GGUF` (revisão `7c41481…`; base `Qwen/Qwen3-8B`), arquivos e sha256 fixos, todos Apache-2.0. Sem fine-tune, sem modelo gated, sem API |
| GPU com até 24 GB de VRAM | dois modelos quantizados carregados juntos: ~12 GB de VRAM; 18,5 s/doc na imagem Docker numa GPU de 16 GB (~9 s/doc só com o principal), mais ~10 min de partida. A camada tem orçamento próprio (média de 40 s/doc, sem contar o 1º documento, que carrega os modelos; 3 h no total) e, se passar, o resto sai só com as regras |
| Offline: nada de internet nem APIs externas na execução | contêiner com `--network none`; o Ollama sobe dentro dele, só em `127.0.0.1` e com `OLLAMA_NO_CLOUD=1` (sem ela, ele tenta acessar `ollama.com` na partida); os pesos vêm de `/models`, conferidos pelo sha256 no início |
| Pesos e dados fora da imagem | `.dockerignore` só deixa entrar `src/`, o manifesto e os Modelfiles (`docker/nlp/`); base e pesos são montados |
| Ambiente declarado (Docker), dependências fixas | imagens-base fixadas por digest (`python:3.12.3-slim`, `ollama/ollama:0.33.2`, `-rocm` para AMD); o pipeline não tem dependência fora da biblioteca padrão |
| Seeds fixas, sem amostragem não determinística | `temperature 0` (gulosa), `seed 42`, um pedido por vez no servidor (`OLLAMA_NUM_PARALLEL=1`), saída JSON restrita por schema. O *texto* gerado na GPU não é idêntico bit a bit entre execuções (64% das respostas iguais); a classe continua decidida pela base: duas execuções seguidas da imagem deram a mesma nota (1,0999, diferença 0,0000), e a variação máxima medida foi 0,004 |
| Disco (referência: ~100 GB) | ~30 GB: pesos 7,5 GB, imagem 4,2 GB e ~15 GB de importação dos pesos dentro do contêiner |

### Nota local

```bash
pip install -r requirements.txt                            # pandas/numpy, só para a métrica
python3 -m tools.evaluate --submission submission.csv      # métrica oficial contra data/goldenset.csv
python3 -m tools.evaluate                                  # atalho: roda o pipeline e pontua o dev
python3 -m tools.verify_official_data                      # base, gabarito e métrica batem com o snapshot oficial?
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

**Reprodutibilidade medida.** Na medição final, cada resposta ao vivo foi comparada com a resposta guardada para o
mesmo pedido: 64% idênticas, 36% com texto diferente (mesmo modelo, mesmos parâmetros gulosos), por não
determinismo numérico da GPU. Nos seis conjuntos comparados, a nota ficou igual em quatro e variou +0,001 e
+0,004 nos outros dois; τ = 0 em todas as execuções.

**Cache (só laboratório).** As ferramentas de avaliação (`tools.nlp_bench`, `tools.calibrate`) guardam a
resposta bruta do modelo em `.cache/nlp/<modelo>/<sha256 do pedido>.json`. Como a decodificação é determinística,
isso permite reavaliar mudanças nas guardas sem refazer a leitura na GPU. O cache não guarda gabarito nem
decisão, fica fora da imagem (`.dockerignore`) e vem desligado no pipeline (`--nlp-cache` não é passado pelo
entrypoint): na execução oficial, cada documento é lido ao vivo pelo modelo.

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
python3 -m tools.stress.bench                # solução atual × perfis de ruído -> relatorios/stress/benchmark.{json,md}
python3 -m tools.evaluate data/stress/extremo_s1000   # um conjunto pelo avaliador de sempre
```

Medição de 26/09/2026, **antes** das melhorias de robustez abaixo (3 sementes, 192 citações por
conjunto; "edição" = caracteres alterados ou inseridos sobre o total). `final_robust` é uma solução anterior
da equipe, só regex, medida para comparação:

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

Soluções anteriores da equipe foram comparadas e removidas do repositório (ficam no histórico do git):
a prova de conceito original; a `final_robust` (branch `feature/final-robust-solution`, só regex, com
tabelas fixas de súmulas e artigos), que também acerta todo o dev, mas não generaliza (0,05 a 0,64 nos
sintéticos, contra 0,88 a 1,10 da solução atual); e o extrator só de spans da branch `alex`, que acha 181
das 192 citações do dev e, em 12 conjuntos (~6.000 citações), só 1 citação que a solução atual não acha.

## Resultados

Métrica oficial (`kaggle_metric.py`; máximo 1,10 com o bônus de calibração), τ = fração das
inventadas do gabarito preditas como `real` (o erro grave). Versão entregue: conjunto Qwen3-4B-Instruct-2507 +
Qwen3-8B (GGUF Q4_K_M), ~18 s/documento numa GPU de 16 GB com os dois carregados (~9 s/documento só com o 4B).
As tabelas das rodadas 1 e 2 abaixo mostram o 4B, que era a configuração medida naquela época; a versão final
e a rodada 3 mostram também o 8B e o conjunto.

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

**Conjuntos de iteração** (código final, respostas do modelo guardadas; 24 conjuntos, 10.288 citações; onde as decisões foram tomadas):

| Conjunto | só regras | NLP 4B | NLP 8B | **NLP 4B+8B** (entregue) |
|---|---:|---:|---:|---:|
| dev (oficial) | 1,0999 | 1,0999 | 1,0999 | **1,0999** |
| `v5_iter` / `v5_iter_ruido` / `v5_iter_denso` | 1,0979 / 1,0973 / 1,0988 | 1,0995 / 1,0995 / 1,0996 | 1,0987 / 1,0995 / 1,0996 | **1,0995 / 1,0995 / 1,0996** |
| `denso` / `val` | 1,0995 / 1,0987 | 1,0995 / 1,0996 | 1,0995 / 1,0996 | **1,0995 / 1,0996** |
| `v5_holdout` / `v5_holdout_ruido` (holdout da rodada 1) | 1,0801 / 1,0714 | 1,0957 / 1,0845 | 1,0968 / 1,0854 | **1,0968 / 1,0864** |
| `armadilhas` / `poluido` | 1,0985 / 1,0926 | 1,0985 / 1,0926 | 1,0985 / 1,0936 | **1,0985 / 1,0936** |
| `ineditos` / `ineditos_poluido` | 1,0971 / 1,0925 | 1,0996 / 1,0934 | 1,0996 / 1,0925 | **1,0996 / 1,0934** |
| `ineditos_v3` / `ineditos_v3_poluido` | 1,0955 / 1,0868 | 1,0962 / 1,0885 | 1,0972 / 1,0892 | **1,0962 / 1,0892** |
| `llm_iter` / `llm_holdout` (peças do Qwen2.5) | 1,0210 / 0,9999 | 1,0241 / 1,0821 | 1,0237 / 1,0821 | **1,0191 / 1,0821** |
| `llm_llama_iter` (peças do Llama) | 0,9255 | 0,9963 | 0,9963 | **1,0056** |
| estresse leve s7700 | 1,0829 | 1,0960 | 1,0960 | **1,0960** |
| estresse moderado s3000 / s7700 | 1,0626 / 1,0658 | 1,0843 / 1,0907 | 1,0955 / 1,0887 | **1,0955 / 1,0927** |
| estresse pesado s3000 / s7700 | 1,0074 / 1,0071 | 1,0700 / 1,0576 | 1,0590 / 1,0557 | **1,0700 / 1,0576** |
| estresse extremo s3000 / s7700 | 0,8706 / 1,0094 | 1,0133 / 1,0444 | 0,9890 / 1,0300 | **1,0350 / 1,0444** |
| **média** | 1,0566 | 1,0794 | 1,0777 | **1,0812** |

τ = 0 em todas as células. Os números dos 39 conjuntos estão em [`docs/benchmarks/`](docs/benchmarks): `tabela_codigo_atual.json` (iteração e rodadas 1 e 2, com as respostas guardadas), `final_ao_vivo_*.json` e `rodada3_*.json` (medições únicas ao vivo), e o resumo com apêndice em `Caca-Alucinacoes_solucao_e_numeros.pdf`.

Depois da rodada 1, as regras ganharam leitura numérica robusta a OCR colado, descritivas de qualquer
classe ("RR de 2016, Rel. Min. …"), preposição com OCR ("cla CLT"), número de lei com hífen e os
apelidos "CR/88" e "NCPC"; a camada ganhou guardas mais tolerantes a OCR. A rodada 2 do holdout
(conjuntos novos, inclusive peças escritas por Llama e Gemma-Gaia) mede a versão final.

**Holdout, rodada 2** (conjuntos gerados depois do congelamento seguinte; peças escritas pelo Gemma-3-Gaia-PT-BR,
autor nunca usado em iteração, e pelo Llama-3.1; medido uma vez com aquele código):

| Conjunto | só regras | regras + NLP | τ |
|---|---:|---:|:---:|
| `v5_holdout2` (formas inéditas) | 1,0775 | **1,0857** | 0 |
| `v5_holdout2_ruido` | 1,0750 | **1,0772** | 0 |
| estresse leve / moderado | 1,0945 / 1,0619 | **1,0999 / 1,0693** | 0 |
| estresse pesado / extremo | 0,9779 / 0,9717 | **1,0322 / 1,0202** | 0 |
| `llm_gaia_holdout` (Gemma-Gaia) | 1,0435 | **1,0834** | 0 |
| `llm_llama_holdout` (Llama) | 1,0324 | **1,0691** | 0 |
| **média** | 1,0418 | **1,0671** | 0 |

**Versão final (a entregue), mesmos conjuntos da rodada 2, medida uma vez AO VIVO** (cache novo e vazio,
um modelo carregado por vez; nenhum erro desses conjuntos foi diagnosticado):

| Conjunto | só regras | NLP 4B | NLP 8B | **NLP 4B+8B** (entregue) | τ |
|---|---:|---:|---:|---:|:---:|
| `v5_holdout2` | 1,0775 | 1,0936 | 1,0959 | **1,0954** | 0 |
| `v5_holdout2_ruido` | 1,0750 | 1,0912 | 1,0908 | **1,0925** | 0 |
| estresse leve / moderado | 1,0945 / 1,0619 | 1,0999 / 1,0705 | 1,0999 / 1,0665 | **1,0999 / 1,0705** | 0 |
| estresse pesado / extremo | 0,9779 / 0,9717 | 1,0322 / 1,0355 | 1,0404 / 1,0246 | **1,0413 / 1,0355** | 0 |
| `llm_gaia_holdout` / `llm_llama_holdout` | 1,0435 / 1,0324 | 1,0834 / 1,0691 | 1,0834 / 1,0691 | **1,0834 / 1,0691** | 0 |
| **média** | 1,0418 | 1,0719 | 1,0713 | **1,0735** | 0 |

O conjunto (união para achar citações, concordância para normalizar, conflito devolve a resposta das regras)
fica >= o 4B sozinho em todos os conjuntos; o 4B sozinho é o caminho quando o 8B falta (`CACA_ENSEMBLE=0`).

**Holdout, rodada 3: a medida limpa** (7 conjuntos gerados depois do congelamento final, nunca usados para
decidir nada; medidos uma única vez, ao vivo, sem olhar os erros):

| Conjunto | só regras | NLP 4B | NLP 8B | **NLP 4B+8B** (entregue) | τ |
|---|---:|---:|---:|---:|:---:|
| `v5_holdout3` (formas inéditas) | 1,0835 | 1,0926 | 1,0926 | **1,0935** | 0 |
| `v5_holdout3_ruido` | 1,0721 | 1,0882 | 1,0929 | **1,0917** | 0 |
| estresse leve / moderado | 1,0918 / 1,0483 | 1,0962 / 1,0631 | 1,0950 / 1,0660 | **1,0962 / 1,0688** | 0 |
| estresse pesado / extremo | 1,0255 / 0,9344 | 1,0875 / 1,0198 | 1,0801 / 1,0109 | **1,0875 / 1,0324** | 0 |
| `llm_gaia_holdout3` (Gemma-Gaia) | 0,9392 | 0,9614 | 0,9395 | **0,9691** | 0 |
| **média** | 1,0278 | 1,0584 | 1,0539 | **1,0627** | 0 |

O ganho sobre as regras se repete em conjuntos nunca vistos: +0,035 na média (rodada 2: +0,032). O conjunto fica
>= o 4B sozinho em todos os conjuntos; o 8B sozinho às vezes fica até 0,001 acima dele, mas perde na média e nos
casos mais difíceis (extremo: 1,0109 contra 1,0324 do conjunto).

**Verificação da imagem Docker** (`tools/verificar_submissao.sh`, AMD RX 9070 XT, os dois modelos carregados juntos,
~11,9 GB de VRAM): imagem sem base nem pesos, `--network none`, 0 problemas de contrato, nota 1,0999 em duas
execuções (diferença 0,0000) e 1,0999 no fallback sem pesos; 18,5 s/doc, mais ~10 min de partida.

## Premissas (não confirmadas pelo dev)

- Descritiva que identifica **um único** registro vira `real`. Nenhum registro vira `incompleta`,
  não `inventada`: nos sintéticos, isso vinha de OCR no nome do relator, e o gabarito não tem
  descritiva inventada.
- Lei citada sem artigo ("Lei nº 13.467/2017") e referências vagas ("reiterados precedentes do
  STJ", "jurisprudência pacífica desta Corte") não são citações: o goldenset atual não as anota. A
  descrição dos dados no Kaggle ainda diz que as vagas são `incompleta`, mas ela é da versão de 01/09
  (225 citações, 31 vagas anotadas); a organização as retirou do gabarito (225 → 195 → 192), e o
  leaderboard confirma a regra atual (29/09: este pipeline tira 1,0999 lá; uma submissão com as vagas
  tirava 1,10 só na versão antiga). Se a regra voltar no conjunto final, as regras perdem ~0,126
  no dev original (medido: 0,9741).
- Número real com relator citado divergente continua `real` (resolve a um registro), mas com
  confiança 0,60.
- `data/` é idêntico ao pacote oficial de 15/09 (base, `goldenset_offsets.csv` → `data/goldenset.csv`,
  26 `.txt`, métrica, conversor e `sample_submission.csv`); `python3 -m tools.verify_official_data`
  confere os hashes. Os textos `gen_n1_003`, `gen_n1_006` e `gen_n1_010` já são os oficiais atualizados
  (com "AgInt no " e "ED no AgR no "), e os 192 spans batem exatamente.

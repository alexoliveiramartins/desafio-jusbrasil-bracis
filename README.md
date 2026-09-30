# Caça-Alucinações — BRACIS 2026 × Jusbrasil

Encontra as citações de jurisprudência e legislação em peças jurídicas e classifica
cada uma como `real` (com o `id_canonico` da base), `inventada` ou `incompleta`,
consultando a base canônica `data/desafio1_bracis.db`.

Há duas versões, com as mesmas regras de extração e resolução:

| Versão | O que faz | Requisitos | Imagem |
|---|---|---|---|
| **regras + NLP** (submissão) | regex + âncoras gerais + resolvedor por índice da base, e um LLM aberto do Hugging Face que revisa o documento: acha citações que as regras perderam e normaliza as que ficaram sem registro por ruído | GPU (ou CPU, mais lento), Ollama 0.33.2 (dentro da imagem) e os pesos de [`model_manifest.json`](model_manifest.json) | `Dockerfile` |
| **só regras** (alternativa e fallback) | regex + âncoras gerais + resolvedor por índice da base | Python ≥ 3.10, só biblioteca padrão; sem GPU | `Dockerfile.rules` |

Nas duas, **quem decide a classe é a consulta determinística à base**; o modelo só lê campos
(tribunal, classe, número, diploma, artigo, ano, relator). A versão só com regras roda os 26
documentos do dev em ~3 s; a camada de NLP acrescenta ~18 s por documento com os dois modelos (~9 s só
com o principal) numa GPU de 16 GB.
Se a camada não puder rodar (sem pesos, sem servidor, sem tempo), a imagem da submissão cai
sozinha para a versão só com regras.

Documentação: [`docs/funcionamento_fim_a_fim.pdf`](docs/funcionamento_fim_a_fim.pdf) (como a solução
funciona, do comando à saída, com um caso real).

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
3. **converte** os JSONs para o CSV de submissão e confere o contrato (`tools/validate_output.py`).

Com GPU, roda regras + o conjunto Qwen3-4B + Qwen3-8B (~12 GB de VRAM, ~18 s por documento, mais
~10 min de partida para conferir e importar os pesos; ~30 GB de disco no total). Sem GPU, ou se a
camada de NLP não subir, segue sozinha só com as regras. Opcionais: `GPU=nvidia|amd|cpu` força o
fabricante; `CACA_ENSEMBLE=0` usa só o modelo principal.

## Entrega

E-mail para desafio-bracis@jusbrasil.com.br até 01/10, 23h59, com o nome da equipe e os integrantes,
o link deste repositório e o hash do commit da versão final. A organização executa o `run.sh` sobre o
`.db` e os documentos do conjunto final. Para conferir antes de entregar, como a organização vai rodar:

```bash
bash run.sh data/desafio1_bracis.db data/txt /tmp/saida.csv
```

## Como rodar

### Com Python

```bash
python3 -m src.main --input data/txt --output resultados --nlp        # regras + NLP (Ollama local com o modelo)
python3 -m src.main --input data/txt --output resultados              # só regras
python3 -m tools.validate_output --input data/txt --output resultados # confere o contrato (0 problemas)
python3 json_to_submission.py resultados submission.csv               # JSONs -> CSV no formato da submissão
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
python3 -m tools.download_models --dest modelos      # com rede, ANTES: revisões fixas do HF + sha256
docker build -t caca-alucinacoes .
docker run --rm --network none --gpus all \
  -v "$PWD/data/desafio1_bracis.db:/data/ref/desafio1_bracis.db:ro" -v "$PWD/modelos:/models:ro" \
  -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out" \
  caca-alucinacoes --input /data/in --output /data/out

# alternativa: só regras
docker build -f Dockerfile.rules -t caca-alucinacoes-rules .
docker run --rm --network none \
  -v "$PWD/data/desafio1_bracis.db:/data/ref/desafio1_bracis.db:ro" \
  -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out" \
  caca-alucinacoes-rules --input /data/in --output /data/out
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
| Seeds fixas, sem amostragem não determinística | `temperature 0` (gulosa), `seed 42`, um pedido por vez no servidor (`OLLAMA_NUM_PARALLEL=1`), saída JSON restrita por schema. O *texto* gerado na GPU pode variar entre execuções (não determinismo numérico da GPU), mas a classe continua decidida pela base: no dev, duas execuções seguidas da imagem tiveram a mesma nota |
| Disco (referência: ~100 GB) | ~30 GB: pesos 7,5 GB, imagem 4,2 GB e ~15 GB de importação dos pesos dentro do contêiner |

### Testes

```bash
python3 -m unittest discover -s tests -t .
```

Testes de unidade do pipeline e da camada de NLP, só com a biblioteca padrão; a camada é testada com
respostas simuladas, sem servidor.

### Dados e arquivos oficiais

`data/` é idêntico ao pacote oficial de 15/09 (base, `goldenset_offsets.csv` → `data/goldenset.csv`,
26 `.txt` e `sample_submission.csv`), assim como a métrica (`kaggle_metric.py`) e o conversor
(`json_to_submission.py`, usado pelo `run.sh`); `python3 -m tools.verify_official_data` confere os hashes.
Os textos `gen_n1_003`, `gen_n1_006` e `gen_n1_010` já são os oficiais atualizados (com "AgInt no " e
"ED no AgR no "). O `requirements.txt` fixa pandas e numpy, que só a métrica usa.

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

A confiança de cada citação vem da taxa de acerto da regra que a decidiu, medida durante o
desenvolvimento em conjuntos sintéticos rotulados pela base (com teto de 0,98).

### Camada de NLP: o que o modelo pode e o que não pode decidir

O modelo **nunca decide a classe**. Os campos que ele lê viram uma forma canônica ("REsp nº
1.741.784/PR (STJ)") que passa pelo mesmo resolvedor das regras. Guardas contra o erro grave
(inventada → real) e contra espúrias:

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

Com o conjunto (Qwen3-4B + Qwen3-8B), as citações novas são a união das achadas pelos dois modelos, e a
normalização exige que os dois concordem; em conflito, vale a resposta das regras.

Decodificação determinística (temperature 0, seed 42, um pedido por vez) e orçamento de tempo: média
acima de 40 s/doc ou total acima de 3 h desliga a camada, e o resto sai só com as regras.

**Cache (desligado na execução oficial).** Com `--nlp-cache <pasta>`, o pipeline guarda a resposta bruta do
modelo por sha256 do pedido, o que permite reprocessar sem refazer a leitura na GPU. O cache não guarda
gabarito nem decisão, fica fora da imagem (`.dockerignore`) e o entrypoint não o passa: na execução oficial,
cada documento é lido ao vivo pelo modelo.

## Premissas (não confirmadas pelo dev)

- Descritiva que identifica **um único** registro vira `real`. Nenhum registro vira `incompleta`,
  não `inventada`: nome de relator sem registro costuma ser OCR no nome, e o gabarito não tem
  descritiva inventada.
- Lei citada sem artigo ("Lei nº 13.467/2017") e referências vagas ("reiterados precedentes do
  STJ", "jurisprudência pacífica desta Corte") não são citações: o goldenset atual não as anota. A
  descrição dos dados no Kaggle ainda diz que as vagas são `incompleta`, mas ela é da versão de 01/09
  (225 citações, 31 vagas anotadas); a organização as retirou do gabarito (225 → 195 → 192), e o
  leaderboard de 29/09 confirma a regra atual. Se a regra antiga voltar no conjunto final, as vagas
  deixam de ser anotadas por esta solução.
- Número real com relator citado divergente continua `real` (resolve a um registro), mas com
  confiança 0,60.

# Protocolo: versão só-spans × versão com camada de NLP

Registrado em 2026-09-28, antes de qualquer ajuste da camada de NLP (`src/nlp.py`) olhando
resultados. As duas versões compartilham as regras (`src/spans.py`, `src/classify.py`); a versão
NLP só acrescenta `--nlp` (LLM aberto do Hugging Face, GGUF, servido pelo Ollama na GPU local).

## Modelos

- Candidatos da camada (família Qwen3, pesos oficiais no Hugging Face):
  `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M`, `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`,
  `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` (raciocínio desligado).
- Autores dos conjuntos escritos por LLM: Qwen2.5-7B (`llm_*` antigos), Llama-3.1-8B e
  Gemma-3-Gaia-PT-BR-4B (novos). Nenhum autor é candidato da camada: o benchmark não mede a
  memória do próprio autor.

## Iteração (pode diagnosticar e ajustar; só fenômenos gerais)

- `dev` (regressão: tem de continuar 1,0999 e τ = 0 nas duas versões);
- `v5_iter` (semente 1), `v5_iter_ruido` (2, ruído 1,8), `v5_iter_denso` (3): `tools.synth.official_v5`
  sem as formas de holdout;
- `llm_iter`, `llm_llama_iter` (semente 3100), `llm_gaia_iter` (3200), `llm_gaia_adv_iter` (3300);
- `data/stress/{moderado,pesado,extremo}_s3000`;
- os sintéticos de iteração antigos (`val`, `armadilhas`, `poluido`, `ineditos_v3_poluido`...).

## Holdout (gerado só DEPOIS de congelar `src/`; medido uma vez; sem --diagnose)

- `v5_holdout` (sementes 101–103) e `v5_holdout_ruido` (201–203, ruído 1,8), com `--holdout`
  (formas exclusivas escritas antes da camada);
- `llm_llama_holdout` (semente 4100), `llm_gaia_holdout` (4200), `llm_llama_adv_holdout` (4300);
- `tools.stress` sobre o dev, perfis `leve` a `extremo`, sementes 7700–7702;
- `llm_holdout` antigo (já existente, nunca diagnosticado nesta rodada).

## Regra de decisão

A versão NLP só é recomendada se, no holdout: (1) τ não piorar em nenhum conjunto; (2) o dev
continuar 1,0999; (3) a média do score subir. O modelo escolhido é o de melhor média nos
conjuntos de iteração, com desempate por latência.

Se algo em `src/` mudar depois da medição do holdout, ele vira iteração e é preciso gerar outro
(sementes 8800+).

## Congelamento e holdout — rodada 1 (registrado em 2026-09-28 17:38, antes de medir)

Código congelado (`sha256sum src/*.py`); modelo: o de `model_manifest.json` (Qwen3-4B-Instruct-2507, empatou com o Qwen3-8B nas amostras de iteração e é mais rápido).

```
dd99fe257d9faa4a909459bfcf1b9eaf473162ef6569b15f1f7a10126f338409  src/classify.py
d929b4e656c78b74be9a049a26254fc1b3539d1eebe39ff4e766197280b078d2  src/__init__.py
6313c7432bb785af2396d7b5184bfcb1641962e88ea3815b81ef8e8362ba5a98  src/launcher.py
f849a1f25b2ca685840fb68a9baf24cb27df3a44696f23342574b60b5d6e3f05  src/main.py
61cfd292a268ae0e0fd4277eeb4eb3d58905095369f98b0e4edec2363e7e058f  src/nlp.py
5e02cf96b3c3a523bcc64bdbeae434ccf392dbf19a044816395a918b0939a4f3  src/normalize.py
6956ad663b6411188ec054e62289b0516ba7ea663a6551f9e30f282eac216d64  src/spans.py
```

Subconjunto medido nesta rodada (o restante do holdout acima fica pendente): `v5_holdout` (semente 101), `v5_holdout_ruido` (201, ruído 1,8), `tools.stress` perfis `leve`, `moderado`, `pesado`, `extremo` com semente 7700, e `llm_holdout` (antigo). Variantes: `spans` e `nlp`, conjuntos completos.

Alteração posterior ao congelamento (sem efeito nas saídas): `OllamaClient` passa a recusar servidor
que não seja loopback (regra "nenhuma chamada externa em runtime"). Os scores das amostras de iteração
reavaliados pelo cache ficam idênticos aos anteriores.
Remoção de variável sem uso em `NLPLayer.refine` (sem efeito nas saídas).

### Resultado da rodada 1 (medido uma vez, código congelado acima)

| Conjunto | só regras | regras + NLP | τ |
|---|---:|---:|:---:|
| v5_holdout (101) | 1,0716 | 1,0957 | 0 |
| v5_holdout_ruido (201) | 1,0714 | 1,0841 | 0 |
| stress leve s7700 | 1,0771 | 1,0960 | 0 |
| stress moderado s7700 | 1,0645 | 1,0882 | 0 |
| stress pesado s7700 | 0,9952 | 1,0482 | 0 |
| stress extremo s7700 | 0,9859 | 1,0257 | 0 |
| llm_holdout | 0,9999 | 1,0821 | 0 |
| média | 1,0379 | 1,0743 | 0 |

Decisão: regra atendida (τ nunca piorou, dev 1,0999, média +0,036). A versão com NLP passa a ser a
imagem da submissão (`Dockerfile`); a só com regras fica em `Dockerfile.regras`.

## Rodada 2 (depois das melhorias de 28/09 à noite)

As regras e a camada mudaram depois da rodada 1 (leitura numérica robusta nas regras, descritivas de
qualquer classe, preposição "cla", número de lei com hífen, marcador "riº", "CR/88"/"NCPC"). Pela regra
acima, os conjuntos da rodada 1 viram iteração. A validação final usa conjuntos novos, gerados só depois
do novo congelamento: v5 com sementes 8801 (formas de holdout) e 8802 (ruído 1,8), estresse do dev
com semente 8800 (leve a extremo) e peças escritas por LLM: Gemma-3-Gaia-PT-BR-4B (`llm_gaia_holdout`,
semente 4200), autor nunca usado em iteração, e Llama-3.1-8B (`llm_llama_holdout`, semente 4100), autor
de um conjunto de iteração (`llm_llama_iter`, semente 3100): mede o mesmo estilo com peças novas.

### Congelamento da rodada 2 (registrado em 2026-09-28 21:06, antes de gerar os conjuntos)

```
79728ef8940213e7a7055be5c03f3fba71c9ffde5b2575a9a96a17eb9b74ac59  src/classify.py
d929b4e656c78b74be9a049a26254fc1b3539d1eebe39ff4e766197280b078d2  src/__init__.py
6313c7432bb785af2396d7b5184bfcb1641962e88ea3815b81ef8e8362ba5a98  src/launcher.py
f849a1f25b2ca685840fb68a9baf24cb27df3a44696f23342574b60b5d6e3f05  src/main.py
8e1b44be6cfc580645bb55c57e635ecb31bc19c6d4013440340f5274a0fc4b96  src/nlp.py
ff6c614f2928c093cbb51bd5d48fbeb0d963889e011011e16a5ca00502a5164a  src/normalize.py
939d91c9e594664f9f9805638568ae64158006d58d40a8fcfe4218cd84f4f1a1  src/spans.py
```

Iteração (conjuntos completos, ~7.000 citações; relatorios/nlp_bench_iteracao_final.json): a versão com NLP é
igual ou melhor que a só com regras nos 13 conjuntos, τ = 0 em todos. O conjunto com armadilhas
(`ineditos_v3_poluido`) revelou τ = 0,0167 numa versão intermediária (evidência emprestada de citação
vizinha quando o modelo junta duas; datas e "Resolução nº" lidos como processo; lei como "tema"); as
guardas foram corrigidas antes deste congelamento.
Alteração posterior ao congelamento da rodada 2 (sem efeito nas saídas): `src/launcher.py` informa o acelerador
encontrado pelo Ollama (CUDA/ROCm/CPU); `Dockerfile` parametrizado por `--build-arg GPU=nvidia|amd|cpu`.

### Resultado da rodada 2 (código congelado acima, medido uma vez)

| Conjunto | só regras | regras + NLP | τ |
|---|---:|---:|:---:|
| v5_holdout2 (8801) | 1,0775 | 1,0857 | 0 |
| v5_holdout2_ruido (8802) | 1,0750 | 1,0772 | 0 |
| stress leve s8800 | 1,0945 | 1,0999 | 0 |
| stress moderado s8800 | 1,0619 | 1,0693 | 0 |
| stress pesado s8800 | 0,9779 | 1,0322 | 0 |
| stress extremo s8800 | 0,9717 | 1,0202 | 0 |
| llm_gaia_holdout (4200) | 1,0435 | 1,0834 | 0 |
| llm_llama_holdout (4100) | 1,0324 | 1,0691 | 0 |
| média | 1,0418 | 1,0671 | 0 |

Regra atendida de novo. Durante a medição, a iteração nos conjuntos da rodada 1 (que viraram iteração) mostrou que
as guardas desse congelamento recusavam formas legítimas ("Recurso Especial Cível", "Reclamação Constitucional",
"tema de repercussão geral nº …", lei por apelido); as guardas foram equilibradas (palavras de bloqueio em vez de
lista de permissão, diploma por evidência única, lei por apelido com guarda de diploma da base). A versão final
será medida uma única vez nos mesmos conjuntos da rodada 2, AO VIVO (cache novo e vazio), sem que nenhum erro
desses conjuntos tenha sido diagnosticado: só os placares agregados acima foram vistos.

## Versão final (congelada em 2026-09-28 23:12, antes da medição final)

Configuração: regras + camada de NLP com o conjunto Qwen3-4B-Instruct-2507 (principal) + Qwen3-8B (model_manifest.json).
Decidido na iteração (20 conjuntos para o 4B; 9 para o conjunto): o conjunto é >= o 4B em todos, τ = 0.

```
79728ef8940213e7a7055be5c03f3fba71c9ffde5b2575a9a96a17eb9b74ac59  src/classify.py
d929b4e656c78b74be9a049a26254fc1b3539d1eebe39ff4e766197280b078d2  src/__init__.py
e24b6a75e47a556a60fa4c1c0877a4f1531a96f9629967bb16e79c6642a50233  src/launcher.py
37e8f70c7f51f1b798cac42c50dc74fa78b1842f06f70d6aea807d98754b2806  src/main.py
e0e3f81a20a1593d7937867ed063d22a1d9764ecaa5329026c8134aa859d476b  src/nlp.py
ff6c614f2928c093cbb51bd5d48fbeb0d963889e011011e16a5ca00502a5164a  src/normalize.py
939d91c9e594664f9f9805638568ae64158006d58d40a8fcfe4218cd84f4f1a1  src/spans.py
```

Medição final: uma vez, nos 8 conjuntos da rodada 2, AO VIVO (pastas de cache novas e vazias), 4B e 8B lidos
em sequência (um modelo carregado por vez) e combinados. Também se mede o 4B sozinho (caminho de fallback se o 8B
faltar) e se compara cada resposta ao vivo com a resposta do cache antigo para o mesmo pedido.

### Resultado da medição final (ao vivo, cache novo e vazio, medido uma vez)

| Conjunto | só regras | 4B | 8B | 4B+8B | τ |
|---|---:|---:|---:|---:|:---:|
| v5_holdout2 | 1,0775 | 1,0936 | 1,0959 | 1,0954 | 0 |
| v5_holdout2_ruido | 1,0750 | 1,0912 | 1,0908 | 1,0925 | 0 |
| stress leve s8800 | 1,0945 | 1,0999 | 1,0999 | 1,0999 | 0 |
| stress moderado s8800 | 1,0619 | 1,0705 | 1,0665 | 1,0705 | 0 |
| stress pesado s8800 | 0,9779 | 1,0322 | 1,0404 | 1,0413 | 0 |
| stress extremo s8800 | 0,9717 | 1,0355 | 1,0246 | 1,0355 | 0 |
| llm_gaia_holdout | 1,0435 | 1,0834 | 1,0834 | 1,0834 | 0 |
| llm_llama_holdout | 1,0324 | 1,0691 | 1,0691 | 1,0691 | 0 |
| média | 1,0418 | 1,0719 | 1,0713 | **1,0735** | 0 |

O conjunto fica >= o 4B em todos os 8 conjuntos, τ = 0. Fidelidade do 4B: das 409 respostas ao vivo, 264 são
idênticas às do cache antigo para o mesmo pedido e 145 diferem (não determinismo da GPU com temperatura 0);
nos placares isso aparece como diferenças de até 0,004 (ruido 1,0899 no cache × 1,0912 ao vivo).

Correção de empacotamento (2026-09-29 01:07, sem tocar em `src/`): a verificação da imagem mostrou que o
`Dockerfile` copiava só o Modelfile do 4B, e o launcher seguia sem o 8B (com aviso). Agora a imagem leva
`docker/nlp/` inteira, e `test_image_ships_every_modelfile` confere que todo Modelfile do manifesto entra nela.
Os hashes de `src/` acima continuam valendo.

## Rodada 3 — medida limpa (registrada em 2026-09-28 23:58, antes de gerar os conjuntos)

Código: a versão final acima (hashes conferidos agora, sem mudança). Conjuntos nunca vistos, gerados agora:
v5 com sementes 9901 (formas de holdout) e 9902 (formas de holdout + ruído 1,8), estresse do dev com semente
9900 (leve, moderado, pesado, extremo) e peças escritas pelo Gemma-3-Gaia-PT-BR-4B com semente 4300 (30 docs).
Medição única, ao vivo (cache novo e vazio), sem diagnóstico, depois da verificação da imagem Docker. Nenhuma
decisão depende deste resultado: ele é o número do relatório de entrega.

### Incidente na rodada 3 e congelamento final (registrado em 2026-09-29 14:36, antes de medir de novo)

A primeira tentativa (14:33) parou ANTES de qualquer leitura do modelo: a saída **só com regras** de
`llm_gaia_holdout3` fez a métrica oficial recusar o arquivo inteiro ("[llm_gaia_holdout3_n1_010] citações #5 e #6
se sobrepõem com IoU >= 0.5"). Só essa mensagem foi vista; o documento não foi aberto. Antes da parada, o bench
imprimiu os placares só com regras dos outros 6 conjuntos (v5_holdout3 1,0835; ruido 1,0721; leve 1,0918;
moderado 1,0483; pesado 1,0255; extremo 0,9344); nenhuma decisão usa esses números. As leituras parciais do 8B
dessa tentativa foram descartadas.

Correção genérica de contrato, sem olhar o dado: `without_duplicates` em `src/main.py` deixa uma só citação de
cada grupo com IoU >= 0,5 (maior confiança, depois a mais longa, depois a primeira; ordem mantida), e
`tools/validar_saida.py` passou a acusar esse caso. Varredura nos 35 conjuntos fora da rodada 3 (1.478 documentos,
dev incluído): 0 pares com IoU >= 0,5, com regras e com o conjunto 4B+8B. A guarda é inerte neles, então nenhum
resultado anterior muda (dev 1,0999, 192/192).

Código da rodada 3 (muda só `src/main.py`):

```
79728ef8940213e7a7055be5c03f3fba71c9ffde5b2575a9a96a17eb9b74ac59  src/classify.py
d929b4e656c78b74be9a049a26254fc1b3539d1eebe39ff4e766197280b078d2  src/__init__.py
e24b6a75e47a556a60fa4c1c0877a4f1531a96f9629967bb16e79c6642a50233  src/launcher.py
19dc52898e3ef806ce8e27a7519cad7d4d03cf48174c8c4ed3f783a29b9fa057  src/main.py
e0e3f81a20a1593d7937867ed063d22a1d9764ecaa5329026c8134aa859d476b  src/nlp.py
ff6c614f2928c093cbb51bd5d48fbeb0d963889e011011e16a5ca00502a5164a  src/normalize.py
939d91c9e594664f9f9805638568ae64158006d58d40a8fcfe4218cd84f4f1a1  src/spans.py
```

Medição: recomeça do zero, ao vivo (caches novos e vazios), um modelo carregado por vez no Ollama do host.

### Resultado da rodada 3 (código congelado acima, ao vivo, medido uma vez, sem diagnóstico)

O 4B foi lido de 14:37 a 15:13 e o 8B de 15:50 a 16:31 (29/09), um modelo carregado por vez no Ollama do host,
cada um com pasta de cache nova e vazia; entre os dois, a GPU atendeu a verificação da imagem Docker. A combinação
só usa as leituras dessa medição (servidor desligado).

| Conjunto | só regras | 4B | 8B | 4B+8B | τ |
|---|---:|---:|---:|---:|:---:|
| v5_holdout3 (semente 9901) | 1,0835 | 1,0926 | 1,0926 | 1,0935 | 0 |
| v5_holdout3_ruido (9902) | 1,0721 | 1,0882 | 1,0929 | 1,0917 | 0 |
| stress leve s9900 | 1,0918 | 1,0962 | 1,0950 | 1,0962 | 0 |
| stress moderado s9900 | 1,0483 | 1,0631 | 1,0660 | 1,0688 | 0 |
| stress pesado s9900 | 1,0255 | 1,0875 | 1,0801 | 1,0875 | 0 |
| stress extremo s9900 | 0,9344 | 1,0198 | 1,0109 | 1,0324 | 0 |
| llm_gaia_holdout3 (4300) | 0,9392 | 0,9614 | 0,9395 | 0,9691 | 0 |
| média | 1,0278 | 1,0584 | 1,0539 | **1,0627** | 0 |

O conjunto fica >= o 4B em todos os 7 conjuntos (+0,0043 na média) e acima do melhor modelo sozinho em 4; no
`v5_holdout3_ruido`, o 8B sozinho fica 0,0012 acima dele (na rodada 2, o mesmo no `v5_holdout2`, 0,0005). Ganho
sobre as regras: +0,035 na média, contra +0,032 na rodada 2.

### Verificação da imagem Docker (AMD, 29/09 15:14–15:50)

`GPU=amd bash tools/verificar_submissao.sh nlp`, com o conjunto e os dois modelos carregados juntos
(`CACA_MAX_LOADED_MODELS=2`, o padrão; ~11,9 GB de VRAM numa RX 9070 XT de 16 GB): imagem sem base e sem pesos,
`--network none`, acelerador ROCm, sha256 dos dois pesos conferido, 26 documentos, 0 problemas de contrato, nota
1,0999; segunda execução 1,0999 (diferença 0,0000); sem os pesos, cai para as regras e dá 1,0999. 18,5 s/doc no
processamento, mais ~10 min de partida (conferência do sha256 e importação dos GGUF pelo Ollama).

Na madrugada, a mesma verificação com `CACA_MAX_LOADED_MODELS=1` (troca 4B <-> 8B a cada documento) travou a
máquina depois de avisos crescentes do driver (`svm_range_deferred_list_work [amdgpu]`); com os dois modelos
carregados juntos não houve aviso durante o processamento.

Depois da verificação, `OLLAMA_NO_CLOUD=1` entrou no `ENV` do `Dockerfile`: sem ela, o Ollama 0.33.2 tenta acessar
`ollama.com` na partida (catálogo e recomendações de modelos; com `--network none` a tentativa falha sem efeito);
com ela, nenhuma tentativa (conferido em contêiner sem rede e sem GPU). `src/` não mudou.

### Limitação conhecida do conjunto (achada na iteração, depois da rodada 3; código não alterado)

No `llm_iter` (iteração), o conjunto fica 0,005 abaixo do 4B sozinho (1,0191 × 1,0241) por uma única citação: o
4B lê "Ag. Reg. no STJ, rel. Og Fernandes, 2019" como julgado descritivo e a base o confirma (real); o 8B alinha
um trecho mais curto, que dispara o recorte do span longo das regras, e para o span recortado a leitura do 4B já
não é aceita (vira incompleta). Recortar antes de normalizar não muda o resultado. Nos 11 conjuntos de iteração
com as duas leituras guardadas, o conjunto fica acima do 4B em 5, igual em 5 e abaixo só neste (média 1,0596 →
1,0623). Corrigir exigiria mudar o código congelado e uma nova rodada de holdout; fica para uma versão futura.

## Benchmark completo do conjunto (29/09, código congelado da rodada 3)

Regras, 4B, 8B e 4B+8B em 39 conjuntos: os 7 da rodada 3 (ao vivo, acima) e 32 conjuntos de iteração e das
rodadas 1 e 2, com as respostas dos modelos guardadas e cobertura completa (nenhuma leitura faltando;
1.432 documentos, 12.435 citações). Nos 32: médias 1,0529 (regras) · 1,0774 (4B) · 1,0761 (8B) · 1,0792
(4B+8B); o conjunto fica acima do 4B em 11, igual em 20 e abaixo em 1 (`llm_iter`, limitação acima), e
acima do 8B em 13, igual em 16, abaixo em 3. τ = 0 em todas as células. Números em `docs/benchmarks/`.

## Regras de entrega de 29/09 (e-mail da organização)

A avaliação final roda o código da equipe sobre um `.db` novo e documentos novos, no formato do dev; a nota
vem dessa execução (o leaderboard do Kaggle e a comparação de CSVs não entram no ranking). Pedem um ponto de
entrada único que receba o `.db` e a pasta dos `.txt` e gere a saída no formato da submissão: `run.sh`
(prepara pesos e imagem se faltarem, executa a imagem sem rede, converte com o `json_to_submission.py`
oficial e valida o contrato). O índice da base é montado a partir do `.db` recebido a cada execução.

Gabarito: o pacote oficial de 15/09 (`goldenset_offsets.csv`, 192 citações) é idêntico a `data/`; o
leaderboard confirma (este pipeline tira 1,09992 lá em 29/09). A descrição dos dados ainda é a de 01/09
(225 citações, com 31 referências vagas anotadas como incompletas); contra aquele gabarito, as regras tiram
0,9741. A regra do conjunto final para as referências vagas segue em aberto (pergunta sugerida à organização).

### Correção do orçamento de tempo (29/09, achada no teste de ponta a ponta do `run.sh`)

Na primeira execução do `run.sh`, o 1º documento levou 101,6 s (carregar os dois modelos na GPU, com o disco
disputado por outro processo); com o 2º e o 3º (20,1 s e 16,5 s), a média de 46,1 s passou do orçamento de
40 s/doc e a camada se desligou: os outros 23 documentos saíram só com as regras. No dev isso não aparece na
nota (as regras já tiram 1,0999), mas no conjunto final tiraria todo o ganho da camada. Agora o 1º documento
conta para o total (3 h), mas fica fora da média por documento (`NLPLayer._charge`); teste novo
`test_cold_start_does_not_turn_layer_off`. Não muda nenhuma saída enquanto a camada está ligada.
`src/nlp.py` passa a ter o sha256 `05a30af2c626e320312eef28b971d5db1f353a364f1ba578711f29d4d3ee3796`; os demais arquivos de `src/` não mudam.
O `run.sh` também passou a reconstruir a imagem quando o código muda (etiqueta com o hash do código).

### Limpeza de código sem uso (30/09; saídas idênticas)

Removidos: o léxico opcional (`src/lexicon.json` nunca existiu, então todas as medições rodaram sem ele) e o seu
gerador `tools/lexicon.py`; as tabelas curadas de `src/classify.py` para a versão antiga da base (nenhum registro da
base atual as usava, e elas estavam presas a ids da base de dev); a segunda passada da camada (opção desligada e
nunca avaliada: `--nlp-second-pass`, `uncovered_windows`). Prova de equivalência: impressão digital (sha256)
das saídas antes e depois, em 1.732 documentos só com regras (42 conjuntos) e em 9 conjuntos com 4B+8B
(respostas guardadas): 51 de 51 idênticas. Novos sha256 de `src/`:

```
17dc0f9193c7c7b0803da3abeafa498e5984fe7e248438c32da790f88cf99d93  src/classify.py
d929b4e656c78b74be9a049a26254fc1b3539d1eebe39ff4e766197280b078d2  src/__init__.py
e24b6a75e47a556a60fa4c1c0877a4f1531a96f9629967bb16e79c6642a50233  src/launcher.py
e562fcdb5283a4b5e590a1d4a1b8460e97aee771c4175e4028b49dff1fa98e02  src/main.py
2c72344292b55e7c1b0c00d711ba969818b25aa0fd68036ff883c475e19b8c3d  src/nlp.py
9578af75c8e19a569a11322c6a512b23a9660fab34a29b74a318a1de3e58326e  src/normalize.py
b9cba2f1bf905b689bf70014db0250b2bfa44a0823890c5308d687c263b86891  src/spans.py
```

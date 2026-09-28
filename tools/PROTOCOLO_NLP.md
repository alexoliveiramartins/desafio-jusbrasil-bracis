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

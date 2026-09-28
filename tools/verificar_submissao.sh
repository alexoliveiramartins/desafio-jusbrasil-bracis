#!/usr/bin/env bash
# Verificação da submissão, de ponta a ponta, como a organização vai executar:
#   bash tools/verificar_submissao.sh            # as duas versões
#   bash tools/verificar_submissao.sh regras     # só a versão sem modelo
#
# Para cada imagem: build; conferência de que base e pesos NÃO estão na imagem; execução no dev
# com --network none e base/pesos montados; validação do contrato (tools.validar_saida); nota pelo
# fluxo oficial (json_to_submission.py -> kaggle_metric.py); tempo por documento; e, na versão com
# NLP, uma segunda execução para conferir que a saída é idêntica (decodificação determinística).
set -euo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-todas}"
PY="${PYTHON:-venv/bin/python}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
MANIFEST_FILE="$(python3 -c 'import json;print(json.load(open("model_manifest.json"))["modelo"]["arquivo"])')"

# GPU disponível: AMD (ROCm) ou NVIDIA; sem GPU, a imagem NLP roda em CPU (lenta; o orçamento a desliga).
GPU_ARGS=()
NLP_BUILD_ARGS=()
if [ -e /dev/kfd ]; then
  GPU_ARGS=(--device /dev/kfd --device /dev/dri --group-add video)
  NLP_BUILD_ARGS=(--build-arg "OLLAMA_IMAGE=$(python3 -c 'import json;print(json.load(open("model_manifest.json"))["runtime"]["imagem_rocm"])')")
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

run_image() {  # imagem, pasta de saída, [volumes extras...]
  local image="$1" dest="$2"; shift 2
  mkdir -p "$dest"
  local start end
  start=$(date +%s)
  docker run --rm --network none "$@" \
    -v "$PWD/data/desafio1_bracis.db:/data/ref/desafio1_bracis.db:ro" \
    -v "$PWD/data/txt:/data/in:ro" -v "$dest:/data/out" \
    "$image" --input /data/in --output /data/out 2> "$dest.log"
  end=$(date +%s)
  echo "  tempo: $((end - start)) s para $(ls data/txt/*.txt | wc -l) documentos (inclui subir o servidor)"
  grep -E "launcher|nlp\]|documentos," "$dest.log" | sed 's/^/  /' || true
}

score() {  # pasta de saída
  "$PY" -m tools.validar_saida --input data/txt --output "$1" | sed 's/^/  /'
  "$PY" json_to_submission.py "$1" "$1.csv" > /dev/null
  "$PY" -m tools.evaluate --submission "$1.csv" | grep '"score_final"' | sed 's/^ */  /'
}

no_data_inside() {  # imagem
  if docker run --rm --entrypoint sh "$1" -c 'find / -xdev \( -name "*.db" -o -name "*.gguf" \) 2>/dev/null | grep -q .'; then
    echo "  ERRO: a imagem contém base ou pesos"; exit 1
  fi
  echo "  ok: nenhuma base (.db) nem pesos (.gguf) dentro da imagem"
}

if [ "$WHICH" != "nlp" ]; then
  echo "== versão só com regras (Dockerfile.regras)"
  docker build -q -f Dockerfile.regras -t caca-alucinacoes-regras . > /dev/null
  no_data_inside caca-alucinacoes-regras
  run_image caca-alucinacoes-regras "$OUT/regras"
  score "$OUT/regras"
fi

if [ "$WHICH" != "regras" ]; then
  echo "== versão com NLP (Dockerfile)"
  if [ ! -f "modelos/$MANIFEST_FILE" ]; then
    "$PY" -m tools.baixar_modelo --dest modelos
  fi
  docker build -q "${NLP_BUILD_ARGS[@]}" -t caca-alucinacoes . > /dev/null
  no_data_inside caca-alucinacoes
  run_image caca-alucinacoes "$OUT/nlp1" "${GPU_ARGS[@]}" -v "$PWD/modelos:/models:ro"
  score "$OUT/nlp1"
  echo "  segunda execução (determinismo):"
  run_image caca-alucinacoes "$OUT/nlp2" "${GPU_ARGS[@]}" -v "$PWD/modelos:/models:ro" > /dev/null
  if diff -r "$OUT/nlp1" "$OUT/nlp2" > /dev/null; then echo "  ok: saídas idênticas"; else echo "  ERRO: saídas diferentes"; exit 1; fi
  echo "  sem os pesos (fallback para as regras):"
  run_image caca-alucinacoes "$OUT/sem_pesos" > /dev/null
  score "$OUT/sem_pesos"
fi

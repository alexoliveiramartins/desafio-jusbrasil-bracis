#!/usr/bin/env bash
# Ponto de entrada único da avaliação final:
#
#   bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>
#
# Lê a base canônica <caminho_db> (no formato original) e os .txt de <pasta_txt>, e grava <arquivo_saida>
# no formato de submissão (documento_id,citacoes), o mesmo do json_to_submission.py oficial.
#
#   1. preparação (só se faltar algo; é a única etapa com internet): baixa os pesos declarados em
#      model_manifest.json (HF id + revisão fixa, sha256 conferido) para ./modelos e constrói a imagem
#      Docker da GPU disponível;
#   2. execução, sem rede (--network none): base e textos montados só para leitura; o índice da base é
#      montado a partir de <caminho_db> a cada execução (nada é pré-calculado sobre a base de dev);
#   3. conversão: um JSON por documento -> <arquivo_saida> (json_to_submission.py oficial) e validação
#      do contrato (tools/validar_saida.py).
#
# Precisa de Docker e python3 (só biblioteca padrão). Variáveis opcionais: GPU=nvidia|amd|cpu (senão,
# detecta), CACA_ENSEMBLE=0 (só o modelo principal), CACA_MAX_LOADED_MODELS (padrão 2), IMAGEM (nome
# da imagem; padrão caca-alucinacoes:<gpu>). Sem GPU, roda só com as regras.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "uso: bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>" >&2
  exit 2
fi
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "erro: python3 não encontrado" >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "erro: docker não encontrado" >&2; exit 2; }
[ -f "$1" ] || { echo "erro: base não encontrada: $1" >&2; exit 2; }
[ -d "$2" ] || { echo "erro: pasta de textos não encontrada: $2" >&2; exit 2; }
DB="$(realpath "$1")"
TXT="$(realpath "$2")"
SAIDA="$(realpath -m "$3")"
N_TXT="$(find "$TXT" -maxdepth 1 -name '*.txt' | wc -l)"
[ "$N_TXT" -gt 0 ] || { echo "erro: nenhum .txt em $2" >&2; exit 2; }
mkdir -p "$(dirname "$SAIDA")"

# GPU disponível (pode ser forçada com GPU=nvidia|amd|cpu).
GPU="${GPU:-}"
if [ -z "$GPU" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then GPU=nvidia
  elif [ -e /dev/kfd ]; then GPU=amd
  else GPU=cpu; fi
fi
case "$GPU" in
  nvidia) GPU_ARGS=(--gpus all) ;;
  amd)    GPU_ARGS=(--device /dev/kfd --device /dev/dri --group-add video) ;;
  cpu)    GPU_ARGS=() ;;
  *)      echo "erro: GPU=$GPU (use nvidia, amd ou cpu)" >&2; exit 2 ;;
esac
IMAGEM="${IMAGEM:-caca-alucinacoes:$GPU}"
echo "[run] base: $DB"
echo "[run] textos: $TXT ($N_TXT documentos)"
echo "[run] GPU: $GPU · imagem: $IMAGEM"

# 1. Preparação: pesos e imagem (com internet, só na primeira vez).
cd "$RAIZ"
if [ "$GPU" != cpu ]; then
  FALTANDO="$("$PY" - <<'EOF'
import json
from pathlib import Path
m = json.load(open("model_manifest.json"))
modelos = [m["modelo"]] + [c for c in m.get("conjunto", [])]
print(" ".join(x["arquivo"] for x in modelos if not (Path("modelos") / x["arquivo"]).is_file()))
EOF
)"
  if [ -n "$FALTANDO" ]; then
    echo "[run] baixando os pesos declarados (revisão fixa do Hugging Face): $FALTANDO"
    "$PY" -m tools.baixar_modelo --dest modelos \
      || echo "[run] aviso: download dos pesos falhou; a execução segue só com as regras" >&2
  fi
fi
# A imagem leva o hash do código-fonte: reconstrói se ela faltar ou se o código mudou (sem mudança, não
# reconstrói nem acessa a rede).
FONTE="$(cat Dockerfile .dockerignore model_manifest.json \
  $(find src docker/nlp -type f -not -path '*/__pycache__/*' | LC_ALL=C sort) | sha256sum | cut -c1-16)"
if [ "$(docker image inspect -f '{{index .Config.Labels "caca.fonte"}}' "$IMAGEM" 2>/dev/null)" != "$FONTE" ]; then
  echo "[run] construindo a imagem $IMAGEM (código $FONTE)"
  docker build --build-arg "GPU=$GPU" --label "caca.fonte=$FONTE" -t "$IMAGEM" "$RAIZ"
fi

# 2. Execução sem rede.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/json" "$TMP/sem_pesos"
MODELOS="$RAIZ/modelos"
if [ "$GPU" = cpu ]; then
  MODELOS="$TMP/sem_pesos"   # sem GPU, os modelos seriam lentos demais: só as regras
  echo "[run] sem GPU: execução só com as regras"
fi
executar() {  # pasta de pesos, argumentos de GPU...
  local pesos="$1"; shift
  docker run --rm --network none "$@" \
    -e CACA_ENSEMBLE="${CACA_ENSEMBLE:-1}" -e CACA_MAX_LOADED_MODELS="${CACA_MAX_LOADED_MODELS:-2}" \
    -v "$DB:/data/ref/desafio1_bracis.db:ro" -v "$pesos:/models:ro" \
    -v "$TXT:/data/in:ro" -v "$TMP/json:/data/out" \
    "$IMAGEM" --input /data/in --output /data/out
}
INICIO=$(date +%s)
if ! executar "$MODELOS" "${GPU_ARGS[@]}"; then
  if [ "${#GPU_ARGS[@]}" -gt 0 ]; then
    echo "[run] aviso: a execução com GPU falhou; repetindo só com as regras" >&2
    find "$TMP/json" -name '*.json' -delete
    executar "$TMP/sem_pesos"
  else
    exit 1
  fi
fi
echo "[run] execução: $(( $(date +%s) - INICIO )) s"

# 3. Conversão para o formato de submissão e validação do contrato.
"$PY" "$RAIZ/json_to_submission.py" "$TMP/json" "$SAIDA"
"$PY" -m tools.validar_saida --input "$TXT" --output "$TMP/json" --db "$DB" \
  || echo "[run] aviso: o validador acusou problemas (ver acima)" >&2
echo "[run] saída: $SAIDA"

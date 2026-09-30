# Imagem da submissão: regras + camada de NLP com LLM aberto do Hugging Face (model_manifest.json),
# servido pelo Ollama DENTRO do contêiner, só em 127.0.0.1. Roda com --network none. Sem GPU, sem
# pesos ou sem tempo, cai sozinha para a versão só com regras (Dockerfile.regras).
#
#   python3 -m tools.baixar_modelo --dest modelos      # antes, com rede: revisão fixa + sha256
#
#   NVIDIA (padrão):  docker build -t caca-alucinacoes .
#                     docker run --rm --network none --gpus all <volumes> caca-alucinacoes --input /data/in --output /data/out
#   AMD (ROCm):       docker build --build-arg GPU=amd -t caca-alucinacoes .
#                     docker run --rm --network none --device /dev/kfd --device /dev/dri <volumes> caca-alucinacoes ...
#   Sem GPU:          docker build --build-arg GPU=cpu -t caca-alucinacoes .   (lento; o orçamento de tempo desliga
#                     a camada se passar de 40 s/doc e o resto sai só com as regras)
#
#   <volumes>: -v "$PWD/data/desafio1_bracis.db:/data/ref/desafio1_bracis.db:ro" -v "$PWD/modelos:/models:ro"
#              -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out"
# Pesos e base não vão na imagem (regra da competição): são montados. Imagens-base fixadas por digest; com o
# BuildKit, só a variante escolhida é baixada.
ARG GPU=nvidia
FROM ollama/ollama:0.33.2@sha256:020e4134285e2ef4d8fd801234176de3b4faadc992a3eb06c8e66a2f9d4c4ba2 AS ollama-nvidia
FROM ollama/ollama:0.33.2-rocm@sha256:eaddb04a9e21fa3950c279792f04752db7addeebc5e9f314d9abc0dda33bca24 AS ollama-amd
FROM ollama-nvidia AS ollama-cpu
FROM ollama-${GPU} AS ollama

FROM python:3.12.3-slim@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    CACA_DB=/data/ref/desafio1_bracis.db \
    CACA_MODEL_DIR=/models \
    OLLAMA_MODELS=/tmp/ollama-models \
    OLLAMA_NO_CLOUD=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
COPY src/ /app/src/
COPY model_manifest.json /app/model_manifest.json
COPY docker/nlp/ /app/docker/nlp/

ENTRYPOINT ["python", "-m", "src.launcher"]
CMD ["--input", "/data/in", "--output", "/data/out"]

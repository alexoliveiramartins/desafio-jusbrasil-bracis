# Execução offline do pipeline determinístico (só biblioteca padrão, sem GPU).
#   docker build -t caca-alucinacoes .
#   docker run --rm --network none -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out" \
#     caca-alucinacoes --input /data/in --output /data/out
# A base canônica vai dentro da imagem (data/desafio1_bracis.db); para usar
# outra, monte-a e passe --db /caminho/da/base.db.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /app
COPY src/ /app/src/
COPY data/desafio1_bracis.db /app/data/desafio1_bracis.db

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--input", "/data/in", "--output", "/data/out"]

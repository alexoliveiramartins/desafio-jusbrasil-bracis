# syntax=docker/dockerfile:1

FROM python:3.12-slim

LABEL org.opencontainers.image.title="BRACIS 2026 - Verificador de citacoes juridicas" \
      org.opencontainers.image.description="Extrai, classifica e resolve citacoes contra a base canonica do desafio" \
      org.opencontainers.image.source="regex.py"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# A base canonica faz parte da cobertura congelada e permite execucao offline.
# Os documentos finais nao sao copiados: devem ser montados em /data/txt.
COPY regex.py desafio1_bracis.db ./

RUN mkdir -p /data/txt /output \
    && python -m py_compile regex.py

VOLUME ["/data/txt", "/output"]

ENTRYPOINT ["python", "/app/regex.py"]
CMD ["--txt-dir", "/data/txt", "--db", "/app/desafio1_bracis.db", "--output-dir", "/output"]

"""Pipeline por documento e CLI do contrato de execução.

    python -m src.main --input <pasta com .txt> --output <pasta de saída> [--db base.db]

Para cada .txt: identifica os spans (spans.py), classifica cada citação contra
a base (classify.py) e grava <documento_id>.json no formato do contrato:

    {"documento_id": "...", "citacoes": [
        {"inicio": 0, "fim": 10, "trecho": "...", "tipo": "lei|jurisprudencia",
         "classificacao": "real|inventada|incompleta",
         "resolucao": {"id_canonico": "123"} | null,
         "confianca": 0.97}
    ]}

Os offsets são codepoints do texto ORIGINAL (arquivo lido com newline="").
Só biblioteca padrão. Sem --nlp, nenhuma chamada de rede; com --nlp, só ao
servidor LLM local (src/nlp.py), e sem ele o pipeline segue só com as regras.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

from .classify import INCOMPLETA, REAL, CanonicalIndex, Resolution, resolve
from .nlp import DEFAULT_MODEL, DEFAULT_URL, NLPLayer, from_args
from .spans import clean, expand_spans, extract_citations, find_anchors, to_original


@lru_cache(maxsize=4)
def _name_words(known_relators: tuple[frozenset, ...]) -> frozenset[str]:
    """Todos os tokens de nomes de relatores da base (para juntar/separar nomes na limpeza)."""
    return frozenset().union(*known_relators)


def extract(content: str, known_relators: list[frozenset] = ()) -> list[dict]:
    """Citações do documento, com spans no texto original.

    O regex (formas catalogadas) e as âncoras (formas novas) rodam no texto
    limpo; o mapa de offsets devolve cada span ao original. `trecho` é o texto
    original; `trecho_norm`, o limpo, é o que o resolvedor lê.
    """
    cleaned, mapping = clean(content, _name_words(tuple(known_relators)))
    found = expand_spans(cleaned, extract_citations(cleaned))
    found = sorted(found + find_anchors(cleaned, found, known_relators), key=lambda c: c["inicio"])
    citations = []
    for c in found:
        start, end = to_original(mapping, c["inicio"], c["fim"])
        citations.append(c | {"inicio": start, "fim": end, "trecho": content[start:end], "trecho_norm": c["trecho"]})
    return citations


# A métrica oficial recusa a submissão INTEIRA se duas citações de um documento se sobrepõem com IoU >= 0,5.
DUPLICATE_IOU = 0.5


def _iou(a: dict, b: dict) -> float:
    inter = max(0, min(a["fim"], b["fim"]) - max(a["inicio"], b["inicio"]))
    return inter / ((a["fim"] - a["inicio"]) + (b["fim"] - b["inicio"]) - inter) if inter else 0.0


def without_duplicates(citations: list[dict]) -> list[dict]:
    """Garantia do contrato: de um grupo de citações com IoU >= 0,5 fica uma só.

    Fica a de maior confiança; no empate, a mais longa; depois, a primeira. A ordem original se mantém.
    """
    ranked = sorted(range(len(citations)), key=lambda i: (-citations[i]["resolution"].confianca,
                                                         -(citations[i]["fim"] - citations[i]["inicio"]),
                                                         citations[i]["inicio"], i))
    kept = []
    for i in ranked:
        if all(_iou(citations[i], citations[k]) < DUPLICATE_IOU for k in kept):
            kept.append(i)
    return [citations[i] for i in sorted(kept)]


def process_text(documento_id: str, content: str, index: CanonicalIndex, debug: bool = False,
                 nlp: NLPLayer | None = None) -> dict:
    citations = []
    for citation in extract(content, index.relator_tokens):
        # Contexto para desempate (ementa) e checagem do relator citado após o número.
        citation["contexto"] = content[max(0, citation["inicio"] - 400):citation["fim"] + 400]
        citation["contexto_depois"] = content[citation["fim"]:citation["fim"] + 160]
        try:
            citation["resolution"] = resolve(citation, index)
        except Exception:  # noqa: BLE001 — uma citação nunca derruba o documento
            print(f"[erro] {documento_id} {citation['trecho']!r}\n{traceback.format_exc()}", file=sys.stderr)
            citation["resolution"] = Resolution(INCOMPLETA, None, 0.40, "erro_interno")
        citations.append(citation)
    if nlp is not None:
        citations = nlp.refine(content, citations, index)
    citations = without_duplicates(citations)
    output = []
    for citation in citations:
        resolution = citation["resolution"]
        output.append({
            "inicio": citation["inicio"],
            "fim": citation["fim"],
            "trecho": citation["trecho"],
            "tipo": citation["tipo"],
            "classificacao": resolution.classificacao,
            "resolucao": {"id_canonico": resolution.id_canonico} if resolution.classificacao == REAL else None,
            "confianca": resolution.confianca,
        })
        if debug:
            output[-1]["_regra"] = resolution.regra
    return {"documento_id": documento_id, "citacoes": output}


def process_file(path: Path, index: CanonicalIndex, debug: bool = False, nlp: NLPLayer | None = None) -> dict:
    # newline="" preserva CRLF: os offsets são em codepoints do texto original.
    with path.open(encoding="utf-8", newline="") as stream:
        content = stream.read()
    return process_text(path.stem, content, index, debug, nlp)


# A base não vai dentro da imagem (regra da competição): no contêiner ela é montada e indicada por CACA_DB.
DEFAULT_DB = Path(os.environ.get("CACA_DB") or Path(__file__).resolve().parent.parent / "data" / "desafio1_bracis.db")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Caça-Alucinações: extrai e classifica citações.")
    parser.add_argument("--input", type=Path, default=Path("data/txt"), help="pasta com os .txt")
    parser.add_argument("--output", type=Path, default=Path("resultados"), help="pasta de saída dos JSONs")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="base canônica SQLite")
    parser.add_argument("--nlp", action="store_true", help="liga a camada de NLP (LLM local via Ollama)")
    parser.add_argument("--nlp-model", default=DEFAULT_MODEL, help="modelo GGUF do Hugging Face no Ollama")
    parser.add_argument("--nlp-url", default=DEFAULT_URL, help="servidor Ollama")
    parser.add_argument("--nlp-cache", type=Path, default=None, help="pasta de cache das respostas do modelo")
    parser.add_argument("--nlp-extra-model", action="append", default=[],
                        help="outro modelo (conjunto): união para citações novas, acordo para normalizar")
    parser.add_argument("--nlp-budget-doc", type=float, default=40.0,
                        help="média máxima de segundos por documento na camada (depois ela se desliga)")
    parser.add_argument("--nlp-budget-total", type=float, default=3 * 3600.0,
                        help="segundos totais na camada (depois ela se desliga)")
    args = parser.parse_args(argv)
    if not args.db.is_file():
        print(f"base canônica não encontrada: {args.db} (monte-a e passe --db ou CACA_DB)", file=sys.stderr)
        return 2
    nlp = from_args(args.nlp, args.nlp_url, args.nlp_model, args.nlp_cache,
                    budget_doc_s=args.nlp_budget_doc, budget_total_s=args.nlp_budget_total,
                    extra_models=tuple(args.nlp_extra_model))

    files = sorted(args.input.glob("*.txt"))
    if not files:
        print(f"nenhum .txt em {args.input}", file=sys.stderr)
        return 1
    start = time.monotonic()
    index = CanonicalIndex.from_sqlite(args.db)
    print(f"base indexada em {time.monotonic() - start:.1f}s", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    total = 0
    for n, path in enumerate(files, 1):
        doc_start = time.monotonic()
        document = process_file(path, index, nlp=nlp)
        classes = [c["classificacao"] for c in document["citacoes"]]
        total += len(classes)
        print(f"[{n:>3}/{len(files)}] {path.stem}: {len(classes)} citações "
              f"(real {classes.count('real')}, inventada {classes.count('inventada')}, "
              f"incompleta {classes.count('incompleta')}) {time.monotonic() - doc_start:.2f}s", file=sys.stderr)
        (args.output / f"{path.stem}.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.monotonic() - start
    print(f"{len(files)} documentos, {total} citações, {elapsed:.1f}s -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

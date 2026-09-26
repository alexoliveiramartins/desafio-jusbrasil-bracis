"""Experimento separado: não altera o extrator nem os JSONs de submissão."""

import csv
import json
from pathlib import Path
from time import perf_counter

import regex

if __package__:
    from .benchmark_extractor import compare_citations, print_summary
    from .extractor import extract_citations
    from .re_patterns import PATTERNS
else:
    from benchmark_extractor import compare_citations, print_summary
    from extractor import extract_citations
    from re_patterns import PATTERNS


# Palavras longas e marcadores textuais. Números, UFs, tribunais, nomes e
# siglas curtas ficam fora dos grupos fuzzy na variante localizada.
TEXT_FRAGMENTS = (
    "Súmula", "Vinculante", "Constituição", "Federal", "República",
    "Código", "Processo", "Civil", "Penal", "Consumidor", "Complementar",
    "Reclamação", "Recurso", "Especial", "Agravo", "Regimental",
    "Interno", "Embargos", "Declaração", "acórdão", "precedente",
    "proferido", "julgado", r"relatoria\s+de",
)


def compile_patterns(mode: str, max_errors: int) -> list:
    patterns = []
    for family, citation_type, original in PATTERNS:
        source = original.pattern
        if mode == "text":
            for fragment in TEXT_FRAGMENTS:
                source = source.replace(fragment, rf"(?:{fragment}){{e<=1}}")
        else:
            # e = substituições, inserções e deleções.
            source = rf"(?:{source}){{e<={max_errors}}}"
        pattern = regex.compile(
            source, regex.IGNORECASE | regex.VERBOSE | regex.ENHANCEMATCH | regex.VERSION0
        )
        patterns.append((family, citation_type, pattern))
    return patterns


def extract_fuzzy(text: str, patterns: list, max_errors: int) -> list[dict]:
    candidates = []
    for family, citation_type, pattern in patterns:
        # Se exceder o limite, o experimento falha explicitamente. Não trata
        # uma busca interrompida como documento sem citações.
        for match in pattern.finditer(text, timeout=5):
            # Na versão localizada, cada fragmento admite um erro, mas a soma
            # de erros da citação também precisa respeitar o orçamento total.
            if sum(match.fuzzy_counts) > max_errors or match.start() == match.end():
                continue
            candidates.append({
                "inicio": match.start(),
                "fim": match.end(),
                "trecho": match.group(),
                "tipo": citation_type,
                "familia": family,
                "erros": dict(zip(
                    ("substituicoes", "insercoes", "delecoes"), match.fuzzy_counts
                )),
                "posicoes_erros": dict(zip(
                    ("substituicoes", "insercoes", "delecoes"), match.fuzzy_changes
                )),
            })

    # Mesma política de contenção usada pelo extrator atual.
    candidates.sort(key=lambda item: (item["inicio"], -item["fim"]))
    citations = []
    for candidate in candidates:
        if not any(
            previous["inicio"] <= candidate["inicio"]
            and candidate["fim"] <= previous["fim"]
            for previous in citations
        ):
            citations.append(candidate)
    return citations


def load_documents() -> list[dict]:
    documents = {}
    with Path("data/goldenset.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            document_id = row["documento_id"]
            if document_id not in documents:
                with Path("data/txt", f"{document_id}.txt").open(
                    encoding="utf-8", newline=""
                ) as text_file:
                    text = text_file.read()
                documents[document_id] = {
                    "documento_id": document_id, "nivel": int(row["nivel"]),
                    "texto": text, "esperadas": [],
                }
            start, end = int(row["inicio"]), int(row["fim"])
            documents[document_id]["esperadas"].append({
                "citacao_id": row["citacao_id"], "inicio": start, "fim": end,
                "trecho": documents[document_id]["texto"][start:end],
            })
    return list(documents.values())


def main() -> None:
    documents = load_documents()
    experiments = []
    for label, mode, budget in [
        ("Baseline re", "baseline", 0),
        ("Texto: 1 edição total", "text", 1),
        ("Global: 1 edição", "global", 1),
    ]:
        patterns = compile_patterns(mode, budget) if mode != "baseline" else []
        started = perf_counter()
        results = []
        for document in documents:
            predictions = (
                extract_citations(document["texto"]) if mode == "baseline"
                else extract_fuzzy(document["texto"], patterns, budget)
            )
            results.append({
                "documento_id": document["documento_id"], "nivel": document["nivel"],
                **compare_citations(document["esperadas"], predictions),
            })
        elapsed = perf_counter() - started
        exact_spans = sum(
            pair["iou"] == 1 for doc in results for pair in doc["acertos"]
        )
        print(f"\n{label} — {elapsed:.3f}s (extração + comparação, sem carga/compilação)", flush=True)
        print_summary("TOTAL", results)
        for level in (1, 2):
            print_summary(f"Nível {level}", [doc for doc in results if doc["nivel"] == level])
        print(f"Spans exatamente iguais: {exact_spans}", flush=True)
        experiments.append({
            "variante": label, "segundos": elapsed,
            "spans_exatos": exact_spans, "documentos": results,
        })

    report = Path("relatorios/fuzzy.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "regex_version": regex.__version__,
        "nota": "Avaliação de desenvolvimento, apenas spans; sem resolução ou classificação.",
        "experimentos": experiments,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalhes, faltantes e posições dos erros: {report}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

if __package__:
    from .re_patterns import PATTERNS
else:
    from re_patterns import PATTERNS

def extract_citations(content: str) -> list[dict]:
    candidates = []
    for _, citation_type, pattern in PATTERNS:
        for match in pattern.finditer(content):
            candidates.append({
                "inicio": match.start(),
                "fim": match.end(),
                "trecho": match.group(),
                "tipo": citation_type,
                # Provisório: a classificação será feita pelo resolvedor.
                "classificacao": "inventada",
                "resolucao": None,
            })

    # Remove candidatos contidos em outro, como Rcl dentro de AgInt na Rcl.
    # Ocorrências da mesma referência em posições diferentes são preservadas.
    candidates.sort(key=lambda c: (c["inicio"], -c["fim"]))
    citations = []
    for candidate in candidates:
        if any(
            previous["inicio"] <= candidate["inicio"]
            and candidate["fim"] <= previous["fim"]
            for previous in citations
        ):
            continue
        citations.append(candidate)
    return citations


def extract_document(file: Path) -> dict:
    # Preserva CRLF, quando presente, para não deslocar os offsets originais.
    with file.open(encoding="utf-8", newline="") as stream:
        content = stream.read()

    return {"documento_id": file.stem, "citacoes": extract_citations(content)}


def main() -> None:
    citations_count = 0
    folder_path = Path("data/txt")
    output_path = Path("resultados")
    output_path.mkdir(parents=True, exist_ok=True)

    documents = []

    for file in sorted(folder_path.glob("*.txt")):
        document = extract_document(file)
        citations_count += len(document["citacoes"])

        output_file = output_path / f"{file.stem}.json"
        output_file.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        documents.append(document)
        
    print(json.dumps({"documentos": documents}, ensure_ascii=False, indent=2))
    print(f"Citações extraidas: {citations_count}")


if __name__ == "__main__":
    main()

"""Avalia apenas spans; classificação, tipo e ID não entram nos acertos."""

import csv
import json
from pathlib import Path

if __package__:
    from .extractor import extract_document
else:
    from extractor import extract_document


def compare_citations(expected: list[dict], extracted: list[dict]) -> dict:
    # Mesmo pareamento guloso por maior IoU de kaggle_metric._casar.
    candidates = []
    for gold_index, gold in enumerate(expected):
        for prediction_index, prediction in enumerate(extracted):
            intersection = max(
                0,
                min(gold["fim"], prediction["fim"])
                - max(gold["inicio"], prediction["inicio"]),
            )
            union = (
                gold["fim"] - gold["inicio"]
                + prediction["fim"] - prediction["inicio"] - intersection
            )
            iou = intersection / union if union else 0
            if iou >= 0.5:
                candidates.append((-iou, gold_index, prediction_index))

    matched_gold = set()
    matched_predictions = set()
    matches = []
    for negative_iou, gold_index, prediction_index in sorted(candidates):
        if gold_index in matched_gold or prediction_index in matched_predictions:
            continue
        matched_gold.add(gold_index)
        matched_predictions.add(prediction_index)
        matches.append({
            "esperada": expected[gold_index],
            "extraida": extracted[prediction_index],
            "iou": -negative_iou,
        })

    return {
        "acertos": matches,
        "sem_correspondencia": [
            citation for index, citation in enumerate(extracted)
            if index not in matched_predictions
        ],
        "faltantes": [
            citation for index, citation in enumerate(expected)
            if index not in matched_gold
        ],
    }


def print_summary(label: str, documents: list[dict]) -> None:
    correct = sum(len(doc["acertos"]) for doc in documents)
    extra = sum(len(doc["sem_correspondencia"]) for doc in documents)
    missing = sum(len(doc["faltantes"]) for doc in documents)
    precision = correct / (correct + extra) if correct + extra else 0
    recall = correct / (correct + missing) if correct + missing else 0
    print(
        f"{label}: {correct} acertos | {extra} sem correspondência | "
        f"{missing} faltantes | precisão {precision:.1%} | recall {recall:.1%}"
    )


def main() -> None:
    folder_path = Path("data/txt")
    gold_path = Path("data/goldenset.csv")
    report_path = Path("relatorios/extracao.json")

    gold_by_document = {}
    levels = {}
    with gold_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            document_id = row["documento_id"]
            levels[document_id] = int(row["nivel"])
            citation = {
                "citacao_id": row["citacao_id"],
                "inicio": int(row["inicio"]),
                "fim": int(row["fim"]),
                "tipo": row["tipo"],
            }
            gold_by_document.setdefault(document_id, []).append(citation)

    documents = []
    for document_id, expected in sorted(gold_by_document.items()):
        file = folder_path / f"{document_id}.txt"
        # Executa a versão atual do extrator, sem depender de JSONs antigos.
        extracted = extract_document(file)["citacoes"]
        with file.open(encoding="utf-8", newline="") as stream:
            text = stream.read()
        for citation in expected:
            # Recupera quebras reais, evitando depender do escape usado no CSV.
            citation["trecho"] = text[citation["inicio"]:citation["fim"]]
        documents.append({
            "documento_id": document_id,
            "nivel": levels[document_id],
            **compare_citations(expected, extracted),
        })

    print("Extração: IoU >= 0,5, uma correspondência por citação.")
    print("Classificação, tipo e ID canônico não são avaliados.")
    print("Toda extração sem par é listada, inclusive componentes extras tolerados pela métrica oficial.")
    print_summary("TOTAL", documents)
    for level in sorted(set(levels.values())):
        print_summary(f"Nível {level}", [doc for doc in documents if doc["nivel"] == level])

    for level in sorted(set(levels.values())):
        level_documents = [doc for doc in documents if doc["nivel"] == level]
        print(f"\nNÍVEL {level}")
        for field, title in [
            ("faltantes", "CITAÇÕES FALTANTES"),
            ("sem_correspondencia", "EXTRAÇÕES SEM CORRESPONDÊNCIA"),
        ]:
            count = sum(len(doc[field]) for doc in level_documents)
            print(f"\n{title} ({count})")
            if count == 0:
                print("  Nenhuma.")
            for document in level_documents:
                for citation in document[field]:
                    print(
                        f"  {document['documento_id']} "
                        f"[{citation['inicio']}:{citation['fim']}] {citation['trecho']!r}"
                    )

    # Fora de resultados/, para não ser confundido com JSON de submissão.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"documentos": documents}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nRelatório completo: {report_path}")


if __name__ == "__main__":
    main()

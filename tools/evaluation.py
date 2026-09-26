"""Avaliação local com a métrica oficial (kaggle_metric.avaliar).

Funções usadas por tools.evaluate, tools.battery e tools.calibrate: lê um
goldenset no formato oficial, roda o pipeline em processo e pontua.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from json_to_submission import encode
from kaggle_metric import avaliar
from src.classify import CanonicalIndex
from src.main import process_file

DB = Path("data/desafio1_bracis.db")
DEV = (Path("data/txt"), Path("data/goldenset.csv"))
SYNTHETIC = Path("data/synthetic")


@dataclass
class Split:
    name: str
    txt: Path
    gold: Path


def split_paths(name: str) -> Split:
    if name == "dev":
        return Split(name, *DEV)
    folder = Path(name) if Path(name).is_dir() else SYNTHETIC / name
    return Split(name, folder / "txt", folder / "goldenset.csv")


def load_gold(path: Path) -> dict[str, list[dict]]:
    """Linhas do goldenset por documento (utf-8-sig: o goldenset oficial tem BOM)."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            by_doc[row["documento_id"]].append(row)
    return dict(by_doc)


def run_split(split: Split, index: CanonicalIndex, debug: bool = False) -> dict[str, dict]:
    """Saída do pipeline (formato do contrato) para cada documento do goldenset."""
    gold = load_gold(split.gold)
    return {doc_id: process_file(split.txt / f"{doc_id}.txt", index, debug=debug) for doc_id in gold}


def score(gold: dict[str, list[dict]], outputs: dict[str, dict]) -> dict:
    """Resultado de kaggle_metric.avaliar + recall de spans (IoU >= 0,5) e de spans exatos."""
    solution, submission = [], []
    found = exact = total = 0
    for doc_id, rows in gold.items():
        doc = outputs.get(doc_id, {"citacoes": []})
        submission.append({"documento_id": doc_id, "citacoes": encode(doc)})
        solution.append({
            "documento_id": doc_id,
            "nivel": int(rows[0]["nivel"]),
            "citacoes": "|".join(
                f"{r['inicio']},{r['fim']},{r['classificacao']},{r['id_canonico'] or '-'}" for r in rows
            ),
        })
        spans = [(c["inicio"], c["fim"]) for c in doc["citacoes"]]
        for r in rows:
            total += 1
            gold_span = (int(r["inicio"]), int(r["fim"]))
            found += any(iou(gold_span, s) >= 0.5 for s in spans)
            exact += gold_span in spans
    result = avaliar(pd.DataFrame(solution), pd.DataFrame(submission, dtype=str))
    result["recall_spans"] = found / total if total else 0.0
    result["exact_spans"] = exact / total if total else 0.0
    return result


def score_submission(submission_csv: Path, gold_csv: Path) -> dict:
    """Pontua um submission.csv já gerado (fluxo oficial: JSON -> CSV -> métrica)."""
    gold = load_gold(gold_csv)
    solution = pd.DataFrame([{
        "documento_id": doc_id,
        "nivel": int(rows[0]["nivel"]),
        "citacoes": "|".join(
            f"{r['inicio']},{r['fim']},{r['classificacao']},{r['id_canonico'] or '-'}" for r in rows
        ),
    } for doc_id, rows in gold.items()])
    submission = pd.read_csv(submission_csv, dtype=str, keep_default_na=False)
    return avaliar(solution, submission)


def summary(result: dict) -> str:
    """Uma linha: score final, por nível F1 por classe e τ."""
    parts = []
    for level, n in sorted(result["niveis"].items()):
        f1 = " ".join(f"{c[:4]}={v:.3f}" for c, v in n["f1_por_classe"].items())
        parts.append(f"N{level} {n['score']:.3f} ({f1}) τ={n['tau']:.3f}")
    recall = f" | spans {result['recall_spans']:.3f}" if "recall_spans" in result else ""
    if "exact_spans" in result:
        recall += f" (exatos {result['exact_spans']:.3f})"
    return f"score={result['score_final']:.4f} | " + " | ".join(parts) + recall


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def diagnose(gold: dict[str, list[dict]], outputs: dict[str, dict]) -> Counter:
    """Imprime faltantes, erros de classe/id e espúrias; devolve a matriz de confusão.

    Mesmo critério de pareamento da métrica: IoU >= 0,5, um para um.
    """
    confusion = Counter()
    for doc_id, rows in sorted(gold.items()):
        preds = outputs.get(doc_id, {"citacoes": []})["citacoes"]
        used = set()
        for row in rows:
            span = (int(row["inicio"]), int(row["fim"]))
            options = [(iou(span, (p["inicio"], p["fim"])), i) for i, p in enumerate(preds) if i not in used]
            best_iou, best = max(options, default=(0.0, None))
            if best is None or best_iou < 0.5:
                confusion[(row["classificacao"], "—")] += 1
                print(f"FALTANTE  {doc_id} {row['trecho']!r} ({row['classificacao']})")
                continue
            used.add(best)
            pred = preds[best]
            pred_id = (pred.get("resolucao") or {}).get("id_canonico") or ""
            confusion[(row["classificacao"], pred["classificacao"])] += 1
            wrong_id = row["classificacao"] == "real" == pred["classificacao"] and pred_id != row["id_canonico"]
            if pred["classificacao"] != row["classificacao"] or wrong_id:
                print(f"ERRO      {doc_id} {row['trecho']!r} -> {pred['trecho']!r}: gold={row['classificacao']} "
                      f"{row['id_canonico'] or ''} pred={pred['classificacao']} {pred_id} {pred.get('_regra', '')}")
        for i, pred in enumerate(preds):
            if i not in used:
                confusion[("—", pred["classificacao"])] += 1
                print(f"ESPÚRIA   {doc_id} {pred['trecho']!r} ({pred['classificacao']})")
    return confusion

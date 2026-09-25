
import argparse
import csv
from pathlib import Path
import pandas as pd

def parse_submission(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    by_doc = {}
    for _, row in df.iterrows():
        doc = row["documento_id"]
        items = []
        raw = row.get("citacoes", "")
        if raw:
            for part in raw.split("|"):
                fields = part.split(",")
                if len(fields) < 5:
                    raise ValueError(f"Formato inválido em {doc}: {part!r}")
                inicio, fim, classe, doc_id, confianca = fields[:5]
                items.append({
                    "inicio": int(inicio),
                    "fim": int(fim),
                    "classificacao": classe,
                    "id_canonico": None if doc_id == "-" else doc_id,
                    "confianca": float(confianca),
                })
        by_doc[doc] = items
    return by_doc

def iou(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = pd.read_csv(args.gold)
    pred = parse_submission(args.submission)

    rows = []
    exact = 0
    matched = 0
    unmatched_gold = 0
    extra_pred = 0

    for doc, gdf in gold.groupby("documento_id", sort=False):
        preds = pred.get(doc, [])
        unused = set(range(len(preds)))

        pairs = []
        for _, g in gdf.iterrows():
            best = None
            for j in unused:
                p = preds[j]
                score = iou(int(g.inicio), int(g.fim), p["inicio"], p["fim"])
                if best is None or score > best[0]:
                    best = (score, j, p)

            if best is None or best[0] < 0.5:
                unmatched_gold += 1
                rows.append({
                    "documento_id": doc,
                    "citacao_id": g["citacao_id"],
                    "status": "missing",
                    "gold_inicio": int(g.inicio),
                    "gold_fim": int(g.fim),
                    "pred_inicio": "",
                    "pred_fim": "",
                    "delta_inicio": "",
                    "delta_fim": "",
                    "iou": 0.0,
                    "gold_trecho": g["trecho"],
                })
                continue

            score, j, p = best
            unused.remove(j)
            matched += 1
            is_exact = p["inicio"] == int(g.inicio) and p["fim"] == int(g.fim)
            if is_exact:
                exact += 1
            else:
                rows.append({
                    "documento_id": doc,
                    "citacao_id": g["citacao_id"],
                    "status": "boundary_mismatch",
                    "gold_inicio": int(g.inicio),
                    "gold_fim": int(g.fim),
                    "pred_inicio": p["inicio"],
                    "pred_fim": p["fim"],
                    "delta_inicio": p["inicio"] - int(g.inicio),
                    "delta_fim": p["fim"] - int(g.fim),
                    "iou": round(score, 6),
                    "gold_trecho": g["trecho"],
                })

        for j in sorted(unused):
            p = preds[j]
            extra_pred += 1
            rows.append({
                "documento_id": doc,
                "citacao_id": "",
                "status": "extra",
                "gold_inicio": "",
                "gold_fim": "",
                "pred_inicio": p["inicio"],
                "pred_fim": p["fim"],
                "delta_inicio": "",
                "delta_fim": "",
                "iou": 0.0,
                "gold_trecho": "",
            })

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    total = len(gold)
    print(f"gold_total: {total}")
    print(f"matched_iou>=0.5: {matched}")
    print(f"exact_spans: {exact}")
    print(f"boundary_mismatches: {matched - exact}")
    print(f"missing_gold: {unmatched_gold}")
    print(f"extra_pred: {extra_pred}")
    print(f"exact_span_rate: {exact / total:.4%}")
    print(f"report: {args.out}")

if __name__ == "__main__":
    main()

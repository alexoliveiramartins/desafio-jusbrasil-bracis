"""Benchmark de robustez: soluções × perfis de ruído × sementes, métrica oficial.

    python -m tools.stress.bench                                  # tudo (gera o que faltar)
    python -m tools.stress.bench --profiles limpo extremo --seeds 5
    python -m tools.stress.bench --solutions lucas                # só a solução atual

Soluções:
  lucas         src.main (a solução do repositório), lendo como o CLI lê;
  final_robust  baseline/final_robust/final_solution.py, lendo como ele lê
                (Path.read_text, que converte CRLF e desloca offsets).

Por execução: score oficial (kaggle_metric.avaliar), F1 por classe e τ por
nível, recall de spans (IoU >= 0,5), spans exatos, predições espúrias (sem par
e fora da regra EXTRA da métrica), matriz gabarito -> predição, documentos que
lançaram exceção e tempo. Salva relatorios/stress/benchmark.{json,md}.

É instrumento de MEDIDA: ajustar regras olhando estes erros transforma o
conjunto em iteração (ver README, "Avaliação sem overfitting").
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import statistics
import subprocess
import time
import traceback
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

import kaggle_metric as km
from src.classify import CanonicalIndex
from src.main import process_file
from tools.evaluation import DB, load_gold, score

from . import BASE_SEED, COMPOSITE, OUT, PROFILES, generate, seeds_for, source_paths, split_dir

REPORT = Path("relatorios/stress")
FINAL_ROBUST = Path("baseline/final_robust/final_solution.py")


# ------------------------------------------------------------------ soluções

class Lucas:
    name = "lucas"

    def __init__(self):
        self.index = CanonicalIndex.from_sqlite(DB)

    def run(self, path: Path) -> dict:
        return process_file(path, self.index)


class FinalRobust:
    name = "final_robust"

    def __init__(self):
        spec = importlib.util.spec_from_file_location("final_solution", FINAL_ROBUST)
        self.fs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fs)
        self.conn = sqlite3.connect(DB)
        self.conn.row_factory = sqlite3.Row

    def run(self, path: Path) -> dict:
        text = path.read_text(encoding="utf-8")  # igual a final_solution.executar
        return {"documento_id": path.stem, "citacoes": [self.fs.validar(self.conn, c) for c in self.fs.extrair(text)]}


SOLUTIONS = {"lucas": Lucas, "final_robust": FinalRobust}


# ------------------------------------------------------------------ medida

def run_solution(solution, folder: Path, gold: dict) -> tuple[dict, list[str], float]:
    outputs, errors = {}, []
    start = time.monotonic()
    for doc_id in gold:
        try:
            outputs[doc_id] = solution.run(folder / "txt" / f"{doc_id}.txt")
        except Exception:  # noqa: BLE001 — um documento não derruba a bateria
            errors.append(f"{doc_id}: {traceback.format_exc(limit=1).strip().splitlines()[-1]}")
            outputs[doc_id] = {"documento_id": doc_id, "citacoes": []}
    return outputs, errors, time.monotonic() - start


def confusion(gold: dict, outputs: dict) -> tuple[Counter, int, int]:
    """Matriz gabarito -> predição com o pareamento da métrica; espúrias e total de predições."""
    matrix, spurious, total = Counter(), 0, 0
    for doc_id, rows in gold.items():
        golds = [dict(inicio=int(r["inicio"]), fim=int(r["fim"]), classe=r["classificacao"],
                      doc_ids=frozenset(filter(None, r["id_canonico"].split(":")))) for r in rows]
        preds = [dict(inicio=c["inicio"], fim=c["fim"], classe=c["classificacao"],
                      id_canonico=str((c.get("resolucao") or {}).get("id_canonico") or ""))
                 for c in outputs[doc_id]["citacoes"]]
        total += len(preds)
        pairs, g_free, p_free = km._casar(golds, preds)
        for gi, pi in pairs:
            g, p = golds[gi], preds[pi]
            wrong_link = g["classe"] == p["classe"] == "real" and p["id_canonico"] not in g["doc_ids"]
            matrix[(g["classe"], "real (id errado)" if wrong_link else p["classe"])] += 1
        for gi in g_free:
            matrix[(golds[gi]["classe"], "não extraída")] += 1
        matched = [golds[gi] for gi, _ in pairs]
        for pi in p_free:
            if not any(km._contida(preds[pi], g) for g in matched):
                spurious += 1
                matrix[("espúria", preds[pi]["classe"])] += 1
    return matrix, spurious, total


def measure(solution, folder: Path) -> dict:
    gold = load_gold(folder / "goldenset.csv")
    outputs, errors, seconds = run_solution(solution, folder, gold)
    n_gold = sum(map(len, gold.values()))
    matrix, spurious, n_pred = confusion(gold, outputs)
    try:
        result = score(gold, outputs)
    except km.ParticipantVisibleError as exc:  # submissão que o Kaggle rejeitaria
        return {"score": 0.0, "invalida": str(exc), "erros": errors, "segundos": seconds}
    niveis = result["niveis"]
    return {
        "score": result["score_final"],
        "niveis": {k: {"score": v["score"], "macro_f1": v["macro_f1"], "tau": v["tau"], "b": v["b"],
                       "f1": v["f1_por_classe"]} for k, v in niveis.items()},
        "tau": max(v["tau"] for v in niveis.values()),
        "recall_spans": result["recall_spans"],
        "exact_spans": result["exact_spans"],
        "espurias_por_100": 100 * spurious / n_gold,
        "predicoes": n_pred,
        "matriz": {f"{g} -> {p}": n for (g, p), n in sorted(matrix.items())},
        "erros": errors,
        "segundos": seconds,
    }


# ------------------------------------------------------------------ agregação

def mean(runs: list[dict], key) -> float:
    values = [key(r) for r in runs if "invalida" not in r]
    return statistics.mean(values) if values else float("nan")


def aggregate(runs: list[dict]) -> dict:
    scores = [r["score"] for r in runs]
    f1 = {c: mean(runs, lambda r, c=c: statistics.mean(n["f1"].get(c, 0.0) for n in r["niveis"].values()))
          for c in km.CLASSES}
    matrix = Counter()
    for r in runs:
        matrix.update(r.get("matriz", {}))
    return {
        "media": statistics.mean(scores),
        "desvio": statistics.pstdev(scores),
        "pior": min(scores),
        "tau_max": max((r["tau"] for r in runs if "tau" in r), default=float("nan")),
        "recall_spans": mean(runs, lambda r: r["recall_spans"]),
        "exact_spans": mean(runs, lambda r: r["exact_spans"]),
        "espurias_por_100": mean(runs, lambda r: r["espurias_por_100"]),
        "f1_medio": f1,
        "matriz_total": dict(sorted(matrix.items())),
        "documentos_com_erro": sum(len(r["erros"]) for r in runs),
        "invalidas": sum("invalida" in r for r in runs),
        "segundos_medio": statistics.mean(r["segundos"] for r in runs),
    }


# ------------------------------------------------------------------ relatório

def fmt(x: float, digits: int = 3) -> str:
    return "—" if x != x else f"{x:.{digits}f}"


def markdown(report: dict) -> str:
    meta, res, sols = report["meta"], report["resultados"], report["meta"]["solucoes"]
    lines = [
        "# Benchmark de robustez a ruído",
        "",
        f"Gerado em {meta['data']} (commit `{meta['commit']}`), {meta['sementes']} sementes por perfil "
        f"(base {meta['semente_base']}), {meta['documentos']} documentos e {meta['citacoes']} citações por conjunto.",
        f"Os textos são os de `{meta.get('fonte', 'data/txt')}` degradados por `tools.stress`; "
        "os rótulos do gabarito não mudam.",
        "Score = métrica oficial (`kaggle_metric.avaliar`, máx. 1,10). Spans = recall com IoU ≥ 0,5; "
        "exatos = início e fim idênticos; espúrias = predições sem par por 100 citações do gabarito.",
        "",
    ]

    def block(title: str, profiles: list[str], delta: bool) -> None:
        lines.extend([f"## {title}", ""])
        head = "| perfil | edição | citações alteradas |"
        sep = "|---|---:|---:|"
        for s in sols:
            head += f" {s} score{' (Δ limpo)' if delta else ''} | {s} spans / exatos | {s} τ máx | {s} espúrias |"
            sep += "---:|---:|---:|---:|"
        lines.extend([head, sep])
        for p in profiles:
            if p not in res:
                continue
            m = res[p]["manifesto"]
            row = f"| {p} | {m['taxa_edicao']:.1%} | {m['citacoes_alteradas']:.0%} |"
            for s in sols:
                a = res[p][s]
                sc = f"{a['media']:.3f} ± {a['desvio']:.3f}"
                if delta and "limpo" in res:
                    sc += f" ({a['media'] - res['limpo'][s]['media']:+.3f})"
                row += (f" {sc} | {fmt(a['recall_spans'])} / {fmt(a['exact_spans'])} | {fmt(a['tau_max'])} |"
                        f" {fmt(a['espurias_por_100'], 1)} |")
            lines.append(row)
        lines.append("")

    ablation = sorted((p for p in res if p.startswith("so_")),
                      key=lambda p: res[p][sols[0]]["media"])
    block("Severidade (todas as famílias juntas)", ["limpo"] + COMPOSITE, delta=False)
    block("Ablação (uma família por vez, intensidade \"pesado\"; da pior para a melhor)", ablation, delta=True)

    lines.extend(["## F1 médio por classe", "", "| perfil | " + " | ".join(
        f"{s} real / inventada / incompleta" for s in sols) + " |", "|---|" + "---:|" * len(sols)])
    for p in ["limpo"] + COMPOSITE + ablation:
        if p in res:
            lines.append(f"| {p} | " + " | ".join(
                " / ".join(fmt(res[p][s]["f1_medio"][c]) for c in km.CLASSES) for s in sols) + " |")
    lines.append("")

    for p in ("pesado", "extremo"):
        if p not in res:
            continue
        lines.extend([f"## Matriz gabarito → predição em `{p}` (soma das sementes)", ""])
        for s in sols:
            lines.append(f"**{s}**")
            lines.append("")
            lines.extend(f"- {k}: {v}" for k, v in res[p][s]["matriz_total"].items()
                         if not (k.split(" -> ")[0] == k.split(" -> ")[1]))
            lines.append("")

    lines.extend(["## Falhas de execução e tempo", "", "| solução | documentos com exceção | submissões inválidas | "
                  "tempo médio por conjunto (s) |", "|---|---:|---:|---:|"])
    for s in sols:
        errs = sum(res[p][s]["documentos_com_erro"] for p in res)
        inval = sum(res[p][s]["invalidas"] for p in res)
        secs = statistics.mean(res[p][s]["segundos_medio"] for p in res)
        lines.append(f"| {s} | {errs} | {inval} | {secs:.1f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    parser.add_argument("--solutions", nargs="+", default=list(SOLUTIONS), choices=list(SOLUTIONS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--regen", action="store_true", help="regera os conjuntos mesmo se existirem")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--source", type=Path, default=None, help="pasta com txt/ e goldenset.csv (padrão: dev)")
    parser.add_argument("--out", type=Path, default=OUT, help="onde ficam os conjuntos gerados")
    args = parser.parse_args()
    txt_dir, gold_csv = source_paths(args.source)

    solutions = [SOLUTIONS[name]() for name in args.solutions]
    report = {"meta": {
        "data": date.today().isoformat(),
        "commit": subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip(),
        "sementes": args.seeds, "semente_base": args.base_seed, "solucoes": args.solutions,
        "fonte": str(args.source or "data/txt"),
    }, "resultados": {}}
    start = time.monotonic()
    for profile in args.profiles:
        runs = {s.name: [] for s in solutions}
        manifests = []
        for seed in seeds_for(profile, args.base_seed, args.seeds):
            folder = split_dir(profile, seed, args.out)
            if args.regen or not (folder / "manifest.json").exists():
                generate(profile, seed, folder, txt_dir, gold_csv)
            manifests.append(json.loads((folder / "manifest.json").read_text(encoding="utf-8")))
            for s in solutions:
                runs[s.name].append(measure(s, folder))
        report["meta"]["documentos"] = manifests[0]["documentos"]
        report["meta"]["citacoes"] = manifests[0]["citacoes"]
        entry = {"manifesto": {
            "taxa_edicao": statistics.mean(m["taxa_edicao"] for m in manifests),
            "citacoes_alteradas": statistics.mean(m["citacoes_alteradas"] for m in manifests),
            "familias": manifests[0]["familias"],
        }}
        for name, rs in runs.items():
            entry[name] = aggregate(rs) | {"execucoes": rs}
        report["resultados"][profile] = entry
        print(f"{profile:<16} edição {entry['manifesto']['taxa_edicao']:>5.1%}  " + "  ".join(
            f"{n}={entry[n]['media']:.3f}±{entry[n]['desvio']:.3f} (spans {fmt(entry[n]['recall_spans'])}, "
            f"τ {fmt(entry[n]['tau_max'])})" for n in runs), flush=True)

    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.report / "benchmark.md").write_text(markdown(report), encoding="utf-8")
    print(f"\n{time.monotonic() - start:.0f}s -> {args.report}/benchmark.{{json,md}}")


if __name__ == "__main__":
    main()

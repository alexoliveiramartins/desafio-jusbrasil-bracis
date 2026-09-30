"""Benchmark: versão só-spans (regras) × versões com a camada de NLP, por modelo e conjunto.

    python -m tools.nlp_bench --sets dev v5_iter llm_llama_iter --models hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
    python -m tools.nlp_bench --sets ... --models A B --variants spans nlp --workers 2

Variantes:
  spans       só as regras (a solução de hoje);
  nlp_norm    regras + normalização pelo modelo das citações não resolvidas;
  nlp_recall  regras + citações que só o modelo achou;
  nlp         as duas.

As respostas do modelo ficam em cache (.cache/nlp/<modelo>), então todas as
variantes leem a mesma resposta e só a primeira passada custa GPU. O tempo por
documento reportado é o da primeira passada (cache frio), a {workers} por vez.
Um modelo por vez; o servidor descarrega o modelo ao trocar (`ollama stop`).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.classify import CanonicalIndex
from src.nlp import DEFAULT_URL, NLPLayer, OllamaClient

from src.main import process_file

from .evaluation import DB, load_gold, score, split_paths

VARIANTS = {"spans": None, "nlp_norm": (True, False), "nlp_recall": (False, True), "nlp": (True, True)}


def sample(gold: dict[str, list[dict]], docs: int | None) -> dict[str, list[dict]]:
    """Primeiros `docs` documentos, metade de cada nível (determinístico)."""
    if not docs:
        return gold
    by_level = {}
    for doc_id in sorted(gold):
        by_level.setdefault(gold[doc_id][0]["nivel"], []).append(doc_id)
    keep = {d for ids in by_level.values() for d in ids[:max(1, docs // len(by_level))]}
    return {d: rows for d, rows in gold.items() if d in keep}


def run(split, gold: dict, index: CanonicalIndex, nlp: NLPLayer | None = None) -> dict[str, dict]:
    return {d: process_file(split.txt / f"{d}.txt", index, debug=True, nlp=nlp) for d in gold}


def cache_dir(model: str) -> Path:
    return Path(".cache/nlp") / re.sub(r"[^\w.-]+", "_", model)


def warm(layer: NLPLayer, texts: list[str], workers: int) -> float:
    """Lê todos os documentos (enche o cache); devolve segundos por documento."""
    start = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(layer.read, texts))
    return (time.monotonic() - start) / max(1, len(texts))


def changes(outputs: dict[str, dict]) -> Counter:
    counts = Counter()
    for doc in outputs.values():
        for c in doc["citacoes"]:
            regra = c.get("_regra", "")
            counts["novas"] += regra.endswith("+novo")
            counts["normalizadas"] += regra in ("real_nlp", "inventada_nlp")
    return counts


def row(result: dict) -> dict:
    niveis = result["niveis"]
    return {"score": round(result["score_final"], 4)} | {
        f"N{level}": round(n["score"], 4) for level, n in sorted(niveis.items())} | {
        "tau": round(max(n["tau"] for n in niveis.values()), 4), "spans": round(result["recall_spans"], 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--docs", type=int, default=None, help="amostra balanceada por nível (seleção de modelo)")
    parser.add_argument("--second-pass", action="store_true", help="segunda passada nas frases com pistas não cobertas")
    parser.add_argument("--extra-models", nargs="*", default=[], help="modelos do conjunto (além de --models)")
    parser.add_argument("--fresh-cache", type=Path, default=None,
                        help="pasta nova e vazia para o cache: toda resposta vem ao vivo do modelo (sem reaproveitar "
                             "leituras antigas); as variantes da mesma execução compartilham essa pasta")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--report", type=Path, default=Path("relatorios/nlp_bench.json"))
    args = parser.parse_args()

    index = CanonicalIndex.from_sqlite(DB)
    report = json.loads(args.report.read_text()) if args.report.exists() else {}

    def save() -> None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    splits = {}
    for name in args.sets:
        split = split_paths(name)
        gold = sample(load_gold(split.gold), args.docs)
        texts = [(split.txt / f"{d}.txt").open(encoding="utf-8", newline="").read() for d in gold]
        splits[name] = (split, gold, texts)
        if "spans" in args.variants:
            report.setdefault(name, {})["spans"] = row(score(gold, run(split, gold, index)))
            print(f"{name:<22} {'spans':<48} {report[name]['spans']}", flush=True)
    save()
    if args.fresh_cache:
        if args.fresh_cache.exists() and any(args.fresh_cache.iterdir()):
            parser.error(f"--fresh-cache precisa ser uma pasta nova ou vazia: {args.fresh_cache}")

    def cache_for(model: str) -> Path:
        return args.fresh_cache / re.sub(r"[^\w.-]+", "_", model) if args.fresh_cache else cache_dir(model)

    for model in args.models if set(args.variants) - {"spans"} else []:  # um modelo por vez na GPU
        client = OllamaClient(args.url, model, cache_dir=cache_for(model))
        for name, (split, gold, texts) in splits.items():
            seconds = warm(NLPLayer(client), texts, args.workers)
            for variant in args.variants:
                if VARIANTS[variant] is None:
                    continue
                normalize, recall = VARIANTS[variant]
                extras = tuple(OllamaClient(args.url, m, cache_dir=cache_for(m)) for m in args.extra_models)
                layer = NLPLayer(client, normalize=normalize, recall=recall, second_pass=args.second_pass,
                                 extra_clients=extras)
                outputs = run(split, gold, index, layer)
                entry = row(score(gold, outputs)) | dict(changes(outputs)) | {"s_por_doc": round(seconds, 2)}
                label = variant + ("+2p" if args.second_pass else "") + "".join(f"+{m.split('/')[-1]}" for m in args.extra_models)
                report.setdefault(name, {})[f"{label}@{model}"] = entry
                print(f"{name:<22} {label + '@' + model.split('/')[-1]:<48} {entry}", flush=True)
            save()
        subprocess.run(["ollama", "stop", model], check=False, capture_output=True)


if __name__ == "__main__":
    main()

"""Compara LLMs locais como normalizadores de citação (sem treino), em dados de ITERAÇÃO.

    python -m tools.model_bench --models qwen2.5:7b-instruct-q4_K_M llama3.1:8b qwen3:8b

Tarefa: dado o trecho de uma citação `real` dos conjuntos de iteração, o
modelo devolve em JSON tipo, tribunal, número, diploma, artigo e ano. O
gabarito vem do registro da base apontado pelo rótulo (id_canonico), não de
texto escrito à mão. Métrica principal: os campos resolveriam o registro certo
(processo: número + tribunal; súmula: número + tribunal; artigo: diploma +
artigo; descritiva: tribunal + ano). Também: JSON válido e segundos por citação.

Um modelo por vez; o servidor descarrega o modelo entre execuções
(`ollama stop`). Holdouts nunca entram aqui.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.classify import CanonicalIndex

from .evaluation import DB, load_gold, split_paths
from .llm import ChatClient, LLMError

SYSTEM = "Você é um assistente jurídico brasileiro. Responda somente em JSON."
PROMPT = """Extraia os campos da citação jurídica abaixo (ela pode ter erros de OCR, abreviações ou apelidos).
Campos:
- "tipo": "processo", "sumula", "artigo" ou "julgado" (julgado citado sem número, por tribunal, ano e relator);
- "tribunal": "STF", "STJ", "TST", "TSE", "STM" ou null;
- "numero": só os dígitos do número do processo ou da súmula (corrija OCR como l→1, O→0, S→5), ou null;
- "diploma": "CF", "CC", "CPC", "CDC", "CPP", "CLT", "CE", "LC64/1990", "CPM", "OUTRA" ou null;
- "artigo": número do artigo, ou null;
- "ano": ano do julgado, ou null.
Citação: {trecho}"""


def build_items(splits: list[str], index: CanonicalIndex, per_split: int, seed: int) -> list[dict]:
    """Citações `real` com os campos esperados, tirados do registro da base."""
    dispositivo_of = {doc: key for key, doc in index.dispositivos.items()}
    sumula_of = {doc: key for key, doc in index.sumulas.items()}
    number_of = {doc: key for key, docs in index.by_short.items() for doc in docs}
    number_of |= {doc: key for key, docs in index.by_cnj.items() for doc in docs}
    rng = random.Random(seed)
    items = []
    for name in splits:
        rows = [r for rows in load_gold(split_paths(name).gold).values() for r in rows if r["classificacao"] == "real"]
        rng.shuffle(rows)
        for r in rows[:per_split]:
            doc = r["id_canonico"]
            record = index.records[doc]
            trecho = r["trecho"].replace("\\n", "\n")
            if doc in dispositivo_of:
                diploma, artigo = dispositivo_of[doc]
                expected = {"tipo": "artigo", "diploma": diploma, "artigo": str(artigo)}
            elif doc in sumula_of:
                tribunal, numero, _ = sumula_of[doc]
                expected = {"tipo": "sumula", "tribunal": tribunal, "numero": str(numero)}
            elif re.search(r"\d{4,}|\d\.\d{3}", re.sub(r"\b(?:19|20)\d\d\b", "", trecho)) and doc in number_of:
                key = number_of[doc]
                digits = str(key) if isinstance(key, int) else f"{key[0]}{key[1]}"
                expected = {"tipo": "processo", "tribunal": record.tribunal, "numero": digits}
            else:
                expected = {"tipo": "julgado", "tribunal": record.tribunal, "ano": str(record.ano)}
            items.append({"split": name, "trecho": trecho, "expected": expected})
    return items


def _norm_number(value) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits.lstrip("0")


def correct(answer: dict, expected: dict) -> bool:
    """Os campos do modelo resolveriam o registro certo?"""
    for field, want in expected.items():
        if field == "tipo":
            continue
        got = answer.get(field)
        if field in ("numero", "artigo", "ano"):
            if _norm_number(got) != _norm_number(want):
                return False
        elif str(got or "").upper() != str(want).upper():
            return False
    return True


def run_model(model: str, items: list[dict], url: str) -> dict:
    client = ChatClient(url, model, max_tokens=300, timeout_s=120, cache_dir=Path(".cache/llm-bench"))
    suffix = " /no_think" if model.startswith("qwen3") else ""  # Qwen3: sem modo de raciocínio (latência)
    ok, valid, seconds = Counter(), 0, []
    by_kind = defaultdict(lambda: [0, 0])
    for item in items:
        start = time.monotonic()
        try:
            raw = client.chat(SYSTEM, PROMPT.format(trecho=item["trecho"]) + suffix, json_mode=True)
        except LLMError as error:
            print(f"[llm] {model}: {error}")
            raw = ""
        seconds.append(time.monotonic() - start)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        try:
            answer = json.loads(raw)
            valid += isinstance(answer, dict)
        except json.JSONDecodeError:
            answer = {}
        hit = isinstance(answer, dict) and correct(answer, item["expected"])
        kind = item["expected"]["tipo"]
        by_kind[kind][0] += hit
        by_kind[kind][1] += 1
        ok[item["split"]] += hit
    return {
        "modelo": model,
        "acerto": sum(v[0] for v in by_kind.values()) / len(items),
        "por_tipo": {k: round(h / n, 3) for k, (h, n) in sorted(by_kind.items())},
        "json_valido": valid / len(items),
        "s_por_citacao": sum(seconds) / len(seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["llm_iter", "llm_adv_iter", "ineditos_v3_poluido"])
    parser.add_argument("--per-split", type=int, default=60)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--report", type=Path, default=Path("relatorios/model_bench.json"))
    args = parser.parse_args()

    index = CanonicalIndex.from_sqlite(DB)
    items = build_items([s for s in args.splits if split_paths(s).gold.exists()], index, args.per_split, args.seed)
    print(f"{len(items)} citações: {dict(Counter(i['expected']['tipo'] for i in items))}", flush=True)
    results = []
    for model in args.models:
        result = run_model(model, items, args.url)
        subprocess.run(["ollama", "stop", model], check=False, capture_output=True)  # um modelo por vez
        results.append(result)
        print(f"{model:<55} acerto {result['acerto']:.3f}  JSON {result['json_valido']:.2f}  "
              f"{result['s_por_citacao']:.2f}s/cit  {result['por_tipo']}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"itens": len(items), "resultados": results}, ensure_ascii=False, indent=2),
                           encoding="utf-8")


if __name__ == "__main__":
    main()

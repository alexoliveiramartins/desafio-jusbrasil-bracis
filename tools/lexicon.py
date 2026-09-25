"""Destila o vocabulário de LLMs locais em src/lexicon.json (usado pelo pipeline, sem GPU e sem treino).

    python -m tools.lexicon --models llama3.1:8b qwen3:8b qwen3.5:9b <gaia> --min-votes 2

Cada professor lista como advogados brasileiros se referem a cada tribunal,
diploma, classe e recurso coberto pela base. Só entram formas que passam em
filtros AUTOMÁTICOS (o léxico não é editado à mão):

  * consenso: propostas por pelo menos `--min-votes` professores diferentes
    para a mesma entidade (um professor sozinho alucina com consistência);
  * não ambíguas: forma proposta para entidades diferentes sai;
  * sem dígitos, sem inglês, e palavra única só se for sigla;
  * sem conflito com o que o normalizador já sabe ("agravo de instrumento" já
    é a classe AI; não pode virar Agravo Interno).

Os professores devem ser modelos DIFERENTES do autor dos holdouts
(tools.synth_llm usa Qwen2.5), para o holdout não medir a memória do autor.
Rejeições e votos vão para relatorios/lexicon.json.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from src.normalize import appeal_chain, class_code, diploma_key, fold
from src.spans import courts_in

from .llm import ChatClient, LLMError

# Entidades cobertas pela base (a cobertura é congelada): chave -> nome completo.
ENTITIES = {
    "tribunal": {
        "STF": "Supremo Tribunal Federal", "STJ": "Superior Tribunal de Justiça",
        "TST": "Tribunal Superior do Trabalho", "TSE": "Tribunal Superior Eleitoral",
        "STM": "Superior Tribunal Militar",
    },
    "diploma": {
        "CF": "Constituição Federal de 1988", "CC": "Código Civil (Lei 10.406/2002)",
        "CPC": "Código de Processo Civil (Lei 13.105/2015)", "CDC": "Código de Defesa do Consumidor",
        "CPP": "Código de Processo Penal", "CLT": "Consolidação das Leis do Trabalho",
        "CE": "Código Eleitoral (Lei 4.737/1965)", "LC64/1990": "Lei Complementar 64/1990 (Lei de Inelegibilidade)",
        "CPM": "Código Penal Militar",
    },
    "classe": {
        "REsp": "Recurso Especial", "AREsp": "Agravo em Recurso Especial", "RE": "Recurso Extraordinário",
        "ARE": "Recurso Extraordinário com Agravo", "Rcl": "Reclamação", "HC": "Habeas Corpus",
        "RHC": "Recurso Ordinário em Habeas Corpus", "RMS": "Recurso Ordinário em Mandado de Segurança",
        "AR": "Ação Rescisória", "RO": "Recurso Ordinário", "AI": "Agravo de Instrumento",
        "APL": "Apelação (criminal, Justiça Militar)", "RSE": "Recurso em Sentido Estrito",
        "REspe": "Recurso Especial Eleitoral", "RR": "Recurso de Revista",
        "AIRR": "Agravo de Instrumento em Recurso de Revista",
    },
    "recurso": {
        "AGINT": "Agravo Interno", "AGRG": "Agravo Regimental", "ED": "Embargos de Declaração",
        "EDV": "Embargos de Divergência",
    },
}
SYSTEM = "Você é um jurista brasileiro experiente e responde em JSON quando pedido."
LIST_PROMPT = """Liste de 8 a 20 formas diferentes como advogados e juízes brasileiros se referem, em peças
jurídicas, ao seguinte {what}: "{name}". Inclua siglas, abreviações usuais, apelidos e perífrases
(expressões que um jurista reconheceria como referência a ele). Não inclua números.
Responda em JSON: {{"formas": ["...", "..."]}}"""
ENGLISH = {"court", "code", "supreme", "law", "appeal", "the", "of", "and", "labor", "labour", "procedure",
           "constitution", "consumer", "electoral", "military", "special", "extraordinary", "review",
           "complaint", "writ", "federal court", "civil code", "action", "motion", "tribunal court"}
ARTICLES = {"o", "a", "os", "as", "do", "da", "de", "dos", "das", "e", "em", "no", "na"}


def propose(client: ChatClient, model: str) -> dict[str, dict[str, set[str]]]:
    """{tipo: {forma normalizada: {chaves propostas}}} e as formas originais de um professor."""
    suffix = " /no_think" if model.startswith("qwen3") else ""
    found: dict[str, dict[str, set[str]]] = {kind: defaultdict(set) for kind in ENTITIES}
    raw: dict[str, dict[str, str]] = {kind: {} for kind in ENTITIES}
    for kind, entities in ENTITIES.items():
        for key, name in entities.items():
            for temperature in (0.0, 0.7):  # determinística + uma amostra para diversidade
                try:
                    answer = client.chat(SYSTEM, LIST_PROMPT.format(what=kind, name=name) + suffix,
                                         json_mode=True, temperature=temperature)
                    forms = json.loads(re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)).get("formas", [])
                except (LLMError, json.JSONDecodeError, AttributeError):
                    forms = []
                for form in forms if isinstance(forms, list) else []:
                    if isinstance(form, str) and 2 <= len(form.strip()) <= 60:
                        norm = fold(form).strip(" .,;")
                        found[kind][norm].add(key)
                        raw[kind].setdefault(norm, form.strip())
    return found, raw


def reject_reason(kind: str, norm: str, raw: str, key: str) -> str | None:
    words = [w for w in norm.replace(".", " ").split() if w not in ARTICLES]
    if re.search(r"\d", norm):
        return "tem dígitos"
    if not words:
        return "vazia"
    if any(w in ENGLISH for w in words) or any(e in norm for e in ENGLISH if " " in e):
        return "inglês"
    if len(words) == 1 and not (re.fullmatch(r"[A-Z][A-Za-z.]{1,7}", raw) and sum(c.isupper() for c in raw) >= 2):
        return "palavra única que não é sigla"
    # Conflito com o que o normalizador já sabe.
    if kind == "diploma" and (known := diploma_key(norm)) and known != key:
        return f"normalizador já lê como {known}"
    if kind == "classe" and (known := class_code(norm)) and known != key:
        return f"normalizador já lê como {known}"
    if kind == "recurso":
        chain, known = appeal_chain(norm), class_code(norm)
        if (chain and key not in chain) or (known and not chain):
            return f"normalizador já lê como {chain or known}"
    if kind == "tribunal" and (known := courts_in(norm)) and known != [key]:
        return f"normalizador já lê como {known}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--models", nargs="+", required=True, help="professores (modelos diferentes do autor)")
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("src/lexicon.json"))
    parser.add_argument("--report", type=Path, default=Path("relatorios/lexicon.json"))
    args = parser.parse_args()

    votes: dict[str, dict[str, dict[str, set[str]]]] = {k: defaultdict(lambda: defaultdict(set)) for k in ENTITIES}
    raw_forms: dict[str, dict[str, str]] = {kind: {} for kind in ENTITIES}
    for model in args.models:
        client = ChatClient(args.url, model, max_tokens=800, timeout_s=300, cache_dir=Path(".cache/llm-lexicon"))
        found, raw = propose(client, model)
        for kind in ENTITIES:
            for norm, keys in found[kind].items():
                for key in keys:
                    votes[kind][norm][key].add(model)
                raw_forms[kind].setdefault(norm, raw[kind][norm])
        print(f"{model}: {sum(len(v) for v in found.values())} formas propostas", flush=True)
        try:
            import subprocess
            subprocess.run(["ollama", "stop", model], check=False, capture_output=True)  # um modelo por vez
        except OSError:
            pass

    lexicon: dict[str, dict[str, str]] = {kind: {} for kind in ENTITIES}
    report = []
    for kind, forms in votes.items():
        for norm, by_key in sorted(forms.items()):
            if len(by_key) > 1:
                reason = f"ambígua: {sorted(by_key)}"
            else:
                key, teachers = next(iter(by_key.items()))
                reason = (f"só {len(teachers)} voto(s)" if len(teachers) < args.min_votes
                          else reject_reason(kind, norm, raw_forms[kind][norm], key))
            report.append({"tipo": kind, "forma": norm, "votos": {k: sorted(v) for k, v in by_key.items()},
                           "aceita": reason is None, "motivo": reason})
            if reason is None:
                lexicon[kind][norm] = key
    args.out.write_text(json.dumps({
        "_fonte": f"consenso de {args.min_votes}+ entre {', '.join(args.models)} (tools/lexicon.py; não editar à mão)",
        **lexicon,
    }, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    accepted = sum(len(v) for v in lexicon.values())
    print(f"léxico: {accepted} formas aceitas de {len(report)} -> {args.out}")


if __name__ == "__main__":
    main()

"""Conjuntos escritos por um LLM local: a SUPERFÍCIE das citações não é minha.

    python -m tools.synth_llm --split llm_holdout --docs 40 --seed 5000
    python -m tools.synth_llm --split llm_iter --docs 40 --seed 1000

Os catálogos v1–v4 foram escritos por quem escreveu o pipeline, então medem
generalização de forma otimista. Aqui o rótulo continua vindo da base (mesmas
garantias de tools/synth/sources.py), mas quem decide COMO cada citação é
escrita é o modelo (Qwen2.5-7B-Instruct via Ollama, temperatura > 0, semente
fixa por documento):

  1. sorteio uma matéria (cível, penal, trabalhista, eleitoral, militar) e
     fatos rotulados coerentes com ela (processo real/inventado, súmula,
     artigo, julgado descritivo);
  2. o modelo escreve cada fato como citação, no estilo dele (1ª chamada, JSON);
     valido sem usar o pipeline: dígitos intactos, tribunal/UF/diploma
     coerentes; descritivas são rerrotuladas pelo nome que o modelo escreveu;
  3. o modelo redige a peça com marcadores {C1}, {C2}… no lugar das citações
     (2ª chamada) e eu substituo: o span do gabarito é exato. Peça com
     marcador faltando/repetido ou com outra citação no corpo é descartada.

`llm_iter` é para diagnosticar; `llm_holdout` (outra semente) só se mede.

Modo adversarial (`--mode adversarial`, conjuntos `llm_adv_*`): o modelo é
instruído a escrever as citações de um jeito DIFÍCIL para um programa
(apelidos e perífrases de tribunais e leis, abreviações raras, ordem incomum),
mantendo os dígitos. Sem evidência explícita do tribunal/diploma, uma segunda
chamada (verificador, temperatura 0, só com a citação) diz a que tribunal ou
lei ela se refere; a citação só entra se bater com o fato.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

from src.normalize import fold, name_tokens

from .llm import ChatClient, LLMError
from .synth.noise import ocr_noise, pollute_citation
from .synth.sources import COURT_LONG, SUMULA_REAL, Sources, clean_relator, perturb_number

MATTERS = {
    "cível": {"courts": {"STJ": {"REsp", "AREsp", "RMS", "AR"}, "STF": {"RE", "ARE", "Rcl"}},
              "diplomas": ["CC", "CPC", "CDC", "CF"], "sumulas": ["STJ", "STF"]},
    "penal": {"courts": {"STJ": {"HC", "RHC", "REsp", "AREsp"}, "STF": {"HC", "Rcl"}},
              "diplomas": ["CPP", "CF"], "sumulas": ["STJ", "STF"]},
    "trabalhista": {"courts": {"TST": {"RR", "AIRR", "ARR"}}, "diplomas": ["CLT", "CF"], "sumulas": ["TST"]},
    "eleitoral": {"courts": {"TSE": {"REspe", "AI", "RO", "AR", "Rcl"}}, "diplomas": ["CE", "LC64/1990", "CF"],
                  "sumulas": ["TSE"]},
    "militar": {"courts": {"STM": {"APL", "RSE", "HC", "AgInt", "Rcl"}}, "diplomas": ["CPM", "CF"], "sumulas": []},
}
DOC_TYPES = ["petição de recurso", "contestação", "parecer do Ministério Público", "decisão monocrática",
             "memorial", "contrarrazões", "voto de relator"]
CLASS_NAMES = {
    "REsp": "Recurso Especial", "AREsp": "Agravo em Recurso Especial", "RMS": "Recurso em Mandado de Segurança",
    "AR": "Ação Rescisória", "RE": "Recurso Extraordinário", "ARE": "Recurso Extraordinário com Agravo",
    "Rcl": "Reclamação", "HC": "Habeas Corpus", "RHC": "Recurso em Habeas Corpus", "RR": "Recurso de Revista",
    "AIRR": "Agravo de Instrumento em Recurso de Revista", "ARR": "Recurso de Revista com Agravo",
    "REspe": "Recurso Especial Eleitoral", "AI": "Agravo de Instrumento", "RO": "Recurso Ordinário",
    "APL": "Apelação", "RSE": "Recurso em Sentido Estrito", "AgInt": "Agravo Interno",
}
CHAIN_NAMES = {"AGINT": "Agravo Interno", "AGRG": "Agravo Regimental", "ED": "Embargos de Declaração",
               "EDV": "Embargos de Divergência", "E": "Embargos", "AG": "Agravo", "PEXT": "Pedido de Extensão"}
DIPLOMA_NAMES = {"CF": "da Constituição Federal", "CC": "do Código Civil", "CPC": "do Código de Processo Civil",
                 "CDC": "do Código de Defesa do Consumidor", "CPP": "do Código de Processo Penal",
                 "CLT": "da Consolidação das Leis do Trabalho", "CE": "do Código Eleitoral",
                 "LC64/1990": "da Lei Complementar nº 64/1990", "CPM": "do Código Penal Militar"}
# Validação do diploma escrito pelo modelo (lista própria desta ferramenta, não do pipeline).
DIPLOMA_EVIDENCE = {
    "CF": ["constitui", "cf", "crfb", "carta"], "CC": ["codigo civil", "cc", "10.406", "10406"],
    "CPC": ["processo civil", "cpc", "13.105", "13105"], "CDC": ["consumidor", "cdc", "8.078", "8078"],
    "CPP": ["processo penal", "cpp", "3.689", "3689"], "CLT": ["clt", "consolidacao", "5.452", "5452"],
    "CE": ["codigo eleitoral", "4.737", "4737"], "LC64/1990": ["64/", "64,", "inelegibilidade", "complementar"],
    "CPM": ["penal militar", "cpm", "1.001", "1001"],
}
COURT_EVIDENCE = {sigla: [sigla.lower(), fold(name)] for sigla, name in COURT_LONG.items()}
# A classe filtra o tribunal só em número curto (STJ/STF): exige evidência da classe certa.
CLASS_EVIDENCE = {
    "REsp": ["resp", "esp", "especial"], "AREsp": ["aresp", "agravo em", "ag. em", "ag em"],
    "RHC": ["rhc", "habeas"], "HC": ["hc", "habeas"], "RMS": ["rms", "mandado de seguranca", "ms"],
    "AR": ["ar", "rescis"], "RE": ["re", "extraordin"], "ARE": ["are", "extraordin"],
    "Rcl": ["rcl", "recl", "reclama"], "RO": ["ro", "ordinario"], "AI": ["ai", "instrumento"],
    "AgInt": ["agint", "agravo interno", "ag. int"], "SLS": ["sls", "suspensao"],
}
FIELD_COPY = re.compile(r"(?<!\w)(?:tribunal|classe|numero|ano|recursos(?: na cadeia)?|uf)\s*:")
COURTS = ["STF", "STJ", "TST", "TSE", "STM"]
UFS = set("AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO".split())

SYSTEM = ("Você é advogado(a) e redator(a) jurídico(a) brasileiro(a). Escreve em português formal, "
          "como se faz na prática forense brasileira.")
CITE_PROMPT = """Escreva como cada referência abaixo apareceria citada numa peça jurídica brasileira.
Use o estilo que achar natural e VARIE entre elas: siglas (REsp, AgInt, HC...), abreviações
(Rec. Esp., Ag. Reg....) ou nomes por extenso; com ou sem tribunal e UF.
Regras: copie os números com os mesmos dígitos (número curto pode ter pontos de milhar);
não troque tribunal, UF, ano, relator nem diploma; escreva só a citação, sem relator nem data,
exceto nas referências sem número, que se identificam por tribunal, ano e relator.
Responda em JSON: {{"citacoes": ["<citação 1>", "<citação 2>", ...]}}, na mesma ordem.

Referências:
{items}"""
CITE_PROMPT_ADV = """Escreva cada referência abaixo como ela poderia aparecer numa peça jurídica brasileira,
de um jeito DIFÍCIL para um programa de computador reconhecer, mas claro para um jurista experiente:
use apelidos ou perífrases do tribunal e da lei, abreviações pouco usuais, ordem incomum dos elementos,
e embuta tribunal, ano e relator na própria expressão quando fizer sentido. Varie entre as referências.
Regras: copie os números com os mesmos dígitos (número curto pode ter pontos de milhar); não troque o
tribunal, a UF, o ano, o relator nem a lei (mude só a forma de escrever); nas referências sem número,
não acrescente número de processo nem classe processual.
Responda em JSON: {{"citacoes": ["<citação 1>", "<citação 2>", ...]}}, na mesma ordem.

Referências:
{items}"""
VERIFY_PROMPT = """A citação jurídica abaixo se refere a qual {what}? Responda só com uma destas opções: {options}.

Citação: {span}"""
BODY_PROMPT = """Redija um(a) {doc_type} de matéria {matter}, com 250 a 450 palavras, com cabeçalho e fecho.
Na argumentação, use os marcadores abaixo, cada um exatamente uma vez, no lugar de uma citação
(o marcador será substituído pela citação indicada ao lado; escreva a frase de modo que ela faça sentido):
{items}
Não escreva nenhuma outra citação, número de processo, artigo de lei ou súmula além dos marcadores.
Responda só com o texto da peça."""


def _fmt_cnj(key) -> str:
    seq, rest = key
    return f"{seq:07d}-{rest[:2]}.{rest[2:6]}.{rest[6]}.{rest[7:9]}.{rest[9:]}"


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


class Fact:
    """Um fato rotulado: o que o modelo deve citar e como validar o que ele escreveu."""

    def __init__(self, kind: str, label: str, doc_id: str, prompt: str, tipo: str, **check):
        self.kind, self.label, self.doc_id, self.prompt, self.tipo, self.check = kind, label, doc_id, prompt, tipo, check


def _process_fact(rng: random.Random, src: Sources, court: str, classes: set[str], invent: bool) -> Fact | None:
    ids = [d for d in src.unique_short + src.unique_cnj
           if src.index.records[d].tribunal == court and src.index.records[d].classe in classes]
    if not ids:
        return None
    doc_id = rng.choice(ids)
    record = src.index.records[doc_id]
    key = src.key_of(doc_id)
    if invent and (key := perturb_number(rng, src, key)) is None:
        return None
    number = _fmt_cnj(key) if isinstance(key, tuple) else str(key)
    digits = _digits(number) if isinstance(key, tuple) else str(key)
    chain = [CHAIN_NAMES[c] for c in dict.fromkeys(record.chain) if c in CHAIN_NAMES]
    parts = [f"tribunal: {court}", f"classe: {CLASS_NAMES.get(record.classe, record.classe)}"]
    if chain:
        parts.append("recursos na cadeia: " + ", ".join(chain))
    parts.append(f"número: {number}")
    if record.uf:
        parts.append(f"UF: {record.uf}")
    return Fact("processo", "inventada" if invent else "real", "" if invent else doc_id,
                "processo — " + "; ".join(parts), "jurisprudencia", digits=digits, court=court, uf=record.uf,
                classe=record.classe, cnj=isinstance(key, tuple))


def _sumula_fact(rng: random.Random, src: Sources, court: str, invent: bool) -> Fact:
    real = [(t, n, v) for t, n, v in SUMULA_REAL if t == court]
    if real and not invent:
        trib, num, vinc = rng.choice(real)
        doc_id = src.index.sumulas[(trib, num, vinc)]
    else:
        trib, vinc = court, court == "STF" and rng.random() < 0.4
        existing = {n for (t, n, v) in src.index.sumulas if t == trib and v == vinc}
        num = rng.choice([n for n in range(1, 700) if n not in existing])
        doc_id = ""
    what = f"Súmula Vinculante {num} do STF" if vinc else f"súmula nº {num} do {trib}"
    return Fact("sumula", "real" if doc_id else "inventada", doc_id, f"súmula — {what}", "jurisprudencia",
                digits=str(num), court=None if vinc else trib)


def _article_fact(rng: random.Random, src: Sources, diploma: str, invent: bool) -> Fact:
    real = [(d, n) for d, n in src.index.dispositivos if d == diploma]
    if real and not invent:
        _, num = rng.choice(real)
        doc_id = src.index.dispositivos[(diploma, num)]
    else:
        existing = {n for d, n in real}
        num = rng.choice([n for n in range(2, 300) if n not in existing])
        doc_id = ""
    return Fact("artigo", "real" if doc_id else "inventada", doc_id,
                f"artigo — art. {num} {DIPLOMA_NAMES[diploma]}", "lei", digits=str(num), diploma=diploma)


def _descriptive_fact(rng: random.Random, src: Sources, court: str) -> Fact | None:
    groups = [(g, records) for g, records in src.desc_groups if g[0] == court]
    if not groups:
        return None
    (tribunal, ano, relator), _ = rng.choice(groups)
    name = clean_relator(relator)
    return Fact("descritiva", "", "", f"julgado sem número — tribunal: {tribunal}; ano: {ano}; relator(a): {name}",
                "jurisprudencia", court=tribunal, ano=ano, relator=name)


def draw_facts(rng: random.Random, src: Sources, matter: str) -> list[Fact]:
    cfg = MATTERS[matter]
    facts: list[Fact] = []
    n = rng.randint(5, 8)
    while len(facts) < n:
        court, classes = rng.choice(sorted(cfg["courts"].items()))
        r = rng.random()
        if r < 0.45:
            fact = _process_fact(rng, src, court, classes, invent=rng.random() < 0.45)
        elif r < 0.65:
            fact = _article_fact(rng, src, rng.choice(cfg["diplomas"]), invent=rng.random() < 0.45)
        elif r < 0.8 and cfg["sumulas"]:
            fact = _sumula_fact(rng, src, rng.choice(cfg["sumulas"]), invent=rng.random() < 0.5)
        else:
            fact = _descriptive_fact(rng, src, court)
        if fact and all(f.prompt != fact.prompt for f in facts):
            facts.append(fact)
    return facts


# Citação/número fora dos marcadores: a peça teria citação sem rótulo.
BODY_CITATION = re.compile(r"(?<!\w)(?:arts?\.|artigos?|sumulas?|leis? n|decretos?|resp|rcl|hc|agint|agrg"
                           r"|rel\.|relator|relatora|relatoria|min\.|ministr[oa])(?!\w)|\d\.\d{3}|\d{4,}")


def surfaces(raw: str, facts: list[Fact]) -> list[str | None]:
    """Citações escritas pelo modelo (1ª chamada), uma por fato, na ordem."""
    import json
    try:
        items = json.loads(raw).get("citacoes", [])
    except (json.JSONDecodeError, AttributeError):
        return [None] * len(facts)
    items = [_unwrap(i.strip().rstrip(".;,:").strip()) if isinstance(i, str) else None for i in items]
    return (items + [None] * len(facts))[:len(facts)]


WRAPPERS = re.compile(r"^(?:<(.*)>|\*\*(.*)\*\*|\"(.*)\"|“(.*)”|«(.*)»)$", re.DOTALL)


def _unwrap(item: str) -> str:
    """Tira delimitadores que o modelo copia do exemplo do prompt ("<citação 1>") ou de markdown."""
    while m := WRAPPERS.match(item):
        item = next(g for g in m.groups() if g is not None).strip().rstrip(".;,:").strip()
    return item


MARKER = re.compile(r"\[\[C(\d+)\]\]|\{C(\d+)\}|\[C(\d+)\]")


def assemble(body: str, cites: list[tuple[str, Fact, str, str]]):
    """(texto, [(início, fim, fato, rótulo, id)], motivo) — texto None se a peça não serve.

    Cada citação entra onde o modelo pôs o marcador {Cn} ou onde ele copiou a
    própria citação; todas as ocorrências são anotadas. Parágrafo com número
    ou citação fora dos spans (citação sem rótulo) é removido da peça.
    """
    def substitute(m: re.Match) -> str:
        n = int(next(g for g in m.groups() if g))
        if not 1 <= n <= len(cites):
            return m.group()
        before = " " if m.start() and body[m.start() - 1].isalnum() else ""  # "precedente{C4}"
        after = " " if m.end() < len(body) and body[m.end()].isalnum() else ""
        return before + cites[n - 1][0] + after

    text = MARKER.sub(substitute, body).strip()
    kept, rows, cursor = [], [], 0
    for para in text.split("\n"):
        spans = []
        for surface, fact, label, doc_id in cites:
            for m in re.finditer(re.escape(surface), para):
                if not any(s < m.end() and m.start() < e for s, e, *_ in spans):
                    spans.append((m.start(), m.end(), fact, label, doc_id))
        outside = "".join(para[a:b] for a, b in _gaps(spans, len(para)))
        if BODY_CITATION.search(fold(re.sub(r"\b(?:19|20)\d\d\b", "", outside))):
            continue  # citação ou número sem rótulo: o parágrafo sai
        rows += [(cursor + s, cursor + e, f, l, d) for s, e, f, l, d in sorted(spans, key=lambda r: r[0])]
        kept.append(para)
        cursor += len(para) + 1
    if len({id(r[2]) for r in rows}) < 3:
        return None, [], "menos de 3 citações no corpo"
    return "\n".join(kept), rows, ""


def _gaps(spans, length):
    pos = 0
    for s, e, *_ in sorted(spans, key=lambda r: r[0]):
        yield pos, s
        pos = e
    yield pos, length


def _has(low: str, evidence: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(e)}" + (r"(?!\w)" if len(e) <= 3 else ""), low) for e in evidence)


CLASS_WORDS = re.compile(r"(?<!\w)(?:resp|aresp|respe|rcl|recl|hc|rhc|rms|are|apl|rse|rr|airr|arr|agint|agrg)(?!\w)|"
                         + "|".join(fold(n) for n in CLASS_NAMES.values()))


# Classes cuja evidência contradiz o tribunal do registro (modo adversarial só rejeita contradição).
CLASS_COURT = {"REsp": "STJ", "AREsp": "STJ", "RE": "STF", "ARE": "STF", "Rcl": None, "HC": None, "RHC": None,
               "RMS": None, "AR": None, "RO": None, "AI": None, "AgInt": None, "SLS": "STJ"}


def _verified(span: str, what: str, options: list[str], expected: str, verify) -> bool:
    """Evidência explícita decide; sem ela, o verificador (LLM) precisa concordar com o fato."""
    if verify is None:
        return False
    return verify(span, what, options) == expected


def validate(span: str, fact: Fact, src: Sources, verify=None) -> tuple[str, str] | None:
    """(rótulo, id) se o que o modelo escreveu ainda corresponde ao fato; None se não.

    `verify` (modo adversarial) é chamado quando falta evidência explícita de
    tribunal/diploma: apelidos e perífrases passam só se o verificador concordar.
    """
    low = fold(span)
    if len(span) > (140 if fact.kind == "descritiva" else 90):
        return None  # forma longa demais para ser só a citação
    if FIELD_COPY.search(low):
        return None  # o modelo copiou os campos em vez de escrever uma citação
    if fact.kind == "processo":
        if fact.check["digits"] not in _digits(span):
            return None
        classe = fact.check["classe"]
        if not fact.check["cnj"] and classe in CLASS_EVIDENCE and not _has(low, CLASS_EVIDENCE[classe]):
            # Adversarial: aceita classe escrita de outro jeito, desde que não contradiga o tribunal.
            if verify is None or any(_has(low, CLASS_EVIDENCE[c]) for c, court in CLASS_COURT.items()
                                     if court and court != fact.check["court"]):
                return None
        courts = [c for c, ev in COURT_EVIDENCE.items() if _has(low, ev)]
        if courts and courts != [fact.check["court"]]:
            return None
        if not courts and verify is not None and verify(span, "tribunal", COURTS + ["NENHUM"]) not in (
                fact.check["court"], "NENHUM", None):
            return None
        ufs = set(re.findall(r"(?:[/(–-]\s*|,\s*)([A-Z]{2})\b(?!\s*[-–]\s*\d)", span)) & UFS
        if ufs and ufs != {fact.check["uf"]}:
            return None
        return fact.label, fact.doc_id
    if fact.kind == "sumula":
        if re.findall(r"\d+", span) != [fact.check["digits"]]:
            return None
        court = fact.check["court"]
        if court and not _has(low, COURT_EVIDENCE[court]) and not _verified(span, "tribunal", COURTS, court, verify):
            return None
        return fact.label, fact.doc_id
    if fact.kind == "artigo":
        numbers = re.findall(r"\d+", span.replace(".", ""))
        if not numbers or numbers[0] != fact.check["digits"]:
            return None
        diploma = fact.check["diploma"]
        if not _has(low, DIPLOMA_EVIDENCE[diploma]):
            others = [d for d, ev in DIPLOMA_EVIDENCE.items() if d != diploma and _has(low, ev)]
            if others or not _verified(span, "lei ou código", list(DIPLOMA_EVIDENCE) + ["OUTRA"], diploma, verify):
                return None
        return fact.label, fact.doc_id
    # Descritiva: sem número de processo; tribunal e ano presentes; rótulo pelo nome escrito.
    if re.search(r"\d{5,}|\d\.\d{3}", span) or str(fact.check["ano"]) not in span or CLASS_WORDS.search(low):
        return None
    court = fact.check["court"]
    if not _has(low, COURT_EVIDENCE[court]) and not _verified(span, "tribunal", COURTS, court, verify):
        return None
    cited = [t for t in name_tokens(fact.check["relator"]) if t in name_tokens(span)]
    if not cited:
        return None
    matches = src.same_relator(fact.check["court"], fact.check["ano"], " ".join(cited))
    if not matches:
        return None
    return ("real", matches[0].doc_id) if len(matches) == 1 else ("incompleta", "")


def noisy_document(rng: random.Random, text: str, rows: list, level: int):
    """Nível 2: OCR e poluição (v1) dentro das citações, com os spans recalculados."""
    if level == 1:
        return text, rows
    out, new_rows, pos = [], [], 0
    for start, end, fact, label, doc_id in sorted(rows, key=lambda r: r[0]):
        out.append(text[pos:start])
        cite = pollute_citation(rng, ocr_noise(rng, text[start:end], 0.08), 0.3)
        begin = sum(len(p) for p in out)
        out.append(cite)
        new_rows.append((begin, begin + len(cite), fact, label, doc_id))
        pos = end
    out.append(text[pos:])
    return "".join(out), new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True)
    parser.add_argument("--docs", type=int, default=40)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/desafio1_bracis.db"))
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--mode", choices=["normal", "adversarial"], default="normal")
    args = parser.parse_args()

    src = Sources(args.db)
    client = ChatClient(args.url, args.model, max_tokens=1400, timeout_s=300, cache_dir=Path(".cache/llm-synth"))
    dest = args.out / args.split
    (dest / "txt").mkdir(parents=True, exist_ok=True)
    for old in (dest / "txt").glob("*.txt"):
        old.unlink()
    adversarial = args.mode == "adversarial"

    def verify(span: str, what: str, options: list[str]) -> str | None:
        answer = client.chat(SYSTEM, VERIFY_PROMPT.format(what=what, options=", ".join(options), span=span))
        found = [o for o in options if re.search(rf"(?<![\w/]){re.escape(o)}(?![\w/])", answer)]
        return found[0] if len(found) == 1 else None

    rows_out, made, attempts, dropped = [], 0, 0, Counter()
    while made < args.docs and attempts < args.docs * 6:
        attempts += 1
        # Semente por tentativa: mudar a validação não muda os prompts seguintes (cache estável).
        rng = random.Random(f"{args.split}-{args.seed}-{attempts}")
        matter = rng.choice(sorted(MATTERS))
        facts = draw_facts(rng, src, matter)
        items = "\n".join(f"{i}. {f.prompt}" for i, f in enumerate(facts, 1))
        if adversarial:  # sem o rótulo "julgado sem número", que o modelo copiava
            items = items.replace("julgado sem número — ", "precedente citado sem número de processo — ")
        client.seed = rng.randint(1, 10**6)
        try:
            raw = client.chat(SYSTEM, (CITE_PROMPT_ADV if adversarial else CITE_PROMPT).format(items=items),
                              json_mode=True, temperature=0.8)
            cites = []
            for surface, fact in zip(surfaces(raw, facts), facts):
                label = validate(surface, fact, src, verify if adversarial else None) if surface else None
                if label:
                    cites.append((surface, fact, *label))
            dropped["citação"] += len(facts) - len(cites)
            if len(cites) < 3:
                dropped["menos de 3 citações válidas"] += 1
                continue
            markers = "\n".join(f"{{C{i}}} = {c[0]}" for i, c in enumerate(cites, 1))
            body = client.chat(SYSTEM, BODY_PROMPT.format(doc_type=rng.choice(DOC_TYPES), matter=matter,
                                                          items=markers), temperature=0.8)
        except LLMError as error:
            print(f"[llm] {error}")
            continue
        text, rows, reason = assemble(body, cites)
        if text is None:
            dropped[reason] += 1
            continue
        made += 1
        level = 1 if made <= args.docs // 2 else 2
        text, rows = noisy_document(rng, text, rows, level)
        doc_id = f"{args.split}_n{level}_{made:03d}"
        (dest / "txt" / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="")
        for j, (start, end, fact, label, cid) in enumerate(rows, 1):
            rows_out.append({"nivel": level, "documento_id": doc_id, "citacao_id": f"g{j}", "inicio": start,
                             "fim": end, "trecho": text[start:end].replace("\n", "\\n"), "tipo": fact.tipo,
                             "classificacao": label, "id_canonico": cid})
        print(f"[{made}/{args.docs}] {doc_id}: {len(rows)} citações de {len(facts)} fatos ({matter})", flush=True)
    with (dest / "goldenset.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    counts = Counter(r["classificacao"] for r in rows_out)
    print(f"{args.split}: {made} documentos, {len(rows_out)} citações {dict(counts)}; descartes {dict(dropped)}")


if __name__ == "__main__":
    main()

"""Catálogo v3: conjunto de ITERAÇÃO para formatos inéditos.

Gramática ampla (siglas, nomes por extenso, abreviações usuais em peças,
marcadores de número, honoríficos, rabos de citação, ordens livres) com
vocabulário próprio, diferente dos holdouts v2 e v4. Diagnostique aqui; meça
generalização nos holdouts.
"""

from __future__ import annotations

import random

from .sources import COURT_LONG, SUMULA_REAL, Catalog, Cite, Sources, clean_relator, fmt_thousands, perturb_number

# Abreviações usuais em peças (não truncamento aleatório: isso é o v2).
USUAL_ABBREV = {
    "Recurso": ["Rec.", "Rec"], "Especial": ["Esp.", "Especial"], "Extraordinário": ["Extr.", "Extraord."],
    "Ordinário": ["Ord.", "Ordin."], "Agravo": ["Agr.", "Ag."], "Regimental": ["Reg.", "Regim."],
    "Interno": ["Int.", "Interno"], "Embargos": ["Emb.", "Embs."], "Declaração": ["Decl.", "Declar."],
    "Divergência": ["Div.", "Diverg."], "Instrumento": ["Instr.", "Inst."], "Reclamação": ["Recl.", "Reclam."],
    "Apelação": ["Apel.", "Ap."], "Criminal": ["Crim.", "Criminal"], "Mandado": ["Mand.", "MS"],
    "Segurança": ["Seg.", "Segurança"], "Rescisória": ["Resc.", "Rescis."], "Eleitoral": ["Eleit.", "Eleitoral"],
    "Habeas": ["Hab.", "Habeas"], "Corpus": ["Corp.", "Corpus"], "Sentido": ["Sent.", "Sentido"],
    "Estrito": ["Estr.", "Estrito"], "Ação": ["Ação", "Ac."], "Pedido": ["Ped.", "Pedido"],
    "Extensão": ["Ext.", "Extensão"],
}


def usual_abbrev(rng: random.Random, phrase: str) -> str:
    out = []
    for word in phrase.split():
        options = USUAL_ABBREV.get(word)
        out.append(rng.choice(options) if options and rng.random() < 0.6 else word)
    return " ".join(out)


def recase(rng: random.Random, text: str) -> str:
    r = rng.random()
    return text.upper() if r < 0.1 else text.lower() if r < 0.15 else text


# ------------------------------------------------------------------ processos

V3_CLASS = {
    "REsp": ("Recurso Especial", ["REsp", "RESP", "Resp", "R. Esp."]),
    "AREsp": ("Agravo em Recurso Especial", ["AREsp", "ARESP", "A. REsp"]),
    "RHC": ("Recurso em Habeas Corpus", ["RHC", "RHC"]),
    "RMS": ("Recurso em Mandado de Segurança", ["RMS", "RMS"]),
    "HC": ("Habeas Corpus", ["HC", "HC"]),
    "Rcl": ("Reclamação", ["Rcl", "Rcl.", "RCL"]),
    "RE": ("Recurso Extraordinário", ["RE", "RE"]),
    "ARE": ("Recurso Extraordinário com Agravo", ["ARE", "ARE"]),
    "REspe": ("Recurso Especial Eleitoral", ["REspe", "RESPE", "REsp Eleitoral"]),
    "RO": ("Recurso Ordinário", ["RO", "RO"]),
    "AI": ("Agravo de Instrumento", ["AI", "AI"]),
    "APL": ("Apelação Criminal", ["APL", "Ap. Crim."]),
    "RSE": ("Recurso em Sentido Estrito", ["RSE", "RSE"]),
    "AR": ("Ação Rescisória", ["AR", "AR"]),
    "AgInt": ("Agravo Interno", ["AgInt", "AgInt"]),
}
V3_CHAIN = {
    "AGINT": ("Agravo Interno", ["AgInt", "Ag. Int.", "AgInt."]),
    "AGRG": ("Agravo Regimental", ["AgRg", "AgR", "Ag. Rg."]),
    "ED": ("Embargos de Declaração", ["EDcl", "ED", "EDs"]),
    "EDV": ("Embargos de Divergência", ["EDv", "EDv."]),
    "PEXT": ("Pedido de Extensão", ["PExt", "PExt."]),
}
FEMININE = {"Rcl", "APL", "AR"}
V3_MARKERS = ["nº ", "n. ", "n.º ", "Nº ", "n° ", "No ", "Nr. ", "nro ", "núm. ", "número ", "sob nº ",
              "de número ", "", ""]
V3_COURTS = {
    "STJ": ["STJ", "STJ", "Eg. STJ", "Colendo STJ", COURT_LONG["STJ"], "Superior Tribunal de Justiça (STJ)"],
    "STF": ["STF", "STF", "Excelso Pretório", "C. STF", COURT_LONG["STF"], "Supremo Tribunal Federal (STF)"],
    "TST": ["TST", "TST", "Eg. TST", COURT_LONG["TST"]],
    "TSE": ["TSE", "TSE", "Col. TSE", COURT_LONG["TSE"]],
    "STM": ["STM", "STM", "E. STM", COURT_LONG["STM"]],
}
ORGAOS = ["T1", "T2", "T3", "T4", "T5", "T6", "Segunda Turma", "Quarta Turma", "Corte Especial", "Plenário",
          "Segunda Seção", "Terceira Seção", "Pleno"]


def _date(rng: random.Random, year: int | None = None) -> str:
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), year or rng.randint(2010, 2025)
    return rng.choice([f"{d:02d}.{m:02d}.{y}", f"{d:02d}-{m:02d}-{y}", f"{d}/{m}/{y}"])


def _element(rng: random.Random, full: str, acronyms: list[str]) -> str:
    r = rng.random()
    if r < 0.5:
        return rng.choice(acronyms)
    if r < 0.7:
        return full
    return usual_abbrev(rng, full)


def _number(rng: random.Random, key) -> str:
    if isinstance(key, tuple):
        seq, rest = key
        dv, ano, j, tr, orig = rest[:2], rest[2:6], rest[6], rest[7:9], rest[9:]
        return rng.choice([
            f"{seq:07d}-{dv}.{ano}.{j}.{tr}.{orig}", f"{seq:07d}-{dv}.{ano}.{j}.{tr}.{orig}",
            f"{seq}-{dv}.{ano}.{j}.{tr}.{orig}", f"{seq:07d}-{dv}.{ano}.{j}.{tr}.{orig}".replace("-", " - "),
            f"{seq:07d}-{dv}.{ano}.{j}.{tr}.\n{orig}", f"{seq:07d}{dv}{ano}{j}{tr}{orig}",
        ])
    return rng.choice([fmt_thousands(key), fmt_thousands(key), str(key), fmt_thousands(key, " "),
                       fmt_thousands(key).replace(".", ".\n", 1)])


def _tst(rng: random.Random, record, key) -> str | None:
    marks = {"ED": "ED", "E": "E", "AG": "Ag"}
    if any(w not in marks for w in record.chain):
        return None
    prefix = {"RR": "RR", "AIRR": "AIRR", "ARR": "ARR"}.get(record.classe or "", "RR")
    chain = "-".join([marks[w] for w in record.chain] + [prefix])
    number = _number(rng, key)
    forms = [f"{chain}-{number}", f"TST-{chain}-{number}", f"TST - {chain} - {number}", f"{chain} n. {number}",
             f"Proc. nº TST-{chain}-{number}"]
    if not record.chain and prefix == "RR":
        forms += [f"Recurso de Revista nº {number}", f"Rec. de Revista {number}"]
    if not record.chain and prefix == "AIRR":
        forms += [f"Agravo de Instrumento em Recurso de Revista nº {number}"]
    return rng.choice(forms)


def _tail(rng: random.Random, relator: str | None) -> str:
    rel = relator or "Fulano"
    return rng.choice([
        f", Relator(a) Min. {rel}, {rng.choice(ORGAOS)}, DJe de {_date(rng)}",
        f", Min. {rel}, {rng.choice(ORGAOS)}, DJ {_date(rng)}",
        f", j. em {_date(rng)}",
        f", {rng.choice(ORGAOS)}, Rel. Min. {rel}",
        f" - Rel. Min. {rel} - DJe {_date(rng)}",
    ])


def gen_v3_process(rng: random.Random, src: Sources, invent: bool) -> Cite | None:
    ix = src.index
    doc_id = rng.choice(src.unique_short + src.unique_cnj)
    record = ix.records[doc_id]
    key = src.key_of(doc_id)
    if key is None or (invent and (key := perturb_number(rng, src, key)) is None):
        return None
    if record.tribunal == "TST":
        text = _tst(rng, record, key)
    else:
        if record.classe not in V3_CLASS or len(record.chain) > 3 or any(w not in V3_CHAIN for w in record.chain):
            return None
        chain = sorted(set(record.chain), key=record.chain.index)
        parts = [_element(rng, *V3_CHAIN[w]) for w in chain] + [_element(rng, *V3_CLASS[record.classe])]
        text = parts[0]
        for i, part in enumerate(parts[1:], 1):
            following = chain[i] if i < len(chain) else record.classe
            natural = " na " if following in FEMININE else " nos " if following == "ED" else " no "
            text += rng.choice([natural, natural, natural, " em ", "-" if len(parts[i - 1]) < 7 else natural]) + part
        text = recase(rng, text)
        uf = rng.choice(["/{u}", "/{u}", "-{u}", " – {u}", " ({u})", ", {u}", ""]).format(u=record.uf) if record.uf else ""
        text = f"{text} {rng.choice(V3_MARKERS)}{_number(rng, key)}{uf}"
        if rng.random() < 0.35:
            court = rng.choice(V3_COURTS[record.tribunal])
            text = rng.choice([f"{court}, {text}", f"{text} ({court})", f"{text}, do {court}", f"{court} – {text}",
                               f"{court}: {text}"])
    if not text:
        return None
    after = _tail(rng, clean_relator(record.relator or "") or None) if rng.random() < 0.4 else ""
    return Cite(text, "jurisprudencia", "inventada" if invent else "real", "" if invent else doc_id, after)


# ------------------------------------------------------------------ súmulas

V3_SUMULA = ["Súmula n. {n}/{t}", "Súmula {n} do {h} {t}", "enunciado sumular {n}/{t}", "Súmula {n} - {t}",
             "Súm. {n}, {t}", "verbete {n} da Súmula/{t}", "{t} - Súmula {n}", "Súmula {n} do {L}",
             "Súmula nº {n} do {L} ({t})"]
V3_TST = ["Súmula {n}, I, do TST", "Súmula nº {n}, item I, do TST", "Súmula {n}/TST, IV"]
V3_VINC = ["Súmula Vinculante n. {n}", "SV {n}", "enunciado vinculante nº {n}", "verbete vinculante {n} do STF",
           "Súmula Vinculante {n} do Pretório Excelso", "SÚMULA VINCULANTE Nº {n}"]
HONORIFIC = ["Eg.", "Col.", "C.", "E.", "Egrégio", "Colendo"]


def gen_v3_sumula(rng: random.Random, src: Sources, invent: bool) -> Cite:
    if not invent:
        trib, num, vinc = rng.choice(SUMULA_REAL)
        doc_id = src.index.sumulas[(trib, num, vinc)]
    else:
        trib, vinc = rng.choice([("STJ", False), ("STF", False), ("TST", False), ("TSE", False), ("STF", True)])
        existing = {n for (t, n, v) in src.index.sumulas if t == trib and v == vinc}
        num = rng.choice([n for n in range(1, 650) if n not in existing])
        doc_id = ""
    if vinc:
        form = rng.choice(V3_VINC)
    elif trib == "TST" and rng.random() < 0.4:
        form = rng.choice(V3_TST)
    else:
        form = rng.choice(V3_SUMULA)
    text = form.format(n=num, t=trib, L=COURT_LONG[trib], h=rng.choice(HONORIFIC))
    return Cite(recase(rng, text), "jurisprudencia", "inventada" if invent else "real", doc_id)


# ------------------------------------------------------------------ artigos

V3_DIPLOMAS = {
    "CF": [("Constituição Federal", "da"), ("CF/1988", "da"), ("Carta Política", "da"), ("Constituição", "da"),
           ("Magna Carta", "da"), ("CR/88", "da"), ("CRFB", "da")],
    "CPC": [("CPC/2015", "do"), ("Código Processual Civil", "do"), ("Código de Processo Civil de 2015", "do"),
            ("CPC vigente", "do"), ("Lei nº 13.105/2015", "da")],
    "CLT": [("CLT", "da"), ("Consolidação das Leis do Trabalho", "da"), ("Decreto-Lei nº 5.452/1943", "do"),
            ("CLT/1943", "da")],
    "CPP": [("CPP", "do"), ("Código de Processo Penal Brasileiro", "do"), ("Código Processual Penal", "do")],
    "CPM": [("CPM", "do"), ("Código Penal Militar (CPM)", "do"), ("Decreto-Lei 1.001/1969", "do")],
    "CC": [("CC", "do"), ("Código Civil Brasileiro", "do"), ("CC/2002", "do"), ("Código Civil (Lei 10.406/2002)", "do")],
    "CDC": [("CDC", "do"), ("Código de Defesa do Consumidor", "do"), ("Lei 8.078/1990", "da")],
    "CE": [("Código Eleitoral", "do"), ("Lei nº 4.737/65", "da"), ("Código Eleitoral (Lei 4.737/1965)", "do")],
    "LC64/1990": [("LC 64/1990", "da"), ("Lei Complementar 64/90", "da"), ("LC nº 64/90", "da")],
}
V3_ENUM = ["", "", ", caput", ", inc. I", ", incisos I a III", ", § 2º", ", §§ 1º e 2º", ", par. único",
           ", alínea b", ", I, 'a'", " caput", ", inciso XXXV"]
V3_ARTICLE = ["art. {a}{e} {p} {d}", "art. {a}{e}, {p} {d}", "{d}, art. {a}{e}", "{d} - art. {a}{e}",
              "artigo {a}{e} {p} {d}", "Art. {a}{e} {p} {d}", "artigo {a}, {d}"]


def gen_v3_article(rng: random.Random, src: Sources, invent: bool) -> Cite:
    if not invent:
        (diploma, num), doc_id = rng.choice(sorted(src.index.dispositivos.items()))
    else:
        diploma = rng.choice(sorted(V3_DIPLOMAS))
        existing = {n for (d, n) in src.index.dispositivos if d == diploma}
        num = rng.choice([n for n in range(2, 400) if n not in existing])
        doc_id = ""
    name, prep = rng.choice(V3_DIPLOMAS[diploma])
    art = rng.choice([f"{num}º", f"{num}°", f"{num}o"]) if num < 10 else rng.choice([fmt_thousands(num), str(num)])
    text = rng.choice(V3_ARTICLE).format(a=art, e=rng.choice(V3_ENUM), p=prep, d=name)
    return Cite(recase(rng, text), "lei", "inventada" if invent else "real", doc_id)


# ------------------------------------------------------------------ descritivas

V3_NOUN = ["julgado", "precedente", "acórdão", "aresto", "julgamento", "decisum"]
V3_REL = ["Rel. Min. {r}", "Relatora Ministra {r}", "Min. Rel. {r}", "rel. {r}", "de relatoria de {r}",
          "relatoria: Min. {r}", "Ministro {r} (relator)", "Min.ª {r}"]


def _v3_year(rng: random.Random, ano: int) -> str:
    return rng.choice([f"em {ano}", f"de {ano}", f"({ano})", f"julgado em {_date(rng, ano)}",
                       f"em {rng.randint(1, 12):02d}/{ano}", f"{ano}"])


def gen_v3_descriptive(rng: random.Random, src: Sources, invent: bool) -> Cite | None:
    (tribunal, ano, _), records = rng.choice(src.desc_groups)
    record = rng.choice(records)
    words = clean_relator(record.relator).split()
    name = f"{words[0]} {words[-1]}" if len(words) >= 3 and rng.random() < 0.3 else " ".join(words)
    name = name.upper() if rng.random() < 0.25 else name
    classe = record.classe if record.classe in V3_CLASS and rng.random() < 0.3 else None
    matches = src.same_relator(tribunal, ano, name, classe)
    if not matches:
        return None
    court = rng.choice(V3_COURTS[tribunal])
    court_part = rng.choice([f"do {court}", f"no {court}", f"({court})", f"{court},", court])
    year, rel = _v3_year(rng, ano), rng.choice(V3_REL).format(r=name)
    head = rng.choice(V3_CLASS[classe][1]) if classe else rng.choice(V3_NOUN)
    r = rng.random()
    if r < 0.2:
        text = f"{head} {court_part} {year}, {rel}"
    elif r < 0.35:
        text = f"{head} ({court}, {rel}, {year})"
    else:
        parts = [court_part.rstrip(","), year, rel]
        rng.shuffle(parts)
        text = f"{head} " + rng.choice([", ", " ", " – "]).join(parts)
    text = recase(rng, text)
    if len(matches) == 1:
        return Cite(text, "jurisprudencia", "real", matches[0].doc_id)
    return Cite(text, "jurisprudencia", "incompleta")


V3_GENERATORS = [
    (lambda rng, src, t: gen_v3_process(rng, src, False), 0.28),
    (lambda rng, src, t: gen_v3_process(rng, src, True), 0.22),
    (lambda rng, src, t: gen_v3_descriptive(rng, src, False), 0.16),
    (lambda rng, src, t: gen_v3_sumula(rng, src, rng.random() < 0.5), 0.12),
    (lambda rng, src, t: gen_v3_article(rng, src, rng.random() < 0.45), 0.22),
]

# ------------------------------------------------------------------ contexto

V3_PARAGRAPHS = [
    "Sobre o tema: {c}.",
    "Aplicável, pois, {c}.",
    "Como se viu em {c}, a tese não prospera.",
    "{c}.",
    "Consoante {c}, o recurso não merece trânsito.",
    "Ex vi {c}.",
    "A questão já foi dirimida (cf. {c}).",
    "Ver, ainda: {c}.",
]
V3_DENSE = [
    "Nesse sentido: {c}; e {c2}.",
    "Vide {c} e {c2}.",
    "{c}; {c2}.",
    "Confiram-se: {c}, bem como {c2}.",
]
V3_DISTRACTORS = [
    "Petição protocolada sob o nº {prot} em {data}.",
    "Nos autos do Processo nº {cnj8}, a sentença foi proferida em {data}.",
    "O Ministro {r1}, em palestra proferida em {ano}, defendeu a tese contrária.",
    "Recurso Especial admitido na origem em {data}.",
    "Incidem as regras da Lei nº 8.112/90 (Regime Jurídico Único).",
    "O STJ, em {ano}, editou a Resolução nº {n3}.",
    "Tema de repercussão geral ainda pendente de julgamento.",
    "Recurso ordinário nº de ordem {n3} na pauta de {data}.",
]
RELATORS = ["Nancy Andrighi", "Rogerio Schietti Cruz", "Mauro Campbell Marques", "Rosa Weber", "Luiz Fux"]


def _distractor(rng: random.Random, filler) -> str:
    return rng.choice(V3_DISTRACTORS).format(
        prot=f"{rng.randint(10, 99)}.{rng.randint(100, 999)}/{rng.randint(2015, 2025)}",
        cnj8=f"{rng.randint(1000000, 9999999)}-{rng.randint(10, 99)}.20{rng.randint(10, 25)}.8."
             f"{rng.randint(1, 27):02d}.{rng.randint(1, 9999):04d}",
        data=_date(rng), ano=rng.randint(2010, 2025), n3=rng.randint(100, 999), r1=rng.choice(RELATORS),
    )


CATALOG = Catalog(generators=V3_GENERATORS, paragraphs=V3_PARAGRAPHS, distractor=_distractor,
                  dense_paragraphs=V3_DENSE)

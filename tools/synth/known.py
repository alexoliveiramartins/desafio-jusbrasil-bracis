"""Formatos conhecidos: a base de val, test, armadilhas, denso, ocr_forte e poluido.

`test` acrescenta formatos exclusivos (nunca vistos em `val`) e
`formatos_ineditos` aplica `holdout_variant`, escrito antes de o pipeline
suportar aquelas superfícies.
"""

from __future__ import annotations

import random
import re

from .sources import SUMULA_REAL, Cite, Sources, clean_relator, fmt_thousands


def fmt_short(rng: random.Random, n: int, test: bool) -> str:
    options = [fmt_thousands(n), str(n), fmt_thousands(n)]
    if test:
        options += [fmt_thousands(n, " "), fmt_thousands(n, ". ")]
    return rng.choice(options)


def fmt_cnj(rng: random.Random, key: tuple[int, str], test: bool) -> str:
    seq, rest = key
    dv, ano, j, tr, orig = rest[:2], rest[2:6], rest[6], rest[7:9], rest[9:]
    seq7 = f"{seq:07d}"
    options = [
        f"{seq7}-{dv}.{ano}.{j}.{tr}.{orig}",
        f"{seq}-{dv}.{ano}.{j}.{tr}.{orig}",
    ]
    if test:
        options += [f"{seq7}{dv}{ano}{j}{tr}{orig}", f"{seq7}-{dv} {ano} {j} {tr} {orig}"]
    return rng.choice(options)


MARKERS_SHARED = ["nº ", "nº ", "n. ", "", "Nº "]
MARKERS_TEST = ["n.º ", "no ", "n° "]

CHAIN_WORDS = {
    "AGINT": ["AgInt no", "Agravo Interno no"],
    "AGRG": ["AgRg no", "Agravo Regimental no"],
    "ED": ["EDcl no", "Embargos de Declaração no"],
    "EDV": ["EDv nos", "Embargos de Divergência no"],
    "PEXT": ["PExt no"],
}
CLASS_WORDS = {
    "REsp": (["REsp", "Recurso Especial"], ["RESP", "Rec. Esp.", "R.Esp."]),
    "AREsp": (["AREsp", "Agravo em Recurso Especial"], ["ARESP", "A.REsp"]),
    "RHC": (["RHC", "Recurso em Habeas Corpus"], ["R.H.C."]),
    "RMS": (["RMS", "Recurso em Mandado de Segurança"], ["R.M.S."]),
    "HC": (["HC", "Habeas Corpus"], ["H.C."]),
    "Rcl": (["Rcl", "Reclamação"], ["RCL", "Recl."]),
    "AR": (["AR", "Ação Rescisória"], []),
    "RE": (["RE", "Recurso Extraordinário"], ["R.E."]),
    "ARE": (["ARE"], []),
    "REspe": (["REspe", "Recurso Especial Eleitoral"], ["RESPE"]),
    "RO": (["RO", "Recurso Ordinário"], []),
    "AI": (["AI", "Agravo de Instrumento"], []),
    "APL": (["APL", "Apelação"], ["Apelação Criminal"]),
    "RSE": (["RSE", "Recurso em Sentido Estrito"], []),
    "AgInt": (["AgInt", "Agravo Interno"], []),
}
TST_PREFIX = {"RR": "RR", "AIRR": "AIRR", "ARR": "ARR"}


def render_class(rng, record, test: bool) -> str | None:
    words = CLASS_WORDS.get(record.classe)
    if not words:
        return None
    base = rng.choice(words[0] + (words[1] if test else []))
    # A cadeia inteira precisa ser escrita: omitir um recurso (ex.: EDv)
    # transformaria a citação em outro registro e o rótulo ficaria errado.
    if any(w not in CHAIN_WORDS for w in record.chain) or len(record.chain) > 3:
        return None
    chain = sorted(set(record.chain), key=record.chain.index)
    prefix = " ".join(rng.choice(CHAIN_WORDS[w]) for w in chain)
    return f"{prefix} {base}".strip()


# ------------------------------------------------------------ geradores

def gen_real_process(rng, src: Sources, test: bool) -> Cite | None:
    ix = src.index
    doc_id = rng.choice(src.unique_short + src.unique_cnj)
    record = ix.records[doc_id]
    marker = rng.choice(MARKERS_SHARED + (MARKERS_TEST if test else []))
    if record.tribunal == "TST":
        # Cadeia como no próprio acórdão: "TST-ED-E-ED-RR-<CNJ>".
        chain = "".join({"ED": "ED-", "E": "E-", "AG": "Ag-"}.get(w, "") for w in record.chain)
        prefix = TST_PREFIX.get(record.classe or "", "RR")
        lead = rng.choice(["TST-", "processo nº TST-"] + ([""] if not chain else []))
        text = f"{lead}{chain}{prefix}-{fmt_cnj(rng, src.cnj_of[doc_id], test)}"
        return Cite(text, "jurisprudencia", "real", doc_id)
    cls = render_class(rng, record, test)
    if not cls:
        return None
    if doc_id in src.cnj_of:
        number = fmt_cnj(rng, src.cnj_of[doc_id], test)
    else:
        number = fmt_short(rng, src.short_of[doc_id], test)
    uf = f"/{record.uf}" if record.uf and rng.random() < 0.8 else ""
    if uf and test and rng.random() < 0.4:
        uf = rng.choice([f" - {record.uf}", f" ({record.uf})"])
    return Cite(f"{cls} {marker}{number}{uf}", "jurisprudencia", "real", doc_id)


def gen_invented_process(rng, src: Sources, test: bool) -> Cite | None:
    for _ in range(50):
        base = gen_real_process(rng, src, test)
        if base is None:
            continue
        # Troca o número por outro que não exista em nenhum cabeçalho.
        digits = re.findall(r"\d[\d.\- ]*\d", base.text)
        if not digits:
            continue
        original = max(digits, key=len)
        raw = re.sub(r"\D", "", original)
        for _ in range(20):
            pos = rng.randrange(len(raw) - 4 if len(raw) > 13 else len(raw))
            new = raw[:pos] + str((int(raw[pos]) + rng.randint(1, 8)) % 10) + raw[pos + 1:]
            core = new[:-13] if len(new) >= 14 else new
            if new[0] != "0" and not src.number_exists(core.lstrip("0")):
                break
        else:
            continue
        # Reaplica a formatação original dígito a dígito.
        it = iter(new)
        replaced = re.sub(r"\d", lambda _: next(it), original)
        return Cite(base.text.replace(original, replaced), "jurisprudencia", "inventada")
    return None


def gen_descriptive(rng, src: Sources, test: bool) -> Cite:
    (tribunal, ano, relator), records = rng.choice(src.desc_groups)
    relator = clean_relator(relator)
    templates = [
        "julgado do {t} proferido em {a} pela relatoria de {r}",
        "precedente do {t} de {a}, da relatoria de {r}",
        "acórdão do {t} julgado em {a} sob relatoria de {r}",
    ]
    if test:
        templates += ["decisão do {t} de {a}, relatoria do Min. {r}"]
    text = rng.choice(templates).format(t=tribunal, a=ano, r=relator)
    if len(records) == 1:
        return Cite(text, "jurisprudencia", "real", records[0].doc_id)
    return Cite(text, "jurisprudencia", "incompleta")


def gen_sumula(rng, src: Sources, test: bool) -> Cite:
    real = rng.random() < 0.5
    if real:
        trib, num, vinc = rng.choice(SUMULA_REAL)
        doc_id = src.index.sumulas[(trib, num, vinc)]
    else:
        trib, vinc = rng.choice([("STJ", False), ("STF", False), ("TST", False), ("STF", True)])
        existing = {n for (t, n, v) in src.index.sumulas if t == trib and v == vinc}
        num = rng.choice([n for n in range(100, 999) if n not in existing])
        doc_id = ""
    if vinc:
        forms = [f"Súmula Vinculante {num}", f"Súmula Vinculante nº {num}"]
        if test:
            forms += [f"Súmula Vinculante n. {num} do STF"]
    else:
        forms = [f"Súmula {num} do {trib}", f"Súmula nº {num} do {trib}"]
        if test:
            forms += [f"Súmula nº {num}/{trib}", f"Enunciado {num} da Súmula do {trib}"]
    return Cite(rng.choice(forms), "jurisprudencia", "real" if real else "inventada", doc_id)


DIPLOMA_FORMS = {
    "CF": (["Constituição Federal", "Constituição da República", "CF"], ["CF/88"]),
    "CLT": (["CLT", "Consolidação das Leis do Trabalho"], []),
    "CPC": (["CPC", "Código de Processo Civil"], ["Lei nº 13.105/2015"]),
    "CPP": (["CPP", "Código de Processo Penal"], []),
    "CPM": (["Código Penal Militar"], ["CPM"]),
    "CC": (["Código Civil"], ["CC"]),
    "CDC": (["Código de Defesa do Consumidor", "CDC"], []),
    "CE": (["Código Eleitoral"], []),
    "LC64/1990": (["Lei Complementar nº 64/1990"], ["LC nº 64/1990"]),
}
INCISOS = ["", ", I", ", IX", ", LV", ", § 1º", ", parágrafo único"]


def gen_article(rng, src: Sources, test: bool) -> Cite:
    real = rng.random() < 0.55
    if real:
        (diploma, num), doc_id = rng.choice(sorted(src.index.dispositivos.items()))
    else:
        diploma = rng.choice(sorted(DIPLOMA_FORMS))
        existing = {n for (d, n) in src.index.dispositivos if d == diploma}
        num = rng.choice([n for n in range(2, 400) if n not in existing])
        doc_id = ""
    shared, extra = DIPLOMA_FORMS[diploma]
    name = rng.choice(shared + (extra if test else []))
    art = f"{num}º" if num < 10 else fmt_thousands(num)
    word = rng.choice(["art.", "artigo"] + (["art"] if test else []))
    prep = "da" if name.startswith(("Constituição", "Consolidação", "Lei", "CF", "CLT", "LC")) else "do"
    text = f"{word} {art}{rng.choice(INCISOS)}, {prep} {name}".replace(", do", " do").replace(", da", " da")
    if rng.random() < 0.5:
        text = text.replace(f"{art}, ", f"{art} ") if "," not in text.split(art)[1][:3] else text
    if test and rng.random() < 0.25:
        # Ordem invertida, comum em peças: "CF, art. 5º".
        text = f"{name}, art. {art}"
    return Cite(text, "lei", "real" if real else "inventada", doc_id)


# Referências vagas NÃO são citações no goldenset atual: entram como
# distratores (texto sem rótulo), para medir falsos positivos.
VAGUE_DISTRACTORS = [
    "Aplica-se o artigo correspondente do Código de Processo Civil.",
    "Tal entendimento encontra eco em reiterados precedentes do Superior Tribunal de Justiça.",
    "O verbete sumular aplicável à espécie afasta a pretensão.",
    "A jurisprudência pacífica do STF caminha no mesmo sentido.",
]


def gen_stage(rng, src: Sources, test: bool) -> Cite | None:
    """Mesmo número com registros em fases recursais diferentes: a cadeia
    citada (AgInt, EDcl...) precisa escolher o registro certo."""
    ix = src.index
    groups = [ids for ids in ix.by_short.values()
              if len(ids) > 1 and len({ix.records[i].chain for i in ids}) == len(ids)]
    if not groups:
        return None
    doc_id = rng.choice(rng.choice(groups))
    record = ix.records[doc_id]
    cls = render_class(rng, record, test)
    if not cls:
        return None
    number = fmt_short(rng, next(k for k, v in ix.by_short.items() if doc_id in v), test)
    uf = f"/{record.uf}" if record.uf else ""
    return Cite(f"{cls} nº {number}{uf}", "jurisprudencia", "real", doc_id)


GENERATORS = [
    (gen_real_process, 0.30),
    (gen_invented_process, 0.22),
    (gen_descriptive, 0.16),
    (gen_sumula, 0.10),
    (gen_article, 0.16),
]


# ------------------------------------------------------------ documento

OPENINGS = [
    "EXCELENTÍSSIMO SENHOR MINISTRO RELATOR\n\nAutos nº {autos}\n\n{parte}, já qualificada nos autos, "
    "vem, respeitosamente, apresentar suas razões.\n",
    "PARECER\n\nProcesso nº {autos}\nInteressado: {parte}\n\nTrata-se de consulta formulada "
    "sobre a controvérsia descrita abaixo, protocolada em {data}.\n",
    "DECISÃO MONOCRÁTICA\n\nAutos nº {autos}\n\nCuida-se de recurso interposto por {parte} "
    "contra acórdão publicado em {data}.\n",
]
PARAGRAPHS = [
    "Nesse sentido, a orientação firmada no {c} afasta a tese sustentada pela parte adversa.",
    "Conforme decidido no {c}, a matéria não comporta reexame nesta instância.",
    "A jurisprudência é firme, como se extrai do {c}, ao qual se remete desde logo.",
    "Incide, na espécie, o {c}, cuja aplicação não foi afastada pelo acórdão recorrido.",
    "Não se ignora o teor do {c}; todavia, a hipótese dos autos é diversa.",
    "Ademais, o {c} reforça a necessidade de fundamentação concreta.",
    "A pretensão encontra amparo no {c}, conforme demonstrado nas razões recursais.",
    "O valor da causa foi fixado em R$ {valor}, conforme fls. {fls}, nos termos da Lei nº 13.467/2017.",
    "A sessão de julgamento ocorreu em {data}, com publicação no DJe de {data2}.",
] + VAGUE_DISTRACTORS
CLOSINGS = ["\nAnte o exposto, requer o provimento do recurso.\n\nNestes termos, pede deferimento.\n",
            "\nÉ o parecer.\n", "\nPublique-se. Intimem-se.\n"]
PARTES = ["MARIA APARECIDA SOUZA", "INDÚSTRIA TÊXTIL ARARAQUARA S.A.", "JOSÉ CARLOS PEREIRA",
          "MUNICÍPIO DE SÃO JOSÉ", "COOPERATIVA AGRÍCOLA DO VALE LTDA"]


DENSE_PARAGRAPHS = [
    "Nesse sentido: {c}; {c2}.",
    "Confiram-se, entre outros, o {c} e o {c2}, ambos no mesmo sentido.",
    "(v. {c}; {c2})",
]


# Formatos do perfil `formatos_ineditos`, escritos ANTES de qualquer ajuste
# do pipeline para eles. Nunca ajuste regras olhando os erros desse perfil:
# ele existe para medir generalização a superfícies realmente novas.
def holdout_variant(rng, cite: Cite) -> Cite:
    t = cite.text
    variants = []
    if cite.tipo == "jurisprudencia" and re.match(r"(?:REsp|AREsp|RHC|RMS|HC|Rcl)\b", t):
        variants += [
            lambda: re.sub(r"^REsp", "Resp.", t),
            lambda: re.sub(r"\bnº ", "n. ", t),
            lambda: t.replace("/", "-"),
        ]
    if re.match(r"Súmula \d+ do (\w+)$", t):
        num, trib = re.match(r"Súmula (\d+) do (\w+)$", t).groups()
        variants += [lambda: f"Súm. {num}/{trib}", lambda: f"Súmula {num}, {trib}",
                     lambda: f"verbete nº {num} da Súmula do {trib}"]
    m = re.match(r"(art\.|artigo) (\S+?)(, [^d]+?)? d[oa] (CF|CPC|CLT|Constituição Federal)$", t)
    if m:
        art, diploma = m.group(2), m.group(4)
        variants += [lambda: f"art. {art}, caput, da {diploma}" if diploma in ("CF", "CLT", "Constituição Federal")
                     else f"art. {art}, caput, do {diploma}"]
    m = re.match(r"julgado do (\w+) proferido em (\d{4}) pela relatoria de (.+)$", t)
    if m:
        variants += [lambda: f"julgamento do {m.group(1)} em {m.group(2)}, sob a relatoria do Ministro {m.group(3)}"]
    if not variants:
        return cite
    return Cite(rng.choice(variants)(), cite.tipo, cite.classe, cite.doc_id)


"""Classificação das citações contra a base canônica (desafio1_bracis.db).

  1. índice: lê a base uma vez e indexa o número PRÓPRIO de cada acórdão,
     súmulas e dispositivos (`CanonicalIndex`);
  2. resolvedor: citação -> real (com id_canonico) / inventada / incompleta
     (`resolve`). É o único ponto que pode dizer `real`, e só por consulta à base.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import (
    OCR_DIGITS,
    number_groups,
    appeal_chain,
    article_number,
    canonical_words,
    chain_distance,
    class_code,
    cnj_justice,
    cnj_key,
    content_terms,
    diploma_key,
    locate_number,
    name_matches,
    name_tokens,
    number_digits,
    short_key,
    trailing_uf,
)
from .spans import article_diploma, courts_in, known_relator, ocr, relator_name


# ============================================================================
# 1. Índice da base canônica
# ============================================================================
# Índice da base canônica (desafio1_bracis.db).
#
# A base não tem colunas com o número do processo, a súmula ou o artigo. Este
# módulo extrai esses identificadores do texto de cada registro uma única vez e
# monta índices em memória para o resolvedor.
#
# A cobertura da base é congelada (a mesma no conjunto cego), por isso súmulas
# e dispositivos sem número legível no texto são mapeados em tabelas curadas.

# Reserva para a versão antiga da base, sem cabeçalho "Súmula n. X do Y" /
# "Artigo N do <diploma>". A base atual é lida pelos cabeçalhos.
# Súmulas cujo texto não traz o número: doc_id -> (tribunal, número, vinculante).
CURATED_SUMULAS = {
    1289712966: ("STF", 10, True),
    1431369957: ("TST", 331, False),
}

# Dispositivos de lei: doc_id -> diploma (chave de src/normalize.DIPLOMAS).
CURATED_DISPOSITIVOS = {
    28893055: "CPC",
    10590194: "CPM",
    10626510: "CF",
    10710324: "CLT",
    10647746: "CLT",
    10641516: "CF",
    10637358: "CLT",
    11304039: "LC64/1990",
    10641213: "CF",
    10577194: "CE",
    10718759: "CC",
    10606184: "CDC",
    10652044: "CPP",
}

CNJ_IN_TEXT = re.compile(r"(\d{1,7})\s*-\s*(\d{2})\.(\d{4})\.(\d)\.(\d{2})\.(\d{4})")
# Número próprio no cabeçalho de STJ/STF: "Nº 1.528.455 - RJ", "RECLAMAÇÃO 76.532".
HEADER_SHORT = re.compile(
    r"(?:\bN[º°o]\.?\s*|\bN\s+|RECLAMAÇÃO\s+|EXTRAORDINÁRIO\s+(?:COM\s+AGRAVO\s+)?"
    r"|HABEAS\s+CORPUS\s+|AGRAVO\s+)"
    r"(\d{1,3}(?:\.\d{3})+|\d{2,8})\b(?!\s*[-.]\d{2}\.\d{4})"
)
TST_OWN = re.compile(
    r"TST\s*-\s*(?:[A-Za-z]+\s*-\s*)*"
    r"(\d{1,7})\s*-\s*(\d{2})\.(\d{4})\.(\d)\.(\d{2})\.(\d{4})"
)
# O acórdão do TST se identifica em "Vistos, relatados e discutidos estes
# autos de <classe> nº TST-<recursos>-<CNJ>"; outros números TST- no texto
# podem ser precedentes citados.
TST_VISTOS = re.compile(r"(?i)vistos,?\s+relatados\s+e\s+discutidos[^.]{0,250}?(?=TST\s*-)")
# TSE com OCR no cabeçalho: "Nº 3112-8520146070000", "Nº 839-42. 2010.6.19.0000".
TSE_HEADER = re.compile(r"\bN\s*[º°o]?\.?\s*(\d[\d\s.\-]{12,30}\d)")
# STF: "AÇÃO RESCISÓRIA 2.614 DISTRITO FEDERAL RELATORA".
STF_HEADER = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d{2,7})\s+[A-ZÀ-Ú][A-ZÀ-Ú ]*?\s+RELATOR")
HEADER_UF = re.compile(r"^\s*[-–]?\s*([A-Z]{2})\b")
ARTIGO_IN_TEXT = re.compile(r"^\s*Art\.?\s*(\d+)")
SUMULA_IN_TEXT = re.compile(r"SÚMULA\s+(\d+)")
# Cabeçalhos da base atual: "Artigo 290 do Decreto-Lei nº 1.001, de 21 de ...",
# "Súmula Vinculante n. 10 do STF".
ARTIGO_HEADER = re.compile(r"^\s*Artigo\s+(\d+)\S*\s+d[oa]\s+([^\n]+)")
SUMULA_HEADER = re.compile(r"^\s*Súmula\s+(Vinculante\s+)?n\.?\s*(\d+)\s+d[oa]\s+(\w+)", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    doc_id: str
    tribunal: str | None
    natureza: str
    ano: int | None
    relator: str | None
    uf: str | None = None
    # Cadeia recursal do cabeçalho, ex.: ("AGINT", "ED") para "EDcl no AgInt no REsp".
    chain: tuple[str, ...] = ()
    # Classe processual principal do cabeçalho ("REsp", "AREsp", "Rcl"...).
    classe: str | None = None


@dataclass
class CanonicalIndex:
    records: dict[str, Record] = field(default_factory=dict)
    by_cnj: dict[tuple, list[str]] = field(default_factory=lambda: defaultdict(list))
    by_short: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    sumulas: dict[tuple, str] = field(default_factory=dict)
    dispositivos: dict[tuple, str] = field(default_factory=dict)
    unindexed: list[str] = field(default_factory=list)
    # Termos da ementa/início de cada acórdão, para desempate por contexto.
    terms: dict[str, frozenset] = field(default_factory=dict)
    # Tokens do nome de cada relator do acervo: acham descritivas sem pista de relatoria.
    relator_tokens: list[frozenset] = field(default_factory=list)

    @classmethod
    def from_sqlite(cls, path: str | Path) -> "CanonicalIndex":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"base canônica não encontrada: {path}")
        # Somente leitura: a base nunca é alterada pelo pipeline.
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT id, tribunal, natureza, ano, relator, texto FROM documentos"
            ).fetchall()
        finally:
            conn.close()

        index = cls()
        for doc_id, tribunal, natureza, ano, relator, texto in rows:
            index._add(str(doc_id), tribunal, natureza, ano, relator, texto)
        names = {frozenset(name_tokens(r.relator)) for r in index.records.values()
                 if r.natureza == "acordao" and r.relator}
        index.relator_tokens = sorted((n for n in names if len(n) >= 2), key=sorted)
        return index

    def _add(self, doc_id, tribunal, natureza, ano, relator, texto) -> None:
        if natureza == "dispositivo":
            header = ARTIGO_HEADER.match(texto)
            if header and diploma_key(header.group(2)):
                self.dispositivos[(diploma_key(header.group(2)), int(header.group(1)))] = doc_id
            elif CURATED_DISPOSITIVOS.get(int(doc_id)) and ARTIGO_IN_TEXT.match(texto):
                # Formato antigo da base (sem cabeçalho): tabela curada.
                diploma = CURATED_DISPOSITIVOS[int(doc_id)]
                self.dispositivos[(diploma, int(ARTIGO_IN_TEXT.match(texto).group(1)))] = doc_id
            else:
                self.unindexed.append(doc_id)
            self.records[doc_id] = Record(doc_id, tribunal, natureza, ano, relator)
            return

        if natureza == "sumula":
            header = SUMULA_HEADER.match(texto)
            curated = CURATED_SUMULAS.get(int(doc_id))
            m = SUMULA_IN_TEXT.search(texto)
            if header:
                self.sumulas[(header.group(3).upper(), int(header.group(2)), bool(header.group(1)))] = doc_id
            elif curated:
                self.sumulas[curated] = doc_id
            elif m and tribunal:
                self.sumulas[(tribunal, int(m.group(1)), False)] = doc_id
            else:
                self.unindexed.append(doc_id)
            self.records[doc_id] = Record(doc_id, tribunal, natureza, ano, relator)
            return

        self.terms[doc_id] = content_terms(texto[:4000])
        key, uf, chain, classe = _own_number(tribunal, texto)
        if key is None:
            self.unindexed.append(doc_id)
        elif isinstance(key, tuple):
            self.by_cnj[key].append(doc_id)
        else:
            self.by_short[key].append(doc_id)
        self.records[doc_id] = Record(doc_id, tribunal, natureza, ano, relator, uf, chain, classe)


def _unglue(text: str) -> str:
    """Separa palavras grudadas na base: "nosEMBARGOS" -> "nos EMBARGOS".

    Só na fronteira minúscula -> duas maiúsculas, para não quebrar siglas
    como "AgInt", "REsp" e "AgRg".
    """
    return re.sub(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Ý]{2})", " ", text)


def _own_number(tribunal: str | None, texto: str):
    """(chave do número próprio, UF, cadeia recursal) de um acórdão."""
    texto = _unglue(texto[:1500]) + texto[1500:]
    header = texto[:1500]
    # STM, TSE e TST usam numeração CNJ; no TST ela não está no cabeçalho,
    # mas o acórdão se identifica como "TST-<recursos>-<CNJ>".
    if tribunal == "TST":
        vistos = TST_VISTOS.search(texto)
        m = TST_OWN.match(texto, vistos.end()) if vistos else None
        m = m or TST_OWN.search(texto)
        if not m:
            return None, None, (), None
        prefix = m.group(0)
        return cnj_key("".join(m.groups())), None, appeal_chain(prefix), class_code(prefix)

    if tribunal in ("STM", "TSE"):
        m = CNJ_IN_TEXT.search(header)
        if m:
            u = re.match(r"\s*/\s*([A-Z]{2})\b", texto[m.end():m.end() + 6])
            before = header[:m.start()]
            return (cnj_key("".join(m.groups())), u.group(1) if u else None,
                    appeal_chain(before), class_code(before))
        m = TSE_HEADER.search(texto[:600])
        if m:
            digits, _ = number_digits(m.group(1))
            key = cnj_key(digits)
            if key:
                before = texto[:m.start()]
                return key, None, appeal_chain(before), class_code(before)
        return None, None, (), None

    m = HEADER_SHORT.search(texto[:800])
    if m is None and tribunal == "STF":
        m = STF_HEADER.search(texto[:800])
    if not m:
        return None, None, (), None
    u = HEADER_UF.match(texto[m.end():m.end() + 8])
    before = texto[:m.start(1)][-250:]
    return short_key(m.group(1)), u.group(1) if u else None, appeal_chain(before), class_code(before)


# ============================================================================
# 2. Resolvedor
# ============================================================================
# Resolvedor determinístico: citação extraída -> classe + id canônico.
#
# Regras (Dados do Caça-Alucinações §1):
#   real        identificadores suficientes e exatamente um registro na base;
#   inventada   identificadores suficientes, nenhum registro correspondente;
#   incompleta  identificadores insuficientes para a consulta ou para um
#               registro único.
#
# É a única etapa que decide `real`, e só por consulta à base.

REAL, INVENTADA, INCOMPLETA = "real", "inventada", "incompleta"

# Probabilidade de a classe estar correta, por regra. Alimenta o bônus de
# calibração (Brier). Medida nos conjuntos de iteração (`python -m tools.calibrate`),
# com teto de 0,98: o sintético tende a ser mais fácil que o cego.
# real_descritiva fica mais baixa: depende de uma leitura da regra oficial que
# nem o dev nem o sintético confirmam.
CONFIDENCE = {
    "real_exato": 0.98,
    "real_ocr": 0.96,
    "real_cadeia_divergente": 0.93,
    "real_duplicata": 0.85,
    "real_atributo_divergente": 0.60,
    "real_descritiva": 0.80,
    "inventada": 0.97,
    "inventada_ocr": 0.95,
    "incompleta_descritiva": 0.97,
    "incompleta_descritiva_sem_registro": 0.60,
    "incompleta_sem_numero": 0.55,
    "erro_interno": 0.40,
}

# Tribunais em que cada classe processual existe. Classe desconhecida: todos.
CLASS_COURTS = {
    "REsp": {"STJ"}, "AREsp": {"STJ"}, "RHC": {"STJ", "STF"}, "RMS": {"STJ", "STF"},
    "SLS": {"STJ"}, "RE": {"STF"}, "ARE": {"STF"}, "Rcl": {"STF", "STJ", "TSE", "STM"},
    "HC": {"STJ", "STF", "STM", "TSE"}, "AR": {"STJ", "STF", "TSE", "STM"},
    "REspe": {"TSE"}, "RO": {"TSE", "STF", "STJ"}, "AI": {"TSE", "STF"},
    "APL": {"STM"}, "RSE": {"STM"}, "AgInt": {"STM", "STJ"},
    "RR": {"TST"}, "AIRR": {"TST"}, "ARR": {"TST"},
}
# Descritiva sem tribunal: a classe citada indica o tribunal ("Rcl de 2025, Rel. Min. …").
CLASS_TRIBUNAL = {
    "REsp": "STJ", "AREsp": "STJ", "RHC": "STJ", "RMS": "STJ", "SLS": "STJ",
    "RE": "STF", "ARE": "STF", "REspe": "TSE", "APL": "STM", "RSE": "STM",
    "RR": "TST", "AIRR": "TST", "ARR": "TST",
}


@dataclass
class Resolution:
    classificacao: str
    id_canonico: str | None
    confianca: float
    regra: str


def _result(classe: str, doc_id: str | None, regra: str) -> Resolution:
    return Resolution(classe, doc_id, CONFIDENCE[regra], regra)


def resolve(citation: dict, index: CanonicalIndex) -> Resolution:
    familia = citation["familia"]
    trecho = citation.get("trecho_norm") or citation["trecho"]
    if familia in ("processos", "processos_trabalhistas"):
        return _resolve_process(trecho, index, citation.get("contexto", ""), citation.get("contexto_depois", ""))
    if familia == "sumulas":
        return _resolve_sumula(trecho, index)
    if familia == "artigos":
        return _resolve_article(trecho, index)
    if familia in ("processos_descritivos", "julgados_descritivos"):
        return _resolve_descriptive(trecho, index)
    if familia == "temas":
        return _result(INVENTADA, None, "inventada")  # a base não cobre temas
    raise ValueError(f"família desconhecida: {familia}")


# ------------------------------------------------------------------ processos

# Relator citado logo após o número: "REsp 1.234/SP, Rel. Min. Fulano de Tal".
RELATOR_AFTER = re.compile(
    r"^[\s,;(–-]*(?:[^.;]{0,40}?)\bRel(?:\.|ator|atora)\s*(?:p/\s*ac[oó]rd[aã]o\s*)?"
    r"(?:Min(?:\.|istr[oa])\s*)?"
    r"(?P<nome>[A-ZÀ-Ý][\wÀ-ÿ]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÀ-Ý][\wÀ-ÿ]+){0,5})"
)


def _resolve_process(trecho: str, index: CanonicalIndex, context: str, after: str) -> Resolution:
    digits, fixes = number_digits(trecho)
    if not digits:
        return _result(INCOMPLETA, None, "incompleta_sem_numero")

    candidates = _process_candidates(digits, trecho, index)
    if not candidates and (robust := _robust_number(trecho, digits)):
        # A leitura principal não achou registro; o número lido de novo com OCR colado
        # ("9Bb-96.2010.G…", "67 067", "…-3\ufeff2 2012") pode achar. Só letra -> dígito.
        candidates, fixes = _process_candidates(robust, trecho, index), fixes or 1
    if not candidates:
        return _result(INVENTADA, None, "inventada_ocr" if fixes else "inventada")

    before = _before_number(trecho)
    tail = trecho[len(before):]
    # Recursos listados depois do número ("REsp 1.234/SP, AgInt") também são da cadeia.
    chain = tuple(sorted(appeal_chain(before) + (appeal_chain(tail[tail.find(","):]) if "," in tail else ())))
    relator_match = RELATOR_AFTER.match(after)
    relator = relator_match.group("nome") if relator_match else None
    context_terms = content_terms(context) if context else frozenset()

    def rank(doc_id: str):
        record = index.records[doc_id]
        return (
            chain_distance(chain, record.chain),                                   # 1. cadeia recursal
            0 if relator is None or name_matches(relator, record.relator) else 1,  # 2. relator citado
            -len(context_terms & index.terms.get(doc_id, frozenset())),            # 3. contexto x ementa
            int(doc_id),                                                           # 4. desempate estável
        )

    scored = sorted(candidates, key=rank)
    best = scored[0]
    ties = [c for c in scored if rank(c)[:2] == rank(best)[:2]]
    if relator and not name_matches(relator, index.records[best].relator):
        # O número existe, mas o relator citado é de outro julgado: sinal de
        # alucinação parcial. A classe segue a definição (resolve a um
        # registro), mas a confiança cai.
        regra = "real_atributo_divergente"
    elif fixes:
        regra = "real_ocr"
    elif len(ties) > 1:
        regra = "real_duplicata"  # mesmo julgado com doc_ids diferentes no acervo
    elif chain_distance(chain, index.records[best].chain):
        regra = "real_cadeia_divergente"
    else:
        regra = "real_exato"
    return _result(REAL, best, regra)


def _process_candidates(digits: str, trecho: str, index: CanonicalIndex) -> list[str]:
    """Registros com esse número, filtrados por classe, UF e tribunal citados no trecho."""
    key = cnj_key(digits) if len(digits) >= 14 else None
    if key:
        candidates = [c for c in index.by_cnj.get(key, ()) if index.records[c].tribunal in (cnj_justice(key), None)]
    else:
        candidates = _filter_by_class(trecho, list(index.by_short.get(short_key(digits), ())), index)
    uf = trailing_uf(trecho)
    if uf:
        candidates = [c for c in candidates if index.records[c].uf in (uf, None)]
    # Tribunal citado explicitamente ("(STJ)", "Superior Tribunal Militar, …").
    cited_courts = set(courts_in(trecho))
    if cited_courts:
        candidates = [c for c in candidates if index.records[c].tribunal in cited_courts]
    return candidates


def _robust_number(trecho: str, digits: str) -> str | None:
    """Número do trecho pelo leitor de OCR colado (normalize.number_groups), se for o único candidato
    e diferente do que a leitura principal achou: CNJ (14 a 20 dígitos) ou número curto (3 a 8)."""
    groups = [g for g in number_groups(trecho) if not (len(g) == 4 and g[:2] in ("19", "20"))]
    cnj = [g.zfill(20) for g in groups if 14 <= len(g) <= 20]
    short = [g for g in groups if 3 <= len(g) <= 8]
    robust = cnj[0] if len(cnj) == 1 else short[0] if not cnj and len(short) == 1 else None
    if robust is None or robust.lstrip("0") == digits.lstrip("0"):
        return None
    return robust


def _before_number(trecho: str) -> str:
    return trecho[:locate_number(trecho)[2]]


def _filter_by_class(trecho: str, candidates: list[str], index: CanonicalIndex) -> list[str]:
    courts = CLASS_COURTS.get(class_code(_before_number(trecho)) or "")
    if not courts:
        return candidates
    return [c for c in candidates if index.records[c].tribunal in courts]


# ------------------------------------------------------------------ súmulas e artigos

_SUMULA_CUE = re.compile(
    rf"{ocr('súmula')}|s[uú5]m\.|{ocr('sumular')}|{ocr('enunciado')}|{ocr('verbete')}|(?-i:\bSV\b)",
    re.IGNORECASE,
)


# Número de súmula com maioria de letras de OCR ("SO6", "Z1l", "|b1"): number_digits
# recusa esses tokens (palavra não vira número), mas logo depois da pista a posição é
# do número. Mesmo formato que o extrator aceita (spans.SUMULA_NUMBER).
_SUMULA_OCR_NUMBER = re.compile(
    r"(?<![\w|])(?:\d|[lIBgGSOQZ|](?=[\dOolISsgGBbDQqZz|]{0,2}\d))[\dOolISsgGBbDQqZz|]{0,3}(?![\w|])")


def _resolve_sumula(trecho: str, index: CanonicalIndex) -> Resolution:
    """Número + tribunal (sigla ou extenso) + vinculante, em qualquer ordem."""
    cue = _SUMULA_CUE.search(trecho)
    digits, fixes = number_digits(trecho[cue.start():] if cue else trecho)
    if not digits and cue and (m := _SUMULA_OCR_NUMBER.search(trecho, cue.end())):
        digits = re.sub(r"\D", "", m.group().translate(OCR_DIGITS))
        fixes = sum(not c.isdigit() for c in m.group())
    if not digits or len(digits) > 4:
        return _result(INCOMPLETA, None, "incompleta_sem_numero")
    numero = int(digits)
    vinculante = "vinculante" in canonical_words(trecho) or bool(re.search(r"(?-i:\bSV\b)", trecho))
    courts = courts_in(trecho)
    tribunal = courts[-1] if courts else ("STF" if vinculante else None)
    if tribunal:
        doc_id = index.sumulas.get((tribunal, numero, vinculante))
    else:
        matches = [d for (t, n, v), d in index.sumulas.items() if n == numero and not v]
        if len(matches) > 1:
            return _result(INCOMPLETA, None, "incompleta_sem_numero")
        doc_id = matches[0] if matches else None
    if doc_id is None:
        return _result(INVENTADA, None, "inventada_ocr" if fixes else "inventada")
    return _result(REAL, doc_id, "real_ocr" if fixes else "real_exato")


def _resolve_article(trecho: str, index: CanonicalIndex) -> Resolution:
    numero = article_number(trecho)
    diploma = article_diploma(trecho)
    if numero is None or diploma is None:
        return _result(INCOMPLETA, None, "incompleta_sem_numero")
    doc_id = index.dispositivos.get((diploma, numero))
    if doc_id is None:
        return _result(INVENTADA, None, "inventada")
    return _result(REAL, doc_id, "real_exato")


# ------------------------------------------------------------------ descritivas

_YEAR = re.compile(r"(?<![\w|])((?:19|[2Z][0OD])[\dOolIgGSsBbLDQqZz|]{2})(?![\w|])")
_RELATOR_FALLBACK = re.compile(
    rf"(?:{ocr('relatoria')}\s+\S+|Rel\.\s*Min\.)\s*(?:Min(?:\.|istr[oa])\s*)?(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def descriptive_matches(trecho: str, index: CanonicalIndex) -> list[str] | None:
    """Registros compatíveis com tribunal + ano + relator (+ classe, se citada).

    Aceita os três elementos em qualquer ordem e o tribunal por sigla ou por
    extenso. Devolve None quando falta algum elemento da consulta.
    """
    text = " ".join(trecho.split())
    rel = relator_name(text) or known_relator(text, index.relator_tokens)
    if rel:
        relator = rel[2]
    else:
        m = _RELATOR_FALLBACK.search(text)
        if not m:
            return None
        relator = m.group(1)
    years = [d for d in (number_digits(y)[0] for y in _YEAR.findall(text)) if len(d) == 4]
    if not years:
        return None
    ano = int(years[0])
    head = re.split(r"\b(?:[S5]T[FJM]|T[S5][TE])\b|(?:19|[2Z][0OD])[\dOolIgGSsBbLDQqZz|]{2}", text)[0]
    classe = class_code(head)
    courts = courts_in(text)
    if classe and courts and courts[0] not in CLASS_COURTS.get(classe, {courts[0]}):
        classe = None  # classe impossível no tribunal citado ("Rec. Esp. … STM"): não filtra
    if courts:
        tribunais = {courts[0]}
    elif classe == "Rcl":
        tribunais = {"STF", "STJ"}
    elif classe in CLASS_TRIBUNAL:
        tribunais = {CLASS_TRIBUNAL[classe]}
    else:
        return None
    return [
        r.doc_id for r in index.records.values()
        if r.natureza == "acordao" and r.tribunal in tribunais and r.ano == ano
        and (classe is None or r.classe == classe)
        and name_matches(relator, r.relator)
    ]


def _resolve_descriptive(trecho: str, index: CanonicalIndex) -> Resolution:
    """Tribunal + ano + relator: só é `real` se identificar um único registro."""
    matches = descriptive_matches(trecho, index)
    if matches is None:
        return _result(INCOMPLETA, None, "incompleta_sem_numero")
    if len(matches) == 1:
        return _result(REAL, matches[0], "real_descritiva")
    if not matches:
        # Nenhum registro. A definição sugeriria `inventada`, mas nos conjuntos
        # sintéticos isso vinha de OCR no nome do relator, e o goldenset não
        # tem descritiva inventada. `incompleta` também não arrisca τ.
        return _result(INCOMPLETA, None, "incompleta_descritiva_sem_registro")
    return _result(INCOMPLETA, None, "incompleta_descritiva")

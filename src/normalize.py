"""Normalização de citações: OCR, números de processo e diplomas legais.

Nada aqui altera o texto do documento. As funções recebem o trecho já
extraído e devolvem chaves comparáveis com o índice da base canônica.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path

# Confusões típicas de OCR em posições de dígito.
OCR_DIGITS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "i": "1", "|": "1",
    "S": "5", "s": "5", "L": "1",
    "g": "9", "q": "9",
    "G": "6", "b": "6",
    "B": "8",
    "Z": "2", "z": "2",
})
OCR_LETTERS = frozenset("OoQDlIi|SsgqGbBZzL")

TOKEN = re.compile(r"[0-9A-Za-z|]+")
SEPARATOR = re.compile(r"[\s.\-–—]*")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def fold(text: str) -> str:
    """Minúsculas, sem acentos e com espaços simples."""
    return re.sub(r"\s+", " ", strip_accents(text).lower()).strip()


# Vocabulário fechado das citações, em ordem de prioridade (mais frequente
# primeiro). Serve para dois fins: corrigir OCR ("agrãv0" -> agravo) e expandir
# abreviações por truncamento ("Rec." -> recurso, "Espec." -> especial).
VOCAB_PRIORITY = """
    recurso agravo especial embargos declaracao regimental interno habeas corpus
    reclamacao apelacao criminal mandado seguranca instrumento ordinario
    extraordinario sentido estrito rescisoria acao eleitoral divergencia revista
    suspensao liminar sentenca constituicao constitucional federal
    republica codigo processo processual civil penal militar consumidor defesa
    consolidacao leis trabalho trabalhista trabalhistas celetista inelegibilidade
    inelegibilidades complementar sumula sumular vinculante
    enunciado artigo inciso paragrafo alinea correspondente precedentes relatoria
    relator relatora relatado proferido julgado julgamento ministro ministra
""".split()
LEGAL_VOCABULARY = sorted(set(VOCAB_PRIORITY))


_FIRST_OCR = {"c": "e", "e": "c", "l": "i", "i": "l"}


def ocr_fix_words(text: str) -> str:
    """fold() + correção de palavras do vocabulário jurídico com erro de OCR."""
    def fix(m: re.Match) -> str:
        word = m.group()
        if len(word) < 5 or word in LEGAL_VOCABULARY:
            return word
        plain = word.translate(str.maketrans("0135", "oles"))
        for variant in (plain, plain.replace("rn", "m")):
            if variant in LEGAL_VOCABULARY:
                return variant
        target = plain.replace("rn", "m")
        close = difflib.get_close_matches(target, LEGAL_VOCABULARY, n=3, cutoff=0.75)
        # A 1ª letra também sofre OCR, mas só nas trocas clássicas.
        firsts = {target[0], _FIRST_OCR.get(target[0], target[0])}
        close = [c for c in close if c[0] in firsts]
        return close[0] if close else word

    return re.sub(r"[a-z0-9]*[a-z][a-z0-9]*", fix, fold(text))


def is_subsequence(short: str, word: str) -> bool:
    it = iter(word)
    return all(ch in it for ch in short)


def expand_abbreviation(token: str) -> str | None:
    """'espec' -> 'especial', 'embs' -> 'embargos', 'rg' -> 'regimental'.

    Abreviação jurídica é truncamento (prefixo) ou, mais raramente, esqueleto
    de consoantes (subsequência com a mesma inicial). Em empate vale a ordem de
    VOCAB_PRIORITY. Nada com menos de 2 letras é expandido.
    """
    if len(token) < 2 or token in LEGAL_VOCABULARY or token in _ALIASES:
        return None  # já é palavra ou sigla conhecida ("clt" não é "celetista")
    if re.fullmatch(r"[ivxlc]+", token):
        return None  # numeral romano ("III." de um inciso)
    if "rn" in token:  # "crirn." -> "crim." (OCR de m)
        return expand_abbreviation(token.replace("rn", "m"))
    for word in VOCAB_PRIORITY:
        if word.startswith(token) and word != token:
            return word
    for word in VOCAB_PRIORITY:
        if word[0] == token[0] and len(token) >= 2 and is_subsequence(token, word):
            return word
    if "l" in token[1:]:  # "reglm" é "regim" com OCR
        return expand_abbreviation(token[0] + token[1:].replace("l", "i"))
    return None


# Palavra de 2–8 letras seguida de ponto, sem fazer parte de sigla pontuada
# ("R.Esp.", "H.C." ficam como estão: o ponto antes da palavra as protege).
_ABBREV = re.compile(r"(?<![a-z0-9.])([a-z]{2,8})\.(?![a-z0-9])")


def canonical_words(text: str) -> str:
    """ocr_fix_words() + abreviações por truncamento expandidas.

    "Agr. Int. no Rec. Espec." -> "agravo interno no recurso especial".
    """
    def expand(m: re.Match) -> str:
        return expand_abbreviation(m.group(1)) or m.group(0)

    return _ABBREV.sub(expand, ocr_fix_words(text))


def is_number_marker(token: str) -> bool:
    """Abreviações de 'número': n, nº, n.º, n°, no, nr, nro, num, núm., número (com OCR).

    O "n" lido como "ri" ou "u" só conta com o sinal: "riº", "uº", "u." ("rio" é palavra).
    """
    if re.fullmatch(r"(?:ri|u)\.?\s*[º°]\.?|u\.", token.strip(), re.IGNORECASE):
        return True
    t = re.sub(r"[.º°ª]", "", fold(token).replace("rn", "m").translate(str.maketrans("0c", "oe")))
    return t.startswith("n") and len(t) <= 6 and is_subsequence(t[1:], "umero")


# ------------------------------------------------------------------ léxico destilado

LEXICON_PATH = Path(__file__).with_name("lexicon.json")
# Palavra genérica sozinha não é apelido ("tribunal", "recurso", "lei").
_GENERIC = set(VOCAB_PRIORITY) | {"tribunal", "corte", "superior", "supremo", "suprema", "egregio", "colendo",
                                  "excelso", "excelsa", "justica", "lei", "codigo", "carta", "diploma", "norma"}


def load_lexicon(path: Path = LEXICON_PATH) -> dict[str, dict[str, str]]:
    """Formas destiladas de um LLM por tools/lexicon.py: {tipo: {forma canônica: chave}}.

    Vazio sem o arquivo. Formas de uma palavra genérica são ignoradas.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    lexicon = {}
    for kind, forms in data.items():
        if isinstance(forms, dict):
            canon = {canonical_words(f).strip(" .,;"): k for f, k in forms.items()}
            lexicon[kind] = {f: k for f, k in canon.items() if f and not (" " not in f and f in _GENERIC)}
    return lexicon


def _lexicon_pattern(form: str) -> str:
    return r"\s+".join(re.escape(w) for w in form.split())


def _numeric_token(token: str) -> str | None:
    """Converte um token com dígitos e letras confundíveis, ou devolve None."""
    if not any(c.isdigit() for c in token):
        return None
    letters = [c for c in token if not c.isdigit()]
    if any(c not in OCR_LETTERS for c in letters) or len(letters) * 2 > len(token):
        return None
    return token.translate(OCR_DIGITS)


def number_digits(trecho: str) -> tuple[str, int]:
    """Dígitos do identificador principal do trecho e nº de correções de OCR."""
    digits, fixes, _ = locate_number(trecho)
    return digits, fixes


def locate_number(trecho: str) -> tuple[str, int, int]:
    """(dígitos, correções de OCR, posição inicial) do identificador principal.

    Junta os tokens numéricos consecutivos a partir do primeiro, separados
    apenas por espaço, ponto ou hífen. UF, parênteses e barra encerram o número.
    Palavras com um dígito trocado ("Agrãv0") não contam como número.
    """
    first = None
    pos = 0
    digits = []
    fixes = 0
    started = False
    while pos < len(trecho):
        m = TOKEN.search(trecho, pos)
        if not m:
            break
        converted = _numeric_token(m.group())
        # Dígito encostado numa letra faz parte de uma palavra ("5úmula",
        # "Dcclaraçã0"), não é o número da citação.
        # "º", "°" e "ª" são letras para o Python, mas não colam número em palavra ("nº66516").
        word_before = re.search(r"\w+$", trecho[:m.start()])
        after_marker = (m.group().isdigit() and len(m.group()) >= 2  # "n233" sim; "n0" (= "no") não
                        and word_before is not None and is_number_marker(word_before.group()))
        glued = (m.start() > 0 and trecho[m.start() - 1].isalpha() and trecho[m.start() - 1] not in "ºª"
                 and not after_marker) or (
            m.end() < len(trecho) and trecho[m.end()].isalpha() and trecho[m.end()] not in "ºª")
        if glued:
            converted = None
        if converted is None and started and not glued and (
                _ocr_only(m.group()) or _ocr_block(m.group(), trecho[pos:m.start()], trecho[m.end():m.end() + 3])):
            # Dígito isolado lido como letra no meio do número: "2015.S.24".
            converted = m.group().translate(OCR_DIGITS)
        gap = trecho[pos:m.start()]
        # Vírgula só separa milhar quando vem seguida de exatamente 3 dígitos.
        comma_thousands = gap == "," and len(m.group()) == 3
        # Barra no lugar do ponto entre DV e ano do CNJ: "7000395-11/2022.7.00...".
        slash_cnj = (gap.strip() == "/" and 3 <= len("".join(digits)) <= 9 and converted is not None
                     and re.fullmatch(r"(?:19|20)\d\d", converted)
                     and re.match(r"[\s.\-–]*[\dOolISsDQqZz|]", trecho[m.end():m.end() + 5]))
        # Depois de um espaço, só continua o número um grupo que começa por dígito
        # ou um grupo de milhar com OCR ("60 G81"); "193 D0 STJ" termina em 193.
        space_ok = not gap.isspace() or m.group()[0].isdigit() or (
            len(m.group()) == 3 and sum(c.isdigit() for c in m.group()) >= 2)
        if started and (converted is None or not space_ok or not (SEPARATOR.fullmatch(gap) or comma_thousands
                                                                   or slash_cnj)):
            break
        if converted is not None and not started and not _looks_numeric(m.group()):
            converted = None
        if (converted is None and not started and not glued and m.group() in ("l", "I", "L", "|")
                and re.match(r"[.\s]\s*(?:\d|[OolISsgGBbLDQqZz|]\d)", trecho[m.end():m.end() + 5])):
            converted = "1"  # "l.508.709": o 1 inicial lido como letra (ou "|")
        if converted is not None:
            if not started:
                first = m.start()
            started = True
            digits.append(converted)
            fixes += sum(1 for c in m.group() if not c.isdigit())
        pos = m.end()
    return "".join(digits), fixes, first if first is not None else len(trecho)


def _ocr_only(token: str) -> bool:
    """Grupo curto no meio do número: "S" em "2015.S.24", "0OS", "Gl4".

    Com 2–3 caracteres precisa ter um dígito: palavras ("DO") nunca viram dígitos.
    """
    if not all(c in OCR_LETTERS or c.isdigit() for c in token):
        return False
    return len(token) == 1 or (len(token) <= 3 and any(c.isdigit() for c in token))


def _ocr_block(token: str, before: str, after: str) -> bool:
    """Bloco de CNJ todo lido como letras ("700040O-GG.2023", "7.OO.0000").

    Só entre separadores "-"/"." e com dígito logo depois: "1.234 DO STJ" não conta.
    """
    return (2 <= len(token) <= 4 and all(c in OCR_LETTERS for c in token)
            and before.strip() in ("-", ".", "–") and re.match(r"\s*[.\-–]\s*\d", after) is not None)


def _looks_numeric(token: str) -> bool:
    """O primeiro token do número precisa ser majoritariamente dígitos."""
    return sum(c.isdigit() for c in token) * 2 >= len(token)


# ------------------------------------------------------------------ números com OCR colado

_UNIT = re.compile(r"[0-9OoQDlIi|SsgqGbBZzL]+")
_UF_FIRST = {"8": "B", "5": "S", "0": "O", "6": "G"}  # dígito que o OCR pôs no lugar da 1ª letra da UF
_SEP = set(" \t\n\r.-–—/\u00a0\u2009\u202f\u200b\u200c\u200d\u2060\u00ad\ufeff\u00ac")


def _joins(gap: str) -> bool:
    """Separador entre dois pedaços do mesmo número: pontuação, espaços e invisíveis; vírgula só colada
    ("4O,2023" é um número; "33.235, 2021" são dois)."""
    return all(c in _SEP or (c == "," and (gap[i + 1:i + 2] or "x") not in " \t\n\r")
               for i, c in enumerate(gap))


def number_groups(span: str) -> list[str]:
    return [digits for digits, _, _ in number_group_spans(span)]


def number_group_spans(span: str) -> list[tuple[str, int, int]]:
    """Números do trecho, com OCR letra->dígito, cada um inteiro ("6O.685" -> "60685").

    Pedaços (dígitos ou letras parecidas) ligados só por separadores formam um grupo. Pedaço sem
    dígito de verdade só entra se for curto e encostar em outro com dígito ("l. 627.496", "2012 G 2O"),
    e nunca se for parte de palavra ("no 685" não vira "0685"). Grupo colado em letras ("AgInt7S57430",
    "185do", "Vinculante10") vale se tiver ao menos dois dígitos de verdade: um dígito solto entre
    letras ("RE5P") é OCR de palavra, não número.
    """
    def letter(i: int) -> bool:  # "º" e "ª" são letras para o Python, mas aqui marcam ordinal
        return 0 <= i < len(span) and span[i].isalpha() and span[i] not in "ºª°"

    pieces = []
    for m in _UNIT.finditer(span):
        start, end, text = m.start(), m.end(), m.group()
        # UF colada: o último caractere pode ser a 1ª letra da UF lida como número ("77.14oSP" -> 140 + SP;
        # "- 8A" -> BA). Só quando ele e o seguinte formam uma UF; "86OPE" é 860 + PE.
        if len(text) >= 1 and span[end:end + 1].isupper() and not span[end + 1:end + 2].isalpha():
            first = _UF_FIRST.get(text[-1], text[-1])
            if first + span[end] in UFS and (len(text) > 1 or not letter(start - 1)):
                text, end = text[:-1], end - 1
        if text:
            pieces.append((start, end, text))
    real = [sum(c.isdigit() for c in p[2]) for p in pieces]

    def linked(i: int, j: int) -> bool:
        a, b = sorted((i, j))
        return _joins(span[pieces[a][1]:pieces[b][0]])

    keep = []
    for i, (start, end, text) in enumerate(pieces):
        inside_word = letter(start - 1) or letter(end)
        if real[i]:
            # Um dígito só, com letra antes, é OCR de palavra ("RE5P", "Súmu1a"), a não ser que continue num
            # número ("RESP6 .q89.q16" = 6.989.916); letra depois pode ser a UF.
            continues = i + 1 < len(pieces) and real[i + 1] and linked(i, i + 1)
            keep.append(real[i] >= 2 or not letter(start - 1) or continues)
            continue
        near_digits = any(0 <= j < len(pieces) and real[j] and linked(i, j) for j in (i - 1, i + 1))
        keep.append(len(text) <= 2 and near_digits and not inside_word)
    groups, current = [], []
    for i, ok in enumerate(keep):
        if ok and current and linked(current[-1], i):
            current.append(i)
        else:
            if current:
                groups.append(current)
            current = [i] if ok else []
    if current:
        groups.append(current)
    out = []
    for group in groups:
        digits = sum(real[i] for i in group)
        glued = letter(pieces[group[0]][0] - 1) or letter(pieces[group[-1]][1])
        text = "".join(pieces[i][2] for i in group).translate(OCR_DIGITS)
        raw = span[pieces[group[0]][0]:pieces[group[-1]][1]]
        if digits and (digits >= 2 or not glued) and text.isdigit() and not _DATE.fullmatch(raw):
            out.append((text, pieces[group[0]][0], pieces[group[-1]][1]))
    return out


# Data não é número de processo: "17/5/2021", "20-04-2019", "17.12.2014", "0G/2021" (mês/ano).
_D2 = r"[0-9OoQDlIi|SsgqGbBZzL]{1,2}"
_DATE = re.compile(rf"\s*{_D2}\s*([/.\-])\s*{_D2}\s*\1\s*(?:[0-9OoQDlIi|SsgqGbBZzL]{{2}}|[0-9OoQDlIi|SsgqGbBZzL]{{4}})\s*"
                   rf"|\s*{_D2}\s*/\s*[0-9OoQDlIi|SsgqGbBZzL]{{4}}\s*")


def cnj_key(digits: str) -> tuple[int, str] | None:
    """Chave CNJ: (sequencial sem zeros à esquerda, DV+ano+J+TR+origem)."""
    digits = re.sub(r"\D", "", digits)
    if len(digits) < 14 or len(digits) > 20:
        return None
    return int(digits[:-13]), digits[-13:]


def short_key(number: str) -> int:
    return int(re.sub(r"\D", "", number))


def cnj_justice(key: tuple[int, str]) -> str | None:
    """Segmento J do CNJ: 5 = Trabalho, 6 = Eleitoral, 7 = Militar."""
    return {"5": "TST", "6": "TSE", "7": "STM"}.get(key[1][6])


UFS = frozenset(
    "AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO".split()
)


def trailing_uf(trecho: str) -> str | None:
    m = re.search(r"[/(\-–]\s*([A-Z]{2})\)?\s*$", trecho)
    return m.group(1) if m and m.group(1) in UFS else None


# Diplomas: chave -> formas por extenso ou siglas (já em fold()).
DIPLOMAS = {
    "CF": ["constituicao federal", "constituicao da republica", "cf", "cf/88", "crfb", "cr",
           "constituicao federal de 1988", "constituicao da republica de 1988"],
    "CLT": ["consolidacao das leis do trabalho", "clt", "decreto-lei 5452/1943"],
    "CPC": ["codigo de processo civil", "cpc", "ncpc", "lei 13105/2015"],
    "CPP": ["codigo de processo penal", "cpp", "decreto-lei 3689/1941"],
    "CPM": ["codigo penal militar", "cpm", "decreto-lei 1001/1969"],
    "CP": ["codigo penal", "cp", "decreto-lei 2848/1940"],
    "CC": ["codigo civil", "cc", "lei 10406/2002"],
    "CDC": ["codigo de defesa do consumidor", "cdc", "lei 8078/1990"],
    "CE": ["codigo eleitoral", "lei 4737/1965"],
    "LC64/1990": ["lei complementar 64/1990", "lc 64/1990"],
}
_ALIASES = {alias: key for key, aliases in DIPLOMAS.items() for alias in aliases}
# Carregado depois de _ALIASES: a canonicalização das formas usa expand_abbreviation.
LEXICON = load_lexicon()
for _form, _key in LEXICON.get("diploma", {}).items():
    _ALIASES.setdefault(_form, _key)


# Ano da lei que a base indexa. "CPC/73" ou "Código Civil de 1916" são OUTRAS
# leis: mapeá-las para o código atual transformaria inventada em real (τ).
DIPLOMA_YEAR = {"CF": 1988, "CLT": 1943, "CPC": 2015, "CPP": 1941, "CPM": 1969,
                "CP": 1940, "CC": 2002, "CDC": 1990, "CE": 1965, "LC64/1990": 1990}
_YEAR_SUFFIX = re.compile(r"(?:/\s*|,?\s+de\s+(?:\d{1,2}o?\s+de\s+[a-z]+\s+de\s+)?)(\d{4}|\d{2})$")


def _year4(value: str) -> int:
    year = int(value)
    if year >= 100:
        return year
    return 1900 + year if year >= 30 else 2000 + year


_LAW_NUMBER_OCR = re.compile(r"(?<![A-Za-zÀ-ÿ])(?=[\dOolISsZzGgqQBb|.]*\d)[\dOolISsZzGgqQBb|][\dOolISsZzGgqQBbD|.]*(?=\s*/)")
_LAW_YEAR_OCR = re.compile(r"(?<=/)(?=\s*[\dOolISsZzGgqQBbD|]*\d)\s*[\dOolISsZzGgqQBbD|]{2,4}(?![A-Za-z\d])")


def diploma_key(text: str) -> str | None:
    """Identifica o diploma citado, tolerando OCR, abreviações e sinônimos.

    Ordem: nome/sigla conhecidos -> número da lei -> nome aproximado ->
    radicais ("processual" + "civil" -> CPC; "consumerista" -> CDC).
    """
    # Número e ano de lei com OCR ("13.l0s/2015", "G4/1990", "13.467/Z017"): letras
    # confundíveis viram dígitos com OCR_DIGITS, ainda com caixa (G→6, g→9).
    # Quebra de linha/hífen depois do ponto do milhar ("13.-467", "9.-\n096", "I3. |05").
    text = re.sub(r"(?<=[\dOolISsZzGgqQBb|])\.[\s\-]+(?=[\dOolISsZzGgqQBbD|])", ".", text)
    text = _LAW_NUMBER_OCR.sub(lambda m: m.group().translate(OCR_DIGITS), text)
    text = _LAW_YEAR_OCR.sub(lambda m: m.group().translate(OCR_DIGITS), text)
    t = canonical_words(text)
    t = re.sub(r"\blc[i1l]\b", "lei", t)  # "Lci": "e" lido como "c"
    # fold() já converteu "nº" em "no"; remove o marcador só antes de números.
    t = re.sub(r"\b(?:n|ri)[o.°]?\s*(?=\d)", "", t)  # "riº": OCR de "nº"
    t = re.sub(r"(\d)\.\s?(\d)", r"\1\2", t)
    t = re.sub(r"\s*/\s*", "/", t).strip(" ,.;:")
    # Anos de 2 dígitos em números de lei: "8.078/90" -> "8078/1990".
    t = re.sub(r"(\d{3,})/(\d{2})\b", lambda m: f"{m.group(1)}/{_year4(m.group(2))}", t)
    candidates = [t]
    paren = re.search(r"\(([^)]*)\)", t)
    if paren:  # "cdc (lei 8078/1990)": o nome e o conteúdo dos parênteses
        candidates += [re.sub(r"\s*\([^)]*\)", "", t).strip(), paren.group(1).strip()]
    for candidate in candidates:
        key = _diploma_exact(candidate)
        if key:
            return key
    for candidate in candidates:
        key = _diploma_by_stems(candidate)
        if key:
            return key
    return None


def _diploma_exact(t: str) -> str | None:
    year = None
    stripped = t
    m = _YEAR_SUFFIX.search(stripped)
    if m and not re.fullmatch(r"(?:lei(?: complementar)?|decreto-lei|lc) \d+/\d{4}", stripped):
        year = _year4(m.group(1))
        stripped = stripped[:m.start()].strip(" ,")
    stripped = re.sub(r"\s+(?:brasileir[oa]|vigente|atual)$", "", stripped)
    key = _lookup_diploma(stripped) or _lookup_diploma(t)
    if key and year and not key.startswith("LEI:") and DIPLOMA_YEAR.get(key) not in (None, year):
        return f"{key}/{year}"  # outra lei com o mesmo nome (ex.: CPC/1973)
    return key


def _lookup_diploma(t: str) -> str | None:
    if t in _ALIASES:
        return _ALIASES[t]
    # "Lei nº 4.737, de 15 de julho de 1965" -> "lei 4737/1965"
    t2 = re.sub(r"^((?:lei(?: complementar)?|decreto-lei|lc) \d+),? de .*?(\d{4})$", r"\1/\2", t)
    if t2 in _ALIASES:
        return _ALIASES[t2]
    law = re.fullmatch(r"(lei(?: complementar)?|decreto-lei|lc) (\d+)/(\d{4})", t2)
    if law:
        kind = "lei complementar" if law.group(1) == "lc" else law.group(1)
        normalized = f"{kind} {law.group(2)}/{law.group(3)}"
        return _ALIASES.get(normalized, f"LEI:{law.group(2)}/{law.group(3)}")
    close = difflib.get_close_matches(t2, [a for a in _ALIASES if len(a) > 5], n=1, cutoff=0.85)
    return _ALIASES[close[0]] if close else None


def _diploma_by_stems(t: str) -> str | None:
    """Classificação por radicais de sentido, para nomes não catalogados."""
    words = set(re.findall(r"[a-z]+", t))

    def has(*stems: str) -> bool:
        return any(w.startswith(x) for w in words for x in stems)

    if has("estadua", "estado", "municip", "organic", "regiment"):
        return None  # constituição estadual, lei orgânica, regimento: não são da base
    key = None
    if has("inelegibilidad"):
        key = "LC64/1990"
    elif has("constitu") or ("carta" in words and not has("codigo", "process")) or {"lei", "maior"} <= words:
        key = "CF"
    elif has("consumid"):
        key = "CDC"
    elif has("trabalhis", "celetis") or ("consolidacao" in words and has("trabalh", "leis")):
        key = "CLT"
    else:
        penal, civil, proc = "penal" in words, "civil" in words, has("process")
        code = has("codigo")
        if penal and has("militar"):
            key = "CPM"
        elif penal and proc:
            key = "CPP"
        elif civil and proc:
            key = "CPC"
        elif penal and code:
            key = "CP"
        elif civil and code:
            key = "CC"
        elif "eleitoral" in words and code:
            key = "CE"
    if key is None:
        return None
    years = [_year4(y) for y in re.findall(r"\b(\d{4})\b", t)]
    if years and DIPLOMA_YEAR.get(key) not in years:
        return f"{key}/{years[-1]}"
    return key


def article_number(trecho: str) -> int | None:
    # Mesmo conjunto de OCR do extrator (patterns.ARTICLE_NUMBER): ler
    # menos caracteres truncaria "1S0" em "1", um artigo que existe (τ).
    m = re.search(r"(?<!\w)[aáàâão]r[tf]\w*[\s.\-–]*(?:n[º°o.]*\s*)?([\dlISLBgQZ|][\dOolIgGSBLDQZ|.]*)", trecho, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).rstrip(".")
    # "5o" e "lo" (1º com OCR) são ordinais: o "o" não é um zero.
    if re.fullmatch(r"[\dlI]o", raw):
        raw = raw[:-1]
    digits = re.sub(r"\D", "", raw.translate(OCR_DIGITS))
    return int(digits) if digits else None


# Marcadores da cadeia recursal ("EDcl no AgInt no REsp"), usados para
# escolher entre registros do mesmo número em fases diferentes.
APPEAL_MARKERS = (
    ("AGINT", r"ag\.?\s*int\b|agravo\s+interno"),
    # "agr\b" é a sigla AgR; "Agr." (com ponto) é abreviação de agravo.
    ("AGRG", r"agrg\b|ag\.?\s*reg\b|agr\b(?!\.)|agravo\s+regimental"),
    ("EDV", r"edv\b|emb\.?\s*div\b|embargos\s+(?:de\s+)?diverg[eê]ncia"),
    ("ED", r"edcl\b|eds?\b|emb\.?\s*decl\b|embargos\s+(?:de\s+)?declara[cç][aã]o"),
    ("E", r"(?<![a-z])e(?=\s*-)|recurso\s+de\s+embargos"),
    ("PEXT", r"pext\b"),
    # Agravo nas siglas hifenizadas do TST: "Ag-ARR", "TST-Ag-RR".
    ("AG", r"ag(?=\s*-\s*[a-z])|ag(?=a?i?rr\b|arr\b)"),
)
APPEAL_MARKERS += tuple((key, _lexicon_pattern(form) + r"\b") for form, key in LEXICON.get("recurso", {}).items())
# (?:...) em volta: sem ele o (?<![a-z]) valeria só para a 1ª alternativa.
_MARKERS = [(name, re.compile(rf"(?<![a-z])(?:{pattern})", re.IGNORECASE))
            for name, pattern in APPEAL_MARKERS]


def appeal_chain(text: str) -> tuple[str, ...]:
    """Multiconjunto ordenado dos recursos citados antes do número.

    Conta nas duas leituras (siglas originais e abreviações expandidas) e fica
    com o máximo: "Agr. Int. no EDcl" tem AgInt só na forma expandida.
    """
    readings = (ocr_fix_words(text), canonical_words(text))
    found = []
    for name, pattern in _MARKERS:
        count = max(len(pattern.findall(t)) for t in readings)
        found.extend([name] * count)
    return tuple(sorted(found))


def chain_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Diferença entre cadeias: primeiro pelos recursos presentes (conjunto),
    depois pela contagem. "EDv nos EMBARGOS DE DIVERGÊNCIA" conta EDv duas
    vezes, mas é um recurso só."""
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    return 10 * len(set(a) ^ set(b)) + sum(((ca - cb) + (cb - ca)).values())


# Classe processual principal (a mais próxima do número). A ordem importa:
# formas mais específicas primeiro ("agravo em recurso especial" antes de
# "recurso especial").
CLASS_CODES = (
    ("AREsp", r"agravo\s+[ec]m\s+(?:recurso\s+especial|resp\b)|a\.?\s*r\.?\s*esp\b|agresp\b"),
    ("REspe", r"recurso\s+especial\s+eleitoral|respe\b|arespei\b|(?:recurso|rec\.?|r\.?)\s*esp(?:ecial|\.)?\s+eleit"),
    ("REsp", r"recurso\s+especial|rec\.?\s*esp\b|r\.?\s*esp\b"),
    ("RHC", r"recurso\s+(?:ordinario\s+)?em\s+(?:habeas\s+corpus|hc\b)|rhc\b"),
    ("RMS", r"recurso\s+(?:ordinario\s+)?em\s+(?:mandado\s+(?:de\s+)?seguranca|ms\b)"
            r"|recurso\s+ord\.\s+[ec]m\s+mandado|rms\b"),
    ("HC", r"habeas\s+corpus|h\.?\s*c\b"),
    ("Rcl", r"reclamacao|recl\b|rcl\b"),
    ("ARE", r"recurso\s+extraordinario\s+com\s+agravo|agravo\s+[ec]m\s+re\b|are\b"),
    ("RE", r"recurso\s+extraordinario|re\b"),
    ("AR", r"acao\s+rescisoria|ar\b"),
    ("SLS", r"suspensao\s+de\s+liminar"),
    ("AI", r"agravo\s+de\s+instrumento(?!\s+[ec]m\s+recurso\s+de\s+revista)|ai\b"),
    ("RO", r"recurso\s+ordinario|ro\b"),
    ("RSE", r"recurso\s+[ec]m\s+sentido\s+estrito|rse\b"),
    ("APL", r"apelacao|apl\b"),
    ("AIRR", r"agravo\s+de\s+instrumento\s+[ec]m\s+recurso\s+de\s+revista|airr\b"),
    ("ARR", r"recurso\s+de\s+revista\s+com\s+agravo|arr\b|rrag\b"),
    ("RR", r"recurso\s+de\s+revista|rr\b"),
    ("AgInt", r"agravo\s+interno|agint\b|ag\.\s*int\b"),
)
CLASS_CODES += tuple((key, _lexicon_pattern(form) + r"\b") for form, key in LEXICON.get("classe", {}).items())
# (?:...) em volta: sem ele "ro\b" casaria o fim de "ministro" (classe RO).
_CLASS_PATTERNS = [(code, re.compile(rf"(?<![a-z])(?:{p})")) for code, p in CLASS_CODES]


def class_code(text: str) -> str | None:
    """Classe processual citada mais perto do fim do texto (antes do número).

    Tenta primeiro a leitura com abreviações expandidas ("Rec. Espec." ->
    recurso especial) e cai para as siglas originais ("R.Esp.").
    """
    for reading in (canonical_words(text), ocr_fix_words(text)):
        code = _class_code(reading)
        if code:
            return code
    return None


def _class_code(t: str) -> str | None:
    best = None
    for code, pattern in _CLASS_PATTERNS:
        for m in pattern.finditer(t):
            # Prefere a que termina mais à direita; em empate, a mais longa
            # ("agravo em recurso especial" vence "recurso especial").
            if best is None or (m.end(), m.end() - m.start()) > (best[0], best[1]):
                best = (m.end(), m.end() - m.start(), code)
    if best is None:
        return None
    code = best[2]
    # "AgInt" só é classe quando nada mais aparece (STM: "AGRAVO INTERNO Nº").
    if code == "AgInt":
        others = [c for c, p in _CLASS_PATTERNS if c != "AgInt" and p.search(t)]
        return others[-1] if others else "AgInt"
    return code


_NAME_STOP = {"min", "ministro", "ministra", "de", "da", "do", "dos", "das", "e", "rel", "relator", "relatora"}


def name_tokens(name: str) -> list[str]:
    # Dígitos dentro de nomes são OCR: "VER5IANI", "M0rgana".
    folded = fold(name).translate(str.maketrans("01345", "oleas"))
    # Tokens de 2 letras são preposições (também com OCR: "dc", "d0"); títulos
    # com OCR ("mlnistro", "relãtor") também não são nome.
    return [w for w in re.findall(r"[a-z]+", folded)
            if w not in _NAME_STOP and len(w) > 2
            and not difflib.get_close_matches(w, _TITLES, n=1, cutoff=0.8)]


_TITLES = ["ministro", "ministra", "relator", "relatora", "relatoria"]


def name_matches(cited: str, record: str | None, cutoff: float = 0.75) -> bool:
    """Todos os tokens do nome citado aparecem (com tolerância a OCR) no relator do registro."""
    if not record:
        return False
    rec = name_tokens(record)
    cit = name_tokens(cited)
    if not cit or not rec:
        return False
    return all(difflib.get_close_matches(w, rec, n=1, cutoff=cutoff) for w in cit)


_STOPWORDS = frozenset(
    "a o as os de da do das dos e em no na nos nas por para com que se nao ao aos um uma "
    "ou como mais pelo pela pelos pelas sua seu suas seus art lei nº n recurso".split()
)


def content_terms(text: str) -> frozenset[str]:
    """Termos de conteúdo (sem acento, sem stopwords) para comparar contexto e ementa."""
    return frozenset(
        w for w in re.findall(r"[a-z]{4,}", strip_accents(text).lower()) if w not in _STOPWORDS
    )

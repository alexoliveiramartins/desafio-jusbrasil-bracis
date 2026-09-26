"""Famílias de ruído aplicadas a um texto real, com mapa de offsets.

`NoisyDoc` guarda, para cada caractere do original, o que ele virou e o que
foi inserido antes dele. Assim qualquer combinação de ruídos devolve os spans
do gabarito exatos no texto novo (`render`).

Invariantes que mantêm o rótulo do gabarito válido:
  · dígito nunca vira outro dígito (só letra parecida: 0→O, 1→l, 5→S);
  · nada é inserido em dígitos por troca de letra ('º' não vira '2');
  · lixo de página e numeração de linha só entram FORA dos spans;
  · cada caractere é alterado por no máximo uma família (`pristine`).
"""

from __future__ import annotations

import random
import re
import unicodedata

WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")


class NoisyDoc:
    def __init__(self, text: str, spans: list[tuple[int, int]]):
        self.orig = text
        self.cells = list(text)
        self.before = [""] * (len(text) + 1)
        self.in_span = [False] * len(text)
        self.gap_in_span = [False] * (len(text) + 1)  # posição estritamente dentro de um span
        for start, end in spans:
            for i in range(start, end):
                self.in_span[i] = True
            for i in range(start + 1, end):
                self.gap_in_span[i] = True

    def pristine(self, *idx: int) -> bool:
        return all(0 <= i < len(self.cells) and self.cells[i] == self.orig[i] for i in idx)

    def put(self, i: int, value: str) -> bool:
        if not self.pristine(i):
            return False
        self.cells[i] = value
        return True

    def insert(self, i: int, value: str, outside: bool = False) -> bool:
        if outside and self.gap_in_span[i]:
            return False
        self.before[i] += value
        return True

    def line_starts(self) -> list[int]:
        return [0] + [i + 1 for i, c in enumerate(self.orig) if c == "\n" and i + 1 < len(self.orig)]

    def render(self) -> tuple[str, list[int], list[int]]:
        """Texto final e, por caractere original, início/fim do seu substituto."""
        out, starts, ends, pos = [], [], [], 0
        for i, cell in enumerate(self.cells):
            out.append(self.before[i])
            pos += len(self.before[i])
            starts.append(pos)
            out.append(cell)
            pos += len(cell)
            ends.append(pos)
        out.append(self.before[-1])
        return "".join(out), starts, ends

    def edits(self) -> int:
        """Caracteres alterados + inseridos (medida de severidade)."""
        return sum(c != o for c, o in zip(self.cells, self.orig)) + sum(map(len, self.before))


# ------------------------------------------------------------------ OCR

DIGIT_OCR = {"0": "OOoD", "1": "llI|", "2": "Z", "5": "SSs", "6": "Gb", "8": "B", "9": "gq"}
LETTER_OCR = {
    "e": ["c"], "c": ["e"], "a": ["o", "ã"], "o": ["0", "a"], "i": ["l", "í", "1"], "l": ["1", "I", "i"],
    "I": ["l", "1"], "O": ["0"], "S": ["5"], "B": ["8"], "u": ["n"], "n": ["u", "ri"], "h": ["b", "li"],
    "t": ["f"], "f": ["t"], "é": ["e", "ê", "c"], "ç": ["c", "q"], "ã": ["a", "á"], "õ": ["o"],
    "í": ["i", "l"], "g": ["q"], "º": ["°", "o"],
}
MULTI_OCR = {"rn": "m", "cl": "d", "li": "h"}
SPLIT_OCR = {"m": "rn", "d": "cl"}


def ocr(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    text = doc.orig
    for i, ch in enumerate(text):
        if ch in DIGIT_OCR and rng.random() < p:
            doc.put(i, rng.choice(DIGIT_OCR[ch]))
        elif ch in SPLIT_OCR and rng.random() < p / 2:
            doc.put(i, SPLIT_OCR[ch])
        elif text[i:i + 2] in MULTI_OCR and rng.random() < p / 2 and doc.pristine(i, i + 1):
            doc.put(i, MULTI_OCR[text[i:i + 2]])
            doc.put(i + 1, "")
        elif ch in LETTER_OCR and rng.random() < p / 2:
            doc.put(i, rng.choice(LETTER_OCR[ch]))


# ------------------------------------------------------------------ acentos

def strip_accents(s: str) -> str:
    s = s.replace("º", "o").replace("ª", "a")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def acentos(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    nfd = rng.random() < p / 2  # PDF gerado no macOS: acento como caractere combinante
    for m in WORD.finditer(doc.orig):
        if nfd or rng.random() < p:
            for i in range(m.start(), m.end()):
                ch = doc.orig[i]
                if ord(ch) > 127:
                    doc.put(i, unicodedata.normalize("NFD", ch) if nfd else strip_accents(ch))
    for i, ch in enumerate(doc.orig):
        if ch in "ºª" and rng.random() < p:
            doc.put(i, strip_accents(ch))


# ------------------------------------------------------------------ espaços

ODD_SPACES = ["  ", "   ", " ", " ", " ", "\t", "  "]


def espacos(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    text = doc.orig
    for i, ch in enumerate(text):
        if ch == " ":
            r = rng.random()
            if r < p / 5 and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
                doc.put(i, "")  # palavras coladas ("doSTJ")
            elif r < p:
                doc.put(i, rng.choice(ODD_SPACES))
        elif ch in ".,/-" and 0 < i < len(text) - 1 and text[i - 1].isdigit() and rng.random() < p / 3:
            doc.put(i, rng.choice([f"{ch} ", f" {ch}", f" {ch} "]))  # "5. 230.808", "1.234 /SP"
    for m in WORD.finditer(text):
        if len(m.group()) >= 7 and rng.random() < p / 4:
            doc.insert(m.start() + rng.randint(3, len(m.group()) - 3), " ")  # "Recur so"


# ------------------------------------------------------------------ quebras de linha

def quebras(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    text = doc.orig
    for i, ch in enumerate(text):
        if ch == " ":
            r = rng.random()
            if r < p / 15:
                doc.put(i, "\n\n")  # parágrafo espúrio no meio da frase
            elif r < p / 3:
                doc.put(i, "\n")  # reflow de PDF
        elif ch == "\n" and 0 < i < len(text) - 1 and text[i - 1] != "\n" and text[i + 1] != "\n":
            if rng.random() < p / 3:
                doc.put(i, " ")  # linhas coladas
    for m in WORD.finditer(text):
        if len(m.group()) >= 6 and rng.random() < p / 4:
            cut = m.start() + rng.randint(2, len(m.group()) - 2)
            doc.insert(cut, rng.choice(["-\n", "-\n", "- \n", "¬\n", "­\n", "-\n\n"]))


def fim_de_linha(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    """CRLF (arquivo salvo no Windows): muda offsets de quem lê sem newline=""."""
    if rng.random() >= p:
        return
    mixed = rng.random() < 0.25
    for i, ch in enumerate(doc.orig):
        if ch == "\n" and (not mixed or rng.random() < 0.5):
            doc.put(i, "\r\n")


# ------------------------------------------------------------------ pontuação

THOUSANDS = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{3})+(?![\d])")
CNJ = re.compile(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")
NUMERO = re.compile(r"\bn(?:\.\s?)?[º°]\.?", re.IGNORECASE)
UF_SLASH = re.compile(r"/(?=\s?[A-Z]{2}\b)")


def pontuacao(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    text = doc.orig
    for rx in (THOUSANDS, CNJ):
        for m in rx.finditer(text):
            if rng.random() < p:
                sep = rng.choice(["", "", ",", " ", ". "])
                for i in range(m.start(), m.end()):
                    if text[i] in ".-":
                        doc.put(i, sep if text[i] == "." or rng.random() < 0.5 else text[i])
    for m in NUMERO.finditer(text):
        if rng.random() < p:
            i = text.index("º", m.start()) if "º" in m.group() else text.index("°", m.start())
            doc.put(i, rng.choice(["°", "o", ".", "", "º."]))
    for m in UF_SLASH.finditer(text):
        if rng.random() < p:
            doc.put(m.start(), rng.choice([" / ", "-", " - ", "/ ", "", " "]))
    for i, ch in enumerate(text):
        if doc.in_span[i] or i + 1 >= len(text):
            continue
        if ch == "." and text[i + 1] in " \n" and rng.random() < p / 4:
            doc.put(i, rng.choice(["", "..", " .", ","]))
        elif ch == "," and rng.random() < p / 5:
            doc.put(i, rng.choice(["", ";", " ,", ",,"]))


# ------------------------------------------------------------------ erros de digitação

ROWS = ["qwertyuiop", "asdfghjklç", "zxcvbnm"]
NEIGHBORS: dict[str, str] = {}
for r, row in enumerate(ROWS):
    for c, key in enumerate(row):
        near = [row[j] for j in (c - 1, c + 1) if 0 <= j < len(row)]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(ROWS):
                near += [ROWS[rr][j] for j in (c - 1, c, c + 1) if 0 <= j < len(ROWS[rr])]
        NEIGHBORS[key] = "".join(near)


def typos(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    """Erros humanos em palavras; siglas curtas (STJ, REsp, CPC) ficam intactas."""
    text = doc.orig
    for m in WORD.finditer(text):
        w = m.group()
        if len(w) < 4 or (len(w) <= 5 and sum(c.isupper() for c in w) >= 2) or rng.random() >= p:
            continue
        j = m.start() + rng.randint(1, len(w) - 2)
        kind = rng.choice(["troca", "apaga", "duplica", "vizinha"])
        if kind == "troca" and doc.pristine(j, j + 1):
            a, b = doc.cells[j], doc.cells[j + 1]
            doc.cells[j], doc.cells[j + 1] = b, a
        elif kind == "apaga":
            doc.put(j, "")
        elif kind == "duplica":
            doc.put(j, text[j] * 2)
        elif kind == "vizinha" and text[j].lower() in NEIGHBORS:
            key = rng.choice(NEIGHBORS[text[j].lower()])
            doc.put(j, key.upper() if text[j].isupper() else key)


# ------------------------------------------------------------------ caixa

def caixa(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    text = doc.orig
    starts = doc.line_starts()
    for k, s in enumerate(starts):
        if rng.random() < p / 4:
            e = starts[k + 1] if k + 1 < len(starts) else len(text)
            for i in range(s, e):
                doc.put(i, text[i].upper())
    for m in WORD.finditer(text):
        if rng.random() < p / 3:
            fn = rng.choice([str.upper, str.lower, str.swapcase])
            for i in range(m.start(), m.end()):
                doc.put(i, fn(text[i]))


# ------------------------------------------------------------------ lixo de página

NOMES = ["MARIA APARECIDA SOUZA", "JOÃO CARLOS PEREIRA", "ANA LÚCIA FERREIRA", "PEDRO HENRIQUE ALVES"]
UFS = ["SP", "RJ", "MG", "DF", "RS", "PR", "BA", "SC"]
JUNK = [
    "— {p} —\n",
    "Página {p} de {q}\n",
    "fls. {p}\n",
    "\x0c",
    "Documento assinado eletronicamente por {nome}, OAB/{uf} {oab}. Autenticidade: {hex}\n",
    "Autos nº {cnj}\n",
    "Protocolo nº {ano}.{n7}\n",
    "Tel.: (61) 3{n3}-{n4} — CEP 70.{n3}-000 — Brasília/DF\n",
    "CPF {cpf} — CNPJ {cnpj}\n",
    "______________________________\n",
    "|  ||  ¦ ¦ |\n",
    "[pág. {p}]\n",
    "{p}\n",
    "Valor da causa: R$ {valor}\n",
    "\n\n",
]


def _junk(rng: random.Random) -> str:
    n = lambda k: "".join(rng.choice("0123456789") for _ in range(k))  # noqa: E731
    return rng.choice(JUNK).format(
        p=rng.randint(2, 80), q=rng.randint(80, 200), nome=rng.choice(NOMES), uf=rng.choice(UFS),
        oab=f"{n(3)}.{n(3)}", hex=f"{rng.getrandbits(40):010X}", ano=rng.randint(2015, 2025), n7=n(7),
        n3=n(3), n4=n(4), cpf=f"{n(3)}.{n(3)}.{n(3)}-{n(2)}", cnpj=f"{n(2)}.{n(3)}.{n(3)}/0001-{n(2)}",
        cnj=f"{n(7)}-{n(2)}.{rng.randint(2010, 2025)}.{rng.randint(1, 8)}.{n(2)}.{n(4)}",
        valor=f"{rng.randint(1, 999)}.{n(3)},{n(2)}",
    )


def lixo(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    """Cabeçalho/rodapé de página, carimbos, distratores numéricos e numeração de linha."""
    numbered = rng.random() < p  # peças com numeração de linha na margem
    for k, s in enumerate(doc.line_starts()):
        if doc.gap_in_span[s]:
            continue
        if rng.random() < p / 3:
            doc.insert(s, "".join(_junk(rng) for _ in range(rng.randint(1, 3))), outside=True)
        if numbered and s > 0:
            doc.insert(s, f"{k + 1:>3}  ", outside=True)


# ------------------------------------------------------------------ codificação

def _mojibake(ch: str) -> str:
    raw = ch.encode("utf-8")
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def mojibake(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    """UTF-8 lido como cp1252: "ção" → "Ã§Ã£o", "nº" → "nÂº"."""
    whole = rng.random() < p / 3
    for m in re.finditer(r"\S+", doc.orig):
        if whole or rng.random() < p:
            for i in range(m.start(), m.end()):
                if ord(doc.orig[i]) > 127:
                    doc.put(i, _mojibake(doc.orig[i]))


INVISIBLE = ["​", "‌", "‍", "⁠", "﻿", "­"]


def invisiveis(doc: NoisyDoc, rng: random.Random, p: float) -> None:
    """Caracteres de largura zero em qualquer lugar, inclusive no meio de números."""
    for i in range(1, len(doc.orig)):
        if rng.random() < p / 20:
            doc.insert(i, rng.choice(INVISIBLE))


# Ordem de aplicação: as estruturais primeiro; `pristine` impede que duas
# famílias alterem o mesmo caractere.
FAMILIES = {
    "fim_de_linha": fim_de_linha,
    "lixo": lixo,
    "quebras": quebras,
    "espacos": espacos,
    "pontuacao": pontuacao,
    "typos": typos,
    "caixa": caixa,
    "ocr": ocr,
    "acentos": acentos,
    "mojibake": mojibake,
    "invisiveis": invisiveis,
}

"""Ruído de digitalização: OCR de caracteres, poluição da citação e lixo de página.

`ocr_noise` é o ruído do nível 2 em todos os perfis. `pollute_citation` e
`page_junk` são a poluição v1; `pollute_v4` e `page_junk_v4` foram escritos
junto com o holdout v4, antes de qualquer ajuste do pipeline para eles.
"""

from __future__ import annotations

import random
import re
import unicodedata

OCR_SWAPS = {
    "0": "O", "1": "l", "5": "S", "9": "g", "6": "G", "8": "B",
    "e": "c", "a": "ã", "i": "l", "S": "5", "o": "0", "m": "rn",
}


def ocr_noise(rng: random.Random, text: str, rate: float) -> str:
    out = []
    for i, ch in enumerate(text):
        r = rng.random()
        if ch in OCR_SWAPS and r < rate and 0 < i < len(text) - 1:
            out.append(OCR_SWAPS[ch])
        elif ch == " " and r < rate / 2:
            out.append(rng.choice(["  ", "\n", "\u00a0"]))  # espaço duplo, quebra, NBSP
        elif ch == "." and r < rate:
            out.append(rng.choice([". ", "", "- ."]))
        else:
            out.append(ch)
    return "".join(out)


# ------------------------------------------------------------------ v1

ZERO_WIDTH = ["\u200b", "\u00ad", "\ufeff"]  # zero-width space, soft hyphen, BOM
ODD_SPACES = ["\u00a0", "\u2009", "\u202f", "\t", "  "]  # NBSP, thin space, narrow NBSP


def pollute_citation(rng: random.Random, text: str, rate: float) -> str:
    """Poluição de digitalização DENTRO da citação (além do OCR de caracteres)."""
    out = text
    r = rng.random
    if r() < rate:  # hifenização de fim de linha no meio de uma palavra longa
        words = [m for m in re.finditer(r"[A-Za-zÀ-ÿ]{6,}", out)]
        if words:
            w = rng.choice(words)
            cut = w.start() + rng.randint(2, len(w.group()) - 2)
            out = out[:cut] + "-\n" + out[cut:]
    if r() < rate:  # caractere invisível
        pos = rng.randrange(1, len(out))
        out = out[:pos] + rng.choice(ZERO_WIDTH) + out[pos:]
    if r() < rate:  # espaços estranhos
        out = re.sub(" ", lambda _: rng.choice(ODD_SPACES) if r() < 0.4 else " ", out)
    if r() < rate / 2:  # caixa alta
        out = out.upper()
    if r() < rate / 2:  # pontuação duplicada
        out = re.sub(r"([.,])", lambda m: m.group(1) * 2 if r() < 0.3 else m.group(1), out)
    if r() < rate / 2:  # quebra de linha dupla entre tokens
        out = out.replace(" ", "\n\n", 1)
    return out


PAGE_JUNK = [
    "\n\n— {p} —\n\n",
    "\nPODER JUDICIÁRIO\nDocumento assinado eletronicamente. Autenticidade: {code}\n",
    "\nPágina {p} de {q}\n",
    "\n________________________________________\n",
    "\n|||  ||| ¦ ¦\n",
    "\nfls. {p}\n\x0c\n",
    "\nRua dos Tribunais, {p} – CEP 70.{q}0-000 – Brasília/DF\n",
]


def page_junk(rng: random.Random) -> str:
    return rng.choice(PAGE_JUNK).format(
        p=rng.randint(2, 40), q=rng.randint(40, 99), code=f"{rng.getrandbits(40):010X}"
    )


# ------------------------------------------------------------------ v4 (holdout)

def pollute_v4(rng: random.Random, text: str, rate: float) -> str:
    """Defeitos de extração de PDF que a poluição v1 não produz."""
    out = text
    r = rng.random
    if r() < rate:  # acentos decompostos (NFD), comum em PDF gerado no macOS
        out = unicodedata.normalize("NFD", out)
    if r() < rate:  # quebra de linha logo após um ponto dentro do número
        dots = [m.end() for m in re.finditer(r"\d\.(?=\d)", out)]
        if dots:
            pos = rng.choice(dots)
            out = out[:pos] + "\n" + out[pos:]
    if r() < rate / 2:  # palavra partida por espaço ("Recur so")
        words = [m for m in re.finditer(r"[A-Za-zÀ-ÿ]{7,}", out)]
        if words:
            w = rng.choice(words)
            cut = w.start() + rng.randint(3, len(w.group()) - 3)
            out = out[:cut] + " " + out[cut:]
    if r() < rate:  # perda de acentos e do ordinal
        out = out.translate(str.maketrans("çãáéíóúâêôõºª°", "caaeiouaeooooo"))
    if r() < rate / 2:  # traço de hifenização "¬" do OCR
        words = [m for m in re.finditer(r"[A-Za-zÀ-ÿ]{6,}", out)]
        if words:
            w = rng.choice(words)
            cut = w.start() + rng.randint(2, len(w.group()) - 2)
            out = out[:cut] + "¬\n" + out[cut:]
    if r() < rate / 2:  # espaço antes da pontuação
        out = re.sub(r"([,/])", lambda m: " " + m.group(1) if r() < 0.5 else m.group(1), out)
    return out


PAGE_JUNK_V4 = [
    "\n[pág. {p}]\n",
    "\nEste documento foi assinado digitalmente por {q}{p}. Código {code}.\n",
    "\n* * *\n",
    "\n{p}\n\n",
    "\nSupremo Tribunal Federal — Coordenadoria de Processamento — fl. {p}\n",
]


def page_junk_v4(rng: random.Random) -> str:
    return rng.choice(PAGE_JUNK_V4).format(
        p=rng.randint(2, 60), q=rng.randint(10, 99), code=f"{rng.getrandbits(32):08x}"
    )

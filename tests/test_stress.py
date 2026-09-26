import random
import tempfile
import unittest
from pathlib import Path

from tools.stress import DEV_GOLD, DEV_TXT, PROFILES, generate, noisy_document, read_text
from tools.stress.noise import FAMILIES, NoisyDoc

TEXT = "Ver o AgInt no REsp nº 1.234.567/SP e a Súmula 83 do STJ.\nNo mérito, art. 5º da Constituição Federal.\n"
CITATIONS = ["AgInt no REsp nº 1.234.567/SP", "Súmula 83 do STJ", "art. 5º da Constituição Federal"]
SPANS = [(TEXT.index(c), TEXT.index(c) + len(c)) for c in CITATIONS]


class NoisyDocTest(unittest.TestCase):
    def test_boundary_insertions_stay_outside(self):
        doc = NoisyDoc("ab REsp 12 cd", [(3, 10)])
        doc.insert(3, "<")    # antes do início: fora
        doc.insert(10, ">")   # depois do fim: fora
        doc.insert(7, "​")  # no meio: dentro
        doc.put(8, "l")       # OCR no número
        text, starts, ends = doc.render()
        self.assertEqual(text[starts[3]:ends[9]], "REsp​ l2")

    def test_deleted_char_keeps_span(self):
        doc = NoisyDoc("x REsp 12 y", [(2, 9)])
        doc.put(6, "")
        text, starts, ends = doc.render()
        self.assertEqual(text[starts[2]:ends[8]], "REsp12")


class FamiliesTest(unittest.TestCase):
    def test_every_family_keeps_valid_spans_and_digits(self):
        for family in FAMILIES:
            for seed in range(20):
                rng = random.Random(seed)
                doc = NoisyDoc(TEXT, SPANS)
                FAMILIES[family](doc, rng, 1.0)
                text, starts, ends = doc.render()
                for s, e in SPANS:
                    self.assertLess(starts[s], ends[e - 1], family)
                for i, ch in enumerate(TEXT):
                    cell = doc.cells[i]
                    if ch.isdigit() and cell != ch:
                        # dígito só vira letra parecida (ou separador), nunca outro dígito
                        self.assertFalse(any(c.isdigit() for c in cell), (family, ch, cell))

    def test_page_junk_never_enters_a_citation(self):
        for seed in range(20):
            text, spans, _ = noisy_document(TEXT, SPANS, {"lixo": 1.0}, random.Random(seed))
            self.assertEqual([text[a:b] for a, b in spans], [TEXT[s:e] for s, e in SPANS])


class GenerateTest(unittest.TestCase):
    def test_clean_profile_reproduces_dev(self):
        self.assertEqual(PROFILES["limpo"], {})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = generate("limpo", 1, out)
            self.assertEqual(manifest["taxa_edicao"], 0.0)
            for path in sorted(DEV_TXT.glob("*.txt"))[:3]:
                self.assertEqual(read_text(out / "txt" / path.name), read_text(path))
            self.assertEqual(read_text(out / "goldenset.csv").count("\n"), read_text(DEV_GOLD).count("\n"))

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate("extremo", 7, Path(a))
            generate("extremo", 7, Path(b))
            for path in sorted((Path(a) / "txt").glob("*.txt"))[:5]:
                self.assertEqual(read_text(path), read_text(Path(b) / "txt" / path.name))


if __name__ == "__main__":
    unittest.main()

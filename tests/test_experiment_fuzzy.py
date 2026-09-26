import importlib.util
import unittest

HAS_REGEX = importlib.util.find_spec("regex") is not None
if HAS_REGEX:
    from src.experiment_fuzzy import compile_patterns, extract_fuzzy


@unittest.skipUnless(HAS_REGEX, "Instale requirements-fuzzy.txt para testar o experimento")
class FuzzyExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patterns = compile_patterns("text", 1)

    def test_one_substitution_insertion_or_deletion_in_keyword(self):
        for word, counts in [
            ("5úmula", (1, 0, 0)),
            ("Súmmula", (0, 1, 0)),
            ("Súmla", (0, 0, 1)),
        ]:
            citation = f"{word} 987 do STJ"
            text = "⚖️ Referência: " + citation + "."
            with self.subTest(word=word):
                results = extract_fuzzy(text, self.patterns, 1)
                self.assertEqual(len(results), 1)
                result = results[0]
                self.assertEqual(result["trecho"], citation)
                self.assertEqual(text[result["inicio"]:result["fim"]], citation)
                self.assertEqual(tuple(result["erros"].values()), counts)

    def test_localized_fuzziness_does_not_relax_identifiers(self):
        for text in ["de 2025", "Súmula 98O do STJ", "Súmula 987 do STX"]:
            with self.subTest(text=text):
                results = extract_fuzzy(text, self.patterns, 1)
                # Tribunal é opcional na regra original, portanto STX pode
                # ficar fora da captura, mas não deve ser aceito como STJ.
                self.assertFalse(any(result["trecho"] == text for result in results))
                if text != "Súmula 987 do STX":
                    self.assertEqual(results, [])

    def test_total_budget_across_two_fragments(self):
        text = "art. 123 da Constituiçãx Fedcral"
        self.assertEqual(extract_fuzzy(text, self.patterns, 1), [])
        results = extract_fuzzy(text, self.patterns, 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["trecho"], text)
        self.assertEqual(sum(results[0]["erros"].values()), 2)


if __name__ == "__main__":
    unittest.main()

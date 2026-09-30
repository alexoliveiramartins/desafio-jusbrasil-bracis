import unittest
from pathlib import Path

from src.classify import CanonicalIndex
from src.main import process_text
from src.nlp import (NLPLayer, align, canonical, digits_in_span, number_groups, skeleton, trim_article, trim_process,
                     trim_sumula)

DB = Path("data/desafio1_bracis.db")


class FakeClient:
    """Responde sempre as mesmas citações, como o modelo responderia."""

    def __init__(self, citacoes):
        self.citacoes = citacoes

    def chat_json(self, system, user, schema):
        return {"citacoes": self.citacoes}


def item(trecho, tipo="processo", **fields):
    return {"trecho": trecho, "tipo": tipo, "tribunal": None, "classe": None, "numero": None} | fields


class GuardTest(unittest.TestCase):
    def test_letter_for_digit_only(self):
        self.assertTrue(digits_in_span("1741784", "REsp 1.74l.7B4/PR"))
        self.assertTrue(digits_in_span("1741784", "REsp 1741784 - PR"))
        # Dígito trocado por outro dígito: o modelo "corrigiu" um número inventado.
        self.assertFalse(digits_in_span("1741784", "REsp 1.741.785/PR"))

    def test_no_truncated_number(self):
        # O modelo leu só um pedaço do CNJ: sobram dígitos em volta.
        self.assertFalse(digits_in_span("000018470", "REspe n. 0000184-70 2012 6 20 0033"))
        self.assertTrue(digits_in_span("00001847020126200033", "REspe n. 0000184-70 2012 G 2O\n0033"))

    def test_canonical_prefers_span_evidence(self):
        # Tribunal escrito no trecho prevalece sobre o que o modelo disse.
        family, forms = canonical(item("REsp 1.741.784/PR (STJ)", tribunal="STF", classe="REsp",
                                       numero="1741784", uf="PR"), "REsp 1.741.784/PR (STJ)")
        self.assertEqual(family, "processos")
        self.assertEqual(forms, ["REsp nº 1.741.784/PR (STJ)"])
        self.assertIsNone(canonical(item("art. 5º da CF", "artigo", artigo="6", diploma="CF"), "art. 5º da CF"))


class NumberGroupsTest(unittest.TestCase):
    def test_glued_ocr_numbers(self):
        cases = {
            "AgInt7S57430-50.2018.7.00.0000 / DF": ["75574305020187000000"],  # colado na classe
            "7000449-4O,2023 . 7,00,0000 - RS": ["70004494020237000000"],     # vírgula como separador
            "Hqbcas Corpus no 9396\u200c7": ["93967"],                         # invisível no meio
            "RcI\t88.86OPE": ["88860"],                                        # UF colada (PE)
            "Rcl 77.14oSP, que": ["77140"],                                    # UF colada (SP)
            "APL 7Oo07b1-84 Z0Z1 7 00 0000 - 8A": ["70007618420217000000"],   # "8A" é BA, não dígito
            "Rcl 33.235/RJ, de 2021": ["33235", "2021"],                       # vírgula com espaço separa
            "RE5P n. 1.234": ["1234"],                                         # dígito dentro de palavra
            "Rcl no 685": ["685"],                                             # "no" não vira "0"
        }
        for text, groups in cases.items():
            self.assertEqual(number_groups(text), groups, text)

    def test_digit_glued_to_class_starts_number(self):
        self.assertEqual(number_groups("RESP6 .q89.q16 - RS"), ["6989916"])  # "RESP" + 6.989.916
        self.assertEqual(number_groups("RE5P n. 1.234"), ["1234"])              # o 5 é o S de "RESP"

    def test_letter_only_number(self):
        self.assertTrue(digits_in_span("188", "Sumula Viriculantc lBB"))
        self.assertFalse(digits_in_span("0000005", "RESP 0000005/98.2O12.6.08.0018 - TSE"))

    def test_trim_paragraph_to_article(self):
        text = ("A sentença reconheceu a prescrição parcial da pretensão e julgou parcialmente procedentes os "
                "pedidos. Cumpre lembrar o art.   1 105 clo Codigo de PROCESSO\nC1vil, de aplicação cogente")
        start, end = trim_article(text, 0, len(text), "1105")
        self.assertEqual(text[start:end], "art.   1 105 clo Codigo de PROCESSO\nC1vil")

    def test_trim_sumula_with_mojibake_and_nfd(self):
        import unicodedata
        text = "Reforçao argurneento a   5Ãºmula331\xa0do TST,   de\r\n 56  RESTO AMPLAMENTE CONHECIDA"
        start, end = trim_sumula(text, 0, len(text), "331")
        self.assertEqual(text[start:end], "5Ãºmula331\xa0do TST")
        text = unicodedata.normalize("NFD", "a Súmnla 168do\xa0TSE, no qual a Corte firmou orientação pacificada")
        start, end = trim_sumula(text, 0, len(text), "168")
        self.assertEqual(unicodedata.normalize("NFC", text[start:end]), "Súmnla 168do\xa0TSE")

    def test_trim_sentence_to_citation(self):
        text = "Merece regist-\n\nro  o  AgIut nos EDcl uoREsp 214499s / RJ, invocaado descle a peça"
        start, end = trim_process(text, 0, len(text), "")
        self.assertEqual(text[start:end], "AgIut nos EDcl uoREsp 214499s / RJ")


class EvidenceTest(unittest.TestCase):
    def test_heavy_ocr_evidence(self):
        from src.nlp import _class_evidence, _has_evidence
        self.assertTrue(_has_evidence(["vinculante"], "Sumula Viriculantc lBB"))   # "ri" lido por "n"
        self.assertTrue(_class_evidence("RHC", "Reccurso em Hqbcas Corpus"))     # semelhança, não substring
        self.assertFalse(_class_evidence("REsp", "apelo nobre"))
        self.assertFalse(_class_evidence("RE", "Reclamação 1"))

    def test_number_words_blockers_and_unknown_qualifiers(self):
        from src.nlp import _number_after_citation_words as ok
        self.assertTrue(ok("Agravo Regimental no Reclamação Constitucional n. 64.895 – SP", "64895"))  # "Constitucional"
        self.assertTrue(ok("Recurso Especial Cível\nNo 1 925 856/SP. – STJ", "1925856"))                # "Cível"
        self.assertFalse(ok("O STJ, em 2025, editou a Resolução nº 188", "188"))
        self.assertFalse(ok("Recurso ordinário nº de ordem 295", "295"))
        self.assertFalse(ok("fls. 580/845", "580845"))

    def test_tema_and_nicknamed_laws(self):
        self.assertEqual(canonical({"tipo": "tema", "numero": "591"}, "tema de repercussão geral nº 591"),
                         ("temas", ["Tema"]))
        self.assertIsNone(canonical({"tipo": "tema", "numero": "8112"}, "Incidem as regras da Lei nº 8.112/90"))
        self.assertEqual(canonical({"tipo": "artigo", "artigo": "477"}, "art 477 do diploma consolidado"),
                         ("artigos", ["art. 477 da CLT"]))
        self.assertEqual(canonical({"tipo": "artigo", "artigo": "128", "diploma": "OUTRA", "lei": "8.245/1991"},
                                   "art.\n128 da Lei do Inquilinato"), ("artigos", ["art. 128 da Lei nº 8.245/1991"]))
        # O modelo diz que a "Lei do Inquilinato" é o CDC (que está na base): sem evidência do CDC, não decide.
        self.assertIsNone(canonical({"tipo": "artigo", "artigo": "14", "diploma": "OUTRA", "lei": "8.078/1990"},
                                    "art. 14 da Lei do Inquilinato"))

    def test_article_ocr(self):
        family, forms = canonical(item("art. 75daLei CompIemenar no 64/1990", "artigo", artigo="75"),
                                  "art. 75daLei CompIemenar no 64/1990")
        self.assertEqual(forms, ["art. 75 da Lei Complementar nº 64/1990"])  # "CompIemenar": evidência com OCR
        family, forms = canonical(item("art. l da Lei Complementar nº 64/19g0", "artigo", artigo="1",
                                       diploma="LC64/1990"), "art. l da Lei Complementar nº 64/19g0")
        self.assertEqual(forms, ["art. 1 da Lei Complementar nº 64/1990"])


class AlignTest(unittest.TestCase):
    def test_ocr_and_whitespace(self):
        text = "Nesse sentido, o Rec. Esp. nº  1. 57O.531 – CE, de clareza solar."
        sk, index = skeleton(text)
        (start, end), = align(sk, index, "Rec. Esp. nº 1.570.531 - CE")
        self.assertEqual(text[start:end], "Rec. Esp. nº  1. 57O.531 – CE")

    def test_all_occurrences(self):
        text = "Súmula 83 do STJ. Adiante, de novo a Súmula 83 do STJ."
        sk, index = skeleton(text)
        self.assertEqual(len(align(sk, index, "Súmula 83 do STJ")), 2)

    def test_short_strings_are_ignored(self):
        sk, index = skeleton("art. 5 da CF")
        self.assertEqual(align(sk, index, "5"), [])


class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = CanonicalIndex.from_sqlite(DB)

    def run_layer(self, text, citacoes, **kwargs):
        layer = NLPLayer(FakeClient(citacoes), **kwargs)
        return process_text("t", text, self.index, debug=True, nlp=layer)["citacoes"]

    def test_invented_number_never_becomes_real(self):
        # REsp 1.741.784/PR existe; 1.741.785 não. O modelo "corrige" o dígito: nada muda.
        text = "Cumpre destacar o REsp 1.741.785/PR, de clareza solar."
        out = self.run_layer(text, [item("REsp 1.741.785/PR", classe="REsp", numero="1741784", uf="PR",
                                         tribunal="STJ")])
        self.assertEqual([c["classificacao"] for c in out], ["inventada"])

    def test_new_citation_from_model(self):
        # Forma que as regras não conhecem; o modelo acha e a base resolve.
        text = "Veja-se o decidido pelo STJ no feito nº 1741784, oriundo do Paraná, em caso idêntico."
        spans_only = process_text("t", text, self.index)["citacoes"]
        out = self.run_layer(text, [item("STJ no feito nº 1741784, oriundo do Paraná", classe="REsp",
                                         numero="1741784", uf="PR", tribunal="STJ")])
        self.assertEqual(spans_only, [])
        self.assertEqual([(c["classificacao"], (c["resolucao"] or {}).get("id_canonico")) for c in out],
                         [("real", "2566535283")])

    def test_class_disagreement_is_not_decided(self):
        # "apelo nobre" parece APL para a detecção aproximada; o modelo diz REsp: sem acordo, nada entra como real.
        text = "Veja-se o decidido no apelo nobre de número 1741784, oriundo do Paraná, em caso idêntico."
        out = self.run_layer(text, [item("apelo nobre de número 1741784, oriundo do Paraná", classe="REsp",
                                         numero="1741784", uf="PR", tribunal="STJ")])
        self.assertNotIn("real", [c["classificacao"] for c in out])

    def test_header_number_is_not_a_citation(self):
        text = "EGRÉGIO TRIBUNAL\n\nAutos nº 1051967-37.2016.3.02.7661\n\n" + "Texto da peça. " * 10
        out = self.run_layer(text, [item("1051967-37.2016.3.02.7661", numero="10519673720163027661")])
        self.assertEqual(out, [])

    def test_own_autos_number_is_not_a_citation(self):
        # Linha de cabeçalho com ruído (número de margem) antes de "Autos": continua sendo distrator.
        text = "Texto longo da peça, com prosa suficiente para não ser cabeçalho, e mais texto aqui.\n12   Autos nº 0872233-87.2010.7.76.6271\n"
        out = self.run_layer(text, [item("Autos nº 0872233-87.2010.7.76.6271", numero="08722338720107766271")])
        self.assertEqual(out, [])

    def test_number_must_follow_citation_words(self):
        # "Resolução nº 188" e "nº de ordem 295" têm número e tribunal/classe por perto, mas não são citação.
        text = "O STJ, em 2025, editou a Resolução nº 188. O Recurso ordinário nº de ordem 295 foi juntado."
        out = self.run_layer(text, [item("O STJ, em 2025, editou a Resolução nº 188", tribunal="STJ", numero="188"),
                                    item("Recurso ordinário nº de ordem 295", classe="RO", numero="295")])
        self.assertEqual(out, [])

    def test_two_citations_in_one_model_item_do_not_share_numbers(self):
        # O modelo junta "Súmula 3 e Súmula 211": a Súmula 3 (inexistente) não pode herdar o 211 da vizinha.
        text = "Vide STJ - Súmula 3 e Súmula 211 do Egrégio STJ, no ponto."
        out = self.run_layer(text, [item("Vide STJ - Súmula 3 e Súmula 211 do Egrégio STJ", "sumula", numero="211",
                                         tribunal="STJ")])
        by_text = {c["trecho"]: c["classificacao"] for c in out}
        self.assertNotEqual(by_text.get("STJ - Súmula 3"), "real")

    def test_second_pass_rereads_uncovered_sentences(self):
        from src.nlp import uncovered_windows
        text = ("EGRÉGIO TRIBUNAL\n\nAutos nº 1051967-37.2016.3.02.7661\n\n"
                "A matéria foi decidida no REsp 1.741.784/PR, sem ressalvas. Também no feito tombado sob 1.234.567, "
                "oriundo do Paraná. Em 12/03/2020, com valor de R$ 155.135,93 e fls. 234/567, nada mudou.")
        rules = process_text("t", text, self.index)["citacoes"]
        windows = uncovered_windows(text, rules, 70)
        self.assertEqual(len(windows), 1)
        self.assertIn("tombado sob 1.234.567", windows[0])  # número sem citação que o cubra
        self.assertNotIn("R$", windows[0])                   # data, valor, folhas e autos não são pista

    def test_ensemble_disagreement_keeps_rules(self):
        # Dois modelos leem o mesmo número de dois jeitos que levam a registros diferentes: nada muda.
        text = "Cumpre destacar o REsp 1.741.785/PR, de clareza solar."
        layer = NLPLayer(FakeClient([item("REsp 1.741.785/PR", classe="REsp", numero="1741785", uf="PR")]),
                         extra_clients=(FakeClient([item("REsp 1.741.785/PR", classe="REsp", numero="1741785")]),))
        out = process_text("t", text, self.index, debug=True, nlp=layer)["citacoes"]
        self.assertEqual([c["classificacao"] for c in out], ["inventada"])

    def test_recall_off(self):
        text = "Veja-se o decidido no apelo nobre de número 1741784, oriundo do Paraná, em caso idêntico."
        out = self.run_layer(text, [item("apelo nobre de número 1741784", classe="REsp", numero="1741784")],
                             recall=False)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()


class SlowClient(FakeClient):
    """Modelo lento (ex.: contêiner sem GPU): cada leitura "custa" 100 s."""

    def __init__(self, citacoes, clock):
        super().__init__(citacoes)
        self.clock = clock

    def chat_json(self, system, user, schema):
        self.clock[0] += 100.0
        return super().chat_json(system, user, schema)


class ComplianceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = CanonicalIndex.from_sqlite(DB)

    def test_cnj_completed_from_span(self):
        span = "RESP 0000005/98.2O12.6.08.0018 - TSE"
        family, forms = canonical(item(span, classe="REsp", numero="0000005", uf="98"), span)
        self.assertEqual(forms, ["REsp nº 0000005-98.2012.6.08.0018 (TSE)"])

    def test_time_budget_turns_layer_off(self):
        import src.nlp as nlp
        clock = [0.0]
        original = nlp.time.monotonic
        nlp.time.monotonic = lambda: clock[0]
        try:
            layer = NLPLayer(SlowClient([], clock), budget_doc_s=40.0, warmup_docs=3)
            for _ in range(5):
                layer.refine("texto sem citação", [], self.index)
        finally:
            nlp.time.monotonic = original
        self.assertTrue(layer.disabled)
        self.assertEqual(layer.docs, 4)  # 1º doc fora da média + 3 de aquecimento; depois, não consulta mais

    def test_cold_start_does_not_turn_layer_off(self):
        """Partida a frio lenta (carregar os modelos na GPU) não desliga a camada (execução de 29/09)."""
        import src.nlp as nlp
        clock = [0.0]
        durations = iter([101.6, 20.1, 16.5, 21.0, 19.0, 22.0])

        class ColdStart(SlowClient):
            def chat_json(self, system, user, schema):
                self.clock[0] += next(durations)
                return FakeClient.chat_json(self, system, user, schema)

        original = nlp.time.monotonic
        nlp.time.monotonic = lambda: clock[0]
        try:
            layer = NLPLayer(ColdStart([], clock), budget_doc_s=40.0, warmup_docs=3)
            for _ in range(6):
                layer.refine("texto sem citação", [], self.index)
        finally:
            nlp.time.monotonic = original
        self.assertFalse(layer.disabled)
        self.assertEqual(layer.docs, 6)

    def test_only_local_model_server(self):
        from src.nlp import OllamaClient, from_args
        with self.assertRaises(ValueError):
            OllamaClient("http://api.exemplo.com:11434")
        self.assertIsNone(from_args(True, "http://10.0.0.5:11434", "m", None))  # cai para as regras

    def test_launcher_without_weights_runs_rules(self):
        import os
        import tempfile
        from src import launcher
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as empty:
            os.environ["CACA_MODEL_DIR"] = empty
            try:
                code = launcher.main(["--input", "data/txt", "--output", out])
            finally:
                del os.environ["CACA_MODEL_DIR"]
            self.assertEqual(code, 0)
            self.assertEqual(len(list(Path(out).glob("*.json"))), 26)

    def test_image_ships_every_modelfile(self):
        """Cada Modelfile do manifesto entra na imagem (senão o modelo do conjunto some sem erro)."""
        import json
        import re
        manifest = json.loads(Path("model_manifest.json").read_text(encoding="utf-8"))
        copied = re.findall(r"^COPY\s+(?!--)(\S+)\s", Path("Dockerfile").read_text(encoding="utf-8"), re.M)
        for modelfile in [manifest["runtime"]["modelfile"]] + [m["modelfile"] for m in manifest["conjunto"]]:
            self.assertTrue(Path(modelfile).is_file(), modelfile)
            self.assertTrue(any(modelfile == c or (c.endswith("/") and modelfile.startswith(c)) for c in copied),
                            f"{modelfile} fora dos COPY do Dockerfile")

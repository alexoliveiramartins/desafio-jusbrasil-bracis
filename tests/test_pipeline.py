"""Testes do pipeline só com regras: normalização, extração, resolução, âncoras, robustez e contrato."""

import unittest
from pathlib import Path

from src.classify import CanonicalIndex
from src.normalize import appeal_chain, class_code, cnj_key, diploma_key, number_digits, ocr_fix_words
from src.spans import clean, extract_citations
from src.main import process_text
from src.classify import resolve

DB = Path("data/desafio1_bracis.db")


class NormalizeTest(unittest.TestCase):
    """Normalização de números, chaves CNJ, diplomas, cadeia recursal e classe processual."""

    def test_ocr_digits(self):
        """Letras de OCR dentro do número viram dígitos ("21737l8" -> 2173718)."""
        self.assertEqual(number_digits("AgInt no RESP 21737l8 - SP"), ("2173718", 1))
        self.assertEqual(number_digits("AgRg no RESP 1.528.4S5/ RJ")[0], "1528455")
        self.assertEqual(number_digits("Rec. Esp. No 1.880.529 - SP")[0], "1880529")

    def test_uf_is_not_a_digit(self):
        # "SP" não pode virar "5P": só letras dentro de tokens numéricos são trocadas.
        """A UF depois da barra não vira dígito ("SP" não é "5P")."""
        self.assertEqual(number_digits("REsp 1.883.715/SP")[0], "1883715")

    def test_cnj_without_separators(self):
        """CNJ sem separadores e com separadores dão a mesma chave."""
        self.assertEqual(cnj_key("0600316-4920206160182"), cnj_key("600316-49.2020.6.16.0182"))

    def test_diploma_ocr(self):
        """Diploma com OCR, pelo número da lei ou pela sigla."""
        self.assertEqual(diploma_key("Constituição Fedcral"), "CF")
        self.assertEqual(diploma_key("Lei nº 13.105/2015"), "CPC")
        self.assertEqual(diploma_key("Código Penal"), "CP")

    def test_appeal_chain(self):
        """Cadeia recursal lida de fora para dentro, também com OCR ("Agrãv0 Regimental")."""
        self.assertEqual(appeal_chain("EDcl no AgInt no REsp"), ("AGINT", "ED"))
        self.assertEqual(appeal_chain("Agrãv0 Regimental no Rcl"), ("AGRG",))

    def test_class_code(self):
        """Classe processual por extenso ou com OCR; "Reclamação" não vira RE."""
        self.assertEqual(class_code("AgInt no AGRAVO EM RECURSO ESPECIAL"), "AREsp")
        self.assertEqual(class_code("Reclamação"), "Rcl")  # não confundir com RE
        self.assertEqual(class_code("Recurs0 Especial Eleitoral"), "REspe")

    def test_ocr_vocabulary(self):
        """Palavra do vocabulário jurídico com OCR é corrigida ("Súrnula" -> "sumula")."""
        self.assertEqual(ocr_fix_words("Súrnula Vinculante"), "sumula vinculante")


    def test_chain_distance_uses_sets_first(self):
        """A distância entre cadeias compara primeiro os conjuntos de recursos, depois as repetições."""
        from src.normalize import chain_distance
        cited = ("AGINT", "EDV")
        self.assertLess(chain_distance(cited, ("AGINT", "EDV", "EDV")), chain_distance(cited, ("AGINT",)))

    def test_isolated_ocr_digit_inside_cnj(self):
        """Letra de OCR isolada num bloco do CNJ ("S", "O") também vira dígito."""
        self.assertEqual(number_digits("TST-RR-2494S-56.2015.S.24.O091")[0], "24945562015524" + "0091")

    def test_comma_thousands_is_not_truncated(self):
        # "87,020" truncado em 87 casaria com a AR nº 87 (τ).
        """Vírgula de milhar não trunca o número ("87,020" não é 87)."""
        self.assertEqual(number_digits("Reclamação nº 87,020")[0], "87020")
        self.assertEqual(number_digits("RMS 87, julgado")[0], "87")

    def test_article_number_is_not_truncated(self):
        """Número de artigo com OCR ou em caixa alta é lido inteiro."""
        from src.normalize import article_number
        self.assertEqual(article_number("art. 1S0, § 1º da Lei Complementar nº 64/1990"), 150)
        self.assertEqual(article_number("ARTIGO 477 DA CLT"), 477)

    def test_law_header_names(self):
        """Nome completo da lei, com data, leva ao diploma (Decreto-Lei 5.452 -> CLT)."""
        self.assertEqual(diploma_key("Decreto-Lei nº 5.452, de 1º de maio de 1943"), "CLT")
        self.assertEqual(diploma_key("Lei nº 4.737, de 15 de julho de 1965"), "CE")
        self.assertEqual(diploma_key("Constituição Federal de 1988"), "CF")


class ExtractorTest(unittest.TestCase):
    """Padrões catalogados do regex."""

    def test_new_surface_forms(self):
        """Formas catalogadas menos comuns são extraídas com o trecho exato."""
        for text in ["Enunciado 83 da Súmula do STJ", "Súmula nº 83/STJ", "CLT, art. 818",
                     "REspe n.º 0043847-34.2009.6.00.0000", "PExt no RHC nº 90.861/RS",
                     "decisão do STJ de 2023, relatoria do Min. Herman Benjamin"]:
            [c] = extract_citations(f"Ver {text}.")
            self.assertEqual(c["trecho"], text)

    def test_vague_references_are_not_citations(self):
        # Goldenset atual: "artigo correspondente" e "reiterados precedentes" saíram.
        """Referências vagas não são citações (regra do goldenset atual)."""
        self.assertEqual(extract_citations("o artigo correspondente do Código de Processo Civil"), [])
        self.assertEqual(extract_citations("reiterados precedentes do STJ"), [])

    def test_ocr_keywords(self):
        """Palavra-chave com OCR ("5úmula") é reconhecida."""
        spans = [c["trecho"] for c in extract_citations("Conforme a 5úmula 211 do STJ, nego.")]
        self.assertEqual(spans, ["5úmula 211 do STJ"])

    def test_ordinal_appeal(self):
        # O ordinal de recurso repetido é parte do nome ("Terceiro AG.REG na Rcl").
        """Ordinal de recurso repetido faz parte da citação; "Segundo o" é prosa."""
        for text in ["Terceiro AG.REG na Rcl nº 62.425/SP", "Segundos EDcl no AgInt no REsp 1.597.443/PR"]:
            [c] = extract_citations(f"Ver {text}.")
            self.assertEqual(c["trecho"], text)
        [c] = extract_citations("Segundo o REsp 1.883.715/SP, nego.")
        self.assertEqual(c["trecho"], "REsp 1.883.715/SP")

    def test_offsets_are_codepoints(self):
        """Os offsets são em codepoints do texto, também depois de caracteres não ASCII."""
        text = "Ação — ver REsp 1.883.715/SP."
        [citation] = extract_citations(text)
        self.assertEqual(text[citation["inicio"]:citation["fim"]], citation["trecho"])


@unittest.skipUnless(DB.exists(), "base canônica ausente")
class ResolverTest(unittest.TestCase):
    """Resolução contra a base canônica."""

    @classmethod
    def setUpClass(cls):
        """Monta o índice da base uma vez para a classe."""
        cls.index = CanonicalIndex.from_sqlite(DB)

    def classify(self, text):
        """Extrai a única citação do texto e a resolve.

        Parameters
        ----------
        text : str
            Texto com exatamente uma citação.

        Returns
        -------
        src.classify.Resolution
            Resultado do resolvedor.
        """
        [citation] = extract_citations(text)
        return resolve(citation, self.index)

    def test_real(self):
        """Número com OCR que existe na base: real, com o id_canonico."""
        r = self.classify("AgRg no RESP 1.528.4S5/ RJ")
        self.assertEqual((r.classificacao, r.id_canonico), ("real", "2674928810"))

    def test_same_number_different_stage(self):
        # Mesmo número, fases distintas: a cadeia recursal escolhe o registro.
        """Mesmo número em fases recursais distintas: a cadeia citada escolhe o registro."""
        r = self.classify("AgInt no Recurso Especial nº 1.597.443 - PR")
        self.assertEqual(r.id_canonico, "2684973273")

    def test_inventada(self):
        """Processo ou artigo que a base não tem: inventada."""
        self.assertEqual(self.classify("REsp 6.989.916/RS").classificacao, "inventada")
        self.assertEqual(self.classify("art. 290 da Constituição Federal").classificacao, "inventada")

    def test_descriptive_uses_the_base(self):
        # Tribunal + ano + relator com vários registros: incompleta.
        """Descritiva com vários registros na base: regra incompleta_descritiva."""
        r = self.classify("julgado do STF proferido em 2024 pela relatoria de Dias Toffoli")
        self.assertEqual(r.regra, "incompleta_descritiva")

    def test_reversed_article(self):
        """Artigo na ordem invertida ("CF, art. 5º") resolve ao dispositivo."""
        r = resolve({"trecho": "CF, art. 5º", "familia": "artigos"}, self.index)
        self.assertEqual((r.classificacao, r.id_canonico), ("real", "10641516"))

    def test_relator_mismatch_lowers_confidence(self):
        """Relator citado divergente mantém real, pela regra de atributo divergente."""
        r = resolve({"trecho": "AgRg no REsp 1.528.455/RJ", "familia": "processos",
                     "contexto_depois": ", Rel. Min. Fulano Beltrano, julgado em"}, self.index)
        self.assertEqual(r.classificacao, "real")
        self.assertEqual(r.regra, "real_atributo_divergente")

    def test_incompleta(self):
        """Descritiva ambígua sai incompleta."""
        r = self.classify("julgado do STF proferido em 2024 pela relatoria de Dias Toffoli")
        self.assertEqual(r.classificacao, "incompleta")

    def test_output_contract(self):
        """Cada citação da saída tem exatamente os campos do contrato."""
        doc = process_text("x", "Aplica-se a Súmula 443 do STJ.", self.index)
        [c] = doc["citacoes"]
        self.assertEqual(
            set(c), {"inicio", "fim", "trecho", "tipo", "classificacao", "resolucao", "confianca"}
        )
        self.assertEqual(c["resolucao"], {"id_canonico": "1289711022"})



@unittest.skipUnless(DB.exists(), "base canônica ausente")
class AnchorTest(unittest.TestCase):
    """Âncoras gerais: formas fora do catálogo do regex."""

    @classmethod
    def setUpClass(cls):
        """Monta o índice da base uma vez para a classe."""
        cls.index = CanonicalIndex.from_sqlite(DB)

    def run_doc(self, text):
        """Roda o pipeline (só regras) num texto.

        Parameters
        ----------
        text : str
            Texto do documento.

        Returns
        -------
        list of dict
            Citações da saída.
        """
        return process_text("x", text, self.index)["citacoes"]

    def test_truncated_abbreviations(self):
        """Classe e recurso abreviados por truncamento ("Agr. Int. no Rec. Espec.")."""
        [c] = self.run_doc("Conforme o Agr. Int. no Rec. Espec. núm. 1.599.910/PR, nego.")
        self.assertEqual(c["trecho"], "Agr. Int. no Rec. Espec. núm. 1.599.910/PR")
        self.assertEqual(c["classificacao"], "real")

    def test_court_around_citation(self):
        """Tribunal com honorífico antes da citação entra no span ("C. STJ, …")."""
        [c] = self.run_doc("Veja C. STJ, Rec. em HC n. 57.763/PR, que decidiu a questão.")
        self.assertTrue(c["trecho"].startswith("C. STJ,"))

    def test_article_with_caput_and_synonym(self):
        """Artigo com "caput" e sinônimo do diploma ("Código Civil de 2002")."""
        [c] = self.run_doc("Aplica-se o art. 186, caput, do Código Civil de 2002 ao caso.")
        self.assertEqual(c["trecho"], "art. 186, caput, do Código Civil de 2002")
        self.assertEqual(c["classificacao"], "real")

    def test_old_code_is_another_law(self):
        # CPC/1973 não é o CPC da base: não pode virar real (τ).
        """CPC/1973 não é o CPC da base: nunca real."""
        [c] = self.run_doc("Nos termos do art. 373 do CPC/1973, o ônus é do autor.")
        self.assertNotEqual(c["classificacao"], "real")

    def test_descriptive_any_order(self):
        """Descritiva com tribunal, ano e relator em outra ordem."""
        [c] = self.run_doc("No mesmo sentido, o precedente de 2024 do STF (Rel. Min. Dias Toffoli) afasta a tese.")
        self.assertEqual(c["classificacao"], "incompleta")

    def test_sumula_with_court_name(self):
        """Súmula com o nome do tribunal por extenso."""
        [c] = self.run_doc("Incide a Súmula n.º 83 do Superior Tribunal de Justiça na espécie.")
        self.assertEqual((c["classificacao"], c["resolucao"]), ("real", {"id_canonico": "1289710642"}))

    def test_not_citations(self):
        """Número do próprio recurso, CPF, valores, folhas e autos não são citações."""
        texts = [
            "O presente recurso especial, protocolado sob o nº 12.345.678, foi distribuído.",
            "Habeas corpus impetrado em favor de JOSÉ, CPF 123.456.789-00.",
            "Valor da causa: R$ 12.345,00, conforme fls. 345.",
            "Autos nº 7218220-53.2017.8.09.9480",
        ]
        for text in texts:
            self.assertEqual(self.run_doc(text), [], text)


@unittest.skipUnless(DB.exists(), "base canônica ausente")
class GeneralFormsTest(unittest.TestCase):
    """Mecanismos gerais: marcadores, enumerações, rabos de citação, OCR."""

    @classmethod
    def setUpClass(cls):
        """Monta o índice da base uma vez para a classe."""
        cls.index = CanonicalIndex.from_sqlite(DB)

    def run_doc(self, text):
        """Roda o pipeline (só regras) num texto.

        Parameters
        ----------
        text : str
            Texto do documento.

        Returns
        -------
        list of tuple of (str, str, str or None)
            ``(trecho, classificacao, id_canonico)`` de cada citação.
        """
        return [(c["trecho"], c["classificacao"], (c["resolucao"] or {}).get("id_canonico"))
                for c in process_text("x", text, self.index)["citacoes"]]

    def test_number_markers(self):
        """Marcadores de número, também com OCR ("NÚMER0"); "nos" não é marcador."""
        from src.normalize import is_number_marker
        for token in ["nº", "n.º", "Nr.", "nro", "núm.", "NÚMER0", "número", "n°"]:
            self.assertTrue(is_number_marker(token), token)
        self.assertFalse(is_number_marker("nos"))

    def test_article_without_preposition(self):
        """Artigo e diploma sem preposição, nas duas ordens."""
        self.assertEqual(self.run_doc("Vide artigo 477, CLT."), [("artigo 477, CLT", "real", "10710324")])
        self.assertEqual(self.run_doc("Vide CLT - art. 477, inciso I."),
                         [("CLT - art. 477, inciso I", "real", "10710324")])

    def test_enumeration_before_diploma(self):
        """Enumeração de incisos entre o artigo e o diploma."""
        [(trecho, classe, _)] = self.run_doc("Nos termos do art. 5º, incisos I a III, da CF, decide-se.")
        self.assertEqual((trecho, classe), ("art. 5º, incisos I a III, da CF", "real"))

    def test_ordinal_is_not_zero(self):
        # "5o" é 5º; lido como 50 viraria outro artigo.
        """O "o" de "5o" é ordinal, não zero."""
        [(_, classe, doc_id)] = self.run_doc("Aplica-se o artigo 5o, inciso LV da Constituição Federal.")
        self.assertEqual((classe, doc_id), ("real", "10641516"))

    def test_sumula_with_item(self):
        """Súmula do TST com item ("Súmula 331, IV, do TST")."""
        [(trecho, classe, _)] = self.run_doc("Incide a Súmula 331, IV, do TST na espécie.")
        self.assertEqual((trecho, classe), ("Súmula 331, IV, do TST", "real"))

    def test_leading_one_read_as_letter(self):
        """Primeiro dígito lido como letra ("l.508.709")."""
        [(_, classe, doc_id)] = self.run_doc("Cf. Agravo Regimental nos EDs em Rec. Extr. com Ag. Nº l.508.709.")
        self.assertEqual((classe, doc_id), ("real", "5090643218"))

    def test_letter_at_word_end_is_not_a_digit(self):
        """Letra no fim de palavra não vira dígito ("Criminãl 234")."""
        from src.normalize import number_digits
        # "Criminãl 234": o "l" é fim de palavra, não o dígito 1.
        self.assertEqual(number_digits("Apelação Criminãl 234-73.2016.7.11.0211")[0], "2347320167110211")

    def test_descriptive_with_date_and_name_starting_with_de(self):
        """Descritiva com data e nome que começa com "De" (não é partícula)."""
        [(trecho, classe, _)] = self.run_doc(
            "Cf. acórdão relatado pelo Min. Delaide Alves Miranda Arantes no TST em 10/2025.")
        self.assertEqual(trecho, "acórdão relatado pelo Min. Delaide Alves Miranda Arantes no TST em 10/2025")
        self.assertNotEqual(classe, "inventada")

    def test_citation_tail_is_not_a_descriptive(self):
        """Rabo de citação numerada (relator, turma, DJe) não vira descritiva."""
        found = self.run_doc("Cf. REsp 1.501.000/RJ (STJ), Rel. Min. Fulano de Tal, Terceira Turma, DJe 10/10/2020.")
        self.assertEqual([t for t, _, _ in found], ["REsp 1.501.000/RJ (STJ)"])

    def test_court_and_commas_inside_the_phrase(self):
        """Tribunal e vírgulas dentro da frase de número forte; número sem classe não é citação."""
        [(trecho, classe, _)] = self.run_doc("Cf. STJ, Recurso Especial, 1511083/RS, no mesmo sentido.")
        self.assertEqual((trecho, classe), ("STJ, Recurso Especial, 1511083/RS", "real"))
        self.assertEqual(self.run_doc("No agravo, 1.500 páginas foram juntadas em 2020."), [])

    def test_court_of_another_citation_is_not_taken(self):
        """Tribunal separado por conector é de outra citação."""
        [(trecho, _, _)] = self.run_doc("Houve embargos no Eg. TST e Agravo Regimental no RE 1.234.567 do STF.")
        self.assertEqual(trecho, "Agravo Regimental no RE 1.234.567 do STF")

    def test_relator_for_the_judgment(self):
        """Relator para o acórdão ("Rel. p. o ac.") em descritiva."""
        [(trecho, classe, _)] = self.run_doc("Cf. TST, Rel. p. o ac. Marcelo Lamego Pertence, 2024.")
        self.assertEqual((trecho, classe), ("TST, Rel. p. o ac. Marcelo Lamego Pertence, 2024", "real"))

    def test_marker_glued_to_number_normalizes_to_the_same_key(self):
        # "n 76532", "n76532", "no76532", "numero76532", "nº76532": mesma chave, mesmo registro.
        """Marcador colado ao número dá a mesma chave; um dígito de diferença não aproxima."""
        for form in ["Reclamação n 76532", "Reclamação n76532", "Rcl. no76532", "Rcl. numero76532", "Rcl. nº76532"]:
            [(trecho, classe, doc_id)] = self.run_doc(f"Conforme a {form}, nego.")
            self.assertEqual((trecho, classe, doc_id), (form, "real", "6457031728"), form)
        # Um dígito de diferença: não aproxima.
        [(_, classe, _)] = self.run_doc("Conforme a Rcl. nº76533, nego.")
        self.assertEqual(classe, "inventada")
        # "n0" é "no" com OCR, não o processo nº 0.
        self.assertEqual(self.run_doc("Vide AgInt n0 REsp 1.501.000/RJ.")[0][:2], ("AgInt n0 REsp 1.501.000/RJ", "real"))

    def test_appeals_listed_after_the_number(self):
        """Recursos listados depois do número entram no span; item com número é outra citação."""
        [(trecho, classe, _)] = self.run_doc("Cf. Reclamação 45429, Agravo Regimental, STF. Nego.")
        self.assertEqual(trecho, "Reclamação 45429, Agravo Regimental, STF")
        # Item com número é outra citação, não um recurso da primeira.
        found = self.run_doc("Cf. REsp 1.501.000/RJ, AgInt no REsp 1.883.715/SP.")
        self.assertEqual([t for t, _, _ in found], ["REsp 1.501.000/RJ", "AgInt no REsp 1.883.715/SP"])

    def test_descriptive_without_cue_uses_the_base_relators(self):
        """Descritiva sem pista de relatoria, achada pelo nome de um relator da base."""
        [(trecho, classe, _)] = self.run_doc("No mesmo sentido: TSE, 2017, Gilmar Mendes.")
        self.assertEqual(trecho, "TSE, 2017, Gilmar Mendes")
        self.assertIn(classe, ("real", "incompleta"))

    def test_unknown_acronym_before_a_covered_cnj(self):
        """Sigla desconhecida antes de CNJ da Justiça Eleitoral vale; cabeçalho em caixa alta não é sigla."""
        [(trecho, classe, _)] = self.run_doc("Vide REE 0600306-17.2020.6.06.0074, TSE.")
        self.assertEqual((trecho, classe), ("REE 0600306-17.2020.6.06.0074, TSE", "real"))
        # Cabeçalho em caixa-alta não é sigla de classe: o número é o do próprio processo.
        self.assertEqual(self.run_doc("PARECER\n\nProcesso nº 4536167-15.2018.5.01.7399\n\nTrata-se de consulta."), [])


@unittest.skipUnless(DB.exists(), "base canônica ausente")
class RobustnessTest(unittest.TestCase):
    """Digitalização ruim: OCR, espaço perdido/sobrando, mojibake e numeração de margem."""

    @classmethod
    def setUpClass(cls):
        """Monta o índice da base uma vez para a classe."""
        cls.index = CanonicalIndex.from_sqlite(DB)

    def run_doc(self, text):
        """Roda o pipeline (só regras) num texto.

        Parameters
        ----------
        text : str
            Texto do documento.

        Returns
        -------
        list of tuple of (str, str, str or None)
            ``(trecho, classificacao, id_canonico)`` de cada citação.
        """
        return [(c["trecho"], c["classificacao"], (c["resolucao"] or {}).get("id_canonico"))
                for c in process_text("x", text, self.index)["citacoes"]]

    def test_ocr_letters_inside_numbers(self):
        # O extrator aceita as mesmas trocas que normalize.OCR_DIGITS converte (B=8, q=9, Z=2, l=1).
        """Letras de OCR dentro de números (B=8, q=9, Z=2, l=1, G=6) e nas palavras-chave."""
        self.assertEqual(self.run_doc("Vide AgInt no REsp 1.664.Bq3/PR."),
                         [("AgInt no REsp 1.664.Bq3/PR", "real", "2684902370")])
        self.assertEqual(self.run_doc("Conforme a 5úmula Z1l do 5TJ, nego."),
                         [("5úmula Z1l do 5TJ", "real", "1289710776")])
        self.assertEqual(self.run_doc("Aplica-se o art. 1º, IX da Lei Complementar nº G4/1990."),
                         [("art. 1º, IX da Lei Complementar nº G4/1990", "real", "11304039")])

    def test_ocr_that_is_not_a_number(self):
        """OCR que não é número: "d0" é "do", fio de tabela não é "111", "riº" é "nº"."""
        self.assertEqual(self.run_doc("Veja a Súmula d0 STJ sobre o tema."), [])  # "d0" é "do"
        [(trecho, _, _)] = self.run_doc("Rcl 45429 STF.\n\n|||  |||")  # fio de tabela não é "111"
        self.assertEqual(trecho, "Rcl 45429")
        # "riº" é o "nº" com OCR, não abreviação de classe: número do próprio processo.
        self.assertEqual(self.run_doc("Processo riº 9426435-30.2024.8.08.5965\n\nTrata-se."), [])

    def test_ocr_in_keywords(self):
        """OCR nas palavras-chave ("Rd" por "Rcl", "arf." por "art.")."""
        self.assertEqual(self.run_doc("Vide Rd nº 88.860/PE.")[0][:2], ("Rd nº 88.860/PE", "inventada"))
        self.assertEqual([t for t, _, _ in self.run_doc("Ver arf. 93, IX da Constituição Federal.")],
                         ["arf. 93, IX da Constituição Federal"])

    def test_glued_and_split_words(self):
        """Palavras coladas e partidas são corrigidas, com spans no texto original."""
        self.assertEqual(clean("a incidência doart. 303 da CF")[0], "a incidência do art. 303 da CF")
        self.assertEqual(clean("oREspnº 2020005/RJ e a Súmula nº 443do STJ")[0],
                         "o REsp nº 2020005/RJ e a Súmula nº 443 do STJ")
        self.assertEqual(clean("na Suspe nsão de Sent ença")[0], "na Suspensão de Sentença")
        # Caixa trocada não é conector colado: "aGrG" continua uma palavra.
        self.assertEqual(clean("aGrG em Recurso eSPECIAL")[0], "aGrG em Recurso eSPECIAL")
        # Spans continuam no texto original.
        self.assertEqual(self.run_doc("incidência doart. 186 do Código Civil."),
                         [("art. 186 do Código Civil", "real", "10718759")])

    def test_mojibake(self):
        """Mojibake é desfeito para extrair; o span continua no original."""
        self.assertEqual(self.run_doc("A ReclamaÃ§Ã£o nÂº 45429, STF."),
                         [("ReclamaÃ§Ã£o nÂº 45429, STF", "real", "1769282037")])
        # "DÃ" + NBSP é "DA" com OCR, não o mojibake de "à".
        self.assertEqual(clean("DÃ RELATORIA")[0], "DÃ RELATORIA")

    def test_margin_line_numbers(self):
        """Numeração de linha na margem sai; parágrafos numerados ficam."""
        text = "".join(f"{k:>3}  linha {k} do texto\n" for k in range(1, 8)) + "RECURSO ESPECIAL\n  8  Trata-se."
        self.assertEqual(self.run_doc(text), [])
        # Parágrafos numerados ("1.  Dos fatos") não são numeração de margem.
        self.assertEqual(clean("1.  Dos fatos\n2.  Do direito")[0], "1.  Dos fatos\n2.  Do direito")


@unittest.skipUnless(DB.exists(), "base canônica ausente")
class OcrRobustnessRulesTest(unittest.TestCase):
    """Melhorias gerais das regras (valem para as duas versões e para o fallback)."""

    @classmethod
    def setUpClass(cls):
        """Monta o índice da base uma vez para a classe."""
        cls.index = CanonicalIndex.from_sqlite(DB)

    def classes(self, text):
        """Roda o pipeline (só regras) e devolve trecho e classe de cada citação.

        Parameters
        ----------
        text : str
            Texto do documento.

        Returns
        -------
        list of tuple of (str, str)
            ``(trecho, classificacao)`` de cada citação.
        """
        return [(c["trecho"], c["classificacao"]) for c in process_text("t", text, self.index)["citacoes"]]

    def test_robust_number_fallback(self):
        # Número com OCR colado que a leitura principal não acha: o leitor robusto acha, só letra -> dígito.
        """O leitor numérico de reserva acha números com OCR colado; inventado continua inventado."""
        for trecho in ["R-Rp \xa0n° 9Bb-96.2010.G.00.0000", "APL\r\n 70D0761-84 2021 7 00\tooO0 - BA",
                       "TST-ARR-010109\xad0-90 2019 5 O1 00l5"]:
            self.assertEqual(resolve({"familia": "processos", "trecho": trecho}, self.index).classificacao, "real")
        # Número inventado continua inventado.
        self.assertEqual(resolve({"familia": "processos", "trecho": "REsp 1.741.785/PR"}, self.index).classificacao,
                         "inventada")

    def test_descriptive_with_any_class(self):
        """Descritiva com classe e ano ("RR de 2016, Rel. Min. …"), também com OCR na pista."""
        self.assertEqual(self.classes("Veja o RR de 2016, Rel. Min. Augusto Carvalho, no ponto."),
                         [("RR de 2016, Rel. Min. Augusto Carvalho", "real")])
        self.assertEqual(self.classes("Veja o REsp de\n2019, Rcl. Min. Gurgel Faria, no ponto."),
                         [("REsp de\n2019, Rcl. Min. Gurgel Faria", "real")])

    def test_ocr_preposition_and_law_numbers(self):
        """Preposição com OCR ("cla CLT") e número de lei com hífen ou OCR."""
        self.assertEqual(self.classes("Nos termos do art. 818 cla CLT, cabe."), [("art. 818 cla CLT", "real")])
        self.assertEqual(self.classes("Nos termos do artigo 102 da Lei nº 13.-467/2017, cabe."),
                         [("artigo 102 da Lei nº 13.-467/2017", "inventada")])
        self.assertEqual(self.classes("Veja o art 189 da Lei riº 9.504/19g7 no caso."),
                         [("art 189 da Lei riº 9.504/19g7", "inventada")])

    def test_common_diploma_aliases(self):
        """Apelidos comuns de diplomas ("NCPC", "CR/88")."""
        self.assertEqual(self.classes("Veja o art. 373 do NCPC no caso."), [("art. 373 do NCPC", "real")])
        self.assertEqual(self.classes("Veja o art. 7°, caput da CR/88 no caso."), [("art. 7°, caput da CR/88", "real")])

    def test_blank_line_ends_number_and_section_titles(self):
        # O "3." do título da seção seguinte não continua o número nem vira processo.
        """Linha em branco encerra o número; título de seção não é processo."""
        self.assertEqual(self.classes("reafirmada no AgRg no RCL N.\n 67 067.\n\n3. DOS PEDIDOS\n\nEventual"),
                         [("AgRg no RCL N.\n 67 067", "real")])
        self.assertEqual(self.classes("do recurso.\n\n3. DOS PEDIDOS\n\nO acórdão"), [])
        # Quebra simples no identificador, como na amostra oficial, continua valendo.
        self.assertEqual(self.classes("a Rcl n° 33.-\n474 (MA), em 2020")[0][0], "Rcl n° 33.-\n474 (MA)")

    def test_em_with_ocr_in_class_name(self):
        """Nome da classe com "cm" (OCR de "em"): AREsp, não REsp."""
        out = process_text("t", "no Agravo\n cm Recurso Especial do STJ, de 2025, Rel. Min. SÉRGIO KUKINA.",
                           self.index)["citacoes"]
        self.assertEqual(out[0]["resolucao"]["id_canonico"], "5900603883")  # AREsp, não REsp


class OutputContractTest(unittest.TestCase):
    """A métrica recusa a submissão inteira se duas citações de um documento têm IoU >= 0,5."""

    @staticmethod
    def cit(start, end, conf):
        """Citação mínima para o filtro de duplicatas.

        Parameters
        ----------
        start, end : int
            Span.
        conf : float
            Confiança da resolução.

        Returns
        -------
        dict
            ``inicio``, ``fim`` e ``resolution`` (incompleta, regra "teste").
        """
        from src.classify import INCOMPLETA, Resolution
        return {"inicio": start, "fim": end, "resolution": Resolution(INCOMPLETA, None, conf, "teste")}

    def test_duplicates_keep_one_by_confidence_then_length(self):
        """Entre citações com IoU >= 0,5, fica a de maior confiança e, no empate, a mais longa."""
        from src.main import without_duplicates
        a, b, c = self.cit(0, 10, 0.40), self.cit(1, 10, 0.92), self.cit(30, 40, 0.40)
        self.assertEqual(without_duplicates([a, b, c]), [b, c])
        longer, shorter = self.cit(0, 12, 0.85), self.cit(0, 10, 0.85)
        self.assertEqual(without_duplicates([shorter, longer]), [longer])

    def test_partial_overlap_below_threshold_is_kept_in_order(self):
        """Sobreposição com IoU < 0,5 mantém as duas, na ordem."""
        from src.main import without_duplicates
        first, second = self.cit(0, 10, 0.40), self.cit(8, 20, 0.92)  # IoU 2/20
        self.assertEqual(without_duplicates([first, second]), [first, second])

    def test_validator_flags_duplicates(self):
        """O validador acusa duas citações com IoU >= 0,5 no mesmo documento."""
        import json
        import tempfile
        from tools.validate_output import check
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "in").mkdir()
            (tmp / "out").mkdir()
            (tmp / "in" / "d.txt").write_text("REsp 1.234.567/SP e mais", encoding="utf-8")
            cit = {"trecho": "REsp 1.234.567/SP", "tipo": "jurisprudencia", "classificacao": "inventada",
                   "resolucao": None, "confianca": 0.9}
            doc = {"documento_id": "d", "citacoes": [cit | {"inicio": 0, "fim": 17},
                                                      cit | {"inicio": 0, "fim": 14, "trecho": "REsp 1.234.567"}]}
            (tmp / "out" / "d.json").write_text(json.dumps(doc), encoding="utf-8")
            problems = check(tmp / "in", tmp / "out", None)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("IoU", problems[0])


if __name__ == "__main__":
    unittest.main()

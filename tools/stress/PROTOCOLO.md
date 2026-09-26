# Protocolo de avaliação das melhorias de robustez (registrado antes de qualquer ajuste)

Data do registro: 2026-09-26. Código de partida ("antes"): `src/` com os hashes abaixo
(branch `lucas`, commit `b0b624b` + correção do ordinal, ainda não commitada).

```
b68c809b5543fc1d860f4d136c05b0127cfb81e446f002dea0e68469f6c09103  src/classify.py
d929b4e656c78b74be9a049a26254fc1b3539d1eebe39ff4e766197280b078d2  src/__init__.py
2ae8115557f79a581bf0f0b66fbcddc6f88ff2a6edb3318c38f2d5f14e61505e  src/main.py
251262b5f99cdef124ee2ea618482774061a91686371fe193281a2301d8bd39c  src/normalize.py
36af183b246ff45144ff9643a15a8779f7f5fc56bf028aa8fd3dec24fcfe4cd6  src/spans.py
```

## Iteração (pode diagnosticar e corrigir; só fenômenos gerais)

- `data/stress/<perfil>_s1000..s1002`: ruído sobre os 26 textos do dev (já vistos).
- Ruído sobre os sintéticos de **iteração** já existentes (`data/synthetic/val`,
  `data/synthetic/ineditos_v3`, semente 2026), sementes de ruído 1000–1002.
- `dev` e os sintéticos de iteração limpos: regressão (o dev tem de continuar 1,0999, τ = 0).

## Holdout (gerado só DEPOIS de congelar o código; medido uma vez; sem --diagnose)

- **H-dev**: todos os perfis de `tools.stress` sobre `data/txt`, sementes de ruído 9000–9002.
- **H-novo**: documentos inéditos do `tools.synth` (perfis holdout `test` e `ineditos_v4`,
  semente 9100, nunca gerada), com os perfis `limpo`, `leve`, `moderado`, `pesado` e `extremo`,
  sementes de ruído 9000–9002.
- **H-synth**: `tools.battery` em todos os perfis sintéticos, `--base-seed 9200 --seeds 3`.

Antes e depois são medidos nos mesmos conjuntos. Se algo mudar no código depois da medição do
holdout, o holdout vira iteração e é preciso gerar outro, com sementes novas (9500+).

## Ressalva conhecida

Quem corrige o pipeline também escreveu o gerador de ruído (tabela de confusões de OCR etc.).
Ganhos em H-dev e H-novo são, portanto, em parte "dentro da distribuição" do gerador. As medidas
independentes do gerador são H-synth (ruído de `tools/synth/noise.py`, escrito antes) e o dev.

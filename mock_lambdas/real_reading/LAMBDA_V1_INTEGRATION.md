# Adaptador de leitura ECG para `ecg-lambdas-v1` — contrato 2.0

Entrada: `lambda_v1.handler(event, context)`, referenciado pelo shim do v1 em `mock_lambdas/reading/handler.py`.

Este adaptador fica **neste repositório** (ecg-analyser), preservando o v1 como proprietário do envelope/erros de contrato. Não há chamada a recursos congelados de avaliação (`outputs/lote50_01/...`); seis módulos vêm diretamente do `research_methods.registry`. `baseline_peaks` e as utilidades para `prepare`, `grid`, `convert` e `color_gate` usam `frozen_peaks34`, cujo manifesto foi verificado pelos testes. O classificador vem de `selector/score.py` sem retreino.

## Fluxo

1. Validação completa de `request_id`, versão 2.0, layout, velocidade/ganho explícitos e bucket/key; `1. Testes` é proibido.
2. Presença **e SHA-256 do modelo aprovado** (`model_selector.pkl`, prefixo `cb0f395367ec`) verificados antes de baixar e processar a imagem; na ausência/mismatch: `READ_SELECTOR_UNAVAILABLE`.
3. S3 `head_object/get_object` com limite de tamanho, formato e pixels. Decodificação com EXIF preservado por `prepare`; imagem original não é modificada.
4. `prepare`, `estimate_grid` e seis `extract(Context(...))` sequenciais (sem paralelismo, sem dependências de outputs congelados).
5. Integridade dos candidatos `Extraction.validate`; gate cromático **inalterado**, bloqueia exportação de sinais. Grade não válida implica abstinência para evitar amplitudes fictícias.
6. Para cada candidato permitido: `convert` original em unidades físicas; sinal nativo, sem reamostrar a saída; score com **`load_leads`, `score_leads`, `record_score`** do modelo aprovado, sem fallback heurístico.
7. `result.json`, `preview.png` (ou `rejection_preview.png`) e `leads.json` sob `reading/<request_id>/<run_id>/<method>/`, salvos no `RESULTS_BUCKET`. Status e referências retornam em `methods[]`.
8. Envelope final 2.0, com `REFUSED` se todos estiverem bloqueados/abstidos, `ERROR` se nenhum produzir sinal e houver erro.

## Layouts

`3x4+1` e `12x1` têm perfis estáticos e genéricos já disponíveis (`configs/report_normal.json` / `configs/data_12x1.json`), sempre com a calibração sobrescrita **somente** a partir da entrada explícita. Outros layouts do v1 não possuem configuração genérica correspondente nos materiais entregues e retornam `LAYOUT_UNSUPPORTED`. Não foi feito encaixe por nome de arquivo ou por imagem.

## Para testar localmente com os dois repositórios lado a lado

```bash
cd ecg-lambdas-v1
PYTHONPATH=mock_lambdas python -m unittest discover -s mock_lambdas/tests -v
cd ../ecg-analyser
PYTHONPATH=extractor python -m unittest discover -s extractor/tests -p 'test_research_methods.py' -v
PYTHONPATH=extractor python -m unittest discover -s extractor/tests -p 'test_selector.py' -v
```

Na primeira bateria os testes de integração descobrem `../ecg-analyser/extractor`. Se estiver em outro local, defina `ECG_ANALYSER_EXTRACTOR_PATH`. Os testes com `FakeS3` não executam AWS. O teste de exportação utiliza mock **apenas do escore** porque o arquivo `model_selector.pkl` não veio no ZIP. O contrato recusa operação real sem esse arquivo aprovado.

## Deploy: não autorizado, não realizado

Não foi publicado container ou alterado Terraform, IAM, timeout, memória ou bucket. O `mock_lambdas/build.sh` do v1 permanece construindo o pacote da **leitura mock legada**, a partir de `reading/mock_handler.py`, para evitar substituição involuntária de um deploy existente. **Não** use esse build para publicar o adaptador real.

Quando houver decisão de custo/infra, empacotar conjuntamente `ecg_contract`, o módulo `lambda_v1`, módulos `ecg`, `research_methods`, `review`, `configs`, pasta `selector` + `model_selector.pkl` verificado e dependências compatíveis, e apontar o handler para `reading.handler.handler`. O v1 atualmente declara 1769 MB/60 s; ECR, runtime, timeout e memória são decisões pendentes. `extractor/requirements.txt` no ZIP ainda fixa `opencv-python-headless==4.13.0.92`; compatibilidade com OpenCV 5 não foi validada aqui, nem o ambiente AWS.

## Dependências externas para completar

- **Modelo aprovado original** (`selector/model_selector.pkl`, SHA `cb0f395367ec...`) — não incluído e não reconstituído.
- Imagens de `ECG Images/0. Treino` para testar os seis métodos, medir tempos/memória e conferir gate nas amostras 07/38. **Não acessar `1. Testes`.**
- Backend Django: selecionar o método com maior `selector_score` entre `status == "OK"`, baixar `files.leads` e repassar esse JSON na propriedade `signal` da Lambda `interpret` — não implementado neste par de repositórios.
- Deliberar aceitação dos seis métodos ainda categorizados pesquisa e o custo de empacotamento/infra.

**Uso clínico:** trata-se de extração experimental de sinais, com riscos de inexatidão. Nenhum `OK` atesta validade médica ou substitui avaliação clínica.

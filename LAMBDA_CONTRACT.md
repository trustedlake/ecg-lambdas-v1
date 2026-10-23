# Contrato das lambdas de ECG — leitura 2.0, interpretação mock 1.0

O **v1 é a base**; a leitura real usa `ecg-analyser/extractor/lambda_v1.py` sem alterar os seis algoritmos nem o gate cromático. A interpretação continua mock, sem diagnoser. **Nenhum deploy, alteração de memória/timeout, imagem ECR ou custo foi autorizado.**

O código de contrato está em `mock_lambdas/ecg_contract/__init__.py`. O build ZIP atual **continua empacotando `reading/mock_handler.py`**, não o adaptador real. O real requer empacotamento posterior, incluindo dependências e o modelo aprovado (não distribuído no ZIP).

## 1. Envelope

Toda resposta de leitura tem `contract_version: "2.0"`, `operation: "read"`, `request_id`, `status`, `warnings`, `error`, `result`, `meta`.

- `OK`: pelo menos um método produziu sinal e não há avisos.
- `OK_WITH_WARNINGS`: existe sinal, mas há avisos de qualidade e/ou métodos que não concluíram.
- `REFUSED`: nenhum método produziu sinal e **todos** terminaram `BLOCKED`/`ABSTAINED`; `error.details.methods` contém os seis resultados.
- `ERROR`: validação, infraestrutura, seletor indisponível ou erro de execução sem saída viável; `error.details.methods` contém os seis resultados, quando chegaram a ser executados.
- `meta`: `engine: "ecg-analyser"`, versão e `duration_ms` medido. Nenhum valor médico é sintetizado.

Erros esperados retornam no envelope, jamais `FunctionError`. `request_id` é UUID; o `run_id` é criado por execução para impedir sobrescrita. As imagens e os sinais vão ao S3; a resposta síncrona contém referências, **não** todos os dados.

## 2. Request de leitura real

```json
{
  "contract_version": "2.0",
  "request_id": "3f6c1f9e-8f0a-4c1e-9a52-0d2b7e0b6a11",
  "image": {"bucket": "ecg-images", "key": "exam/attempt/original.png"},
  "layout": "3x4+1",
  "calibration": {"speed_mm_s": 25.0, "gain_mm_mv": 10.0}
}
```

- `contract_version`: somente `2.0` no read real. Sem campo, assume 2.0 apenas para compatibilidade do envelope. Versão declarada errada: `UNSUPPORTED_CONTRACT_VERSION`.
- `image`: objeto S3 obrigatório. Somente JPG/PNG/WebP, até 15 MB, 24 megapixels, dimensão mínima 800 × 500 em qualquer orientação; limites são de validação, não de qualidade clínica. `IMAGE_MISSING`, `IMAGE_NOT_FOUND`, `IMAGE_FORMAT_UNSUPPORTED`, `IMAGE_UNREADABLE`, `IMAGE_TOO_LARGE` ou `IMAGE_TOO_SMALL`.
- É proibido ler a origem `ECG Images/1. Testes`, inclusive como key S3 (`IMAGE_SOURCE_FORBIDDEN`). Fontes de testes em produção/treino não são selecionadas automaticamente.
- `layout`: **obrigatório**, sem detecção. Apenas `3x4+1` (configuração genérica `report_normal.json`) e `12x1` (`data_12x1.json`) contam com configurações genéricas compatíveis **neste pacote**. Os demais layouts do contrato antigo (`3x4`, `3x4+3`, `6x2`, `6x2+1`) retornam `LAYOUT_UNSUPPORTED` até ganharem configs explícitas verificadas. Campo faltante: `LAYOUT_MISSING`.
- `calibration`: **objeto obrigatório, com ambos os campos obrigatórios**, sem inferência nem padrão oculto. Velocidades aceitas: 12.5, 25, 50 mm/s. Ganhos aceitos: 5, 10, 20 mm/mV. Valor omitido/incorreto: `CALIBRATION_INVALID`.
- `mock`: ignorado pelo handler real. Cenários sintéticos são exclusivos do antigo `mock_handler.py`.

## 3. Response de leitura real

```json
{
  "contract_version": "2.0",
  "operation": "read",
  "request_id": "3f6c1f9e-8f0a-4c1e-9a52-0d2b7e0b6a11",
  "status": "OK_WITH_WARNINGS",
  "warnings": [{"code": "METHODS_PARTIAL_RESULT", "message": "..."}],
  "error": null,
  "result": {
    "layout": "3x4+1",
    "calibration": {"speed_mm_s": 25.0, "gain_mm_mv": 10.0},
    "units": "mV",
    "methods": [
      {
        "method": "baseline_peaks",
        "version": "peaks34-323b4e6",
        "status": "OK",
        "reason": null,
        "selector_score": 87.5,
        "sampling_rate_hz": 200.0,
        "files": {
          "json": {"bucket": "results", "key": "reading/<request_id>/<run_id>/baseline_peaks/result.json", "content_type": "application/json"},
          "image": {"bucket": "results", "key": "reading/<request_id>/<run_id>/baseline_peaks/preview.png", "content_type": "image/png"},
          "leads": {"bucket": "results", "key": "reading/<request_id>/<run_id>/baseline_peaks/leads.json", "content_type": "application/json"}
        }
      }
    ]
  },
  "meta": {"engine": "ecg-analyser", "engine_version": "dev", "duration_ms": 0}
}
```

**Amostra ilustrativa, não resultado executado.** `methods` sempre tem seis métodos na ordem do registry: `baseline_peaks`, `chromatic_ink`, `faint_ink`, `uneven_photo`, `grayscale_contrast`, `same_appearance`.

| Status individual | Comportamento |
|---|---|
| `OK` | Extração medida e gate cromático permitido; `result.json`, `preview.png` e `leads.json` no S3. `selector_score` real. |
| `BLOCKED` | Gate cromático verdadeiro; `result.json` e `rejection_preview.png` no S3, **nenhum sinal**. |
| `ABSTAINED` | `MethodAbstention`, grade inválida ou nenhum lead medido; JSON e imagem de recusa, sem sinal. |
| `ERROR` | Erro inesperado no método; JSON persistido, `selector_score=null`, sem sinal. |

`selector_score` usa exclusivamente o seletor aprovado (`selector/score.py`: `load_leads`, `score_leads`, `record_score`) sobre os arrays medidos pelo método. Modelo `model_selector.pkl` imutável, SHA-256 começando por `cb0f395367ec`. **Arquivo ausente leva a `READ_SELECTOR_UNAVAILABLE`, não a score fictício.** A seleção pela maior pontuação é responsabilidade do backend. Os seis métodos têm status de pesquisa; `OK` não equivale a validação clínica.

Taxa nativa em Hz = `grid.x.pixels_per_mm × speed_mm_s`; não reamostrar a exportação para 500 Hz. No `leads.json` cada derivação traz `name`, `kind`, `row`, `col`, `start_s`, `duration_s`, `quality`, `coverage`, `bbox_px`, `baseline_y_px`, `samples` (mV). `II_longo` é apresentado como `II`, `kind: rhythm`. Valores NaN e amostras não válidas são exportados como `null`. Nenhum trecho ausente é interpolado para entrega; a reamostragem interna para 150 Hz é **somente para o classificador aprovado**, preservando as lacunas.

Os métodos podem emitir `LEAD_PARTIAL_TRACE`, `AMBIGUOUS_TRACE`, `GRID_CALIBRATION_UNCERTAIN`, `GEOMETRY_FALLBACK`, `ORIENTATION_UNCERTAIN` e `METHODS_PARTIAL_RESULT`. `INCOMPLETE_TRACE` foi mapeado para `LEAD_PARTIAL_TRACE`. Avisos gerais omitem `lead` quando não possuem derivação específica (`lead: null` não é necessário nem emitido).

## 4. Contrato de interpretação permanece mock

O endpoint `operation: "interpret"` continua aceitando `contract_version: "1.0"` e o formato legado de `signal`, e também aceita explicitamente `"2.0"`. O backend **não pode repassar** `result` de leitura 2.0 diretamente para a interpretação: deve selecionar o melhor método com `status: "OK"`, baixar seu `files.leads` e enviar esse objeto como `signal`, junto a `patient.sex` e `request_id`. Nenhuma chamada automática de interpretação foi adicionada ao adaptador.

Regra de compatibilidade de sinal legado: `sampling_rate_hz` entre 100–2000, unidade `mV`, `leads[].name` entre I, II, III, aVR, aVL, aVF, V1–V6, `samples` números ou `null`, pelo menos 1 s válido por derivação; avisos para derivações ausentes. A interpretação é **sintética** e não equivale a diagnóstico médico. Especificação integral do mock anterior: `CONTRATO_LEGADO_V1.md`.

## 5. Catálogo de erros / avisos

Todos os códigos abaixo são definidos em `ecg_contract`; todos os códigos antigos constam de `CONTRATO_LEGADO_V1.md`. Os novos códigos são:

- `IMAGE_SOURCE_FORBIDDEN` — validação: fonte reservada para teste.
- `READ_REFUSED_UNREADABLE_TRACE` — recusa quando todos abstiverem e nenhum gate cromático bloquear.
- `READ_SELECTOR_UNAVAILABLE` — processamento indisponível por falta do modelo aprovado.
- `AMBIGUOUS_TRACE` — amostras com incerteza/ambiguidade.
- `GRID_CALIBRATION_UNCERTAIN` — grade sem calibração consistente.
- `GEOMETRY_FALLBACK` — geometria por fallback.
- `ORIENTATION_UNCERTAIN` — orientação incerta.
- `METHODS_PARTIAL_RESULT` — pelo menos um método não entregou sinal.

## 6. Limitações antes de produção

O ZIP fornecido não inclui `selector/model_selector.pkl`, não inclui imagens da pasta `ECG Images/0. Treino` e não contém imagens do conjunto reservado. Os testes unitários podem verificar estrutura, regras, gate e S3 falso, **não desempenho/precisão clínica** nem o tempo e pico de memória sobre imagens reais. O container/ECR, tamanho/memória/timeout e a integração do backend exigem decisão e avaliação de custo separadas.

### Índice completo de códigos do catálogo compartilhado

| Código | Tipo |
|---|---|
| `INVALID_PAYLOAD` | erro `validation` |
| `UNSUPPORTED_CONTRACT_VERSION` | erro `validation` |
| `REQUEST_ID_INVALID` | erro `validation` |
| `MOCK_SCENARIO_INVALID` | erro `validation` |
| `STORAGE_ERROR` | erro `internal` |
| `INTERNAL_ERROR` | erro `internal` |
| `IMAGE_SOURCE_FORBIDDEN` | erro `validation` |
| `IMAGE_MISSING` | erro `validation` |
| `IMAGE_NOT_FOUND` | erro `validation` |
| `IMAGE_FORMAT_UNSUPPORTED` | erro `validation` |
| `IMAGE_TOO_LARGE` | erro `validation` |
| `IMAGE_TOO_SMALL` | erro `validation` |
| `IMAGE_UNREADABLE` | erro `validation` |
| `LAYOUT_MISSING` | erro `validation` |
| `LAYOUT_UNSUPPORTED` | erro `validation` |
| `READ_REFUSED_UNREADABLE_TRACE` | erro `refused` |
| `READ_SELECTOR_UNAVAILABLE` | erro `processing` |
| `CALIBRATION_INVALID` | erro `validation` |
| `READ_REFUSED_BLURRY` | erro `refused` |
| `READ_REFUSED_CROPPED` | erro `refused` |
| `READ_REFUSED_OCCLUDED` | erro `refused` |
| `READ_REFUSED_LOW_CONTRAST` | erro `refused` |
| `READ_REFUSED_NOT_ECG` | erro `refused` |
| `READ_REFUSED_LAYOUT_MISMATCH` | erro `refused` |
| `READ_GRID_NOT_FOUND` | erro `processing` |
| `READ_TRACE_EXTRACTION_FAILED` | erro `processing` |
| `SEX_MISSING` | erro `validation` |
| `SEX_INVALID` | erro `validation` |
| `SIGNAL_MISSING` | erro `validation` |
| `SIGNAL_INVALID` | erro `validation` |
| `SIGNAL_SAMPLING_RATE_INVALID` | erro `validation` |
| `SIGNAL_LEAD_UNKNOWN` | erro `validation` |
| `SIGNAL_TOO_SHORT` | erro `validation` |
| `INTERP_NO_QRS_DETECTED` | erro `processing` |
| `INTERP_SIGNAL_TOO_NOISY` | erro `processing` |
| `AMBIGUOUS_TRACE` | aviso |
| `GRID_CALIBRATION_UNCERTAIN` | aviso |
| `GEOMETRY_FALLBACK` | aviso |
| `ORIENTATION_UNCERTAIN` | aviso |
| `METHODS_PARTIAL_RESULT` | aviso |
| `LEAD_PARTIAL_TRACE` | aviso |
| `LEAD_SATURATED` | aviso |
| `IMAGE_ROTATION_CORRECTED` | aviso |
| `CALIBRATION_PULSE_NOT_FOUND` | aviso |
| `LEAD_MISSING` | aviso |
| `LEAD_LOW_QUALITY` | aviso |
| `MEASUREMENT_UNRELIABLE` | aviso |

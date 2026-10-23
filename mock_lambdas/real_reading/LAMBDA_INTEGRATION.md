# AWS Lambda ECG Analyser — contrato

| Operação | `status` | `artifacts` | Prefixo S3 |
|---|---|---|---|
| `analyze` | `CONSISTENCY_PASS` / `REVIEW_REQUIRED` | `result`, `signals`, `trace_masks`, `working_image`, `visao_geral` | `ecg-analyser/{attempt_id}/initial/` |
| `analyze` (bloqueado) | `REJECTED_LOW_TRACE_GRID_CONTRAST` | `result`, `rejection_preview` | `ecg-analyser/{attempt_id}/initial/` |
| `refine` | `MANUAL_REVIEW_REQUIRED` | `result`, `signals`, `trace_masks`, `working_image`, `correction_preview` (+ `visao_geral` anterior) | `ecg-analyser/{attempt_id}/{uuid}/` |

Retorno também fornece `warnings`, `grid`, `leads`, `manual_correction`. `refine` recebe máscara/traços em coordenadas `normalized_image_v1`: escala [0,1] sobre **working_image.png** já retificada; o handler arredonda para `round(x*(width-1))`. Rejeições não aceitam pintura, e colunas corrigidas continuam marcadas como ambíguas/invalidas para análise clínica.

Runtime esperado: AWS Lambda container Python + NumPy + OpenCV-headless + Pillow + boto3. Dimensionamento sugerido **3 GB RAM, 240s timeout, /tmp 2 GB**, sujeito a teste real; não foi feito deploy. A imagem Docker foi construída localmente (1.25 GB) e o handler importa corretamente.  Uma Lambda síncrona leva até 240s mais cold start/rede, e Django/Gunicorn/proxy precisam de timeout compatível.

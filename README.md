> **Integração pendente de deploy:** o read real 2.0 está no `ecg-analyser/extractor/lambda_v1.py`; `mock_lambdas/build.sh` continua empacotando o mock antigo. Veja `CONTRATO_LAMBDAS.md`.

# ecg-lambdas-v1

Lambdas de **leitura** e **interpretação** de ECG do projeto ECG, e o contrato v1 entre elas e o backend
([tl-gmiler/ecg-backend](https://github.com/tl-gmiler/ecg-backend)).

| O quê | Onde |
|---|---|
| Contrato v1: request, response, catálogo de erros, cenários e regras para as lambdas reais | [`CONTRATO_LAMBDAS.md`](CONTRATO_LAMBDAS.md) |
| Mocks deployados (`ecg-reading`, `ecg-interpretation`), testes, smoke test e Terraform | [`mock_lambdas/`](mock_lambdas/README.md) |
| Módulo comum (envelope, validação, catálogo), reaproveitável pelas lambdas reais | [`mock_lambdas/ecg_contract/`](mock_lambdas/ecg_contract/__init__.py) |

AWS: conta `785236377332`, região `sa-east-1`, profile `trustedlake`.

```bash
pip install pillow boto3
python -m unittest discover -s mock_lambdas/tests -v
```

Este repo é separado do [trustedlake/ecg-lambdas](https://github.com/trustedlake/ecg-lambdas), o pipeline antigo em Docker
com Step Function. Esta stack só lê os buckets de lá.

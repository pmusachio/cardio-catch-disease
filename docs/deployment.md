# Entrega e consumo

O projeto disponibiliza classificacao de risco cardiovascular por API, app Streamlit e planilha Google Sheets.

## Canais

- **FastAPI:** endpoint `/predict` para receber pacientes em JSON.
- **Streamlit:** `app/streamlit_app.py` para simular um atendimento individual.
- **Google Sheets:** `integrations/google_sheets_appscript.gs` para pontuar linhas da planilha.

## API local

```bash
python -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src python -m cardio_catch_disease.cli train
PYTHONPATH=src uvicorn cardio_catch_disease.api:app --reload
```

## Streamlit

```bash
python -m pip install -r requirements-app.txt
PYTHONPATH=src streamlit run app/streamlit_app.py
```

## Google Sheets

1. Crie uma planilha com as mesmas colunas esperadas pelo modelo.
2. Abra Extensoes > Apps Script.
3. Cole o conteudo de `integrations/google_sheets_appscript.gs`.
4. Defina a propriedade `CARDIO_API_URL` com a URL publica do endpoint `/predict`.
5. Use o menu **Cardio Catch** para pontuar a aba ativa.

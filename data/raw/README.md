# Dados

Fonte Kaggle: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).

Arquivos esperados nesta pasta:

- `cardio_train.csv`

- O arquivo `cardio_train.csv` usa ponto e virgula como separador; o loader detecta isso automaticamente.

## Download via Kaggle API

```bash
mkdir -p data/raw
kaggle datasets download -d sulianova/cardiovascular-disease-dataset --unzip -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```

Mantenha arquivos grandes fora do Git quando necessario e baixe-os novamente no Colab ou no ambiente local.

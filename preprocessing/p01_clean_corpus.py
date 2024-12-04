# %% Carga de bibliotecas
import os
from pathlib import Path
import yaml
import pandas as pd

# %% Establecer el directorio de trabajo
os.chdir(Path(os.path.abspath('')).parent)

# %% Carga de funciones locales
from src.utils import preprocess_text

# %% Configuración general
with open("pipeline/config.yaml", encoding='utf-8') as file:
    config = yaml.safe_load(file)

# %% Carga de los datos
corpus_df = pd.read_csv(config['corpus']['example'], encoding='latin-1')

# %% Preprocesamiento
corpus_clean_df = preprocess_text(corpus_df, "text")

# %% Respaldar el texto procesado
corpus_clean_df.to_csv(config['output']['path_clean_corpus_csv'], index=False)

# %%

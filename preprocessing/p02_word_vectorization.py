# %% Carga de bibliotecas
import os
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import numpy as np

# %% Establecer el directorio de trabajo
os.chdir(Path(os.path.abspath('')).parent)

# %% Configuración general
with open("pipeline/config.yaml", encoding='utf-8') as file:
    config = yaml.safe_load(file)

# %% Carga de los datos
corpus_df = pd.read_csv(config['output']['path_clean_corpus_csv'])

# %% Train, test split
X = corpus_df['text_final']
y = corpus_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %% Generar la codificación para los labels
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.transform(y_test)

# %% Vectorización de palabras (Word Vectorization)
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(X)
vectorizer.get_feature_names_out()

# %% Aplicar la vectorización al train y test de los datos
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# %% Respaldar los datos vectorizados
np.save(config['output']['path_y_train_npy'], y_train)
np.save(config['output']['path_y_test_npy'], y_test)
save_npz(config['output']['path_X_train_vec_npz'], X_train_vec)
save_npz(config['output']['path_X_test_vec_npz'], X_test_vec)

# %%

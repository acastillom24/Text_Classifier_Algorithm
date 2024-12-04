# %% Carga de bibliotecas
import os
from pathlib import Path
import yaml
import numpy as np
from scipy.sparse import load_npz
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# %% Establecer el directorio de trabajo
os.chdir(Path(os.path.abspath('')).parent)

# %% Configuración general
with open("pipeline/config.yaml", encoding='utf-8') as file:
    config = yaml.safe_load(file)

# %% Establecer una semilla
np.random.seed(500)

# %% Carga de los datos
y_train = np.load(config['output']['path_y_train_npy'])
y_test = np.load(config['output']['path_y_test_npy'])
X_train = load_npz(config['output']['path_X_train_vec_npz'])
X_test = load_npz(config['output']['path_X_test_vec_npz'])

# %% Entrenamiento del algoritmo para obtener el modelo
clf = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
clf.fit(X_train, y_train)

# %% Obtener nuevas predicciones a partir del modelo obtenido
y_pred = clf.predict(X_test)
print("Naive Bayes Accuracy Score -> ", accuracy_score(y_pred, y_test)*100)
# %%


import random
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

try:
    from imblearn.over_sampling import SMOTE
    smote_available = True
except ImportError:
    smote_available = False

# ====================== MODELOS DISPONIBLES ======================
modelos_disponibles = {
    "LR": LogisticRegression(random_state=0),
    "RF": RandomForestClassifier(random_state=0),
    "SVM": svm.SVC(),
    "XGB": XGBClassifier(verbosity=0),
    "ANN": MLPClassifier(random_state=1),
    "SGD": SGDClassifier(),
    "BDT": BaggingClassifier(random_state=0),
    "KNN": KNeighborsClassifier()
}

# ======================== CLASE SOLUCION =========================
class Solucion:
    def __init__(self, n):
        self.x = [0] * n
        self.f = 0.0
        self.n = n
        self.numvar = 0

    def GenerarPunto(self):
        for i in range(self.n):
            self.x[i] = random.randint(0, 1)

    def Evaluar(self, X_train, X_test, y_train, y_test, modelo):
        xtrain = X_train.copy()
        self.numvar = 0
        for i in range(self.n):
            self.numvar += self.x[i]
            if self.x[i] == 0:
                xtrain.iloc[:, i] = 0.0
        regressor = modelos_disponibles[modelo]
        if hasattr(y_train, 'values'):
            y_train_fit = y_train.values.ravel()
        else:
            y_train_fit = y_train.ravel() if hasattr(y_train, 'ravel') else y_train
        regressor.fit(xtrain, y_train_fit)
        y_test_p = regressor.predict(X_test)
        # Puedes cambiar accuracy por f1-score si lo prefieres
        self.f = 1 - accuracy_score(y_test, y_test_p)
        return self.f

# ===================== BUSQUEDA LOCAL ============================
def BusquedaLocal(PuntoActual, n, NumEval, X_train, X_test, y_train, y_test, modelo):
    evals = 0
    mejorvec = Solucion(n)
    mejorvec.x = PuntoActual.x[:]
    mejorvec.f = PuntoActual.f
    mejorvec.numvar = PuntoActual.numvar
    while evals < NumEval:
        ind = random.randint(0, n - 1)
        vecino = Solucion(n)
        vecino.x = mejorvec.x[:]
        vecino.x[ind] = 1 - vecino.x[ind]
        vecino.Evaluar(X_train, X_test, y_train, y_test, modelo)
        if vecino.f < mejorvec.f or (vecino.f == mejorvec.f and vecino.numvar < mejorvec.numvar):
            mejorvec = vecino
            evals = 0
            print(f"Mejora: {mejorvec.f:.4f} numvar: {mejorvec.numvar}")
        else:
            evals += 1
    return mejorvec

# ===================== FUNCION PRINCIPAL =========================
def local_search_selection(
    data, modelo_usado="RF", usar_smote='auto', smote_threshold=0.6,
    NumSem=10, NumEval=None
):
    warnings.filterwarnings("ignore")
    x_cols = [col for col in data.columns if col != "SeriousDlqin2yrs"]
    NumVar = len(x_cols)
    xdata = data[x_cols]
    y = data[["SeriousDlqin2yrs"]]

    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(xdata, y, test_size=0.2, random_state=42)

    # --- SMOTE ---
    aplicar_smote = False
    if usar_smote is True:
        aplicar_smote = True
    elif usar_smote == 'auto':
        clase_counts = y_train.value_counts(normalize=True)
        if (clase_counts > smote_threshold).any():
            aplicar_smote = True
    if aplicar_smote and smote_available and len(np.unique(y_train)) == 2:
        smote = SMOTE(random_state=42)
        y_train_array = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
        X_train, y_train = smote.fit_resample(X_train, y_train_array)
        print(f"SMOTE aplicado: dataset de entrenamiento balanceado (umbral={smote_threshold}).")
    else:
        print("SMOTE no aplicado.")

    if NumEval is None:
        NumEval = 10 * NumVar

    Opt = Solucion(NumVar)
    Opt.GenerarPunto()
    Opt.Evaluar(X_train, X_test, y_train, y_test, modelo_usado)
    print(f"Semilla inicial: {Opt.f:.4f} numvar: {Opt.numvar}")

    errores_iter = []
    for sem in range(NumSem):
        print(f"Semilla {sem+1} de {NumSem}")
        PuntoActual = Solucion(NumVar)
        PuntoActual.GenerarPunto()
        PuntoActual.Evaluar(X_train, X_test, y_train, y_test, modelo_usado)
        PuntoActual = BusquedaLocal(PuntoActual, NumVar, NumEval, X_train, X_test, y_train, y_test, modelo_usado)
        if PuntoActual.f < Opt.f or (PuntoActual.f == Opt.f and PuntoActual.numvar < Opt.numvar):
            Opt = PuntoActual
        errores_iter.append(Opt.f)

    print(f"Optimo final {Opt.f:.4f}  numvar: {Opt.numvar}")

    # Evaluación con todas las variables
    todas_las_variables = [1] * NumVar
    OptAll = Solucion(NumVar)
    OptAll.x = todas_las_variables
    OptAll.Evaluar(X_train, X_test, y_train, y_test, modelo_usado)
    print(f"con todas las variables: {OptAll.f:.4f}")

    # Entrenamiento final para métricas
    X_train_masked = X_train.copy()
    X_test_masked = X_test.copy()
    for i in range(NumVar):
        X_train_masked.iloc[:, i] = X_train_masked.iloc[:, i] * Opt.x[i]
        X_test_masked.iloc[:, i] = X_test_masked.iloc[:, i] * Opt.x[i]
    modelo = modelos_disponibles[modelo_usado]
    if hasattr(y_train, 'values'):
        y_train_fit = y_train.values.ravel()
    else:
        y_train_fit = y_train.ravel() if hasattr(y_train, 'ravel') else y_train
    modelo.fit(X_train_masked, y_train_fit)
    y_pred_final = modelo.predict(X_test_masked)

    # Resultados de selección de variables
    df_vars = pd.DataFrame({
        'Variable': x_cols,
        'Seleccionada': Opt.x
    })

    # Métricas de desempeño
    precision = precision_score(y_test, y_pred_final, average='binary')
    recall = recall_score(y_test, y_pred_final, average='binary')
    df_metrics = pd.DataFrame({
        'METRIC': [
            'Best Loss',
            'Num variables',
            'Accuracy',
            'F1-Score',
            'Precision',
            'Recall',
            'Error with all variables'
        ],
        'VALUE': [
            round(Opt.f, 4),
            round(sum(Opt.x), 2),
            round(1 - Opt.f, 4),
            round(f1_score(y_test, y_pred_final, average='binary'), 4),
            round(precision, 4),
            round(recall, 4),
            round(OptAll.f, 4)
        ]
    })

    # Gráfico de convergencia
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, NumSem + 1), errores_iter, marker='o', color='green')
    plt.title('Convergencia del error (1 - accuracy) por semilla')
    plt.xlabel('Semilla')
    plt.ylabel('Mejor error (1 - accuracy)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_vars, df_metrics, modelo_usado

# ======================= CARGAR BD ========================

data = pd.read_excel(r'C:\Users\Bi_analyst\Desktop\Python\TFM\training_modified.xlsx')
data = data.iloc[:1000]

# ======================= PARÁMETROS ========================

modelo_usado = "RF"  # Cambia aquí el modelo
NumSem = 10
NumEval = None  # Por defecto 10*NumVar
smote_threshold = 0.6

# ======================= EJECUCIÓN PRINCIPAL ========================

df_vars, df_metrics, modelo_usado = local_search_selection(
    data,
    modelo_usado=modelo_usado,
    usar_smote='auto',
    smote_threshold=smote_threshold,
    NumSem=NumSem,
    NumEval=NumEval
)

# ======================= EXPORTAR RESULTADOS ========================

import os

output_folder = r"C:\Users\Bi_analyst\Desktop\Python\TFM\BL models - outputs"
os.makedirs(output_folder, exist_ok=True)
output_filename = os.path.join(output_folder, f"{modelo_usado}_output_local_search.xlsx")

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df_vars.to_excel(writer, index=False, sheet_name="Resultados")
    start_row = len(df_vars) + 4
    df_metrics.to_excel(writer, index=False, sheet_name="Resultados", startrow=start_row)

print(f'📁 Resultados guardados en: {output_filename}')
print(df_vars.to_string(index=False))
print('\n', df_metrics.to_string(index=False))



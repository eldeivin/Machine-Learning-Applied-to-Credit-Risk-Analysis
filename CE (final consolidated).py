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

# ======================== FUNCIONES CE ===========================
def Evaluar(x, X_train, X_test, y_train, y_test, NumVar, modelo):
    if modelo not in modelos_disponibles:
        raise ValueError(f"Modelo '{modelo}' no válido. Opciones: {list(modelos_disponibles.keys())}")

    xtrain = X_train.copy()
    for i in range(NumVar):
        xtrain.iloc[:, i] = xtrain.iloc[:, i] * x[i]

    regressor = modelos_disponibles[modelo]
    if hasattr(y_train, 'values'):
        y_train_fit = y_train.values.ravel()
    else:
        y_train_fit = y_train.ravel() if hasattr(y_train, 'ravel') else y_train
    regressor.fit(xtrain, y_train_fit)
    y_test_p = regressor.predict(X_test)
    f1 = f1_score(y_test, y_test_p, average='binary')
    num_vars = sum(x)
    # Penaliza si no selecciona ninguna variable
    if num_vars == 0:
        return 1.0  # peor posible
    return 1 - f1

def GenerarMuestra(Muestra, Prob, NumMuestras, NumElite, NumVar, ProbMin, modo):
    for i in range(NumElite, NumMuestras):
        for j in range(NumVar):
            nivel = max(min(Prob[j], 1 - ProbMin), ProbMin)
            if modo == "binary":
                Muestra[i][j] = 1 if random.random() < nivel else 0
            else:
                Muestra[i][j] = np.clip(np.random.normal(loc=nivel, scale=0.1), 0, 1)
    return Muestra

def GenerarMuestraIni(Muestra, Prob, NumMuestras, NumVar, ProbMin, modo):
    for i in range(NumMuestras):
        for j in range(NumVar):
            nivel = max(min(Prob[j], 1 - ProbMin), ProbMin)
            if modo == "binary":
                Muestra[i][j] = 1 if random.random() < nivel else 0
            else:
                Muestra[i][j] = np.clip(np.random.normal(loc=nivel, scale=0.1), 0, 1)
    return Muestra

def EvaluarMuestra(Muestra, NumMuestras, NumVar, X_train, X_test, y_train, y_test, modelo):
    x = [0] * NumVar
    for i in range(NumMuestras):
        for j in range(NumVar):
            x[j] = Muestra[i][j]
        Muestra[i][NumVar] = Evaluar(x, X_train, X_test, y_train, y_test, NumVar, modelo=modelo)
    return Muestra

def CalcularProb(Prob, df, NumElite, NumVar, modo):
    for i in range(NumVar):
        Prob[i] = df.iloc[:NumElite, i].mean()
    return Prob

# ======================= SCRIPT PRINCIPAL ========================

def cross_entropy_selection(
    data, NumMuestras, NumElite, NumIter, ProbMin, modelo_usado, modo,
    usar_smote='auto', smote_threshold=0.6
):
    """
    Ejecuta el algoritmo de selección de variables por Cross Entropy sobre un DataFrame.
    usar_smote: 'auto', True o False
    smote_threshold: umbral de desbalanceo para aplicar SMOTE (por ejemplo, 0.6 = 60%)
    """
    warnings.filterwarnings("ignore")

    x_cols = [col for col in data.columns if col != "SeriousDlqin2yrs"]
    NumVar = len(x_cols)
    xdata = data[x_cols]
    y = data[["SeriousDlqin2yrs"]]

    Muestra = [[0.0 for _ in range(NumVar)] + [0.0] for _ in range(NumMuestras)]
    Prob = [0.5 for _ in range(NumVar)]

    Muestra = GenerarMuestraIni(Muestra, Prob, NumMuestras, NumVar, ProbMin, modo)

    OptimoFinal = float('inf')
    MejorSolucion = None
    errores_iter = []

    # Split de datos fuera del bucle para consistencia
    X_train, X_test, y_train, y_test = train_test_split(xdata, y, test_size=0.2, random_state=42)

    # --- SMOTE: parametrizado y solo para binario ---
    aplicar_smote = False
    if usar_smote is True:
        aplicar_smote = True
    elif usar_smote == 'auto':
        clase_counts = y_train.value_counts(normalize=True)
        if (clase_counts > smote_threshold).any():
            aplicar_smote = True
    if aplicar_smote:
        # Solo aplicar si es binario
        if smote_available and len(np.unique(y_train)) == 2:
            smote = SMOTE(random_state=42)
            y_train_array = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
            X_train, y_train = smote.fit_resample(X_train, y_train_array)
            print(f"SMOTE aplicado: dataset de entrenamiento balanceado (umbral={smote_threshold}).")
        else:
            print("SMOTE no aplicado: solo se aplica a clasificación binaria y si imblearn está disponible.")
    else:
        print("SMOTE no aplicado.")

    for i in range(NumIter):
        Muestra = GenerarMuestra(Muestra, Prob, NumMuestras, NumElite, NumVar, ProbMin, modo)
        Muestra = EvaluarMuestra(Muestra, NumMuestras, NumVar, X_train, X_test, y_train, y_test, modelo=modelo_usado)
        df = pd.DataFrame(Muestra)
        df = df.sort_values(by=df.columns[NumVar])
        Prob = CalcularProb(Prob, df, NumElite, NumVar, modo)
        errores_iter.append(df.iat[0, NumVar])
        if df.iat[0, NumVar] < OptimoFinal:
            OptimoFinal = df.iat[0, NumVar]
            MejorSolucion = df.iloc[0, :NumVar].tolist()
        print(f"Iter {i + 1}, mejor error: {df.iat[0, NumVar]:.4f}")
        print(f"Probabilidades: {Prob}")

    print(f"Optimo final {OptimoFinal:.4f}  numvar: {sum(MejorSolucion):.2f}")

    if MejorSolucion is None:
        raise ValueError("No se encontró una mejor solución durante las iteraciones.")

    if modo == "binary":
        todas_las_variables = [1] * NumVar
    else:
        todas_las_variables = [1.0] * NumVar

    f_con_todas = Evaluar(todas_las_variables, X_train, X_test, y_train, y_test, NumVar, modelo=modelo_usado)
    print(f"con todas las variables: {f_con_todas:.4f}")

    X_train_masked = X_train.copy()
    X_test_masked = X_test.copy()
    for i in range(NumVar):
        X_train_masked.iloc[:, i] = X_train_masked.iloc[:, i] * MejorSolucion[i]
        X_test_masked.iloc[:, i] = X_test_masked.iloc[:, i] * MejorSolucion[i]

    modelo = modelos_disponibles[modelo_usado]
    if hasattr(y_train, 'values'):
        y_train_fit = y_train.values.ravel()
    else:
        y_train_fit = y_train.ravel() if hasattr(y_train, 'ravel') else y_train
    modelo.fit(X_train_masked, y_train_fit)
    y_pred_final = modelo.predict(X_test_masked)

    if modo == "binary":
        df_vars = pd.DataFrame({
            'Variable': x_cols,
            'Seleccionada': MejorSolucion
        })
    else:
        df_vars = pd.DataFrame({
            'Variable': x_cols,
            'Probabilidad': Prob
        })

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
            round(OptimoFinal, 4),
            round(sum(MejorSolucion), 2),
            round(1 - OptimoFinal, 4),
            round(f1_score(y_test, y_pred_final, average='binary'), 4),
            round(precision, 4),
            round(recall, 4),
            round(f_con_todas, 4)
        ]
    })

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, NumIter + 1), errores_iter, marker='o', color='blue')
    plt.title('Convergencia del error (1 - accuracy) por iteración')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor error (1 - accuracy)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_vars, df_metrics, modelo_usado, modo

# ======================= CARGAR BD ========================

data = pd.read_excel(r'C:\Users\Bi_analyst\Desktop\Python\TFM\training_modified.xlsx')
data = data.iloc[:1000]

# ======================= PARÁMETROS ========================

NumMuestras = 200
NumElite = 20
NumIter = 15
ProbMin = 0.05
modelo_usado = "RF"
modo = "binary"
smote_threshold = 0.6  # <--- parametrizable

# ======================= EJECUCIÓN PRINCIPAL ========================

df_vars, df_metrics, modelo_usado, modo = cross_entropy_selection(
    data,
    NumMuestras=NumMuestras,
    NumElite=NumElite,
    NumIter=NumIter,
    ProbMin=ProbMin,
    modelo_usado=modelo_usado,
    modo=modo,
    usar_smote='auto',
    smote_threshold=smote_threshold
)

# ======================= EXPORTAR RESULTADOS ========================

import os

output_folder = r"C:\Users\Bi_analyst\Desktop\Python\TFM\CE models - outputs"
os.makedirs(output_folder, exist_ok=True)
output_filename = os.path.join(output_folder, f"{modelo_usado}_{modo}_CE_output.xlsx")

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df_vars.to_excel(writer, index=False, sheet_name="Resultados")
    start_row = len(df_vars) + 4
    df_metrics.to_excel(writer, index=False, sheet_name="Resultados", startrow=start_row)

print(f'📁 Resultados guardados en: {output_filename}')
print(df_vars.to_string(index=False))
print('\n', df_metrics.to_string(index=False))




"""
=============================================================================
SOLUCIONES — Taller Titanic (Regresión Logística) con Pipeline + ColumnTransformer
=============================================================================
Copia cada bloque en la celda correspondiente del notebook.
"""


# ============================================================================
# SECCIÓN 4 — Selección de variables
# ============================================================================
# Columnas que NO usaremos como features
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

# Separar features (X) y target (y)
y_train = train_df[TARGET]
X_train = train_df.drop(columns=[TARGET]).drop(columns=DROP_COLS)

y_public = public_df[TARGET]
X_public = public_df.drop(columns=[TARGET]).drop(columns=DROP_COLS)

# private_df NO tiene columna Survived
X_private = private_df.drop(columns=[c for c in DROP_COLS if c in private_df.columns])

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_public:", X_public.shape, "y_public:", y_public.shape)
print("X_private:", X_private.shape)
print("\nColumnas usadas:", list(X_train.columns))


# ============================================================================
# SECCIÓN 5 — Preprocesamiento con ColumnTransformer
# ============================================================================
# Identificar columnas numéricas y categóricas
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

print("Numéricas:", num_cols)
print("Categóricas:", cat_cols)

# Sub-pipeline numérico: imputar faltantes con la mediana + escalar
numeric_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Sub-pipeline categórico: imputar faltantes con la moda + one-hot
categorical_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combinar ambos sub-pipelines
preprocess = ColumnTransformer(transformers=[
    ("num", numeric_preprocess, num_cols),
    ("cat", categorical_preprocess, cat_cols)
])

preprocess


# ============================================================================
# SECCIÓN 6 — Pipeline + Regresión Logística
# ============================================================================
# Crear el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000, random_state=42)

# Pipeline completo: preprocesamiento + modelo
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# Entrenar
pipe.fit(X_train, y_train)

print("Accuracy en train:", round(pipe.score(X_train, y_train), 4))


# ============================================================================
# SECCIÓN 7 — Evaluación en el Public Test
# ============================================================================
# Predecir en el public test
y_pred_public = pipe.predict(X_public)

# Classification report
print(classification_report(y_public, y_pred_public, digits=4))

# Matriz de confusión
cm = confusion_matrix(y_public, y_pred_public)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matriz de Confusión — Public Test")
plt.show()

# ROC-AUC
if hasattr(pipe, "predict_proba"):
    y_proba_public = pipe.predict_proba(X_public)[:, 1]
    print("ROC-AUC:", round(roc_auc_score(y_public, y_proba_public), 4))


# ============================================================================
# SECCIÓN 8 — Entrenamiento final y submission
# ============================================================================
# Combinar train + public para entrenamiento final
full_train_df = pd.concat([train_df, public_df], axis=0, ignore_index=True)
X_full = full_train_df.drop(columns=[TARGET]).drop(columns=DROP_COLS)
y_full = full_train_df[TARGET]

# Re-entrenar pipeline con todos los datos etiquetados
pipe.fit(X_full, y_full)
print("Accuracy en full train:", round(pipe.score(X_full, y_full), 4))

# Predecir el private test
private_pred = pipe.predict(X_private)

# Generar submission.csv
ID_COL = "PassengerId"
assert ID_COL in private_df.columns, "PassengerId debe existir en el private test"

submission = pd.DataFrame({
    ID_COL: private_df[ID_COL].values,
    TARGET: private_pred
})

out_path = os.path.join(DATA_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"Submission guardado en: {out_path}")
print(f"Forma: {submission.shape}")
submission.head()


# ============================================================================
# SECCIÓN 9 — Ejercicios (elige al menos 3)
# ============================================================================

# ------- Ejercicio 1: Variables (incluir/excluir Cabin, Name, Ticket) -------
# Cambio: Incluir Ticket y Cabin junto con las demás features
# (necesitas OneHotEncoder con handle_unknown="ignore" para la alta cardinalidad)

DROP_COLS_EX1 = ["PassengerId", "Name"]  # solo quitamos PassengerId y Name
X_train_ex1 = train_df.drop(columns=[TARGET]).drop(columns=DROP_COLS_EX1)
X_public_ex1 = public_df.drop(columns=[TARGET]).drop(columns=DROP_COLS_EX1)

num_cols_ex1 = X_train_ex1.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols_ex1 = X_train_ex1.select_dtypes(include=["object"]).columns.tolist()

preprocess_ex1 = ColumnTransformer(transformers=[
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols_ex1),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols_ex1)
])
pipe_ex1 = Pipeline([("preprocess", preprocess_ex1), ("model", LogisticRegression(max_iter=1000, random_state=42))])
pipe_ex1.fit(X_train_ex1, y_train)
y_pred_ex1 = pipe_ex1.predict(X_public_ex1)
print("=== Ejercicio 1: Incluir Ticket y Cabin ===")
print(classification_report(y_public, y_pred_ex1, digits=4))
# Explicación: Ticket y Cabin tienen alta cardinalidad y muchos valores faltantes
# (Cabin ~78% NaN). Incluirlos generalmente NO mejora porque el one-hot crea
# cientos de columnas con poca señal, generando sobreajuste o ruido.


# ------- Ejercicio 2: Cabin → HasCabin -------
train_df_ex2 = train_df.copy()
public_df_ex2 = public_df.copy()
train_df_ex2["HasCabin"] = train_df_ex2["Cabin"].notna().astype(int)
public_df_ex2["HasCabin"] = public_df_ex2["Cabin"].notna().astype(int)

DROP_COLS_EX2 = ["PassengerId", "Name", "Ticket", "Cabin"]
X_train_ex2 = train_df_ex2.drop(columns=[TARGET]).drop(columns=DROP_COLS_EX2)
X_public_ex2 = public_df_ex2.drop(columns=[TARGET]).drop(columns=DROP_COLS_EX2)

num_cols_ex2 = X_train_ex2.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols_ex2 = X_train_ex2.select_dtypes(include=["object"]).columns.tolist()

preprocess_ex2 = ColumnTransformer(transformers=[
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols_ex2),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols_ex2)
])
pipe_ex2 = Pipeline([("preprocess", preprocess_ex2), ("model", LogisticRegression(max_iter=1000, random_state=42))])
pipe_ex2.fit(X_train_ex2, y_train)
y_pred_ex2 = pipe_ex2.predict(X_public_ex2)
print("=== Ejercicio 2: HasCabin ===")
print(classification_report(y_public, y_pred_ex2, digits=4))
# Explicación: HasCabin binario captura si el pasajero tenía cabina asignada,
# lo cual correlaciona con la clase socioeconómica. Es más útil que el texto
# original de Cabin porque reduce el ruido de alta cardinalidad a una sola
# señal informativa.


# ------- Ejercicio 3: Título del nombre -------
import re

def extract_title(name):
    match = re.search(r',\s*(\w+)\.', name)
    return match.group(1) if match else "Other"

train_df_ex3 = train_df.copy()
public_df_ex3 = public_df.copy()
train_df_ex3["Title"] = train_df_ex3["Name"].apply(extract_title)
public_df_ex3["Title"] = public_df_ex3["Name"].apply(extract_title)

DROP_COLS_EX3 = ["PassengerId", "Name", "Ticket", "Cabin"]
X_train_ex3 = train_df_ex3.drop(columns=[TARGET]).drop(columns=DROP_COLS_EX3)
X_public_ex3 = public_df_ex3.drop(columns=[TARGET]).drop(columns=DROP_COLS_EX3)

num_cols_ex3 = X_train_ex3.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols_ex3 = X_train_ex3.select_dtypes(include=["object"]).columns.tolist()

preprocess_ex3 = ColumnTransformer(transformers=[
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols_ex3),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols_ex3)
])
pipe_ex3 = Pipeline([("preprocess", preprocess_ex3), ("model", LogisticRegression(max_iter=1000, random_state=42))])
pipe_ex3.fit(X_train_ex3, y_train)
y_pred_ex3 = pipe_ex3.predict(X_public_ex3)
print("=== Ejercicio 3: Título del nombre ===")
print(classification_report(y_public, y_pred_ex3, digits=4))
# Explicación: El título (Mr, Mrs, Miss, Master, etc.) captura información
# sobre género, edad aproximada y estatus social que complementa a Sex y Age.
# "Master" indica niños varones, "Mrs" mujeres casadas, lo que puede mejorar
# la predicción al dar más granularidad que solo "male"/"female".


# ------- Ejercicio 4: Regularización (C = 0.1, 1, 10) -------
print("=== Ejercicio 4: Regularización ===")
for c_val in [0.1, 1, 10]:
    pipe_ex4 = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(C=c_val, max_iter=1000, random_state=42))
    ])
    pipe_ex4.fit(X_train, y_train)
    y_pred_ex4 = pipe_ex4.predict(X_public)
    print(f"\n--- C = {c_val} ---")
    print(classification_report(y_public, y_pred_ex4, digits=4))
# Explicación: C controla la fuerza de regularización (C bajo → más regularización).
# Con C=0.1, el modelo penaliza más los coeficientes grandes, lo que puede
# reducir overfitting pero podría perder recall. Con C=10, el modelo es más
# libre de ajustarse a los datos, pudiendo ganar recall pero arriesgando overfitting.


# ------- Ejercicio 5: class_weight="balanced" -------
pipe_ex5 = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
])
pipe_ex5.fit(X_train, y_train)
y_pred_ex5 = pipe_ex5.predict(X_public)
print("=== Ejercicio 5: class_weight='balanced' ===")
print(classification_report(y_public, y_pred_ex5, digits=4))
# Explicación: class_weight="balanced" asigna pesos inversamente proporcionales
# a la frecuencia de cada clase. Como hay ~62% clase 0 y ~38% clase 1,
# "balanced" aumenta la importancia de la clase minoritaria (Survived=1),
# lo que generalmente mejora el recall de clase 1 pero puede reducir su
# precision si empieza a clasificar más falsos positivos.


# ------- Ejercicio 6: Sin StandardScaler -------
numeric_preprocess_no_scaler = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])
preprocess_no_scaler = ColumnTransformer(transformers=[
    ("num", numeric_preprocess_no_scaler, num_cols),
    ("cat", categorical_preprocess, cat_cols)
])
pipe_ex6 = Pipeline([
    ("preprocess", preprocess_no_scaler),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
pipe_ex6.fit(X_train, y_train)
y_pred_ex6 = pipe_ex6.predict(X_public)
print("=== Ejercicio 6: Sin StandardScaler ===")
print(classification_report(y_public, y_pred_ex6, digits=4))
# Explicación: La regresión logística con regularización (por defecto penalty='l2')
# es sensible a la escala de las features. Sin escalar, variables con rangos
# grandes (como Fare: 0-512) dominan la regularización, mientras que variables
# con rangos pequeños (como SibSp: 0-8) se regularizanen exceso. El escalado
# iguala las escalas para que la regularización actúe equitativamente.

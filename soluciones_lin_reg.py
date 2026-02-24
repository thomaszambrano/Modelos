"""
=============================================================================
SOLUCIONES — Ejercicio Regresión Lineal: Predicción de Ganancias en Startups
=============================================================================
Copia cada bloque en la celda correspondiente del notebook.
"""


# ============================================================================
# SECCIÓN 1 — Importación de librerías
# ============================================================================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ============================================================================
# SECCIÓN 2 — Carga de datos (varias celdas)
# ============================================================================

# --- Celda: Carga el dataset ---
df = pd.read_csv("50_Startups.csv")
print("Shape:", df.shape)

# --- Celda: Primeras filas ---
df.head()

# --- Celda: Tipos de datos y nulos ---
print(df.dtypes)
print("\nValores nulos por columna:")
print(df.isnull().sum())

# --- Celda: Estadísticos descriptivos ---
df.describe()

# --- Celda: Startups por estado ---
df["State"].value_counts()


# ============================================================================
# SECCIÓN 3 — Separación X/y y división train/val/test
# ============================================================================
# Define X e y:
X = df.drop(columns=["Profit"])
y = df["Profit"]

# Paso 1: reserva el 20 % como test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Paso 2: del 80 % restante, separa un 20 % adicional como val set
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42
)

# Verifica los tamaños
print("Train :", X_train.shape, y_train.shape)
print("Val   :", X_val.shape,   y_val.shape)
print("Test  :", X_test.shape,  y_test.shape)


# ============================================================================
# SECCIÓN 4 — Preprocesamiento
# ============================================================================

# --- Celda: Identificar columnas ---
numerical_cols = ["R&D Spend", "Administration", "Marketing Spend"]
categorical_cols = ["State"]

# --- Celda: Pipeline numérico ---
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler())
])

# --- Celda: Pipeline categórico ---
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[["California", "Florida", "New York"]]))
])

# --- Celda: ColumnTransformer ---
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# ============================================================================
# SECCIÓN 5 — Pipeline completo
# ============================================================================
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

pipe


# ============================================================================
# SECCIÓN 6 — Entrenamiento
# ============================================================================

# --- Celda: Entrenar y extraer coeficientes ---
pipe.fit(X_train, y_train)

# Extraer coeficientes
modelo = pipe.named_steps["model"]
feature_names = numerical_cols + categorical_cols

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coeficiente": modelo.coef_
})
print(coef_df)
print(f"\nIntercepto: {modelo.intercept_:.2f}")

# --- Celda: Gráfica de coeficientes ---
colors = ["green" if v > 0 else "red" for v in modelo.coef_]
plt.barh(feature_names, modelo.coef_, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coeficiente")
plt.title("Coeficientes del modelo de Regresión Lineal")
plt.tight_layout()
plt.show()


# ============================================================================
# SECCIÓN 7 — Evaluación en los tres conjuntos
# ============================================================================
import math

# Predicciones en los tres conjuntos
y_pred_train = pipe.predict(X_train)
y_pred_val   = pipe.predict(X_val)
y_pred_test  = pipe.predict(X_test)

# Métricas
results = {
    "Conjunto": ["Train", "Val", "Test"],
    "MAE":  [
        mean_absolute_error(y_train, y_pred_train),
        mean_absolute_error(y_val, y_pred_val),
        mean_absolute_error(y_test, y_pred_test)
    ],
    "RMSE": [
        math.sqrt(mean_squared_error(y_train, y_pred_train)),
        math.sqrt(mean_squared_error(y_val, y_pred_val)),
        math.sqrt(mean_squared_error(y_test, y_pred_test))
    ],
    "R²":   [
        r2_score(y_train, y_pred_train),
        r2_score(y_val, y_pred_val),
        r2_score(y_test, y_pred_test)
    ],
}

df_metrics = pd.DataFrame(results).set_index("Conjunto")
df_metrics = df_metrics.round(2)
print(df_metrics)

# Diagnóstico:
# Si R² Train ≈ R² Val → el modelo generaliza bien (no hay sobreajuste significativo)
# Si R² Train >> R² Val → hay sobreajuste

# --- Celda: Gráficas ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Gráfica 1a — Real vs Predicho (Val)
plt.sca(axes[0, 0])
plt.scatter(y_val, y_pred_val, alpha=0.7)
min_val = min(y_val.min(), y_pred_val.min())
max_val = max(y_val.max(), y_pred_val.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Predicción perfecta")
plt.xlabel("Profit Real")
plt.ylabel("Profit Predicho")
plt.title("Real vs Predicho — Val")
plt.legend()

# Gráfica 1b — Real vs Predicho (Test)
plt.sca(axes[0, 1])
plt.scatter(y_test, y_pred_test, alpha=0.7)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Predicción perfecta")
plt.xlabel("Profit Real")
plt.ylabel("Profit Predicho")
plt.title("Real vs Predicho — Test")
plt.legend()

# Gráfica 2a — Residuos (Val)
plt.sca(axes[1, 0])
residuals_val = y_val - y_pred_val
plt.hist(residuals_val, bins=10, edgecolor="black", alpha=0.7)
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Residuo (Real - Predicho)")
plt.ylabel("Frecuencia")
plt.title("Residuos — Val")

# Gráfica 2b — Residuos (Test)
plt.sca(axes[1, 1])
residuals_test = y_test - y_pred_test
plt.hist(residuals_test, bins=10, edgecolor="black", alpha=0.7)
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Residuo (Real - Predicho)")
plt.ylabel("Frecuencia")
plt.title("Residuos — Test")

plt.tight_layout()
plt.show()

# Reflexión visual:
# Los residuos deben centrarse cerca de 0 en ambos conjuntos.
# Si val y test muestran patrones similares, el modelo generaliza consistentemente.


# ============================================================================
# SECCIÓN 8 — Reflexión sobre preprocesamiento (respuestas para la Markdown)
# ============================================================================
"""
Pregunta 1: MinMaxScaler vs StandardScaler
- MinMaxScaler transforma cada variable al rango [0, 1] usando (x - x_min)/(x_max - x_min).
  Preserva la forma de la distribución y es útil cuando se quiere un rango acotado.
- StandardScaler centra los datos en media=0 y desv.estándar=1 usando (x - μ)/σ.
  Es preferible cuando los datos tienen outliers fuertes (MinMaxScaler es sensible
  a outliers porque depende de min/max).
- Para regresión lineal simple, ambos dan el mismo R² porque el modelo ajusta
  coeficientes libremente, pero los coeficientes cambian de magnitud.

Pregunta 2: OrdinalEncoder en State
- No fue la mejor decisión porque State NO tiene un orden natural. Al asignar
  California=0, Florida=1, New York=2, el modelo interpreta que "New York es el
  doble que Florida" y "Florida es mayor que California", creando una relación
  ordinal artificial. Lo correcto sería usar OneHotEncoder, que crea variables
  binarias independientes sin imponer orden.

Pregunta 3: Variable con más peso
- R&D Spend tiene el coeficiente más grande (positivo), lo que indica que es
  la variable más influyente en la predicción de Profit. Tiene sentido desde
  el punto de vista de negocio: la inversión en I+D genera innovación y
  ventaja competitiva, que se traduce en mayores ganancias.

Pregunta 4: Para qué sirve el val set
- El val set sirve para detectar sobreajuste (comparando métricas de train vs val)
  y para comparar variantes del modelo (ej. distintos encoders, scalers, features)
  sin contaminar el test set. La decisión de cuál modelo es mejor se toma mirando
  solo train y val.

Pregunta 5: ¿Sobreajuste o subajuste?
- Hay que comparar R² train y R² val. Si son similares y altos (>0.9), el modelo
  ajusta bien y generaliza. Si R² train es muy superior a R² val, hay sobreajuste.
  Si ambos son bajos (<0.5), hay subajuste. (El diagnóstico exacto depende de los
  valores que obtengas al ejecutar.)

Pregunta 6: ¿Por qué dividir ANTES de transformar?
- Si transformas todo X antes de dividir, el scaler calcula min/max (o μ/σ) usando
  datos de val y test, lo que constituye data leakage. El modelo tendría acceso
  indirecto a información del futuro (val/test), haciendo que las métricas sean
  artificialmente optimistas y no representen el rendimiento real en datos nuevos.
"""


# ============================================================================
# SECCIÓN 9 — Muestra sintética propia
# ============================================================================

# --- Celda: Diseñar startups ficticias ---
# Startup 1 (Alto I+D, poco marketing): espero Profit alto (~$150,000+)
# Startup 2 (Solo marketing, cero I+D): espero Profit bajo (~$50,000-80,000)
# Startup 3 (Gastos mínimos en todo): espero Profit muy bajo (~$30,000-50,000)
# Startup 4 (Startup promedio): espero Profit medio (~$100,000-120,000)
# Startup 5 (Valores extremos altos): espero Profit muy alto (~$190,000+)

muestra = pd.DataFrame({
    "R&D Spend":       [160000, 0,     10000,  75000,  200000],
    "Administration":  [100000, 90000, 50000,  120000, 170000],
    "Marketing Spend": [20000,  400000, 15000, 200000, 450000],
    "State":           ["California", "New York", "Florida", "California", "New York"]
})

muestra

# --- Celda: Predicciones ---
predicciones = pipe.predict(muestra)
muestra["Profit Predicho"] = predicciones.round(2)
muestra

"""
Reflexión final sobre la muestra sintética (para la Markdown):

1. ¿Las predicciones coinciden con lo que esperabas?
   En general sí: la startup con alto I+D tiene el Profit más alto, y la de
   gastos mínimos tiene el más bajo. R&D Spend domina la predicción.

2. ¿Alguna predicción sorprendente?
   La startup "solo marketing" probablemente muestra un Profit menor al esperado
   porque el modelo aprendió que R&D Spend es mucho más predictivo que Marketing Spend.

3. ¿Cambia mucho el Profit según State con gastos idénticos?
   La diferencia por State es relativamente pequeña porque OrdinalEncoder
   asigna un solo coeficiente lineal. El impacto real de la ubicación
   probablemente no sigue un patrón ordinal.

4. ¿Profit negativo es error del modelo?
   No necesariamente — una startup con gastos muy bajos podría tener pérdidas
   reales. Sin embargo, el modelo extrapola linealmente, así que predicciones
   muy negativas podrían ser artefactos del modelo.
"""


# ============================================================================
# SECCIÓN 10 — Desafíos opcionales
# ============================================================================
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Desafío A: OneHotEncoder vs OrdinalEncoder ---
cat_pipe_ohe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", sparse_output=False))
])

preprocessor_ohe = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numerical_cols),
    ("cat", cat_pipe_ohe, categorical_cols)
])

pipe_ohe = Pipeline(steps=[
    ("preprocess", preprocessor_ohe),
    ("model", LinearRegression())
])

pipe_ohe.fit(X_train, y_train)
y_pred_val_ohe = pipe_ohe.predict(X_val)
print("=== Desafío A: OneHotEncoder ===")
print(f"R² Val (OrdinalEncoder): {r2_score(y_val, y_pred_val):.4f}")
print(f"R² Val (OneHotEncoder):  {r2_score(y_val, y_pred_val_ohe):.4f}")
# OneHotEncoder es más correcto para State porque no impone un orden artificial.
# Con drop='first', California se convierte en la referencia y los coeficientes
# de Florida y New York se interpretan como la diferencia respecto a California.


# --- Desafío B: MinMaxScaler vs StandardScaler ---
num_pipe_ss = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor_ss = ColumnTransformer(transformers=[
    ("num", num_pipe_ss, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

pipe_ss = Pipeline(steps=[
    ("preprocess", preprocessor_ss),
    ("model", LinearRegression())
])

pipe_ss.fit(X_train, y_train)
y_pred_val_ss = pipe_ss.predict(X_val)
print("\n=== Desafío B: StandardScaler ===")
print(f"R² Val (MinMaxScaler):   {r2_score(y_val, y_pred_val):.4f}")
print(f"R² Val (StandardScaler): {r2_score(y_val, y_pred_val_ss):.4f}")
# El R² debería ser muy similar porque la regresión lineal es invariante
# a transformaciones lineales de las features. Los coeficientes cambian
# en magnitud pero las predicciones son equivalentes.


# --- Desafío C: Feature engineering (RD_ratio) ---
df_fe = df.copy()
df_fe["RD_ratio"] = df_fe["R&D Spend"] / (df_fe["R&D Spend"] + df_fe["Marketing Spend"] + 1)

X_fe = df_fe.drop(columns=["Profit"])
y_fe = df_fe["Profit"]

X_trainval_fe, X_test_fe, y_trainval_fe, y_test_fe = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=42
)
X_train_fe, X_val_fe, y_train_fe, y_val_fe = train_test_split(
    X_trainval_fe, y_trainval_fe, test_size=0.2, random_state=42
)

numerical_cols_fe = numerical_cols + ["RD_ratio"]

preprocessor_fe = ColumnTransformer(transformers=[
    ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", MinMaxScaler())]), numerical_cols_fe),
    ("cat", categorical_pipeline, categorical_cols)
])

pipe_fe = Pipeline(steps=[
    ("preprocess", preprocessor_fe),
    ("model", LinearRegression())
])

pipe_fe.fit(X_train_fe, y_train_fe)
y_pred_val_fe = pipe_fe.predict(X_val_fe)
print("\n=== Desafío C: Feature Engineering (RD_ratio) ===")
print(f"R² Val (sin RD_ratio): {r2_score(y_val, y_pred_val):.4f}")
print(f"R² Val (con RD_ratio): {r2_score(y_val_fe, y_pred_val_fe):.4f}")
# Si el R² mejora en val, entonces evalúa en test:
y_pred_test_fe = pipe_fe.predict(X_test_fe)
print(f"R² Test (con RD_ratio): {r2_score(y_test_fe, y_pred_test_fe):.4f}")

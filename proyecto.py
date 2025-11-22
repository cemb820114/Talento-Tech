# ==========
# 1. IMPORTAR LIBRERÍAS
# ==========

# Pandas: manejo de tablas de datos (DataFrame)
import pandas as pd

# NumPy: operaciones numéricas (lo usaremos para algunas funciones)
import numpy as np

# Matplotlib y Seaborn: visualización básica (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Importas la clase OneHotEncoder, que sirve para convertir texto en números mediante One Hot Encoding.
from sklearn.preprocessing import OneHotEncoder

# StandardScaler: escalamiento de variables numéricas
from sklearn.preprocessing import StandardScaler

# train_test_split: separar datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Opcional: un modelo simple para demostrar uso (ej: Regresión Logística)
from sklearn.linear_model import LogisticRegression

import category_encoders as ce
import category_encoders as ce

# ===== IMPORTES ADICIONALES PARA MODELOS Y PIPELINES =====

# Modelos supervisados
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Modelo no supervisado
from sklearn.cluster import KMeans

# Métricas
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Herramientas avanzadas
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Guardar pipeline entrenado
#PIPELINES permite encadenar todos los pasos de tu proceso de Machine Learning en un solo flujo ordenado, desde el preprocesamiento hasta el modelo final.

import joblib


#CARGA DE DATASET
df1_xlsx=pd.read_excel("Datos Zaragoza1.xlsx")
print(df1_xlsx)

#copy
df_csv=df1_xlsx.copy()
print(df_csv)

#BORRAR COLUMNAS
df_csv.drop(columns=['Coordenadas_gps','Municipio',"Estado_edificio","Dias_hasta_expiracion"], inplace=True)
print(df_csv.columns)

#ORDENAR DE MENOR A MAYOR
# Si la columna no esta en formato datetime, primero conviértela
df_csv['Fecha_emision'] = pd.to_datetime(df_csv['Fecha_emision'])
df2 = df_csv.sort_values(by='Fecha_emision')


#FILTRO PARA BORRAR FILAS MAYORES A 2016
print(df2[df2["Anio_emision"]>2016])

# ELIMINAR LAS FILAS EN LAS POSICIONES 
df3 = df2.drop(df2.index[57111:179852])


# ELIMINACION DE DATO ATIPICO
df3=df3.drop(df3.index[0])

# ELIMINACION DE DUPLICADOS DEJANDO SOLO LA PRIMERA APARICION
print(df3.drop_duplicates())
df3 = df3.drop_duplicates()
print(df3.info())

print(df3[df3["Anio_emision"]>2017])
######empty DATA FRAME###################

#filtro para borrar filas con guion y con cero consumo 
print(df3.columns)
df4 = df3[df3['Clasificacion_consumo'] != '-']
df4 = df4[df4['ConsumoKWh/m2/Anio'] != 0]

##### CONVERTIR CATEGORIAS DE CONSUMO Y DE EMISIONES A TIPO BINARIO (0 Y 1) ###############
## CLASIFICACION EMISIONES
####  A, B, C, D Y E SON 0  ####  F Y G SON 1
print(df4.columns)
df4['Clasificacion_Emisiones'] = df4['Clasificacion_Emisiones'].replace({'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,'F': 1, 'G': 1})
print(df4.info())
## CLASIFICACION CONSUMO
####  A, B, C, D Y E SON 0  ####  F Y G SON 1
print(df4.columns)
df4['Clasificacion_consumo'] = df4['Clasificacion_consumo'].replace({'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,'F': 1, 'G': 1})
print(df4.info())

###### CONVERTIR TIPO DE EDIFICIO Y PROVINCIA A NUMEROS ###########
print(df4['Tipo_edificio'].value_counts())
#TIPO DE EDIFICIO
#VIVIENDA INDIVIDUAL 0
#BLOQUE COMPLETO 1
#LOCAL 2
#UNIFAMILIAR 3
#EDIFICIO COMPLETO 4
df4['Tipo_edificio'] = pd.factorize(df4['Tipo_edificio'])[0]

print(df4['Provincia'].value_counts())
#PROVINCIA
#ZARAGOZA 0
#HUESCA 1
#TERUEL 2
df4['Provincia'] = pd.factorize(df4['Provincia'])[0]
print(df4.info())


###### EXPORTAR DATA SET LIMPIO ###########

df4.to_csv("proyecto_limpio.csv", index=False)

df = pd.read_csv("proyecto_limpio_ml.csv")
print(df.info())

#***************GRAFICOS**************************

###########GRAFICO BARRAS EMISION VS TIPO EDIFICACION
plt.bar(df['Tipo_edificio'],df['Emision_CO2'] , color="blue")
plt.title("Grafico de Barras")
plt.xlabel("EDIICACION")
plt.ylabel("EMSION")
plt.legend()
plt.grid(True)
plt.show()


###########GRAFICO BARRAS AÑO CONSTRUCCION VS CONSUMO
plt.bar(df['Anio_construccion'],df['ConsumoKWh/m2/Anio'] ,color="green")
plt.title("Grafico de Barras")
plt.xlabel("AÑO DE CONSTRUCCION")
plt.ylabel("CONSUMO (KWH/M2)")
plt.legend()

plt.xlim(2000, df['Anio_construccion'].max())

plt.grid(True)
plt.show()

###########GRAFICO CIRCULAR EMISIONES
data = df['Emision_CO2']
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color="red", edgecolor="black", alpha=0.7)
plt.title("Histograma de Emisiones de CO2", fontsize=14, fontweight='bold')
plt.xlabel("Emisiones CO2 (kg CO2/m²/año)", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico circular
import matplotlib.pyplot as plt
import pandas as pd

# Calcular cuartiles
Q1 = df['Emision_CO2'].quantile(0.25)
Q2 = df['Emision_CO2'].quantile(0.50)  # Mediana
Q3 = df['Emision_CO2'].quantile(0.75)

# Función de clasificación
def clasificar_cuartiles(valor):
    if valor <= Q1:
        return 'Muy Baja'
    elif valor <= Q2:
        return 'Baja'
    elif valor <= Q3:
        return 'Alta'
    else:
        return 'Muy Alta'

# Aplicar clasificación
df['Categoria_Emision'] = df['Emision_CO2'].apply(clasificar_cuartiles)

# Contar las categorías
categorias = df['Categoria_Emision'].value_counts()

# Ordenar las categorías en el orden deseado
orden = ['Muy Baja', 'Baja', 'Alta', 'Muy Alta']
categorias = categorias.reindex(orden)

# Crear el gráfico circular
plt.figure(figsize=(10, 8))
colores = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']  # Verde, Amarillo, Naranja, Rojo

plt.pie(categorias, 
        labels=categorias.index, 
        autopct='%1.1f%%', 
        colors=colores, 
        startangle=90,
        explode=(0.05, 0.05, 0.05, 0.05),  # Separar ligeramente las porciones
        shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'})

plt.title('Distribución de Emisiones de CO2 por Categoría\n(Clasificación por Cuartiles)', 
          fontsize=16, 
          fontweight='bold',
          pad=20)

# Añadir leyenda con los rangos
leyenda = [
    f'Muy Baja: ≤ {Q1:.2f} kg CO2/m²/año',
    f'Baja: {Q1:.2f} - {Q2:.2f} kg CO2/m²/año',
    f'Alta: {Q2:.2f} - {Q3:.2f} kg CO2/m²/año',
    f'Muy Alta: > {Q3:.2f} kg CO2/m²/año'
]
plt.legend(leyenda, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

plt.axis('equal')  # Para que sea un círculo perfecto
plt.tight_layout()
plt.show()


# Mostrar información adicional
print("="*50)
print("RESUMEN DE CATEGORÍAS")
print("="*50)
print(f"\nRangos de emisiones:")
print(f"  Muy Baja: ≤ {Q1:.2f} kg CO2/m²/año")
print(f"  Baja: {Q1:.2f} - {Q2:.2f} kg CO2/m²/año")
print(f"  Alta: {Q2:.2f} - {Q3:.2f} kg CO2/m²/año")
print(f"  Muy Alta: > {Q3:.2f} kg CO2/m²/año")
print(f"\nDistribución de edificios:")
print(categorias)
print(f"\nTotal de edificios: {len(df)}")

# ============================================================
# PREPARAR DATASET PARA MODELADO (df_modelo)
# ============================================================

print("\n===== 9. PREPARACIÓN DE df_modelo PARA ML =====")

# En este dataset ya NO hay columnas tipo object (sex=0/1, embarked_OHE)
df_modelo = df.copy()

# Confirmar que no queden columnas object
columnas_object = df_modelo.select_dtypes(include=["object"]).columns
if len(columnas_object) > 0:
    print("⚠ Aún hay columnas object. Se eliminarán:", list(columnas_object))
    df_modelo = df_modelo.drop(columns=columnas_object)
else:
    print("✅ No hay columnas object. Todas son numéricas o categóricas codificadas.")

print("\nColumnas finales en df_modelo:")
print(df_modelo.columns)


# ============================================================
# SELECCIÓN DE VARIABLES (X, y)
# ============================================================

print("\n===== 10. SEPARACIÓN DE X (FEATURES) E y (TARGET) =====")

# Para Titanic, la variable objetivo es 'survived'
NOMBRE_TARGET = "clasificacion_emisiones"

if NOMBRE_TARGET not in df_modelo.columns:
    raise ValueError(
        f"ERROR: La columna objetivo '{NOMBRE_TARGET}' no existe en df_modelo.\n"
        f"Columnas disponibles: {list(df_modelo.columns)}"
    )

# y: lo que queremos predecir (sobrevivió o no)
y = df_modelo[NOMBRE_TARGET]

# X: todas las demás columnas (features)
X = df_modelo.drop(NOMBRE_TARGET, axis=1)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

print("\nDistribución de la variable objetivo (y):")
print(y.value_counts(normalize=True))


# ============================================================
# ESCALAMIENTO DE VARIABLES NUMÉRICAS
# ============================================================

print("\n===== 11. ESCALAMIENTO DE VARIABLES NUMÉRICAS =====")

# Todas las columnas de X son numéricas (int/float)
columnas_num_X = X.columns
print("Columnas numéricas a escalar:", list(columnas_num_X))

# Creamos el StandardScaler
escalador = StandardScaler()

# Ajuste y transformación
X_escalado = X.copy()
X_escalado[columnas_num_X] = escalador.fit_transform(X[columnas_num_X])

print("\nPrimeras filas de X_escalado:")
print(X_escalado.head())


# ============================================================
# DIVISIÓN EN TRAIN Y TEST
# ============================================================

print("\n===== 12. DIVISIÓN EN TRAIN / TEST =====")

X_train, X_test, y_train, y_test = train_test_split(
    X_escalado,
    y,
    test_size=0.2,        # 20% test
    random_state=42,
    stratify=y            # mantener proporción de clases
)

print("Shape X_train:", X_train.shape)
print("Shape X_test :", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test :", y_test.shape)



# ============================================================
# ENTRENAMIENTO DE VARIOS MODELOS SUPERVISADOS
# ============================================================

print("\n===== 13. MODELOS SUPERVISADOS (CLASIFICACIÓN) =====")
print("Todos estos modelos son SUPERVISADOS: usan X (features) e y (target).")
print("Problema: clasificación binaria (0 = no sobrevivió, 1 = sobrevivió).")

# Diccionario de modelos a evaluar
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

resultados_modelos = []

for nombre, modelo in modelos.items():
    print("\n--------------------------------------------------------")
    print(f"Entrenando modelo supervisado: {nombre}")

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicción
    y_pred = modelo.predict(X_test)

    # Métrica principal: Accuracy
    acc = accuracy_score(y_test, y_pred)
    resultados_modelos.append((nombre, acc))

    print(f"Accuracy en test: {acc:.4f}")

    # Matriz de confusión
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Reporte de clasificación (precision, recall, f1-score)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

# Resumen comparativo
print("\n===== 13.1 RESUMEN DE ACCURACY DE CADA MODELO =====")
for nombre, acc in resultados_modelos:
    print(f"{nombre:20s} -> Accuracy = {acc:.4f}")

# Elegir el mejor modelo según Accuracy
mejor_modelo_nombre, mejor_acc = max(resultados_modelos, key=lambda t: t[1])
print(f"\n✅ Mejor modelo (según Accuracy): {mejor_modelo_nombre} con {mejor_acc:.4f}")

# ============================================================
# 13. MODELO NO SUPERVISADO: KMEANS (CLUSTERING)
# ============================================================

print("\n===== 14. MODELO NO SUPERVISADO: KMEANS =====")
print("KMeans es NO SUPERVISADO: sólo usa X (no usa y).")

# Entrenar KMeans con 2 clusters (podrían representar 2 grupos de pasajeros)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_escalado)

labels_kmeans = kmeans.labels_

print("\nTamaño de cada cluster:")
valores, cuentas = np.unique(labels_kmeans, return_counts=True)
for cl, c in zip(valores, cuentas):
    print(f"Cluster {cl}: {c} clasificacion")

# Comparar clusters con la variable real clasificacion(solo para análisis)
print("\nTabla cruzada: Cluster vs clasifcacion")
print(pd.crosstab(labels_kmeans, y))

print(df.columns)


# ============================================================xxxxxxxxxxxxxxxxxx
# PIPELINE + GRIDSEARCHCV (FLUJO COMPLETO)
# ============================================================

print("\n===== 15. PIPELINE + GRIDSEARCHCV (KNN) =====")
print("Ahora usamos el df ORIGINAL (con 'emision_co2' y 'superficie_m2' como texto) y un Pipeline completo.")

# X_raw: dataset original sin la columna objetivo
X_raw = df.drop("clasificacion_emisiones", axis=1)
y_raw = df["clasificacion_emisiones"]

# Definimos qué columnas son numéricas y cuáles categóricas en el df original
columnas_numericas = ["clasificacion_consumo", "consumokwhm2anio", "tipo_edificio", "provincia", "anio_emision", "anio_construccion"]
columnas_categoricas_pipe = ["superficie_m2", "emision_co2"]

# Preprocesador: ColumnTransformer aplica transformaciones según el tipo de columna
preprocesador = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), columnas_numericas),               # Escala numéricas
        ("cat", OneHotEncoder(drop="first"), columnas_categoricas_pipe)  # OHE en categóricas
    ]
)

# Definimos el Pipeline: preprocesamiento + modelo KNN
pipeline_knn = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("knn", KNeighborsClassifier())
])

# Dividimos train/test para este flujo
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    random_state=42,
    stratify=y_raw
)

# Definimos la grilla de hiperparámetros para GridSearchCV
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"]
}

# GridSearchCV: búsqueda exhaustiva de mejores hiperparámetros con validación cruzada
grid_knn = GridSearchCV(
    estimator=pipeline_knn,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\nEntrenando GridSearchCV con Pipeline (esto puede tardar un poco)...")
grid_knn.fit(X_train_pipe, y_train_pipe)

print("\nMejores hiperparámetros encontrados:")
print(grid_knn.best_params_)

print("\nMejor Accuracy promedio en validación cruzada:")
print(grid_knn.best_score_)

# Extraer el mejor pipeline ya entrenado
mejor_pipeline_knn = grid_knn.best_estimator_

# Evaluar en el conjunto de prueba
y_pred_pipe = mejor_pipeline_knn.predict(X_test_pipe)

print("\nDesempeño del MEJOR PIPELINE KNN en test:")
print("Accuracy:", accuracy_score(y_test_pipe, y_pred_pipe))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_pipe, y_pred_pipe))
print("\nReporte de clasificación:")
print(classification_report(y_test_pipe, y_pred_pipe))

# Guardar el pipeline a disco
NOMBRE_PIPELINE = "pipeline_proyecto_emisiones.pkl"
joblib.dump(mejor_pipeline_knn, NOMBRE_PIPELINE)
print(f"\n✅ Pipeline guardado como: {NOMBRE_PIPELINE}")


# ============================================================
# PREDICCIONES FINALES Y EXPORTACIÓN
# ============================================================

print("\n===== 16. PREDICCIONES CON EL MEJOR PIPELINE =====")

# Predicciones finales
y_pred_final = mejor_pipeline_knn.predict(X_test_pipe)

# Probabilidades (si el modelo lo soporta)
if hasattr(mejor_pipeline_knn, "predict_proba"):
    y_proba = mejor_pipeline_knn.predict_proba(X_test_pipe)[:, 1]
else:
    y_proba = None

# Construir DataFrame de resultados
df_resultados = pd.DataFrame({
    "y_real": y_test_pipe,
    "y_pred": y_pred_final
}, index=y_test_pipe.index)

if y_proba is not None:
    df_resultados["clasificacion_emisiones"] = y_proba

NOMBRE_RESULTADOS = "predicciones_clasificacion_pipeline_knn.csv"
df_resultados.to_csv(NOMBRE_RESULTADOS, index=True)
print(f"✅ Archivo de predicciones exportado como: {NOMBRE_RESULTADOS}")


# ============================================================
# EXPORTAR DATASET LIMPIO PARA OTROS PROYECTOS
# ============================================================

print("\n===== 17. EXPORTAR DATASET LIMPIO PARA ML =====")

NOMBRE_SALIDA = "dataset_limpio_para_ml.csv"
df_modelo.to_csv(NOMBRE_SALIDA, index=False)
print(f"✅ Dataset limpio exportado como: {NOMBRE_SALIDA}")



































#

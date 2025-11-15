import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1
df_csv=pd.read_csv("Datos Zaragoza.csv")
print(df_csv)

#2
#primeras 5 filas
print(df_csv.head())

#ultimas 5 filas
print(df_csv.tail())

#dimensiones del data set
print(df_csv.shape)

#nombre de las columnas
print(df_csv.columns)

#tipo de datos y valores nulos
print(df_csv.info())

#estadistica basica
print(df_csv.describe())

# Número de valores únicos por columna

print(df_csv.nunique())

# Estadísticas para columnas categóricas (tipo object o string)
print("Estadísticas de variables categóricas:")
print(df_csv.describe(include='object'))

# Conteo de filas duplicadas
print("Número de filas duplicadas:")
print(df_csv.duplicated().sum())

# Ver primeras filas duplicadas si existen
print("Filas duplicadas (si existen):")
print(df_csv[df_csv.duplicated()].head())

# Si quieres ver la correlación entre variables numéricas:
print("Correlación entre variables numéricas:")
print(df_csv.corr(numeric_only=True))


# 3. Limpieza de Datos
# =====================================
# valores Nulos


print(df_csv.isnull().sum())

# Ver porcentaje de nulos por columna
print(df4.isnull().mean() * 100)


# Eliminamos columnas no útiles para análisis inicial
df4.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)


print(df4.columns)

df7=df4.drop(index=[0, 6], inplace=True)

# Eliminar las filas en las posiciones 0 y 2
df8 = df4.drop(df4.index[[0, 6]])

df4(df4.index[0,4])

# Imputación de valores nulos en variables seleccionadas
# 'Age': usamos la mediana porque es robusta ante outliers
df4['Age'].fillna(df4['Age'].median(), inplace=True)  # Mediana para Edad

# 'Embarked': usamos la moda (valor más frecuente) para conservar la categoría dominante
df4['Embarked'].fillna(df4['Embarked'].mode()[0], inplace=True)  # Moda para Embarque

print(df4.isnull().sum())

##Eliminar valores duplicados
#True indica filas duplicadas respecto a la primera ocurrencia
print(df4.duplicated())  # Serie booleana mostrando duplicados

#Número total de filas duplicadas
print(df4.duplicated().sum())  # Conteo de filas duplicadas

# Opcional: eliminar duplicados para continuar con un dataset depurado
df4 = df4.drop_duplicates().copy()  # Eliminamos duplicados y copiamos el resultado para evitar vistas

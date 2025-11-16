import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1
df1_csv=pd.read_excel("Datos Zaragoza1.xlsx")
print(df1_csv)

#copy
df_csv=df1_csv.copy()
print(df_csv)

#borrar columnas
df_csv.drop(columns=['Coordenadas_gps','Municipio',"Estado_edificio","Dias_hasta_expiracion"], inplace=True)
print(df_csv.columns)

#mmenor a myor

# Si la columna no está en formato datetime, primero conviértela
df_csv['Fecha_emision'] = pd.to_datetime(df_csv['Fecha_emision'])
df2 = df_csv.sort_values(by='Fecha_emision')


#filtro para borar filas mayores a 2016
print(df2[df2["Anio_emision"]>2016])

# Eliminar las filas en las posiciones 
df3 = df2.drop(df2.index[57111:103196])
#eliminacion dato atipico
df3=df2.drop(df2.index[0])








#!/usr/bin/env python
# coding: utf-8



# # Descripción del proyecto
# 
# La compañía Sweet Lift Taxi ha recopilado datos históricos sobre pedidos de taxis en los aeropuertos. Para atraer a más conductores durante las horas pico, necesitamos predecir la cantidad de pedidos de taxis para la próxima hora. Construye un modelo para dicha predicción.
# 
# La métrica RECM en el conjunto de prueba no debe ser superior a 48.
# 
# ## Instrucciones del proyecto.
# 
# 1. Descarga los datos y haz el remuestreo por una hora.
# 2. Analiza los datos
# 3. Entrena diferentes modelos con diferentes hiperparámetros. La muestra de prueba debe ser el 10% del conjunto de datos inicial.4. Prueba los datos usando la muestra de prueba y proporciona una conclusión.
# 
# ## Descripción de los datos
# 
# Los datos se almacenan en el archivo `taxi.csv`. 	
# El número de pedidos está en la columna `num_orders`.



# ## Preparación

# In[1]:


#importar todas las librearías
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




#se carga el dataset y de paso se cambia la columna a datetime
taxi = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/taxi.csv?etag=11687de0e23962e5a11c9d8ae13eb630', parse_dates=[0], index_col = 0 )


# In[3]:


#se analizan los datos de las columnas
taxi.info()


# In[4]:


#se observa la base de datos en general
taxi


# observarciones: se puede analizar que hay columnas vacías y también hay una columna de tiempo que se convertirá en datatime, se verá a continuación si hay valores duplicados.

# ## Análisis

# In[5]:


#se analizan valores ausentes
print(taxi.isna().sum())


# In[6]:


#se rellenan los varlores ausentes con fillna, haciendo un bucle en todo el dataframe, ya que hay columnas con número entero y columnas en objeto
def rellenar_datos_ausentes(df):
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            df[col] = df[col].fillna('null')
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna("ausente")
    
    return df

# Llamar la función para rellenar los valores ausentes
taxi = rellenar_datos_ausentes(taxi)

# Mostrar las primeras filas para verificar
print(taxi.head())


# In[7]:


#se observan si existen valores duplicados
print('Número de elementos duplicados:',taxi.duplicated().sum())


# In[8]:


# se eliminan los datos duplicados
taxi = taxi.drop_duplicates()


# In[9]:


# se comprueba que no haya elementos duplicados
print(taxi.duplicated().sum())


# In[10]:


#se observa si las fechas están en orden cronológico

taxi.sort_index(inplace = True)
print(taxi.index.is_monotonic)
print (taxi.info())

#se verá si el data frame está en orden para ello se seleccionarán fechas aleatorias y también se imprimirá una tabla

taxi = taxi['2018-03':'2018-08']
print(taxi.plot())


# In[11]:


#Se hace análisis de fechas para observar el comportamiento de los datos, dividido por Q en el año.
taxi_Q1 = taxi['2018-01':'2018-03']
print(taxi_Q1.head())
print(taxi_Q1.plot())


# Se puede observar que el horario en que son más solicitados los taxis es a las 2:00pm y a las 10:00pm, así como se puede ver que el día en que hubo más solucitud de taxis fue el 24 de Marzo de 2025.
# 

# In[12]:


taxi_Q2 = taxi['2018-04':'2018-06']
print(taxi_Q2.head())
print(taxi_Q2.plot())


# Se puede notar un incremento notable en los pedidos del segundo Q del año, ya que hay pedidos que oscilan entre los 40 a más de 65, siendo el 22 de abril el que más lo solicitan. también se pueden observar varios picos el 6 de mayo, el 20 de mayo.

# In[13]:


taxi_Q3 = taxi['2018-07':'2018-08']
print(taxi_Q3.head())
print(taxi_Q3.plot())


# En el último nálisis se tuvo que dividir por dos meses, julio y agosto, en los últimos dos meses, se puede observar que aumentan los pedidos considerablemente, por ejemplo en comparación con el segundo Q, el mínimo de pedidos es de 40, mientas que el máximo es de 120 el 15 de agosto, igual hay varios picos de pedido, por ejemplo el 22 de agosto y el 25.

# In[14]:


taxi




# se realiza el paso de remuestreo

taxi_hourly = taxi.resample('H').sum() 

print(taxi_hourly.head())


# In[16]:


#remuestreo por mes

taxi_month = taxi.resample('M').sum() 

print(taxi_month.head())


# In[17]:


# remuestreo por día de la semana

taxi_week = taxi.resample('1W').sum() 

print(taxi_week.head())


# In[18]:


#resumen por hora de los datos y pedidos
display(taxi_week.describe())


taxi_hourly.assign(hour=taxi_hourly.index.hour).groupby('hour').sum().plot(kind='bar')
plt.title('Número de pedidos por hora')
plt.xlabel('Fecha y Hora')
plt.ylabel('Número de pedidos')
plt.show()


# Se puede observar que efectivamente los horarios donde hay más solicitudes u ordenes de taxis es en los pedidos de las doce, una y dos, éste último alcanza casi los 1200 pedidos, hay varias horas muertas, por ejemplo en un rango de las cinco a las ocho, de las doce a las quince horas y de las 19 a las 20 horas.

# In[19]:


# se aplica la media móvil  para reducir las fluctuaciones en una serie temporal.
taxi['rolling_mean'] = taxi.rolling(10).mean()
taxi.plot()


# In[20]:


#Se aplica la tendencia y la estacionalidad

# Eliminar filas con valores NaN
taxi_clean = taxi.dropna()

# Ahora puedes hacer la descomposición
from statsmodels.tsa.seasonal import seasonal_decompose

decomposed = seasonal_decompose(taxi_clean['num_orders'], model='additive', period=24)  # Ajusta 'period' según tus datos

# Graficar estacionalidad
decomposed.seasonal['2018-03-01':'2018-08-31'].plot()


# In[21]:


# se hace un muestra de gráficos en división a tendencia, estacional y residual

taxi_clean = taxi.dropna()


from statsmodels.tsa.seasonal import seasonal_decompose

decomposed = seasonal_decompose(taxi_clean['num_orders'], model='additive', period=24) 

# Graficar estacionalidad
decomposed.plot()


# In[22]:


# diferencias entre series temporales

taxi.sort_index(inplace=True)
taxi = taxi['2018-01':'2018-08'].resample('1D').sum()
taxi -= taxi.shift()
taxi['mean'] = taxi['num_orders'].rolling(15).mean()
taxi['std'] = taxi['num_orders'].rolling(15).std()
taxi.plot()


# ## Prueba



# In[23]:


taxi.info()


# In[24]:


# antes de entrenar el modelo se van a reemplazar los valores NAn

def rellenar_datos_ausentes(df):
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            df[col] = df[col].fillna('0')
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna("ausente")
    
    return df

# Llamar la función para rellenar los valores ausentes
taxi = rellenar_datos_ausentes(taxi)

# Mostrar las primeras filas para verificar
print(taxi.head())


# In[25]:


# División de los datos , ya que se pretende predecir el número de ordenes de taxis
X = taxi[['rolling_mean', 'mean', 'std']]  # Variables predictoras
y = taxi['num_orders']  # Variable objetivo


# In[26]:


#90% entrenamiento, 10% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)


# In[30]:


# Crear el modelo de regresión lineal
linear_model = LinearRegression()

# Entrenar el modelo
linear_model.fit(X_train, y_train)

# Hacer las predicciones
y_pred_linear = linear_model.predict(X_test)

# Evaluar con MAE (Error Absoluto Medio)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

print(f'MAE Regresión Lineal: {mae_linear}')


# In[31]:


#evualuación del modelo
r2 = model.score(X_test, y_test)
print(f"R²: {r2}")


# In[32]:


#exactitud del pronóstico

train, test = train_test_split(taxi, shuffle=False, test_size=0.2)

print('Pedido medio de solicitud de taxis:', test['num_orders'].median())


# In[33]:


#utilización de la métrica EAM para exactitud del pronóstico

print('Utilización de taxis:', test['num_orders'].median())

pred_previous = test.shift()
pred_previous.iloc[0] = train.iloc[-1]
print('EAM:',mean_absolute_error(test, pred_previous))


# # Resultados:
# 
# 1.- Frecuencia de los datos: Los datos están organizados a intervalos horarios
# 
# La serie muestra que la demanda de taxis a las 3:00 AM es muy baja (10 pedidos), mientras que a las 12:00 PM puede haber más actividad (aunque no lo has mostrado explícitamente, esto es común en las ciudades). El patrón parece seguir una variabilidad diurna.
# 
# Tendencia: La tendencia a lo largo del tiempo puede mostrar si la demanda de taxis aumenta o disminuye de manera continua. En el caso de los datos que muestras, solo se ve una pequeña muestra de un solo día, por lo que no es posible inferir una tendencia a largo plazo. 
# 
# Estacionalidad: Con los datos a intervalos horarios, puedes detectar patrones estacionales, como la mayor demanda durante las horas pico (por ejemplo, entre las 7:00 AM y las 9:00 AM o entre las 5:00 PM y las 7:00 PM).
# 
# También se puede notar que en meses va aumentando la demanda puesto que se puede observar una creciente elevación del mes de marzo al mes de septiembre.
# 
# Ahora también se observa que el promedio de intervalo de que entre piden un taxi a otro es de 0.79 por hora.
# 
# La desviación estándar es bastante alta en comparación con la media. Esto indica que los pedidos de taxis son muy variables, con muchas fluctuaciones alrededor de la media. En otras palabras, mientras que el promedio es bajo (0.79), hay momentos con valores mucho más altos (como el máximo de 433), lo que sugiere que la demanda puede ser altamente irregular.
# 

# 
# 
# MAE: El modelo tiene un error promedio de 27.05 órdenes de taxi, lo cual es una medida de cuán lejos están las predicciones de los valores reales.
# R²: El modelo tiene un ajuste excelente con un R² de 0.97, lo que significa que casi el 97% de la variabilidad en los datos es explicada por el modelo.
# EAM: nos indica que el modelo de Regresión linear, está funcionando de una manera eficiente
# 

# # Lista de revisión

# - [x]  	
# Jupyter Notebook está abierto.
# - [ ]  El código no tiene errores
# - [ ]  Las celdas con el código han sido colocadas en el orden de ejecución.
# - [ ]  	
# Los datos han sido descargados y preparados.
# - [ ]  Se ha realizado el paso 2: los datos han sido analizados
# - [ ]  Se entrenó el modelo y se seleccionaron los hiperparámetros
# - [ ]  Se han evaluado los modelos. Se expuso una conclusión
# - [ ] La *RECM* para el conjunto de prueba no es más de 48

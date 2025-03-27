**Predicción de Pedidos de Taxi**

#Descripción del Proyecto

Este proyecto tiene como objetivo predecir la cantidad de pedidos de taxis para la próxima hora, con el fin de ayudar a la compañía Sweet Lift Taxi a atraer más conductores durante las horas pico.

Para lograrlo, se ha desarrollado un modelo de aprendizaje automático basado en series temporales, utilizando datos históricos de pedidos de taxis en los aeropuertos.

Alcance del Proyecto

Carga y preprocesamiento de datos: Se descargaron y analizaron los datos, asegurando su integridad mediante la eliminación de valores nulos y duplicados.

Análisis exploratorio: Se examinaron patrones en los datos, incluyendo tendencias, estacionalidad y fluctuaciones horarias en la demanda de taxis.

Remuestreo y agregación de datos: Se realizaron agrupaciones por horas, semanas y meses para entender mejor los patrones de demanda.

Modelado predictivo: Se entrenaron modelos de regresión lineal para predecir la cantidad de pedidos de taxi en la siguiente hora.

Evaluación del modelo: Se midieron las métricas de rendimiento, incluyendo el MAE y R², asegurando que la RAÍZ DEL ERROR CUADRÁTICO MEDIO (RECM) del conjunto de prueba no supere 48.

**Resultados y Conclusiones**

Se identificó que la demanda de taxis tiene picos significativos a las 2:00 PM y 10:00 PM.

Durante el segundo trimestre del año, se observó un aumento en la demanda de taxis, con el 15 de agosto como el día de mayor actividad.

La media móvil y el análisis de tendencias mostraron un incremento en los pedidos a lo largo del tiempo.

El modelo de regresión lineal alcanzó un R² de 0.97, indicando un buen ajuste a los datos.

El MAE del modelo fue de 27.05, lo que sugiere un margen de error razonable para la predicción de la demanda de taxis.

El modelo cumple con la condición de que la RECM en el conjunto de prueba no supere 48.

Tecnologías Utilizadas

Python

Pandas, NumPy para manipulación y análisis de datos

Matplotlib para visualización

Scikit-learn para modelado predictivo

Statsmodels para análisis de series temporales

Instrucciones de Uso

Clonar el repositorio:

git clone <https://github.com/JulietaRiver/Proyectos_Generales.git>
cd <proyecto_14>

Instalar las dependencias necesarias:

pip install -r requirements.txt

Ejecutar el script del modelo:

python proyecto_14.py

Autores

Julieta Rivera

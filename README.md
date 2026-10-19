# Predicción de precios de automóviles con Machine Learning y MLflow

## 1. Descripción del proyecto

Este proyecto utiliza técnicas de Machine Learning para predecir los precios de automóviles a partir de sus características técnicas y comerciales. Se emplea el dataset **Automobile de UCI Machine Learning Repository** para desarrollar un flujo de trabajo que incluye la carga, limpieza y exploración de los datos, el entrenamiento de modelos de regresión y la evaluación de sus resultados.

Además, se utiliza **MLflow** para registrar los parámetros, las métricas, los modelos entrenados y los resultados de cada experimento, facilitando la comparación y reproducibilidad del proceso.

## 2. Dataset

El conjunto de datos utilizado proviene del repositorio oficial de UCI:

- **Nombre:** Automobile
- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/10/automobile)
- **Variable objetivo:** `price`
- **Tipo de problema:** Regresión supervisada.

El dataset contiene información sobre los automóviles, incluyendo fabricante, características del motor, dimensiones, peso, consumo de combustible y precio.

Se seleccionó porque combina variables numéricas y categóricas, contiene valores faltantes y permite analizar cómo las características de un vehículo se relacionan con su precio.

## 3. Objetivos

- Limpiar y preparar los datos para su uso en modelos de Machine Learning.
- Explorar la distribución de los precios y las diferencias entre fabricantes.
- Entrenar y comparar modelos de regresión.
- Optimizar los hiperparámetros de Random Forest mediante búsqueda en cuadrícula.
- Evaluar el desempeño de los modelos con validación cruzada y datos de prueba.
- Registrar los experimentos y sus resultados mediante MLflow.

## 4. Modelos implementados

Se desarrollaron dos modelos para predecir los precios de los automóviles:

**Regresión Lineal (`LinearRegression`)**

Se utiliza como modelo de referencia para estimar la relación entre las características de los vehículos y su precio. Forma parte de un flujo de trabajo que incluye imputación de valores faltantes, estandarización de variables numéricas y codificación One-Hot de variables categóricas.

**Random Forest (`RandomForestRegressor`)**

Utiliza un conjunto de árboles de decisión para capturar relaciones potencialmente no lineales entre las características y el precio. Sus hiperparámetros se optimizan mediante `GridSearchCV`, con validación cruzada de cinco particiones.

Los datos se dividen en un 80 % para entrenamiento y un 20 % para prueba, utilizando `random_state=42` para favorecer la reproducibilidad.

## 5. Métricas de evaluación

Se utilizan las siguientes métricas:

- **CV RMSE:** error cuadrático medio de la raíz promedio obtenido mediante validación cruzada. Un valor menor indica mejores predicciones durante la validación.
- **Test MAE:** error absoluto medio en el conjunto de prueba. Representa la magnitud promedio de los errores, en las mismas unidades que el precio.
- **Test RMSE:** raíz del error cuadrático medio en el conjunto de prueba. Penaliza en mayor medida los errores grandes.
- **Test R²:** coeficiente de determinación que indica la proporción de variabilidad de los precios explicada por el modelo. Un valor cercano a 1 representa un mejor ajuste.

## 6. Resultados obtenidos

La siguiente tabla presenta los resultados de ambos modelos:

| Modelo | CV RMSE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Random Forest ajustado | 2296.45 | 1823.70 | 2787.13 | 0.9365 |
| Regresión Lineal | 3093.31 | 1721.47 | 2802.96 | 0.9358 |

### Análisis comparativo

Ambos modelos presentan un buen desempeño, con valores de R² superiores a 0.93. Esto indica que explican aproximadamente el 93.6 % de la variabilidad de los precios en el conjunto de prueba.

Random Forest obtiene el mejor RMSE en validación cruzada, con 2296.45 frente a 3093.31 de la Regresión Lineal. También presenta un RMSE ligeramente menor en prueba y un R² marginalmente superior.

Por otro lado, la Regresión Lineal consigue el menor MAE, con 1721.47 frente a 1823.70 de Random Forest, lo que indica un menor error absoluto promedio en las predicciones de prueba.

En conjunto, **Random Forest ajustado se considera la alternativa preferida por su mejor desempeño en validación cruzada y su ligera ventaja en RMSE y R² en prueba**. Sin embargo, la diferencia entre ambos modelos en el conjunto de prueba es pequeña, por lo que los resultados también respaldan el uso de la Regresión Lineal como una alternativa más sencilla.

Los resultados corresponden a la partición de prueba utilizada en este experimento y no garantizan el mismo desempeño en otros conjuntos de datos.

## 7. Seguimiento de experimentos con MLflow

MLflow se utiliza para registrar y organizar los experimentos del proyecto. Se configura una base de datos SQLite denominada `mlflow.db` y un experimento llamado `UCI_Automobile_Price_Regression`.

Para cada modelo se registran:

- Nombre del algoritmo.
- Parámetros de configuración e hiperparámetros seleccionados.
- Métricas de validación y prueba.
- Modelo entrenado.
- Archivos JSON con los resultados.
- Identificador único de ejecución (`run_id`).

Esta configuración permite consultar las ejecuciones, comparar los modelos y recuperar los artefactos registrados.

### Consultar la interfaz de MLflow

Desde la carpeta raíz del proyecto, ejecuta:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Después, abre en el navegador:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

Selecciona el experimento `UCI_Automobile_Price_Regression` para consultar las ejecuciones y sus métricas.

**Importante:** ejecuta el comando desde la carpeta que contiene `mlflow.db` para que la interfaz consulte la misma base de datos utilizada durante el entrenamiento.

## 8. Estructura del proyecto

```text
Actividad_Automobile_MLflow/
├── datos/
│   ├── datos_ini/
│   │   └── automobile_original.csv
│   └── datos_limp/
│       ├── automobile_limpio.csv
│       └── resultados_modelos.csv
├── fuentes/
│   ├── datos_prep.py
│   ├── prepara_datos.py
│   ├── entrena.ipynb
│   └── train.py
├── mlflow.db
└── requirements.txt
```

Los archivos CSV y la base de datos de MLflow se generan durante la ejecución del proyecto. La estructura mostrada representa los principales archivos esperados después de completar el flujo de trabajo.

## 9. Instalación y ejecución

Se recomienda utilizar un entorno virtual de Python.

### Crear y activar el entorno

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar el proyecto

1. Abre `fuentes/entrena.ipynb` en Jupyter Notebook.
2. Ejecuta las celdas en orden para cargar y limpiar los datos, explorar sus características y entrenar los modelos.
3. Verifica la tabla comparativa y el archivo `resultados_modelos.csv`.
4. Inicia la interfaz de MLflow para revisar los experimentos registrados.

También se pueden ejecutar los scripts de preparación y entrenamiento desde la terminal, siempre que sus dependencias y rutas estén correctamente configuradas.

## 10. Conclusiones

El proyecto demuestra cómo implementar un flujo de trabajo de Machine Learning para la predicción de precios de automóviles, desde la preparación de los datos hasta la evaluación y el seguimiento de experimentos.

Los resultados muestran que ambos modelos alcanzan un R² cercano a 0.94. Random Forest ajustado destaca por su menor error RMSE en validación cruzada y su ligera ventaja en las métricas RMSE y R² del conjunto de prueba, mientras que la Regresión Lineal obtiene un menor MAE.

Finalmente, la integración con MLflow permite mantener un registro organizado de los experimentos, facilitando la comparación de resultados, la reproducibilidad y la futura mejora de los modelos.

## 11. Tecnologías utilizadas

- Python
- Pandas
- Matplotlib
- Scikit-learn
- UCI Machine Learning Repository
- MLflow
- SQLite
- Jupyter Notebook
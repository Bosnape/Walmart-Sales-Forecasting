# Walmart Store Sales Forecasting

## Propósito del Proyecto

Este proyecto implementa modelos de aprendizaje profundo para la predicción de ventas semanales de tiendas Walmart. El objetivo es desarrollar sistemas de pronóstico capaces de predecir las ventas de cada departamento por tienda, considerando factores temporales, promocionales y económicos.

El proyecto compara dos arquitecturas de redes neuronales: LSTM (Long Short-Term Memory) y Transformer con arquitectura encoder-decoder y 4 cabezas de atención, evaluando su capacidad para capturar patrones temporales y dependencias complejas en series de tiempo de ventas.

Los datos provienen de la competencia de Kaggle "Walmart Recruiting - Store Sales Forecasting" (https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting).

## Autores

- **Pablo Cabrejos**
- **Sofía Flores**
- **Isabella Camacho**
- **Santiago Gómez**

## Estructura del Proyecto

```
Walmart-Sales-Forecasting/
├── data/                          # Datos del proyecto
│   ├── train.csv                  # Conjunto de entrenamiento
│   ├── test.csv                   # Conjunto de prueba (sin etiquetas)
│   ├── stores.csv                 # Información de tiendas
│   └── features.csv               # Características adicionales
│
├── train_results/                 # Resultados de entrenamiento
│   ├── LSTM/                      # Resultados de modelos LSTM
│   └── Transformer/               # Resultados de modelos Transformer
│
├── model_comparisons/             # Visualizaciones comparativas
│   ├── scatter_plots.png          # Gráficos de dispersión
│   └── temporal_series.png        # Series temporales
│
├── predictions/                   # Predicciones en test.csv
│   ├── submission_lstm.csv         # Predicciones LSTM
│   ├── submission_transformer.csv  # Predicciones Transformer
│   ├── predictions_visualization_lstm.png
│   └── predictions_visualization_transformer.png
│
├── data_analysis_and_preparation.ipynb  # Análisis y preparación de datos
├── lstm_baseline.py                # Entrenamiento modelo LSTM
├── transformer_forecasting.py      # Entrenamiento modelo Transformer
├── lstm_test.py                    # Predicciones con LSTM
├── transformer_test.py             # Predicciones con Transformer
├── app.py                          # Dashboard interactivo Streamlit
├── requirements.txt                # Dependencias del proyecto
├── README.md
```

## Dependencias

El proyecto requiere las siguientes librerías de Python:

- **matplotlib**
- **numpy**
- **pandas**
- **scikit-learn**
- **torch** (PyTorch)
- **seaborn**
- **statsmodels**
- **streamlit**
- **plotly**

Para instalar todas las dependencias, ejecute:

```bash
pip install -r requirements.txt
```

## Instrucciones de Ejecución

El proyecto debe ejecutarse en el siguiente orden:

### 1. Análisis y Preparación de Datos

Ejecute el notebook de análisis exploratorio y preparación de datos:

```bash
jupyter notebook data_analysis_and_preparation.ipynb
```

Este notebook realiza:

- Carga y fusión de los datasets (train.csv, stores.csv, features.csv)
- Análisis exploratorio de datos (EDA)
- Procesamiento y transformación de características
- División temporal de datos (80% entrenamiento, 20% validación)
- Guardado de datos procesados en `data/processed_data.pkl`

### 2. Entrenamiento de Modelos

Puede entrenar cualquiera de los dos modelos disponibles:

**Opción A: Entrenar modelo LSTM**

```bash
python lstm_baseline.py
```

**Opción B: Entrenar modelo Transformer**

```bash
python transformer_forecasting.py
```

Ambos scripts:

- Cargan los datos procesados desde `data/processed_data.pkl`
- Crean secuencias temporales agrupadas por Store-Dept
- Entrenan el modelo con early stopping
- Evalúan el modelo en el conjunto de validación
- Guardan el mejor modelo (`.pt`) y resultados (`.json`)

### 3. Generación de Predicciones en Test Set

Una vez entrenados los modelos, genere predicciones para el conjunto de prueba:

**Para modelo LSTM:**

```bash
python lstm_test.py
```

**Para modelo Transformer:**

```bash
python transformer_test.py
```

Estos scripts:

- Cargan el modelo entrenado
- Procesan el conjunto test.csv
- Generan predicciones usando estrategia rolling window
- Crean archivos de submission en formato CSV en `predictions/`
- Generan visualizaciones de las predicciones

### 4. Dashboard Interactivo (Streamlit)

El proyecto incluye un dashboard interactivo para explorar los resultados:

```bash
streamlit run app.py
```

## Resultados Esperados

### Métricas de Validación

Los modelos se evalúan en un conjunto de validación (20% del train.csv) ya que test.csv no contiene etiquetas. Las métricas reportadas incluyen:

- **WMAPE** (Weighted Mean Absolute Percentage Error): Error porcentual medio ponderado
- **WMAE** (Weighted Mean Absolute Error): Error absoluto medio ponderado (peso 5x para semanas festivas)
- **MAE** (Mean Absolute Error): Error absoluto medio
- **RMSE** (Root Mean Squared Error): Raíz del error cuadrático medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual medio absoluto
- **R²** (Coeficiente de determinación): Medida de bondad de ajuste

### Rendimiento de Modelos

Basado en los resultados de múltiples ejecuciones almacenados en `train_results/`:

**Modelo LSTM:**

- WMAPE promedio: ~12.8%
- WMAE promedio: ~$2,040
- MAPE promedio: ~36.2%
- R² promedio: ~0.965
- El modelo LSTM muestra un rendimiento consistente y superior en métricas ponderadas (WMAPE, WMAE)

**Modelo Transformer:**

- WMAPE promedio: ~14.6%
- WMAE promedio: ~$2,320
- MAPE promedio: ~32.6%
- R² promedio: ~0.960
- El modelo Transformer presenta mejor precisión en MAPE, indicando mayor capacidad para capturar cambios y acercarse a los valores reales de ventas

**Conclusión:** El modelo LSTM presenta mejores métricas ponderadas (WMAPE, WMAE), lo que indica mejor rendimiento en semanas festivas. Sin embargo, el modelo Transformer muestra un MAPE menor (~32.6% vs ~36.2%), lo que sugiere que, aunque más conservador y con menor variabilidad, logra captar mejor los cambios en las ventas y proporciona predicciones más precisas a los valores reales. Ambos modelos logran valores de R² superiores a 0.95, indicando una excelente capacidad predictiva.

### Visualizaciones

El proyecto incluye visualizaciones comparativas en `model_comparisons/`:

- **scatter_plots.png**: Comparación de predicciones vs valores reales para ambos modelos
- **temporal_series.png**: Series temporales mostrando la evolución de predicciones y valores reales

Las predicciones finales en test.csv se visualizan en `predictions/`:

- **predictions_visualization_lstm.png**: Visualización de predicciones del modelo LSTM
- **predictions_visualization_transformer.png**: Visualización de predicciones del modelo Transformer

### Archivos de Submission

Los archivos de submission generados (`predictions/submission_lstm.csv` y `predictions/submission_transformer.csv`) contienen las predicciones para el conjunto test.csv en el formato requerido por la competencia de Kaggle, con columnas:

- `Id`: Identificador único (Store_Dept_Date)
- `Weekly_Sales`: Predicción de ventas semanales

## Notas Técnicas

### División de Datos

- **Train (80%)**: 2010-02-05 a 2012-04-06 (335,761 registros)
- **Validation (20%)**: 2012-04-13 a 2012-10-26 (85,809 registros)
- **Test**: 2012-11-02 a 2013-07-26 (115,064 registros)

La división es temporal para respetar la naturaleza de series temporales y evitar data leakage.

### Características del Modelo

Ambos modelos utilizan:

- Embeddings para Store y Dept
- Secuencias temporales de longitud 8 semanas
- Normalización con RobustScaler
- Early stopping para prevenir overfitting
- Pesos diferenciados para semanas festivas (WMAE)

### Arquitectura LSTM

- Hidden size: 256
- Número de capas: 3
- Dropout: 0.3
- Embedding dimensions: Adaptativas según número de stores/depts

### Arquitectura Transformer

- d_model: 256
- Encoder layers: 3
- Decoder layers: 3
- Attention heads: 4
- Feedforward dimension: 512
- Dropout: 0.15

## Créditos

Este proyecto utiliza datos de la competencia de Kaggle:

**Walmart Recruiting - Store Sales Forecasting**

- URL: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
- Descripción: Competencia de reclutamiento de Walmart para predecir ventas de departamentos por tienda

Los datos incluyen información histórica de ventas de 45 tiendas Walmart desde febrero de 2010 hasta julio de 2013, con características adicionales como información de tiendas, variables económicas regionales y datos de promociones (markdowns).

## Licencia

Este proyecto es parte de un curso académico de Inteligencia Artificial en la Universidad EAFIT.

# 🎨 Interfaz Gráfica Interactiva - Walmart Sales Forecasting

## 📋 Descripción

Esta interfaz gráfica proporciona una experiencia interactiva y educativa para explorar el proyecto de pronóstico de ventas de Walmart. Está construida con **Streamlit** y ofrece visualizaciones interactivas con **Plotly**.

## 🚀 Características Principales

### 1. **Dashboard Interactivo**

- Vista general del proyecto con métricas clave
- Resumen de datos y estadísticas principales
- Navegación intuitiva entre secciones

### 2. **Análisis Exploratorio de Datos (EDA)**

- **Resumen General**: Estadísticas descriptivas y distribuciones
- **Series Temporales**: Tendencias y patrones temporales
- **Análisis por Tienda**: Comparación entre tiendas
- **Análisis por Departamento**: Top departamentos por ventas
- **Correlaciones**: Matriz de correlación interactiva

### 3. **Modelos de Machine Learning**

- **Comparación de Modelos**: LSTM vs Transformer
- **Métricas Detalladas**: WMAE, WMAPE, MAE, RMSE, R²
- **Visualizaciones Comparativas**: Gráficos interactivos
- **Arquitectura de Modelos**: Explicación detallada de cada modelo
- **Variabilidad de Resultados**: Análisis de múltiples ejecuciones

### 4. **Predicciones Interactivas**

- **Selector de Modelo**: LSTM, Transformer, o comparación de ambos
- **Filtros Interactivos**: Selección de tienda y departamento
- **Visualizaciones Temporales**: Gráficos de predicciones a lo largo del tiempo
- **Estadísticas de Predicciones**: Métricas por tienda/departamento

### 5. **Guía Educativa**

- **Análisis de Datos**: Explicación del EDA
- **Preprocesamiento**: Pipeline de transformación de datos
- **Modelos de Deep Learning**: Arquitectura y funcionamiento
- **Métricas de Evaluación**: Explicación de cada métrica
- **Flujo Completo**: Diagrama del proceso completo

## 📦 Instalación

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

O instalar Streamlit y Plotly directamente:

```bash
pip install streamlit plotly
```

### 2. Verificar Estructura de Archivos

Asegúrate de que tu proyecto tenga la siguiente estructura:

```
Walmart-Sales-Forecasting/
├── app.py                          # Aplicación Streamlit
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   └── features.csv
├── train_results/
│   ├── LSTM/
│   │   └── lstm_results*.json
│   └── Transformer/
│       └── transformer_results*.json
└── predictions/
    ├── submission_lstm.csv         # Opcional
    └── submission_transformer.csv  # Opcional
```

## 🎯 Uso

### Ejecutar la Interfaz

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Navegación

La interfaz tiene 5 secciones principales accesibles desde el menú lateral:

1. **🏠 Inicio**: Vista general del proyecto
2. **📈 Análisis Exploratorio**: Exploración interactiva de datos
3. **🤖 Modelos de ML**: Comparación y análisis de modelos
4. **🔮 Predicciones**: Visualización de predicciones
5. **📚 Guía Educativa**: Explicación educativa del proyecto

## 🎨 Características de la Interfaz

### Visualizaciones Interactivas

- **Plotly**: Gráficos interactivos con zoom, pan, y tooltips
- **Filtros Dinámicos**: Selección interactiva de tiendas, departamentos, etc.
- **Comparaciones Visuales**: Gráficos comparativos entre modelos

### Diseño Responsivo

- **Layout Ancho**: Optimizado para pantallas grandes
- **Sidebar**: Navegación siempre visible
- **Tabs**: Organización clara de contenido

### Experiencia Educativa

- **Explicaciones Detalladas**: Cada sección incluye explicaciones
- **Código de Ejemplo**: Snippets de código donde es relevante
- **Diagramas**: Visualizaciones del flujo del proyecto

## 📊 Datos Requeridos

### Datos Mínimos

Para que la interfaz funcione completamente, necesitas:

1. **Datos de Entrenamiento** (`data/train.csv`): Obligatorio
2. **Datos de Prueba** (`data/test.csv`): Obligatorio
3. **Información de Tiendas** (`data/stores.csv`): Obligatorio
4. **Características Adicionales** (`data/features.csv`): Obligatorio

### Resultados de Modelos (Opcional)

Para ver comparaciones de modelos:

- `train_results/LSTM/lstm_results*.json`
- `train_results/Transformer/transformer_results*.json`

### Predicciones (Opcional)

Para la sección de predicciones:

- `predictions/submission_lstm.csv`
- `predictions/submission_transformer.csv`

## 🔧 Personalización

### Modificar Colores

Edita la sección de estilos en `app.py`:

```python
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;  # Cambia el color aquí
    }
    </style>
""", unsafe_allow_html=True)
```

### Agregar Nuevas Secciones

1. Agrega una nueva opción en el radio button del sidebar
2. Crea la lógica correspondiente con `if page == "Nueva Sección":`
3. Agrega el contenido deseado

### Agregar Nuevas Visualizaciones

Usa Plotly para crear gráficos interactivos:

```python
import plotly.express as px

fig = px.line(data, x='Date', y='Weekly_Sales')
st.plotly_chart(fig, use_container_width=True)
```

## 🐛 Solución de Problemas

### Error: "No se pudieron cargar los datos"

- Verifica que los archivos CSV estén en `data/`
- Verifica que los nombres de archivos sean correctos
- Verifica que los archivos no estén corruptos

### Error: "No se encontraron resultados de modelos"

- Ejecuta primero el entrenamiento de los modelos
- Verifica que los archivos JSON estén en `train_results/`
- Verifica que los archivos JSON tengan el formato correcto

### La aplicación no se abre

- Verifica que Streamlit esté instalado: `pip install streamlit`
- Verifica que no haya errores en `app.py`
- Intenta ejecutar: `streamlit run app.py --server.port 8501`

### Visualizaciones no se muestran

- Verifica que Plotly esté instalado: `pip install plotly`
- Verifica que los datos estén cargados correctamente
- Revisa la consola para errores

## 📈 Mejoras Futuras

Posibles mejoras que se pueden agregar:

1. **Predicciones en Tiempo Real**: Permitir generar predicciones desde la interfaz
2. **Exportación de Datos**: Descargar gráficos y datos
3. **Análisis de Sensibilidad**: Explorar cómo cambian las predicciones con diferentes inputs
4. **Comparación con Baseline**: Agregar modelos baseline para comparación
5. **Análisis de Errores**: Visualizar dónde el modelo falla más
6. **Feature Importance**: Mostrar qué características son más importantes
7. **Análisis de Residuales**: Visualizar errores del modelo

## 📝 Notas

- La interfaz usa caché de Streamlit para mejorar el rendimiento
- Los datos se cargan una vez y se reutilizan
- Las visualizaciones son interactivas y se pueden exportar

## 🎓 Valor Educativo

Esta interfaz está diseñada para:

1. **Facilitar la Comprensión**: Visualizaciones claras y explicaciones detalladas
2. **Interactividad**: Permitir explorar los datos y resultados
3. **Educación**: Explicar conceptos de ML y series temporales
4. **Presentación**: Facilitar la presentación del proyecto

## 📄 Licencia

Este proyecto es parte de un curso de Inteligencia Artificial.

---

**Desarrollado con ❤️ usando Streamlit, PyTorch y Plotly**

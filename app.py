import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Walmart Sales Forecasting - Dashboard Interactivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.5rem;
    }
    h2 {
        color: #ffffff !important;
    }
    h3 {
        color: #ffffff !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
    }
    .info-box {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #0066cc;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .info-box h3 {
        color: #0066cc;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📊 Walmart Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🧭 Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📈 Análisis Exploratorio", "🤖 Modelos de ML", "🔮 Predicciones", "📚 Guía Educativa"]
)

# Función para cargar datos
@st.cache_data
def load_data():
    """Carga todos los datos necesarios"""
    data = {}
    
    # Cargar datasets
    try:
        data['train'] = pd.read_csv('data/train.csv')
        data['test'] = pd.read_csv('data/test.csv')
        data['stores'] = pd.read_csv('data/stores.csv')
        data['features'] = pd.read_csv('data/features.csv')
        
        # Convertir fechas
        data['train']['Date'] = pd.to_datetime(data['train']['Date'])
        data['test']['Date'] = pd.to_datetime(data['test']['Date'])
        data['features']['Date'] = pd.to_datetime(data['features']['Date'])
        
        # Fusionar datos
        data['train_full'] = data['train'].merge(data['stores'], on='Store', how='left')
        data['train_full'] = data['train_full'].merge(
            data['features'], 
            on=['Store', 'Date', 'IsHoliday'], 
            how='left'
        )
        
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None
    
    return data

# Función para preparar datos del EDA (con caché)
@st.cache_data
def prepare_eda_data(train_full):
    """Prepara los datos para el EDA con características temporales"""
    df = train_full.copy()
    
    # Agregar características temporales si no existen
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # Intentar convertir a datetime si no lo es
        try:
            # Verificar si ya es datetime
            if not hasattr(df['Date'].dtype, 'tz') and df['Date'].dtype != 'datetime64[ns]':
                df['Date'] = pd.to_datetime(df['Date'])
        except:
            df['Date'] = pd.to_datetime(df['Date'])
    
    if 'Year' not in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
    
    return df

# Funciones con caché para cálculos pesados
@st.cache_data
def compute_weekly_sales_trend(train_full):
    """Calcula tendencia de ventas semanales"""
    return train_full.groupby('Date')['Weekly_Sales'].sum().reset_index()

@st.cache_data
def compute_monthly_sales(train_full):
    """Calcula ventas mensuales"""
    monthly = train_full.groupby(['Year', 'Month'])['Weekly_Sales'].mean().reset_index()
    monthly['Date'] = pd.to_datetime(monthly[['Year', 'Month']].assign(Day=1))
    return monthly

@st.cache_data
def compute_correlation_matrix(train_full):
    """Calcula matriz de correlación"""
    numeric_cols = ['Weekly_Sales', 'Store', 'Dept', 'Size', 'Temperature', 
                   'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Quarter']
    numeric_cols = [col for col in numeric_cols if col in train_full.columns]
    corr_data = train_full[numeric_cols].dropna()
    return corr_data.corr(), corr_data

# Función para cargar resultados de modelos
@st.cache_data
def load_model_results():
    """Carga los resultados de los modelos"""
    results = {}
    
    # Cargar resultados LSTM
    lstm_results = []
    lstm_dir = Path('train_results/LSTM')
    if lstm_dir.exists():
        for file in sorted(lstm_dir.glob('*.json')):
            with open(file, 'r') as f:
                lstm_results.append(json.load(f))
        if lstm_results:
            results['LSTM'] = {
                'individual': lstm_results,
                'average': {
                    'wmape': np.mean([r['wmape'] for r in lstm_results]),
                    'wmae': np.mean([r['wmae'] for r in lstm_results]),
                    'mape': np.mean([r['mape'] for r in lstm_results]),
                    'mae': np.mean([r['mae'] for r in lstm_results]),
                    'rmse': np.mean([r['rmse'] for r in lstm_results]),
                    'r2': np.mean([r['r2'] for r in lstm_results])
                }
            }
    
    # Cargar resultados Transformer
    transformer_results = []
    transformer_dir = Path('train_results/Transformer')
    if transformer_dir.exists():
        for file in sorted(transformer_dir.glob('*.json')):
            with open(file, 'r') as f:
                transformer_results.append(json.load(f))
        if transformer_results:
            results['Transformer'] = {
                'individual': transformer_results,
                'average': {
                    'wmape': np.mean([r['wmape'] for r in transformer_results]),
                    'wmae': np.mean([r['wmae'] for r in transformer_results]),
                    'mape': np.mean([r['mape'] for r in transformer_results]),
                    'mae': np.mean([r['mae'] for r in transformer_results]),
                    'rmse': np.mean([r['rmse'] for r in transformer_results]),
                    'r2': np.mean([r['r2'] for r in transformer_results])
                }
            }
    
    return results

# Página de Inicio
if page == "🏠 Inicio":
    st.markdown('<h2 class="section-header">Bienvenido al Dashboard de Pronóstico de Ventas de Walmart</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📦 Tiendas", "45")
        st.metric("📊 Departamentos", "81")
    
    with col2:
        st.metric("📅 Período Entrenamiento", "2010-2012")
        st.metric("🔮 Período Prueba", "2012-2013")
    
    with col3:
        st.metric("📈 Registros Entrenamiento", "421,570")
        st.metric("🧪 Registros Prueba", "115,064")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Objetivo del Proyecto")
    st.markdown("""
    Este proyecto tiene como objetivo predecir las ventas semanales de diferentes departamentos 
    en 45 tiendas Walmart utilizando técnicas avanzadas de Machine Learning, específicamente 
    modelos de Deep Learning como **LSTM** y **Transformer**.
    """)
    
    st.markdown("### 📋 Características del Proyecto")
    st.markdown("""
    1. **Análisis Exploratorio de Datos (EDA)**: Análisis completo de los datos con visualizaciones
    2. **Preprocesamiento de Datos**: Pipeline robusto de transformación de datos
    3. **Modelos de Deep Learning**:
       - **LSTM**: Red neuronal recurrente con embeddings para tiendas y departamentos
       - **Transformer**: Arquitectura encoder-decoder con atención multi-cabeza
    4. **Evaluación**: Métricas especializadas para pronóstico de ventas (WMAE, WMAPE)
    5. **Visualizaciones Interactivas**: Dashboard para explorar resultados
    """)
    
    st.markdown("### 🛠️ Tecnologías Utilizadas")
    st.markdown("""
    - **Python**: Lenguaje principal
    - **PyTorch**: Framework de deep learning
    - **Pandas & NumPy**: Manipulación de datos
    - **Matplotlib & Seaborn**: Visualizaciones estáticas
    - **Plotly**: Visualizaciones interactivas
    - **Streamlit**: Interfaz web interactiva
    """)
    
    # Cargar datos para mostrar preview
    data = load_data()
    if data:
        st.markdown("### Vista Previa de los Datos")
        tab1, tab2, tab3 = st.tabs(["Datos de Entrenamiento", "Información de Tiendas", "Características Adicionales"])
        
        with tab1:
            st.dataframe(data['train'].head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(data['stores'], use_container_width=True)
        
        with tab3:
            st.dataframe(data['features'].head(10), use_container_width=True)

elif page == "📈 Análisis Exploratorio":
    st.markdown('<h2 class="section-header">Análisis Exploratorio de Datos (EDA)</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is None:
        st.error("No se pudieron cargar los datos. Por favor, verifica que los archivos estén en la carpeta 'data/'")
        st.stop()
    
    train_full = prepare_eda_data(data['train_full'])
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Resumen General", 
        "🔍 Valores Faltantes",
        "📈 Distribución de Variables",
        "📅 Series Temporales",
        "🏪 Análisis por Tienda",
        "📦 Análisis por Departamento",
        "🎯 Correlaciones y VIF"
    ])
    
    with tab1:
        st.subheader("Resumen del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Registros", f"{len(train_full):,}")
            st.metric("Tiendas Únicas", f"{train_full['Store'].nunique()}")
        with col2:
            st.metric("Departamentos Únicos", f"{train_full['Dept'].nunique()}")
            st.metric("Período", f"{train_full['Date'].min().date()} a {train_full['Date'].max().date()}")
        with col3:
            st.metric("Media Ventas", f"${train_full['Weekly_Sales'].mean():,.2f}")
            st.metric("Mediana Ventas", f"${train_full['Weekly_Sales'].median():,.2f}")
        with col4:
            st.metric("Desviación Estándar", f"${train_full['Weekly_Sales'].std():,.2f}")
            st.metric("Ventas Máximas", f"${train_full['Weekly_Sales'].max():,.2f}")
        
        # Estadísticas adicionales
        st.subheader("Estadísticas Adicionales de Ventas Semanales")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mínimo", f"${train_full['Weekly_Sales'].min():,.2f}")
        with col2:
            st.metric("Q1 (25%)", f"${train_full['Weekly_Sales'].quantile(0.25):,.2f}")
        with col3:
            st.metric("Q3 (75%)", f"${train_full['Weekly_Sales'].quantile(0.75):,.2f}")
        with col4:
            st.metric("Asimetría", f"{train_full['Weekly_Sales'].skew():.4f}")
        
        st.info(f"**Ventas negativas:** {(train_full['Weekly_Sales'] < 0).sum():,} registros (representan devoluciones)")
        
        # Distribución de ventas
        st.subheader("Distribución de Ventas Semanales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Usar muestreo para histograma más rápido
            sample_size = min(50000, len(train_full))
            sample_data = train_full.sample(n=sample_size, random_state=42) if len(train_full) > sample_size else train_full
            fig = px.histogram(
                sample_data, 
                x='Weekly_Sales',
                nbins=50,  # Reducido de 100 a 50
                title="Histograma de Ventas Semanales",
                labels={'Weekly_Sales': 'Ventas Semanales ($)', 'count': 'Frecuencia'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boxplot con muestreo
            sample_size = min(50000, len(train_full))
            sample_data = train_full.sample(n=sample_size, random_state=42) if len(train_full) > sample_size else train_full
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=sample_data['Weekly_Sales'],
                name='Ventas Semanales',
                boxmean='sd'
            ))
            fig.update_layout(
                title="Diagrama de Caja de Ventas Semanales",
                yaxis_title="Ventas Semanales ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribución logarítmica
        st.subheader("Distribución con Transformación Logarítmica")
        sales_positive = train_full[train_full['Weekly_Sales'] > 0]['Weekly_Sales']
        sample_size = min(50000, len(sales_positive))
        sample_sales = sales_positive.sample(n=sample_size, random_state=42) if len(sales_positive) > sample_size else sales_positive
        fig = px.histogram(
            x=np.log1p(sample_sales),
            nbins=50,  # Reducido de 100 a 50
            title="Distribución Logarítmica (Log(Ventas + 1))",
            labels={'x': 'Log(Ventas Semanales + 1)', 'count': 'Frecuencia'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparación festivos vs no festivos
        st.subheader("Ventas: Días Festivos vs No Festivos")
        holiday_comparison = train_full.groupby('IsHoliday')['Weekly_Sales'].agg(['mean', 'median', 'std', 'count']).reset_index()
        holiday_comparison['IsHoliday'] = holiday_comparison['IsHoliday'].map({True: 'Festivo', False: 'No Festivo'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                holiday_comparison,
                x='IsHoliday',
                y='mean',
                title="Ventas Promedio: Festivos vs No Festivos",
                labels={'mean': 'Ventas Promedio ($)', 'IsHoliday': 'Tipo de Semana'},
                color='IsHoliday',
                color_discrete_map={'Festivo': '#e74c3c', 'No Festivo': '#3498db'},
                text='mean'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boxplot comparativo
            fig = go.Figure()
            for holiday_type in [True, False]:
                data_subset = train_full[train_full['IsHoliday'] == holiday_type]['Weekly_Sales']
                fig.add_trace(go.Box(
                    y=data_subset,
                    name='Festivo' if holiday_type else 'No Festivo',
                    boxmean='sd'
                ))
            fig.update_layout(
                title="Distribución: Festivos vs No Festivos",
                yaxis_title="Ventas Semanales ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Análisis de Valores Faltantes")
        
        # Calcular valores faltantes
        missing_values = train_full.isnull().sum()
        missing_percentage = (missing_values / len(train_full)) * 100
        
        missing_df = pd.DataFrame({
            'Cantidad_Faltante': missing_values,
            'Porcentaje': missing_percentage
        }).sort_values(by='Cantidad_Faltante', ascending=False)
        
        missing_df = missing_df[missing_df['Cantidad_Faltante'] > 0]
        
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualización
            fig = px.bar(
                missing_df.reset_index(),
                x='index',
                y='Porcentaje',
                title="Valores Faltantes por Característica",
                labels={'index': 'Característica', 'Porcentaje': 'Porcentaje de Valores Faltantes (%)'},
                text='Porcentaje'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Estrategia para Manejo de Valores Faltantes:**
            
            1. **MarkDown1-5**: Estas características tienen valores faltantes significativos (>50%). 
               Los valores faltantes probablemente indican semanas sin descuentos promocionales. 
               **Estrategia:** Rellenar con 0 (sin descuento).
            
            2. **CPI y Unemployment**: Pequeño porcentaje de valores faltantes (<1%). 
               **Estrategia:** Rellenado hacia adelante o interpolación basada en continuidad temporal.
            """)
        else:
            st.success("✅ No se encontraron valores faltantes en el dataset.")
    
    with tab3:
        st.subheader("Distribución de Características Clave")
        
        # Características numéricas
        numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
        
        # Selector para mostrar todas o solo algunas
        show_all_features = st.checkbox("Mostrar todas las características", value=False)
        features_to_show = numerical_features if show_all_features else numerical_features[:3]
        
        for feature in features_to_show:
            if feature in train_full.columns:
                data_feature = train_full[feature].dropna()
                
                # Muestreo para visualización más rápida
                sample_size = min(50000, len(data_feature))
                sample_data = data_feature.sample(n=sample_size, random_state=42) if len(data_feature) > sample_size else data_feature
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.histogram(
                        x=sample_data,
                        nbins=30,  # Reducido de 50 a 30
                        title=f"Distribución de {feature}",
                        labels={'x': feature, 'count': 'Frecuencia'}
                    )
                    # Agregar líneas de media y mediana
                    mean_val = data_feature.mean()
                    median_val = data_feature.median()
                    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                                annotation_text=f"Media: {mean_val:.2f}")
                    fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                                annotation_text=f"Mediana: {median_val:.2f}")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Media", f"{mean_val:.2f}")
                    st.metric("Mediana", f"{median_val:.2f}")
                    st.metric("Std", f"{data_feature.std():.2f}")
                    st.metric("Min", f"{data_feature.min():.2f}")
                    st.metric("Max", f"{data_feature.max():.2f}")
        
        # Distribución de Tipo de Tienda
        st.subheader("Distribución de Tipos de Tienda")
        type_counts = train_full['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        fig = px.bar(
            type_counts,
            x='Type',
            y='Count',
            title="Distribución de Tipos de Tienda",
            labels={'Count': 'Conteo', 'Type': 'Tipo de Tienda'},
            color='Type',
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de valores atípicos
        st.subheader("Detección de Valores Atípicos")
        
        def detect_outliers_iqr(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            return outliers, lower_bound, upper_bound
        
        outliers_sales, lower_sales, upper_sales = detect_outliers_iqr(train_full, 'Weekly_Sales')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Registros", f"{len(train_full):,}")
        with col2:
            st.metric("Valores Atípicos", f"{len(outliers_sales):,} ({len(outliers_sales)/len(train_full)*100:.2f}%)")
        with col3:
            st.metric("Límite Superior", f"${upper_sales:,.2f}")
        
        st.info("""
        **Estrategia de Tratamiento de Valores Atípicos:**
        
        Dado que estos son datos de ventas minoristas:
        - **Ventas negativas**: Representan devoluciones y son eventos comerciales válidos - mantenerlos
        - **Ventas muy altas**: A menudo ocurren durante días festivos o eventos especiales - son válidos e importantes para el modelo
        - **Ventas muy bajas**: Pueden representar departamentos nuevos o períodos de liquidación - también válidos
        
        Es importante mantener los valores atípicos ya que representan escenarios comerciales reales que el modelo necesita aprender.
        """)
        
        # Visualización de valores atípicos
        st.subheader("Visualización de Valores Atípicos")
        features_to_check = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        
        # Selector para mostrar todas o solo algunas
        show_all_outliers = st.checkbox("Mostrar todas las características", value=False, key="outliers_checkbox")
        features_to_show = features_to_check if show_all_outliers else ['Weekly_Sales']
        
        for feature in features_to_show:
            if feature in train_full.columns:
                data_feature = train_full[feature].dropna()
                
                # Muestreo para boxplot más rápido
                sample_size = min(50000, len(data_feature))
                sample_data = data_feature.sample(n=sample_size, random_state=42) if len(data_feature) > sample_size else data_feature
                
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=sample_data,
                    name=feature,
                    boxmean='sd'
                ))
                
                # Calcular outliers (usar datos completos para estadísticas)
                Q1 = data_feature.quantile(0.25)
                Q3 = data_feature.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                n_outliers = ((data_feature < lower) | (data_feature > upper)).sum()
                
                fig.update_layout(
                    title=f"Valores Atípicos en {feature} (Atípicos: {n_outliers} ({n_outliers/len(data_feature)*100:.1f}%))",
                    yaxis_title=feature,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Análisis de Series Temporales")
        
        # Ventas totales por fecha (usar función con caché)
        weekly_sales_trend = compute_weekly_sales_trend(train_full)
        
        fig = px.line(
            weekly_sales_trend,
            x='Date',
            y='Weekly_Sales',
            title="Ventas Semanales Totales a lo Largo del Tiempo (Todas las Tiendas)",
            labels={'Weekly_Sales': 'Ventas Totales ($)', 'Date': 'Fecha'}
        )
        
        # Marcar semanas festivas (solo algunas para no sobrecargar)
        holiday_weeks = train_full[train_full['IsHoliday'] == True]['Date'].unique()
        # Limitar a máximo 20 marcadores
        holiday_weeks_limited = holiday_weeks[:20] if len(holiday_weeks) > 20 else holiday_weeks
        for holiday in holiday_weeks_limited:
            if holiday in weekly_sales_trend['Date'].values:
                sales_value = weekly_sales_trend[weekly_sales_trend['Date'] == holiday]['Weekly_Sales'].values[0]
                fig.add_trace(go.Scatter(
                    x=[holiday],
                    y=[sales_value],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='star'),
                    name='Semanas Festivas',
                    showlegend=False
                ))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Ventas promedio mensuales (usar función con caché)
        monthly_sales = compute_monthly_sales(train_full)
        
        fig = px.line(
            monthly_sales,
            x='Date',
            y='Weekly_Sales',
            title="Ventas Semanales Promedio Mensuales",
            labels={'Weekly_Sales': 'Ventas Promedio ($)', 'Date': 'Fecha'},
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de patrones estacionales
        st.subheader("Patrones Estacionales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ventas por mes
            monthly_avg = train_full.groupby('Month')['Weekly_Sales'].mean().sort_index().reset_index()
            fig = px.bar(
                monthly_avg,
                x='Month',
                y='Weekly_Sales',
                title="Ventas Promedio por Mes",
                labels={'Weekly_Sales': 'Ventas Promedio ($)', 'Month': 'Mes'},
                text='Weekly_Sales'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Ventas por trimestre
            quarterly_avg = train_full.groupby('Quarter')['Weekly_Sales'].mean().sort_index().reset_index()
            fig = px.bar(
                quarterly_avg,
                x='Quarter',
                y='Weekly_Sales',
                title="Ventas Promedio por Trimestre",
                labels={'Weekly_Sales': 'Ventas Promedio ($)', 'Quarter': 'Trimestre'},
                color='Quarter',
                text='Weekly_Sales'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ventas por tipo de tienda
            store_type_avg = train_full.groupby('Type')['Weekly_Sales'].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(
                store_type_avg,
                x='Type',
                y='Weekly_Sales',
                title="Ventas Promedio por Tipo de Tienda",
                labels={'Weekly_Sales': 'Ventas Promedio ($)', 'Type': 'Tipo de Tienda'},
                color='Type',
                text='Weekly_Sales'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Ventas por año
            yearly_avg = train_full.groupby('Year')['Weekly_Sales'].mean().reset_index()
            fig = px.bar(
                yearly_avg,
                x='Year',
                y='Weekly_Sales',
                title="Ventas Promedio por Año",
                labels={'Weekly_Sales': 'Ventas Promedio ($)', 'Year': 'Año'},
                text='Weekly_Sales'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Análisis por Tienda")
        
        # Selector de tienda
        store_list = sorted(train_full['Store'].unique())
        selected_stores = st.multiselect(
            "Selecciona tiendas para comparar:",
            store_list,
            default=store_list[:5]
        )
        
        if selected_stores:
            store_data = train_full[train_full['Store'].isin(selected_stores)]
            
            # Ventas promedio por tienda
            store_avg = store_data.groupby('Store')['Weekly_Sales'].mean().reset_index()
            store_avg = store_avg.sort_values('Weekly_Sales', ascending=False)
            
            fig = px.bar(
                store_avg,
                x='Store',
                y='Weekly_Sales',
                title="Ventas Promedio por Tienda",
                labels={'Weekly_Sales': 'Ventas Promedio ($)', 'Store': 'Tienda'},
                text='Weekly_Sales'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Ventas por tipo de tienda
        type_avg = train_full.groupby('Type')['Weekly_Sales'].mean().reset_index()
        fig = px.bar(
            type_avg,
            x='Type',
            y='Weekly_Sales',
            title="Ventas Promedio por Tipo de Tienda",
            labels={'Weekly_Sales': 'Ventas Promedio ($)', 'Type': 'Tipo de Tienda'},
            color='Type',
            text='Weekly_Sales'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.subheader("Análisis por Departamento")
        
        # Top departamentos por ventas
        dept_avg = train_full.groupby('Dept')['Weekly_Sales'].mean().reset_index()
        dept_avg = dept_avg.sort_values('Weekly_Sales', ascending=False).head(20)
        
        fig = px.bar(
            dept_avg,
            x='Dept',
            y='Weekly_Sales',
            title="Top 20 Departamentos por Ventas Promedio",
            labels={'Weekly_Sales': 'Ventas Promedio ($)', 'Dept': 'Departamento'},
            text='Weekly_Sales'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab7:
        st.subheader("Matriz de Correlación")
        
        # Usar función con caché
        correlation_matrix, corr_data = compute_correlation_matrix(train_full)
        
        fig = px.imshow(
            correlation_matrix,
            title="Matriz de Correlación - Características Numéricas",
            color_continuous_scale='RdBu',
            aspect="auto",
            text_auto='.2f'
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlación con ventas
        st.subheader("Correlación con Ventas Semanales")
        sales_corr = correlation_matrix['Weekly_Sales'].sort_values(ascending=False)
        sales_corr = sales_corr[sales_corr.index != 'Weekly_Sales']
        
        fig = px.bar(
            x=sales_corr.index,
            y=sales_corr.values,
            title="Correlación de Variables con Ventas Semanales",
            labels={'x': 'Variable', 'y': 'Correlación'},
            text=sales_corr.values
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # VIF (Factor de Inflación de Varianza)
        st.subheader("Factor de Inflación de Varianza (VIF)")
        st.info("VIF > 10 indica alta multicolinealidad. Esto puede afectar la estabilidad del modelo.")
        
        calculate_vif = st.checkbox("Calcular VIF", value=False)
        
        if calculate_vif:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                
                vif_data = corr_data.drop('Weekly_Sales', axis=1).dropna()
                
                # Muestreo para VIF más rápido
                sample_size = min(10000, len(vif_data))
                vif_sample = vif_data.sample(n=sample_size, random_state=42) if len(vif_data) > sample_size else vif_data
                
                if len(vif_sample) > 0 and len(vif_sample.columns) > 0:
                    with st.spinner("Calculando VIF..."):
                        vif_results = pd.DataFrame()
                        vif_results['Característica'] = vif_sample.columns
                        vif_results['VIF'] = [variance_inflation_factor(vif_sample.values, i) 
                                             for i in range(vif_sample.shape[1])]
                        vif_results = vif_results.sort_values('VIF', ascending=False)
                    
                    st.dataframe(vif_results, use_container_width=True)
                    
                    # Visualización VIF
                    fig = px.bar(
                        vif_results,
                        x='Característica',
                        y='VIF',
                        title="Factor de Inflación de Varianza (VIF) por Característica",
                        labels={'VIF': 'VIF', 'Característica': 'Característica'},
                        text='VIF',
                        color='VIF',
                        color_continuous_scale='Reds'
                    )
                    fig.add_hline(y=10, line_dash="dash", line_color="red", 
                                annotation_text="Umbral VIF = 10")
                    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                    fig.update_layout(height=500, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("statsmodels no está instalado. No se puede calcular VIF. Instala con: `pip install statsmodels`")
            except Exception as e:
                st.warning(f"No se pudo calcular VIF: {e}")

# Página de Modelos
elif page == "🤖 Modelos de ML":
    st.markdown('<h2 class="section-header">Modelos de Machine Learning</h2>', unsafe_allow_html=True)
    
    results = load_model_results()
    
    if not results:
        st.warning("No se encontraron resultados de modelos. Por favor, ejecuta primero el entrenamiento.")
        st.stop()
    
    # Comparación de modelos
    st.subheader("📊 Comparación de Modelos")
    
    if 'LSTM' in results and 'Transformer' in results:
        lstm_avg = results['LSTM']['average']
        trans_avg = results['Transformer']['average']
        
        # Crear tabla comparativa con DataFrame
        comparison_data = {
            'Métrica': ['WMAE', 'WMAPE', 'MAPE', 'MAE', 'RMSE', 'R²'],
            'LSTM': [
                f"${lstm_avg['wmae']:,.2f}",
                f"{lstm_avg['wmape']:.2f}%",
                f"{lstm_avg['mape']:.2f}%",
                f"${lstm_avg['mae']:,.2f}",
                f"${lstm_avg['rmse']:,.2f}",
                f"{lstm_avg['r2']:.4f}"
            ],
            'Transformer': [
                f"${trans_avg['wmae']:,.2f}",
                f"{trans_avg['wmape']:.2f}%",
                f"{trans_avg['mape']:.2f}%",
                f"${trans_avg['mae']:,.2f}",
                f"${trans_avg['rmse']:,.2f}",
                f"{trans_avg['r2']:.4f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Mostrar tabla
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Agregar nota sobre el mejor modelo
        st.info("💡 **Nota:** Para métricas de error (WMAE, WMAPE, MAPE, MAE, RMSE), menor es mejor. Para R², mayor es mejor.")
        
        # Métricas individuales en cards
        st.markdown("### 📈 Métricas Detalladas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔵 LSTM (Long Short-Term Memory)")
            st.metric("WMAE", f"${lstm_avg['wmae']:,.2f}", 
                     delta=f"{((trans_avg['wmae'] - lstm_avg['wmae']) / trans_avg['wmae'] * 100):.1f}% mejor" if lstm_avg['wmae'] < trans_avg['wmae'] else None)
            st.metric("WMAPE", f"{lstm_avg['wmape']:.2f}%",
                     delta=f"{((trans_avg['wmape'] - lstm_avg['wmape']) / trans_avg['wmape'] * 100):.1f}% mejor" if lstm_avg['wmape'] < trans_avg['wmape'] else None)
            st.metric("MAPE", f"{lstm_avg['mape']:.2f}%",
                     delta=f"{((trans_avg['mape'] - lstm_avg['mape']) / trans_avg['mape'] * 100):.1f}% mejor" if lstm_avg['mape'] < trans_avg['mape'] else None)
            st.metric("MAE", f"${lstm_avg['mae']:,.2f}",
                     delta=f"{((trans_avg['mae'] - lstm_avg['mae']) / trans_avg['mae'] * 100):.1f}% mejor" if lstm_avg['mae'] < trans_avg['mae'] else None)
            st.metric("RMSE", f"${lstm_avg['rmse']:,.2f}",
                     delta=f"{((trans_avg['rmse'] - lstm_avg['rmse']) / trans_avg['rmse'] * 100):.1f}% mejor" if lstm_avg['rmse'] < trans_avg['rmse'] else None)
            st.metric("R²", f"{lstm_avg['r2']:.4f}",
                     delta=f"{((lstm_avg['r2'] - trans_avg['r2']) / trans_avg['r2'] * 100):.1f}% mejor" if lstm_avg['r2'] > trans_avg['r2'] else None)
        
        with col2:
            st.markdown("#### 🟢 Transformer")
            st.metric("WMAE", f"${trans_avg['wmae']:,.2f}",
                     delta=f"{((lstm_avg['wmae'] - trans_avg['wmae']) / lstm_avg['wmae'] * 100):.1f}% mejor" if trans_avg['wmae'] < lstm_avg['wmae'] else None)
            st.metric("WMAPE", f"{trans_avg['wmape']:.2f}%",
                     delta=f"{((lstm_avg['wmape'] - trans_avg['wmape']) / lstm_avg['wmape'] * 100):.1f}% mejor" if trans_avg['wmape'] < lstm_avg['wmape'] else None)
            st.metric("MAPE", f"{trans_avg['mape']:.2f}%",
                     delta=f"{((lstm_avg['mape'] - trans_avg['mape']) / lstm_avg['mape'] * 100):.1f}% mejor" if trans_avg['mape'] < lstm_avg['mape'] else None)
            st.metric("MAE", f"${trans_avg['mae']:,.2f}",
                     delta=f"{((lstm_avg['mae'] - trans_avg['mae']) / lstm_avg['mae'] * 100):.1f}% mejor" if trans_avg['mae'] < lstm_avg['mae'] else None)
            st.metric("RMSE", f"${trans_avg['rmse']:,.2f}",
                     delta=f"{((lstm_avg['rmse'] - trans_avg['rmse']) / lstm_avg['rmse'] * 100):.1f}% mejor" if trans_avg['rmse'] < lstm_avg['rmse'] else None)
            st.metric("R²", f"{trans_avg['r2']:.4f}",
                     delta=f"{((trans_avg['r2'] - lstm_avg['r2']) / lstm_avg['r2'] * 100):.1f}% mejor" if trans_avg['r2'] > lstm_avg['r2'] else None)
        
        # Gráfico comparativo
        st.subheader("Comparación Visual de Métricas")
        
        metrics = ['wmae', 'wmape', 'mape', 'mae', 'rmse']
        metric_names = ['WMAE ($)', 'WMAPE (%)', 'MAPE (%)', 'MAE ($)', 'RMSE ($)']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='LSTM',
            x=metric_names,
            y=[lstm_avg[m] for m in metrics],
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            name='Transformer',
            x=metric_names,
            y=[trans_avg[m] for m in metrics],
            marker_color='#2ecc71'
        ))
        
        fig.update_layout(
            title="Comparación de Métricas entre Modelos",
            xaxis_title="Métrica",
            yaxis_title="Valor",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # R² comparativo
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['LSTM', 'Transformer'],
            y=[lstm_avg['r2'], trans_avg['r2']],
            marker_color=['#3498db', '#2ecc71'],
            text=[f"{lstm_avg['r2']:.4f}", f"{trans_avg['r2']:.4f}"],
            textposition='auto'
        ))
        fig.update_layout(
            title="Coeficiente de Determinación (R²)",
            yaxis_title="R²",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detalles de cada modelo
    st.markdown("---")
    st.subheader("📋 Detalles de Arquitectura")
    
    model_tab1, model_tab2 = st.tabs(["LSTM", "Transformer"])
    
    with model_tab1:
        st.markdown("""
        ### Arquitectura LSTM
        
        **Características principales:**
        - **Embeddings**: Representaciones densas para tiendas y departamentos
        - **LSTM**: 3 capas con 256 unidades ocultas
        - **Secuencias**: Longitud de secuencia de 8 semanas
        - **Regularización**: Dropout del 30%
        - **Normalización**: BatchNorm para estabilidad
        
        **Ventajas:**
        - Excelente para capturar dependencias temporales
        - Maneja bien secuencias largas
        - Arquitectura probada en series temporales
        
        **Hiperparámetros:**
        - Learning Rate: 0.001
        - Batch Size: 128
        - Optimizador: AdamW
        - Scheduler: CosineAnnealingWarmRestarts
        """)
    
    with model_tab2:
        st.markdown("""
        ### Arquitectura Transformer
        
        **Características principales:**
        - **Encoder-Decoder**: Arquitectura completa con atención multi-cabeza
        - **Positional Encoding**: Codificación posicional sinusoidal
        - **Attention Heads**: 4 cabezas de atención
        - **Capas**: 3 capas encoder + 3 capas decoder
        - **d_model**: 256 dimensiones
        - **Feedforward**: 512 dimensiones
        
        **Ventajas:**
        - Captura dependencias de largo alcance
        - Atención paralela (más rápido que LSTM)
        - Mejor para patrones complejos
        
        **Hiperparámetros:**
        - Learning Rate: 0.0005
        - Batch Size: 256
        - Optimizador: AdamW con warmup
        - Scheduler: Warmup + Cosine Annealing
        """)
    
    # Variabilidad de resultados
    if 'LSTM' in results and len(results['LSTM']['individual']) > 1:
        st.markdown("---")
        st.subheader("📊 Variabilidad de Resultados (Múltiples Ejecuciones)")
        
        lstm_runs = results['LSTM']['individual']
        trans_runs = results['Transformer']['individual'] if 'Transformer' in results else []
        
        if trans_runs:
            # WMAE por ejecución
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(lstm_runs) + 1)),
                y=[r['wmae'] for r in lstm_runs],
                mode='lines+markers',
                name='LSTM',
                line=dict(color='#3498db', width=2),
                marker=dict(size=10)
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(trans_runs) + 1)),
                y=[r['wmae'] for r in trans_runs],
                mode='lines+markers',
                name='Transformer',
                line=dict(color='#2ecc71', width=2),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="WMAE por Ejecución",
                xaxis_title="Ejecución",
                yaxis_title="WMAE ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Página de Predicciones
elif page == "🔮 Predicciones":
    st.markdown('<h2 class="section-header">Predicciones Interactivas</h2>', unsafe_allow_html=True)
    
    st.info("💡 Esta sección permite explorar las predicciones del modelo. Selecciona parámetros para ver predicciones específicas.")
    
    data = load_data()
    if data is None:
        st.error("No se pudieron cargar los datos.")
        st.stop()
    
    # Cargar predicciones si existen
    predictions_available = False
    lstm_preds = None
    trans_preds = None
    
    def parse_submission_file(df):
        """Parsea un archivo de submission con formato Id,Weekly_Sales"""
        if df is None or df.empty:
            return None
        
        # Verificar si tiene el formato esperado (Id, Weekly_Sales)
        if 'Id' in df.columns and 'Weekly_Sales' in df.columns:
            # Parsear el Id que tiene formato Store_Dept_Date
            df_parsed = df.copy()
            df_parsed[['Store', 'Dept', 'Date']] = df_parsed['Id'].str.split('_', expand=True)
            df_parsed['Store'] = df_parsed['Store'].astype(int)
            df_parsed['Dept'] = df_parsed['Dept'].astype(int)
            df_parsed['Date'] = pd.to_datetime(df_parsed['Date'])
            # Mantener solo las columnas necesarias
            df_parsed = df_parsed[['Store', 'Dept', 'Date', 'Weekly_Sales']]
            return df_parsed
        # Si ya tiene el formato correcto (Store, Dept, Date, Weekly_Sales)
        elif all(col in df.columns for col in ['Store', 'Dept', 'Date', 'Weekly_Sales']):
            df_parsed = df.copy()
            df_parsed['Store'] = df_parsed['Store'].astype(int)
            df_parsed['Dept'] = df_parsed['Dept'].astype(int)
            df_parsed['Date'] = pd.to_datetime(df_parsed['Date'])
            return df_parsed
        else:
            return None
    
    if os.path.exists('predictions/submission_lstm.csv'):
        try:
            lstm_preds_raw = pd.read_csv('predictions/submission_lstm.csv')
            lstm_preds = parse_submission_file(lstm_preds_raw)
            if lstm_preds is not None:
                predictions_available = True
        except Exception as e:
            st.warning(f"Error cargando predicciones LSTM: {e}")
    
    if os.path.exists('predictions/submission_transformer.csv'):
        try:
            trans_preds_raw = pd.read_csv('predictions/submission_transformer.csv')
            trans_preds = parse_submission_file(trans_preds_raw)
            if trans_preds is not None:
                predictions_available = True
        except Exception as e:
            st.warning(f"Error cargando predicciones Transformer: {e}")
    
    if not predictions_available:
        st.warning("No se encontraron archivos de predicciones válidos. Por favor, ejecuta primero los scripts de predicción.")
        
        # Mostrar información sobre cómo generar predicciones
        st.markdown("""
        ### 📝 Para generar predicciones:
        
        1. Ejecuta `lstm_test.py` para generar predicciones del modelo LSTM
        2. Ejecuta `transformer_test.py` para generar predicciones del modelo Transformer
        3. Los archivos se guardarán en `predictions/submission_*.csv`
        
        **Formato esperado:** Los archivos deben tener columnas `Id` (formato: Store_Dept_Date) y `Weekly_Sales`, 
        o columnas `Store`, `Dept`, `Date`, y `Weekly_Sales`.
        """)
        st.stop()
    
    # Selector de modelo
    model_choice = st.radio(
        "Selecciona el modelo:",
        ["LSTM", "Transformer", "Comparar Ambos"]
    )
    
    # Selector de vista
    view_mode = st.radio(
        "Modo de visualización:",
        ["Ver todas las tiendas", "Ver una tienda específica"]
    )
    
    if view_mode == "Ver todas las tiendas (agregado)":
        # Vista agregada de todas las tiendas
        st.markdown("### 📊 Predicciones Agregadas - Todas las Tiendas")
        
        if model_choice == "LSTM" and lstm_preds is not None:
            try:
                # Agregar por fecha (suma de todas las tiendas y departamentos)
                aggregated = lstm_preds.groupby('Date')['Weekly_Sales'].sum().reset_index()
                aggregated = aggregated.sort_values('Date')
                
                fig = px.line(
                    aggregated,
                    x='Date',
                    y='Weekly_Sales',
                    title="Predicciones LSTM - Total de Todas las Tiendas",
                    labels={'Weekly_Sales': 'Ventas Totales Predichas ($)', 'Date': 'Fecha'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ventas Totales Promedio", f"${aggregated['Weekly_Sales'].mean():,.2f}")
                with col2:
                    st.metric("Ventas Totales Máximas", f"${aggregated['Weekly_Sales'].max():,.2f}")
                with col3:
                    st.metric("Ventas Totales Mínimas", f"${aggregated['Weekly_Sales'].min():,.2f}")
                with col4:
                    st.metric("Total de Semanas", len(aggregated))
            except Exception as e:
                st.error(f"Error procesando predicciones LSTM agregadas: {e}")
        
        elif model_choice == "Transformer" and trans_preds is not None:
            try:
                aggregated = trans_preds.groupby('Date')['Weekly_Sales'].sum().reset_index()
                aggregated = aggregated.sort_values('Date')
                
                fig = px.line(
                    aggregated,
                    x='Date',
                    y='Weekly_Sales',
                    title="Predicciones Transformer - Total de Todas las Tiendas",
                    labels={'Weekly_Sales': 'Ventas Totales Predichas ($)', 'Date': 'Fecha'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ventas Totales Promedio", f"${aggregated['Weekly_Sales'].mean():,.2f}")
                with col2:
                    st.metric("Ventas Totales Máximas", f"${aggregated['Weekly_Sales'].max():,.2f}")
                with col3:
                    st.metric("Ventas Totales Mínimas", f"${aggregated['Weekly_Sales'].min():,.2f}")
                with col4:
                    st.metric("Total de Semanas", len(aggregated))
            except Exception as e:
                st.error(f"Error procesando predicciones Transformer agregadas: {e}")
        
        elif model_choice == "Comparar Ambos":
            if lstm_preds is None and trans_preds is None:
                st.warning("No hay predicciones disponibles para comparar.")
            elif lstm_preds is None:
                st.warning("No hay predicciones LSTM disponibles.")
            elif trans_preds is None:
                st.warning("No hay predicciones Transformer disponibles.")
            else:
                try:
                    lstm_agg = lstm_preds.groupby('Date')['Weekly_Sales'].sum().reset_index()
                    trans_agg = trans_preds.groupby('Date')['Weekly_Sales'].sum().reset_index()
                    lstm_agg = lstm_agg.sort_values('Date')
                    trans_agg = trans_agg.sort_values('Date')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=lstm_agg['Date'],
                        y=lstm_agg['Weekly_Sales'],
                        mode='lines',
                        name='LSTM',
                        line=dict(color='#3498db', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=trans_agg['Date'],
                        y=trans_agg['Weekly_Sales'],
                        mode='lines',
                        name='Transformer',
                        line=dict(color='#2ecc71', width=2)
                    ))
                    fig.update_layout(
                        title="Comparación de Predicciones Agregadas - Todas las Tiendas",
                        xaxis_title="Fecha",
                        yaxis_title="Ventas Totales Predichas ($)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### LSTM")
                        st.metric("Promedio Total", f"${lstm_agg['Weekly_Sales'].mean():,.2f}")
                        st.metric("Máximo Total", f"${lstm_agg['Weekly_Sales'].max():,.2f}")
                        st.metric("Mínimo Total", f"${lstm_agg['Weekly_Sales'].min():,.2f}")
                    with col2:
                        st.markdown("### Transformer")
                        st.metric("Promedio Total", f"${trans_agg['Weekly_Sales'].mean():,.2f}")
                        st.metric("Máximo Total", f"${trans_agg['Weekly_Sales'].max():,.2f}")
                        st.metric("Mínimo Total", f"${trans_agg['Weekly_Sales'].min():,.2f}")
                except Exception as e:
                    st.error(f"Error comparando predicciones agregadas: {e}")
    
    else:
        # Vista de una tienda específica (código original)
        # Selector de tienda y departamento
        col1, col2 = st.columns(2)
        
        with col1:
            store_list = sorted(data['train']['Store'].unique())
            selected_store = st.selectbox("Selecciona una tienda:", store_list)
        
        with col2:
            dept_list = sorted(data['train'][data['train']['Store'] == selected_store]['Dept'].unique())
            selected_dept = st.selectbox("Selecciona un departamento:", dept_list)
        
        # Filtrar predicciones para una tienda específica
        if model_choice == "LSTM" and lstm_preds is not None:
            try:
                filtered_preds = lstm_preds[
                    (lstm_preds['Store'] == selected_store) & 
                    (lstm_preds['Dept'] == selected_dept)
                ].copy()
                
                if not filtered_preds.empty:
                    filtered_preds = filtered_preds.sort_values('Date')
                    
                    # Gráfico de predicciones
                    fig = px.line(
                        filtered_preds,
                        x='Date',
                        y='Weekly_Sales',
                        title=f"Predicciones LSTM - Tienda {selected_store}, Dept {selected_dept}",
                        labels={'Weekly_Sales': 'Ventas Predichas ($)', 'Date': 'Fecha'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Estadísticas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Ventas Promedio", f"${filtered_preds['Weekly_Sales'].mean():,.2f}")
                    with col2:
                        st.metric("Ventas Máximas", f"${filtered_preds['Weekly_Sales'].max():,.2f}")
                    with col3:
                        st.metric("Ventas Mínimas", f"${filtered_preds['Weekly_Sales'].min():,.2f}")
                    with col4:
                        st.metric("Total Predicciones", len(filtered_preds))
                else:
                    st.info(f"No se encontraron predicciones para la Tienda {selected_store} y Departamento {selected_dept} con el modelo LSTM.")
            except Exception as e:
                st.error(f"Error procesando predicciones LSTM: {e}")
        
        elif model_choice == "Transformer" and trans_preds is not None:
            try:
                filtered_preds = trans_preds[
                    (trans_preds['Store'] == selected_store) & 
                    (trans_preds['Dept'] == selected_dept)
                ].copy()
                
                if not filtered_preds.empty:
                    filtered_preds = filtered_preds.sort_values('Date')
                    
                    # Gráfico de predicciones
                    fig = px.line(
                        filtered_preds,
                        x='Date',
                        y='Weekly_Sales',
                        title=f"Predicciones Transformer - Tienda {selected_store}, Dept {selected_dept}",
                        labels={'Weekly_Sales': 'Ventas Predichas ($)', 'Date': 'Fecha'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Estadísticas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Ventas Promedio", f"${filtered_preds['Weekly_Sales'].mean():,.2f}")
                    with col2:
                        st.metric("Ventas Máximas", f"${filtered_preds['Weekly_Sales'].max():,.2f}")
                    with col3:
                        st.metric("Ventas Mínimas", f"${filtered_preds['Weekly_Sales'].min():,.2f}")
                    with col4:
                        st.metric("Total Predicciones", len(filtered_preds))
                else:
                    st.info(f"No se encontraron predicciones para la Tienda {selected_store} y Departamento {selected_dept} con el modelo Transformer.")
            except Exception as e:
                st.error(f"Error procesando predicciones Transformer: {e}")
        
        elif model_choice == "Comparar Ambos":
            if lstm_preds is None and trans_preds is None:
                st.warning("No hay predicciones disponibles para comparar. Por favor, genera predicciones primero.")
            elif lstm_preds is None:
                st.warning("No hay predicciones LSTM disponibles. Solo se mostrarán las predicciones del Transformer.")
                model_choice = "Transformer"
            elif trans_preds is None:
                st.warning("No hay predicciones Transformer disponibles. Solo se mostrarán las predicciones del LSTM.")
                model_choice = "LSTM"
            else:
                try:
                    lstm_filtered = lstm_preds[
                        (lstm_preds['Store'] == selected_store) & 
                        (lstm_preds['Dept'] == selected_dept)
                    ].copy()
                    
                    trans_filtered = trans_preds[
                        (trans_preds['Store'] == selected_store) & 
                        (trans_preds['Dept'] == selected_dept)
                    ].copy()
                    
                    if not lstm_filtered.empty and not trans_filtered.empty:
                        lstm_filtered = lstm_filtered.sort_values('Date')
                        trans_filtered = trans_filtered.sort_values('Date')
                        
                        # Gráfico comparativo
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=lstm_filtered['Date'],
                            y=lstm_filtered['Weekly_Sales'],
                            mode='lines',
                            name='LSTM',
                            line=dict(color='#3498db', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=trans_filtered['Date'],
                            y=trans_filtered['Weekly_Sales'],
                            mode='lines',
                            name='Transformer',
                            line=dict(color='#2ecc71', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"Comparación de Predicciones - Tienda {selected_store}, Dept {selected_dept}",
                            xaxis_title="Fecha",
                            yaxis_title="Ventas Predichas ($)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Comparación de estadísticas
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### LSTM")
                            st.metric("Promedio", f"${lstm_filtered['Weekly_Sales'].mean():,.2f}")
                            st.metric("Máximo", f"${lstm_filtered['Weekly_Sales'].max():,.2f}")
                            st.metric("Mínimo", f"${lstm_filtered['Weekly_Sales'].min():,.2f}")
                        
                        with col2:
                            st.markdown("### Transformer")
                            st.metric("Promedio", f"${trans_filtered['Weekly_Sales'].mean():,.2f}")
                            st.metric("Máximo", f"${trans_filtered['Weekly_Sales'].max():,.2f}")
                            st.metric("Mínimo", f"${trans_filtered['Weekly_Sales'].min():,.2f}")
                    else:
                        if lstm_filtered.empty:
                            st.info(f"No se encontraron predicciones LSTM para la Tienda {selected_store} y Departamento {selected_dept}.")
                        if trans_filtered.empty:
                            st.info(f"No se encontraron predicciones Transformer para la Tienda {selected_store} y Departamento {selected_dept}.")
                except Exception as e:
                    st.error(f"Error comparando predicciones: {e}")

# Página Educativa
elif page == "📚 Guía Educativa":
    st.markdown('<h2 class="section-header">Guía Educativa del Proyecto</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎓 Introducción
    
    Este proyecto demuestra cómo aplicar técnicas avanzadas de Machine Learning para resolver 
    un problema real de pronóstico de ventas. A continuación, encontrarás una explicación detallada 
    de cada componente del proyecto.
    """)
    
    # Tabs educativos
    edu_tab1, edu_tab2, edu_tab3, edu_tab4, edu_tab5 = st.tabs([
        "📊 Análisis de Datos",
        "🔧 Preprocesamiento",
        "🧠 Modelos de Deep Learning",
        "📈 Métricas de Evaluación",
        "🚀 Flujo Completo"
    ])
    
    with edu_tab1:
        st.markdown("""
        ### 📊 Análisis Exploratorio de Datos (EDA)
        
        **¿Qué es el EDA?**
        El Análisis Exploratorio de Datos es el primer paso en cualquier proyecto de Machine Learning. 
        Consiste en entender los datos antes de construir modelos.
        
        **Análisis Realizados en el Dashboard:**
        
        1. **Resumen General**
           - Estadísticas descriptivas completas (media, mediana, desviación estándar, cuartiles)
           - Distribución de ventas semanales (histograma y boxplot)
           - Distribución logarítmica para normalización
           - Comparación de ventas: días festivos vs no festivos
           - 45 tiendas Walmart, 81 departamentos únicos
           - 421,570 registros de entrenamiento (2010-2012)
        
        2. **Valores Faltantes**
           - Análisis detallado de valores faltantes por característica
           - MarkDowns: ~64-73% de valores faltantes (imputados con 0)
           - CPI y Unemployment: <1% faltantes (interpolación temporal)
           - Visualización de porcentajes de valores faltantes
        
        3. **Distribución de Variables**
           - Distribuciones de características numéricas (Temperature, Fuel_Price, CPI, Unemployment, Size)
           - Distribución de tipos de tienda (A, B, C)
           - Detección de valores atípicos usando IQR (Interquartile Range)
           - Estrategia: Mantener valores atípicos por ser eventos comerciales válidos
        
        4. **Series Temporales**
           - Tendencias de ventas totales a lo largo del tiempo
           - Patrones estacionales (por mes, trimestre, año)
           - Impacto de días festivos (marcados en visualizaciones)
           - Comparación por tipo de tienda
        
        5. **Análisis por Tienda**
           - Comparación de ventas promedio por tienda
           - Análisis por tipo de tienda
           - Visualizaciones interactivas con filtros
        
        6. **Análisis por Departamento**
           - Top departamentos por ventas promedio
           - Identificación de departamentos más rentables
        
        7. **Correlaciones y VIF**
           - Matriz de correlación completa entre variables numéricas
           - Correlación de cada variable con ventas semanales
           - Factor de Inflación de Varianza (VIF) para detectar multicolinealidad
           - Size (tamaño de tienda): Mayor correlación con ventas (0.24)
        """)
    
    with edu_tab2:
        st.markdown("""
        ### 🔧 Preprocesamiento de Datos
        
        **¿Por qué preprocesar?**
        Los datos crudos rara vez están listos para ser usados directamente en modelos de ML. 
        El preprocesamiento transforma los datos a un formato que los modelos pueden entender.
        
        **Pipeline de Preprocesamiento:**
        
        1. **Manejo de Valores Faltantes**
           ```python
           - MarkDowns: Imputar con 0 (ausencia de promoción)
           - CPI/Unemployment: Interpolación temporal
           ```
        
        2. **Codificación de Variables Categóricas**
           ```python
           - Type (A, B, C): One-Hot Encoding
           - IsHoliday: Booleano a numérico (0/1)
           ```
        
        3. **Ingeniería de Características**
           ```python
           - Extracción temporal: Year, Month, Week, Quarter, DayOfYear
           - Total: 20 características finales
           ```
        
        4. **Normalización**
           ```python
           - StandardScaler para variables numéricas
           - RobustScaler para variable objetivo (ventas)
           ```
        
        5. **División Temporal**
           ```python
           - Train: 80% (2010-02-05 a 2012-04-06)
           - Validation: 20% (2012-04-13 a 2012-10-26)
           - Test: Separado (2012-11-02 a 2013-07-26)
           ```
        
        **Importante:** La división es temporal (no aleatoria) para respetar la naturaleza 
        de series temporales y evitar data leakage.
        """)
    
    with edu_tab3:
        st.markdown("""
        ### 🧠 Modelos de Deep Learning
        
        #### 🔵 LSTM (Long Short-Term Memory)
        
        **¿Qué es LSTM?**
        LSTM es un tipo de Red Neuronal Recurrente (RNN) diseñada para capturar dependencias 
        de largo plazo en secuencias temporales.
        
        **Arquitectura utilizada:**
        ```
        Input → Embeddings (Store, Dept) → LSTM (3 capas, 256 unidades) 
        → BatchNorm → FC Layers → Output
        ```
        
        **Características clave:**
        - **Embeddings**: Convierten IDs de tiendas/departamentos en vectores densos
        - **Secuencias**: Usa 8 semanas históricas para predecir la siguiente
        - **Memoria**: Mantiene información de largo plazo mediante células especiales
        
        **Ventajas:**
        - Excelente para series temporales
        - Maneja dependencias de largo alcance
        - Arquitectura probada y estable
        
        **Desventajas:**
        - Procesamiento secuencial (más lento)
        - Puede tener problemas con secuencias muy largas
        
        ---
        
        #### 🟢 Transformer
        
        **¿Qué es Transformer?**
        Transformer es una arquitectura basada en mecanismos de atención que revolucionó 
        el procesamiento de secuencias.
        
        **Arquitectura utilizada:**
        ```
        Input → Embeddings → Positional Encoding → Encoder (3 capas)
        → Decoder (3 capas) → FC Layers → Output
        ```
        
        **Características clave:**
        - **Attention Multi-Cabeza**: 4 cabezas de atención
        - **Positional Encoding**: Codifica posición temporal
        - **Encoder-Decoder**: Captura contexto y genera predicciones
        
        **Ventajas:**
        - Procesamiento paralelo (más rápido)
        - Captura dependencias de largo alcance eficientemente
        - Mecanismo de atención poderoso
        
        **Desventajas:**
        - Requiere más datos para entrenar bien
        - Más complejo que LSTM
        """)
    
    with edu_tab4:
        st.markdown("""
        ### 📈 Métricas de Evaluación
        
        **¿Por qué múltiples métricas?**
        Diferentes métricas capturan diferentes aspectos del rendimiento del modelo. 
        Para pronóstico de ventas, usamos métricas especializadas.
        
        #### Métricas Utilizadas:
        
        1. **WMAE (Weighted Mean Absolute Error)**
           ```
           WMAE = Σ(weights × |y_true - y_pred|) / Σ(weights)
           ```
           - **Pesos**: 5 para semanas festivas, 1 para regulares
           - **Interpretación**: Error promedio ponderado (menor es mejor)
           - **Uso**: Métrica principal de la competencia Kaggle
        
        2. **WMAPE (Weighted Mean Absolute Percentage Error)**
           ```
           WMAPE = Σ|y_true - y_pred| / Σ|y_true| × 100
           ```
           - **Interpretación**: Error porcentual ponderado (menor es mejor)
           - **Uso**: Fácil de interpretar para stakeholders
        
        3. **MAPE (Mean Absolute Percentage Error)**
           ```
           MAPE = mean(|y_true - y_pred| / |y_true|) × 100
           ```
           - **Interpretación**: Error porcentual promedio simple (menor es mejor)
           - **Uso**: Complementa WMAPE para tener una visión no ponderada del error
        
        4. **MAE (Mean Absolute Error)**
           ```
           MAE = mean(|y_true - y_pred|)
           ```
           - **Interpretación**: Error promedio en dólares
           - **Uso**: Métrica estándar fácil de entender
        
        5. **RMSE (Root Mean Squared Error)**
           ```
           RMSE = sqrt(mean((y_true - y_pred)²))
           ```
           - **Interpretación**: Penaliza más los errores grandes
           - **Uso**: Útil cuando errores grandes son críticos
        
        6. **R² (Coeficiente de Determinación)**
           ```
           R² = 1 - (SS_res / SS_tot)
           ```
           - **Interpretación**: Proporción de varianza explicada (mayor es mejor, máximo 1.0)
           - **Uso**: Medida de bondad de ajuste general
        
        **Visualización en el Dashboard:**
        - Tabla comparativa con todas las métricas lado a lado
        - Métricas detalladas con deltas que muestran cuánto mejor es cada modelo
        - Gráficos comparativos interactivos (barras agrupadas)
        - Variabilidad entre múltiples ejecuciones (si están disponibles)
        
        **Resultados del Proyecto:**
        - **LSTM**: WMAE ~$2,036, WMAPE ~12.77%, MAPE ~36.19%, R² ~0.965
        - **Transformer**: WMAE ~$2,256, WMAPE ~14.35%, MAPE ~32.64%, R² ~0.960
        - **Mejor modelo**: Depende de la métrica (LSTM mejor en WMAE/WMAPE, Transformer mejor en MAPE)
        """)
    
    with edu_tab5:
        st.markdown("""
        ### 🚀 Flujo Completo del Proyecto
        
        ```
        ┌─────────────────────────────────────────────────────────┐
        │                   1. CARGA DE DATOS                      │
        │  • train.csv, test.csv, stores.csv, features.csv        │
        └─────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────────────────────────┐
        │             2. ANÁLISIS EXPLORATORIO (EDA)              │
        │  • Estadísticas descriptivas                            │
        │  • Visualizaciones                                      │
        │  • Detección de valores atípicos                        │
        │  • Análisis de correlaciones                             │
        └─────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────────────────────────┐
        │           3. PREPROCESAMIENTO DE DATOS                   │
        │  • Manejo de valores faltantes                          │
        │  • Codificación de variables categóricas                 │
        │  • Ingeniería de características                        │
        │  • Normalización                                        │
        │  • División temporal (train/val/test)                    │
        └─────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────────────────────────┐
        │        4. CREACIÓN DE SECUENCIAS TEMPORALES              │
        │  • Agrupación por Store-Dept                            │
        │  • Secuencias de 8 semanas                              │
        │  • Preparación para modelos de deep learning            │
        └─────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────────────────────────┐
        │           5. ENTRENAMIENTO DE MODELOS                    │
        │  ┌──────────────┐         ┌──────────────┐              │
        │  │     LSTM     │         │ Transformer  │              │
        │  │  • Embeddings│         │  • Attention │              │
        │  │  • 3 capas   │         │  • Enc-Dec   │              │
        │  │  • 256 units │         │  • 256 dim   │              │
        │  └──────────────┘         └──────────────┘              │
        └─────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────────────────────────┐
        │             6. EVALUACIÓN Y COMPARACIÓN                  │
        │  • Métricas: WMAE, WMAPE, MAE, RMSE, R²                │
        │  • Comparación entre modelos                             │
        │  • Análisis de resultados                                │
        └─────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────────────────────────┐
        │           7. PREDICCIONES EN TEST SET                    │
        │  • Generación de predicciones finales                   │
        │  • Formato para submission                              │
        │  • Visualizaciones de predicciones                      │
        └─────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────────────────────────┐
        │        8. DASHBOARD INTERACTIVO (Streamlit)              │
        │  • Visualización de resultados                          │
        │  • Comparación de modelos                               │
        │  • Exploración de predicciones                          │
        │  • Análisis interactivo de datos                        │
        └─────────────────────────────────────────────────────────┘
        ```
        
        **Tecnologías y Herramientas:**
        - **Python 3.x**: Lenguaje principal
        - **PyTorch**: Framework de deep learning
        - **Pandas/NumPy**: Manipulación de datos
        - **Matplotlib/Seaborn**: Visualizaciones estáticas
        - **Plotly**: Visualizaciones interactivas
        - **Streamlit**: Dashboard web interactivo
        
        **Características del Dashboard:**
        
        1. **Análisis Exploratorio Completo**
           - 7 tabs con análisis detallados
           - Visualizaciones interactivas con Plotly
           - Optimizado con caché para mejor rendimiento
        
        2. **Comparación de Modelos**
           - Tabla comparativa con todas las métricas
           - Visualización de deltas (cuánto mejor es cada modelo)
           - Gráficos comparativos interactivos
           - Detalles de arquitectura de cada modelo
        
        3. **Predicciones Interactivas**
           - **Vista agregada**: Ver todas las tiendas sumadas (análisis corporativo)
           - **Vista específica**: Ver una tienda y departamento específicos
           - Comparación entre modelos LSTM y Transformer
           - Filtros interactivos por tienda y departamento
        
        4. **Guía Educativa**
           - Explicación detallada de cada componente
           - Conceptos de Machine Learning aplicados
           - Flujo completo del proyecto
        
        **Tiempo Estimado:**
        - Análisis y preprocesamiento: ~2-3 horas
        - Entrenamiento LSTM: ~1-2 horas (depende de GPU)
        - Entrenamiento Transformer: ~1-2 horas (depende de GPU)
        - Desarrollo de dashboard: ~3-4 horas
        
        **Lecciones Aprendidas:**
        1. La división temporal es crucial para series temporales
        2. Los embeddings mejoran significativamente el rendimiento
        3. Transformer puede superar a LSTM en algunas métricas
        4. El preprocesamiento correcto es fundamental
        5. Las métricas especializadas (WMAE) son importantes
        6. Un dashboard interactivo facilita la interpretación de resultados
        7. La visualización agregada ayuda a entender tendencias generales
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p>📊 Walmart Sales Forecasting Dashboard | Proyecto de Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)


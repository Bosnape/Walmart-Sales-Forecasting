import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

from transformer_forecasting import TransformerForecaster

def load_and_process_test_data(test_path, preprocessor, 
                               features_path='data/features.csv',
                               stores_path='data/stores.csv'):
    """Carga y procesa test.csv usando el mismo preprocessor de entrenamiento"""
    
    print("\nCargando datos de test:")
    
    test_df = pd.read_csv(test_path)
    print(f"- Test shape: {test_df.shape}")
    
    # Procesar IsHoliday
    test_df['IsHoliday'] = test_df['IsHoliday'].map({'TRUE': 1, 'True': 1, 'FALSE': 0, 'False': 0})
    
    # Merge con stores
    stores_df = pd.read_csv(stores_path)
    test_df = test_df.merge(stores_df, on='Store', how='left')
    
    # Merge con features
    features_df = pd.read_csv(features_path)
    features_df['IsHoliday'] = features_df['IsHoliday'].map({'TRUE': 1, 'True': 1, 'FALSE': 0, 'False': 0})
    
    test_df = test_df.merge(features_df, on=['Store', 'Date'], how='left', suffixes=('', '_feat'))
    
    if 'IsHoliday_feat' in test_df.columns:
        test_df = test_df.drop('IsHoliday_feat', axis=1)
    
    print(f"- Test shape después de merge: {test_df.shape}")
    
    # Convertir Date y extraer features temporales
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df['Year'] = test_df['Date'].dt.year
    test_df['Month'] = test_df['Date'].dt.month
    test_df['Week'] = test_df['Date'].dt.isocalendar().week
    test_df['Quarter'] = test_df['Date'].dt.quarter
    test_df['DayOfYear'] = test_df['Date'].dt.dayofyear
    
    required_cols = ['Store', 'Dept', 'Type', 'Size', 'Temperature', 'Fuel_Price', 
                     'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                     'CPI', 'Unemployment', 'IsHoliday', 
                     'Year', 'Month', 'Week', 'Quarter', 'DayOfYear']
    
    missing_cols = set(required_cols) - set(test_df.columns)
    if missing_cols:
        raise ValueError(f"Columnas faltantes: {missing_cols}")
    
    X_test = test_df[required_cols]
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"- X_test_processed shape: {X_test_processed.shape}")
    print(f"- NaNs en X_test_processed: {np.isnan(X_test_processed).sum()}")
    print(f"- Infs en X_test_processed: {np.isinf(X_test_processed).sum()}")
    
    # Reemplazar NaN e Inf con 0
    X_test_processed = np.nan_to_num(X_test_processed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_test_processed, test_df

def prepare_test_sequences_rolling(X_test, test_df_full, X_train_full, y_train_full,
                                   train_df_original, feature_names, store_min, dept_min, 
                                   scaler_y, seq_len=8):
    """
    Estrategia rolling: usa historia de train, luego va agregando predicciones
    de test como historia para las siguientes predicciones.
    """
    print(f"\nPreparando secuencias de test con estrategia rolling (seq_len={seq_len}):")
    
    # Reordenar columnas
    store_idx = feature_names.index('Store')
    dept_idx = feature_names.index('Dept')
    other_indices = [i for i in range(len(feature_names)) if i not in [store_idx, dept_idx]]
    new_order = other_indices + [store_idx, dept_idx]
    
    X_test = X_test[:, new_order]
    X_train_full = X_train_full[:, new_order]
    
    # Convertir a 0-based
    X_test[:, -2] = X_test[:, -2] - store_min
    X_test[:, -1] = X_test[:, -1] - dept_min
    X_train_full[:, -2] = X_train_full[:, -2] - store_min
    X_train_full[:, -1] = X_train_full[:, -1] - dept_min
    
    # Limpiar NaN/Inf
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Preparar DataFrames
    test_df = test_df_full.copy()
    test_df = test_df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    test_df['test_idx'] = np.arange(len(test_df))
    
    train_df = train_df_original.copy()
    train_df = train_df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    train_df['train_idx'] = np.arange(len(train_df))
    
    # Desnormalizar y_train_full para tener valores reales
    y_train_denorm = scaler_y.inverse_transform(y_train_full)
    
    X_sequences = []
    prediction_rows = []
    
    # Procesar por Store-Dept
    store_dept_groups = test_df.groupby(['Store', 'Dept'])
    
    for (store, dept), test_group in store_dept_groups:
        test_group = test_group.sort_values('Date').reset_index(drop=True)
        
        # Historia de train
        train_mask = (train_df['Store'] == store) & (train_df['Dept'] == dept)
        train_group = train_df[train_mask].sort_values('Date').reset_index(drop=True)
        
        if len(train_group) == 0:
            continue
        
        # Obtener últimas seq_len observaciones de train
        train_history_indices = train_group['train_idx'].values[-seq_len:]
        train_history = X_train_full[train_history_indices]
        
        # Buffer para predicciones de este grupo
        test_predictions = []
        
        for i, row in test_group.iterrows():
            test_idx = row['test_idx']
            
            # Construir secuencia
            if i < seq_len:
                # Usar mix de train history y predicciones previas de test
                n_from_train = seq_len - i
                n_from_test = i
                
                sequence = np.vstack([
                    train_history[-n_from_train:] if n_from_train > 0 else np.array([]).reshape(0, X_test.shape[1]),
                    test_predictions[-n_from_test:] if n_from_test > 0 and len(test_predictions) > 0 else np.array([]).reshape(0, X_test.shape[1])
                ])
            else:
                # Usar solo las últimas seq_len predicciones de test
                sequence = np.array(test_predictions[-seq_len:])
            
            # Validar secuencia
            if sequence.shape[0] != seq_len:
                # Padding con última observación de train si falta
                while sequence.shape[0] < seq_len:
                    sequence = np.vstack([train_history[-1:], sequence])
                sequence = sequence[-seq_len:]
            
            # Verificar NaN/Inf
            if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
                sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_sequences.append(sequence)
            prediction_rows.append({
                'test_idx': test_idx,
                'Store': store,
                'Dept': dept,
                'Date': row['Date']
            })
            
            # Agregar la observación actual de test al buffer
            test_predictions.append(X_test[test_idx])
    
    X_sequences = np.array(X_sequences)
    
    print(f"- Combinaciones Store-Dept: {store_dept_groups.ngroups}")
    print(f"- Secuencias creadas: {X_sequences.shape}")
    print(f"- NaNs en secuencias: {np.isnan(X_sequences).sum()}")
    print(f"- Infs en secuencias: {np.isinf(X_sequences).sum()}")
    
    return X_sequences, prediction_rows

def predict_test(model, X_test_seq, scaler_y, device):
    """Genera predicciones para test con validación"""
    
    print("\nGenerando predicciones:")
    
    model.eval()
    
    # Validar input
    print(f"- Input shape: {X_test_seq.shape}")
    print(f"- Input NaN: {np.isnan(X_test_seq).sum()}")
    print(f"- Input Inf: {np.isinf(X_test_seq).sum()}")
    print(f"- Input min/max: [{X_test_seq.min():.4f}, {X_test_seq.max():.4f}]")
    
    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    
    predictions = []
    batch_size = 512
    
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            
            # Verificar batch
            if torch.isnan(batch).any() or torch.isinf(batch).any():
                print(f"  WARNING: NaN/Inf en batch {i}")
                batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
            
            pred = model(batch).cpu().numpy()
            predictions.append(pred)
    
    predictions = np.concatenate(predictions)
    
    print(f"\nPredicciones raw (normalizadas):")
    print(f"- Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
    print(f"- Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
    
    # Desnormalizar
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
    print(f"\nPredicciones desnormalizadas")
    
    # Limpiar
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=0.0)
    predictions = np.maximum(predictions, 0)
    
    print(f"\nPredicciones finales generadas: {len(predictions)}")
    print(f"- Rango: [${predictions.min():,.2f}, ${predictions.max():,.2f}]")
    print(f"- Media: ${predictions.mean():,.2f}")
    print(f"- Std: ${predictions.std():,.2f}")
    
    return predictions.flatten()

def create_submission(test_df_full, predictions, prediction_info, output_path='submission_transformer.csv'):
    """Crea archivo de submission"""
    
    print("\nCreando archivo de submission:")
    
    # Crear DataFrame con predicciones
    pred_df = pd.DataFrame(prediction_info)
    pred_df['Weekly_Sales'] = predictions
    
    # Crear submission base
    submission = test_df_full[['Store', 'Dept', 'Date']].copy()
    submission = submission.reset_index(drop=True)
    submission['test_idx'] = np.arange(len(submission))
    
    # Merge con predicciones
    submission = submission.merge(
        pred_df[['test_idx', 'Weekly_Sales']],
        on='test_idx',
        how='left'
    )
    
    # Verificar NaNs
    nan_count = submission['Weekly_Sales'].isna().sum()
    if nan_count > 0:
        mean_sales = submission['Weekly_Sales'].mean()
        if np.isnan(mean_sales):
            mean_sales = 5000.0  # Fallback
        print(f"- Warning: {nan_count} filas sin predicción")
        print(f"- Rellenando con: ${mean_sales:,.2f}")
        submission['Weekly_Sales'].fillna(mean_sales, inplace=True)
    
    # Crear Id
    submission['Date_str'] = pd.to_datetime(submission['Date']).dt.strftime('%Y-%m-%d')
    submission['Id'] = (submission['Store'].astype(str) + '_' + 
                       submission['Dept'].astype(str) + '_' + 
                       submission['Date_str'])
    
    # Guardar
    submission[['Id', 'Weekly_Sales']].to_csv(output_path, index=False)
    
    print(f"- Filas en submission: {len(submission)}")
    print(f"- Valores NaN finales: {submission['Weekly_Sales'].isna().sum()}")
    print(f"- Rango final: [${submission['Weekly_Sales'].min():,.2f}, ${submission['Weekly_Sales'].max():,.2f}]")
    print(f"- Archivo guardado: {output_path}")
    
    return submission

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("Predicción con Transformer - Test Set")
    print("="*70)
    
    # Cargar datos
    print("\nCargando datos procesados:")
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    preprocessor = data['preprocessor']
    feature_names = data['feature_names']
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train'].reshape(-1, 1)
    y_val = data['y_val'].reshape(-1, 1)
    
    # Combinar train + val
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])
    
    print(f"- Features: {len(feature_names)}")
    print(f"- X_train_full: {X_train_full.shape}")
    
    # Limpiar train
    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_full = np.nan_to_num(y_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    store_idx = feature_names.index('Store')
    dept_idx = feature_names.index('Dept')
    store_min = X_train_full[:, store_idx].min()
    dept_min = X_train_full[:, dept_idx].min()
    
    print(f"- store_min: {store_min}, dept_min: {dept_min}")
    
    # Cargar train original
    print("\nCargando train.csv original:")
    train_df_orig = pd.read_csv('data/train.csv')
    train_df_orig['Date'] = pd.to_datetime(train_df_orig['Date'])
    print(f"- train.csv: {train_df_orig.shape}")
    
    # Cargar test
    X_test, test_df_full = load_and_process_test_data(
        test_path='data/test.csv',
        preprocessor=preprocessor
    )
    
    # Scaler
    scaler_y = RobustScaler()
    scaler_y.fit(y_train_full)
    
    # Preparar secuencias
    SEQ_LEN = 8
    X_test_seq, prediction_info = prepare_test_sequences_rolling(
        X_test=X_test,
        test_df_full=test_df_full,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        train_df_original=train_df_orig,
        feature_names=feature_names,
        store_min=store_min,
        dept_min=dept_min,
        scaler_y=scaler_y,
        seq_len=SEQ_LEN
    )
    
    # Configuración del modelo (debe coincidir con entrenamiento)
    n_stores = int(X_train_full[:, store_idx].max() - store_min) + 1
    n_depts = int(X_train_full[:, dept_idx].max() - dept_min) + 1
    input_size = X_test_seq.shape[2] - 2
    store_emb_dim = min(50, max(20, int(np.sqrt(n_stores)) * 4))
    dept_emb_dim = min(50, max(20, int(np.sqrt(n_depts)) * 3))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n- Device: {device}")
    print(f"- n_stores: {n_stores}, embedding_dim: {store_emb_dim}")
    print(f"- n_depts: {n_depts}, embedding_dim: {dept_emb_dim}")
    print(f"- input_size: {input_size}")
    
    # Crear modelo con misma configuración que entrenamiento
    model = TransformerForecaster(
        input_size=input_size,
        n_stores=n_stores,
        n_depts=n_depts,
        store_emb_dim=store_emb_dim,
        dept_emb_dim=dept_emb_dim,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.15,
        seq_len=SEQ_LEN
    ).to(device)
    
    # Cargar pesos del modelo entrenado
    model.load_state_dict(torch.load('transformer_best_model.pt', map_location=device))
    print(f"- Modelo Transformer cargado desde transformer_best_model.pt")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"- Total parámetros: {total_params:,}")
    
    # Predicciones
    predictions = predict_test(model, X_test_seq, scaler_y, device)
    
    # Submission
    submission = create_submission(test_df_full, predictions, prediction_info)
    
    print(f"\n" + "="*70)
    print("Proceso completado")
    print("="*70)
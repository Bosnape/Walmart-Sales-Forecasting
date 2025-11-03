import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. MODELO LSTM CON DIAGNÓSTICOS
# ==========================================

class LSTMForecaster(nn.Module):
    """LSTM con embeddings para Store y Dept"""
    def __init__(
        self, 
        input_size,
        n_stores=45,
        n_depts=99,
        store_emb_dim=16,
        dept_emb_dim=16,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,    
        seq_len=4       
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # Embeddings
        self.store_embedding = nn.Embedding(n_stores, store_emb_dim)
        self.dept_embedding = nn.Embedding(n_depts, dept_emb_dim)
        
        # LSTM input
        lstm_input_size = input_size + store_emb_dim + dept_emb_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Extraer Store y Dept
        store_ids = x[:, :, -2].long()
        dept_ids = x[:, :, -1].long()
        x_numeric = x[:, :, :-2]
        
        # Embeddings
        store_emb = self.store_embedding(store_ids)
        dept_emb = self.dept_embedding(dept_ids)
        
        # Concatenar
        x_combined = torch.cat([x_numeric, store_emb, dept_emb], dim=2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_combined)
        last_output = lstm_out[:, -1, :]
        last_output = self.bn(last_output)
        
        # FC layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        prediction = self.fc2(out)
        
        return prediction


# ==========================================
# 2. DIAGNÓSTICO DE DATOS
# ==========================================

def diagnose_data(X, y, name="Data"):
    """Diagnóstico completo de los datos"""
    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO: {name}")
    print(f"{'='*70}")
    
    print(f"\n📊 SHAPES:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    
    print(f"\n📈 TARGET (y) - WEEKLY SALES:")
    print(f"  Min:    ${y.min():,.2f}")
    print(f"  Max:    ${y.max():,.2f}")
    print(f"  Mean:   ${y.mean():,.2f}")
    print(f"  Median: ${y.median() if hasattr(y, 'median') else np.median(y):,.2f}")
    print(f"  Std:    ${y.std():,.2f}")
    print(f"  Q25:    ${np.percentile(y, 25):,.2f}")
    print(f"  Q75:    ${np.percentile(y, 75):,.2f}")
    
    # Distribución
    zeros = (y == 0).sum()
    negatives = (y < 0).sum()
    low = ((y > 0) & (y < 100)).sum()
    mid = ((y >= 100) & (y < 10000)).sum()
    high = (y >= 10000).sum()
    
    print(f"\n📉 DISTRIBUCIÓN:")
    print(f"  Zeros:      {zeros:,} ({zeros/len(y)*100:.2f}%)")
    print(f"  Negativos:  {negatives:,} ({negatives/len(y)*100:.2f}%)")
    print(f"  0-$100:     {low:,} ({low/len(y)*100:.2f}%)")
    print(f"  $100-$10k:  {mid:,} ({mid/len(y)*100:.2f}%)")
    print(f"  >$10k:      {high:,} ({high/len(y)*100:.2f}%)")
    
    print(f"\n🏪 FEATURES (últimas 2 son Store y Dept):")
    for i in range(X.shape[1]):
        col = X[:, i] if len(X.shape) == 2 else X[:, 0, i]
        print(f"  Feature {i:2d}: min={col.min():8.2f}, max={col.max():8.2f}, "
              f"mean={col.mean():8.2f}, std={col.std():8.2f}")
    
    # Store y Dept específicamente
    if len(X.shape) == 2:
        stores = X[:, -2]
        depts = X[:, -1]
    else:
        stores = X[:, 0, -2]  # Primera secuencia
        depts = X[:, 0, -1]
    
    print(f"\n🏬 STORE:")
    print(f"  Únicos: {len(np.unique(stores))}")
    print(f"  Range:  [{stores.min():.0f} - {stores.max():.0f}]")
    
    print(f"\n🏷️  DEPT:")
    print(f"  Únicos: {len(np.unique(depts))}")
    print(f"  Range:  [{depts.min():.0f} - {depts.max():.0f}]")
    
    print(f"{'='*70}\n")


def diagnose_sequences(X_seq, y_seq, name="Sequences"):
    """Diagnóstico de secuencias temporales"""
    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO: {name}")
    print(f"{'='*70}")
    
    print(f"\n📦 SHAPES:")
    print(f"  X_seq: {X_seq.shape} - [samples, seq_len, features]")
    print(f"  y_seq: {y_seq.shape}")
    
    print(f"\n🔍 VERIFICACIÓN DE INTEGRIDAD:")
    
    # Verificar que Store y Dept son constantes dentro de cada secuencia
    stores_first = X_seq[:, 0, -2]
    stores_last = X_seq[:, -1, -2]
    depts_first = X_seq[:, 0, -1]
    depts_last = X_seq[:, -1, -1]
    
    store_changes = (stores_first != stores_last).sum()
    dept_changes = (depts_first != depts_last).sum()
    
    print(f"  Secuencias donde Store cambia: {store_changes} ({store_changes/len(X_seq)*100:.2f}%)")
    print(f"  Secuencias donde Dept cambia:  {dept_changes} ({dept_changes/len(X_seq)*100:.2f}%)")
    
    if store_changes > 0 or dept_changes > 0:
        print(f"  ⚠️  PROBLEMA: Las secuencias NO deben mezclar Store/Dept diferentes!")
    else:
        print(f"  ✅ OK: Todas las secuencias son de un mismo Store-Dept")
    
    # Sample de una secuencia
    print(f"\n📝 SAMPLE SECUENCIA (índice 0):")
    sample = X_seq[0]
    print(f"  Store en timesteps: {sample[:, -2]}")
    print(f"  Dept en timesteps:  {sample[:, -1]}")
    print(f"  Primera feature en timesteps: {sample[:, 0]}")
    print(f"  Target (y): {y_seq[0][0]:.2f}")
    
    # Verificar varianza temporal
    print(f"\n📊 VARIANZA TEMPORAL (primera feature):")
    first_feature = X_seq[:, :, 0]  # [samples, seq_len]
    temporal_std = first_feature.std(axis=1)  # Std de cada secuencia
    print(f"  Std promedio entre timesteps: {temporal_std.mean():.4f}")
    print(f"  Secuencias con Std=0 (sin cambio temporal): {(temporal_std == 0).sum()} ({(temporal_std == 0).sum()/len(X_seq)*100:.2f}%)")
    
    if temporal_std.mean() < 0.01:
        print(f"  ⚠️  PROBLEMA: Muy poca varianza temporal - features pueden estar mal normalizadas")
    
    print(f"{'='*70}\n")


# ==========================================
# 3. CREACIÓN DE SECUENCIAS CON DIAGNÓSTICO
# ==========================================

def create_sequences_grouped(data, seq_len=4, verbose=True):
    """Crea secuencias agrupadas por Store-Dept CON diagnóstico"""
    
    X, y = [], []
    
    # Extraer identificadores
    stores = data[:, -3].astype(int)
    depts = data[:, -2].astype(int)
    
    # Crear grupo único
    group_ids = stores * 100 + depts
    unique_groups = np.unique(group_ids)
    
    if verbose:
        print(f"  Total combinaciones Store-Dept: {len(unique_groups)}")
    
    # Estadísticas de grupos
    group_sizes = []
    sequences_per_group = []
    
    for group_id in unique_groups:
        mask = group_ids == group_id
        group_data = data[mask]
        group_size = len(group_data)
        n_sequences = max(0, group_size - seq_len)
        
        group_sizes.append(group_size)
        sequences_per_group.append(n_sequences)
        
        # Crear secuencias
        for i in range(n_sequences):
            X.append(group_data[i:i+seq_len, :-1])
            y.append(group_data[i+seq_len, -1])
    
    if verbose:
        print(f"  Total secuencias creadas: {len(X)}")
        print(f"\n  Estadísticas de grupos:")
        print(f"    Tamaño promedio por Store-Dept: {np.mean(group_sizes):.1f} timesteps")
        print(f"    Min timesteps: {np.min(group_sizes)}")
        print(f"    Max timesteps: {np.max(group_sizes)}")
        print(f"    Secuencias promedio por grupo: {np.mean(sequences_per_group):.1f}")
        print(f"    Grupos sin secuencias (size <= seq_len): {(np.array(sequences_per_group) == 0).sum()}")
    
    if len(X) == 0:
        raise ValueError("No se pudieron crear secuencias!")
    
    return np.array(X), np.array(y).reshape(-1, 1)


class WalmartDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 4. TRAINING CON DIAGNÓSTICO
# ==========================================

def train_lstm_forecaster(X_train, y_train, X_val, y_val, 
                         n_stores, n_depts,
                         seq_len=4, epochs=100, batch_size=64, lr=0.0001):
    """Entrena LSTM con diagnósticos detallados"""
    
    # DataLoaders
    train_dataset = WalmartDataset(X_train, y_train)
    val_dataset = WalmartDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, drop_last=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"DEVICE: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")
    
    # Model config
    input_size = X_train.shape[2] - 2
    store_emb_dim = min(50, max(8, int(np.sqrt(n_stores)) + 5))
    dept_emb_dim = min(50, max(8, int(np.sqrt(n_depts)) + 5))
    
    print(f"\n🔧 CONFIGURACIÓN DEL MODELO:")
    print(f"  Input size (sin Store/Dept): {input_size}")
    print(f"  n_stores: {n_stores}, embedding_dim: {store_emb_dim}")
    print(f"  n_depts: {n_depts}, embedding_dim: {dept_emb_dim}")
    print(f"  Hidden size: 128")
    print(f"  Seq_len: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    
    model = LSTMForecaster(
        input_size=input_size,
        n_stores=n_stores,
        n_depts=n_depts,
        store_emb_dim=store_emb_dim,
        dept_emb_dim=dept_emb_dim,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        seq_len=seq_len
    ).to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parámetros: {total_params:,}")
    print(f"  Parámetros entrenables: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*70}")
    print(f"INICIANDO ENTRENAMIENTO")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = 0
        train_preds = []
        train_actuals = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Guardar predicciones para diagnóstico
            if epoch == 0 and batch_idx == 0:
                train_preds.append(predictions.detach().cpu().numpy())
                train_actuals.append(y_batch.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # VALIDATION
        model.eval()
        val_loss = 0
        val_preds = []
        val_actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
                
                val_preds.append(predictions.cpu().numpy())
                val_actuals.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Concatenar todas las predicciones
        val_preds = np.concatenate(val_preds)
        val_actuals = np.concatenate(val_actuals)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print con más detalles
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val Preds - Min: {val_preds.min():.4f}, Max: {val_preds.max():.4f}, "
                  f"Mean: {val_preds.mean():.4f}, Std: {val_preds.std():.4f}")
            print(f"  Val Actual- Min: {val_actuals.min():.4f}, Max: {val_actuals.max():.4f}, "
                  f"Mean: {val_actuals.mean():.4f}, Std: {val_actuals.std():.4f}")
            
            # DIAGNÓSTICO: Si std es muy bajo, hay problema
            if val_preds.std() < 0.01:
                print(f"  ⚠️  WARNING: Std de predicciones muy bajo - modelo colapsando!")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'lstm_diagnosed_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('lstm_diagnosed_best.pt'))
    
    return model, train_losses, val_losses


# ==========================================
# 5. EVALUACIÓN CON DIAGNÓSTICO
# ==========================================

def evaluate_lstm(model, X_test, y_test, scaler=None, verbose=True):
    """Evalúa con diagnósticos detallados"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
    
    # Shapes
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    
    # ANTES de desnormalizar
    if verbose:
        print(f"\n{'='*70}")
        print(f"PREDICCIONES (NORMALIZADAS)")
        print(f"{'='*70}")
        print(f"  Min:  {predictions.min():.4f}")
        print(f"  Max:  {predictions.max():.4f}")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std:  {predictions.std():.4f}")
    
    # Desnormalizar
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)
    
    # Eliminar negativos
    predictions = np.maximum(predictions, 0)
    
    # Métricas
    mae = np.mean(np.abs(y_test - predictions))
    rmse = np.sqrt(np.mean((y_test - predictions)**2))
    wmape = np.sum(np.abs(y_test - predictions)) / np.sum(np.abs(y_test)) * 100
    
    threshold = 100
    mask = y_test > threshold
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
    else:
        mape = float('inf')
    
    ss_res = np.sum((y_test - predictions)**2)
    ss_tot = np.sum((y_test - y_test.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*70}")
    print(f"MAE:    ${mae:,.2f}")
    print(f"RMSE:   ${rmse:,.2f}")
    print(f"WMAPE:  {wmape:.2f}%")
    print(f"MAPE:   {mape:.2f}% (solo ventas >${threshold})")
    print(f"R²:     {r2:.4f}")
    print(f"\nRango predicciones: [${predictions.min():,.2f}, ${predictions.max():,.2f}]")
    print(f"Rango reales:       [${y_test.min():,.2f}, ${y_test.max():,.2f}]")
    print(f"Media predicciones: ${predictions.mean():,.2f}")
    print(f"Media reales:       ${y_test.mean():,.2f}")
    print(f"Std predicciones:   ${predictions.std():,.2f}")
    print(f"Std reales:         ${y_test.std():,.2f}")
    print(f"{'='*70}")
    
    # DIAGNÓSTICO FINAL
    if predictions.std() < y_test.std() * 0.1:
        print(f"\n⚠️  PROBLEMA CRÍTICO: Std predicciones es {predictions.std() / y_test.std() * 100:.1f}% del std real")
        print(f"    El modelo NO está capturando la varianza de los datos")
        print(f"    Posibles causas:")
        print(f"      1. Normalización incorrecta")
        print(f"      2. Features sin varianza temporal")
        print(f"      3. Modelo colapsando a la media")
        print(f"      4. Learning rate muy alto/bajo")
    
    return {
        'mape': mape,
        'wmape': wmape,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'actuals': y_test
    }


# ==========================================
# 6. SCRIPT PRINCIPAL CON DIAGNÓSTICOS
# ==========================================

if __name__ == "__main__":
    import pickle
    import os
    
    print("\n" + "="*70)
    print("LSTM FORECASTER - VERSIÓN CON DIAGNÓSTICOS COMPLETOS")
    print("="*70)
    
    # Cargar datos
    print("\n📁 CARGANDO DATOS...")
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train'].reshape(-1, 1)
    y_val = data['y_val'].reshape(-1, 1)
    feature_names = data['feature_names']
    
    # DIAGNÓSTICO 1: Datos originales
    diagnose_data(X_train, y_train, "X_train / y_train ORIGINAL")
    diagnose_data(X_val, y_val, "X_val / y_val ORIGINAL")
    
    # Reordenar columnas
    print(f"\n🔄 REORDENANDO COLUMNAS...")
    store_idx = feature_names.index('Store')
    dept_idx = feature_names.index('Dept')
    other_indices = [i for i in range(len(feature_names)) if i not in [store_idx, dept_idx]]
    new_order = other_indices + [store_idx, dept_idx]
    
    X_train = X_train[:, new_order]
    X_val = X_val[:, new_order]
    feature_names = [feature_names[i] for i in new_order]
    print(f"  ✅ Últimas 2 features: {feature_names[-2:]}")
    
    # Convertir a 0-based
    print(f"\n🔢 CONVIRTIENDO ÍNDICES A 0-BASED...")
    store_min = X_train[:, -2].min()
    dept_min = X_train[:, -1].min()
    
    X_train[:, -2] = X_train[:, -2] - store_min
    X_val[:, -2] = X_val[:, -2] - store_min
    X_train[:, -1] = X_train[:, -1] - dept_min
    X_val[:, -1] = X_val[:, -1] - dept_min
    
    n_stores = int(X_train[:, -2].max()) + 1
    n_depts = int(X_train[:, -1].max()) + 1
    print(f"  n_stores: {n_stores}, n_depts: {n_depts}")
    
    # Limpiar negativos
    print(f"\n🧹 LIMPIANDO NEGATIVOS...")
    mask_train = y_train.flatten() >= 0
    mask_val = y_val.flatten() >= 0
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    X_val = X_val[mask_val]
    y_val = y_val[mask_val]
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    
    # DIAGNÓSTICO 2: Después de limpieza
    diagnose_data(X_train, y_train, "X_train / y_train DESPUÉS DE LIMPIEZA")
    
    # Normalización
    print(f"\n📏 NORMALIZANDO TARGET...")
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    print(f"  y_train_scaled: min={y_train_scaled.min():.4f}, max={y_train_scaled.max():.4f}, "
          f"mean={y_train_scaled.mean():.4f}, std={y_train_scaled.std():.4f}")
    
    # Combinar
    train_data_scaled = np.concatenate([X_train_scaled, y_train_scaled], axis=1)
    val_data_scaled = np.concatenate([X_val_scaled, y_val_scaled], axis=1)
    
    # Crear secuencias
    SEQ_LEN = 4
    print(f"\n🔗 CREANDO SECUENCIAS (seq_len={SEQ_LEN})...")
    print(f"\n  TRAIN:")
    X_train_seq, y_train_seq = create_sequences_grouped(train_data_scaled, seq_len=SEQ_LEN, verbose=True)
    print(f"\n  VALIDATION:")
    X_val_seq, y_val_seq = create_sequences_grouped(val_data_scaled, seq_len=SEQ_LEN, verbose=True)
    
    # DIAGNÓSTICO 3: Secuencias
    diagnose_sequences(X_train_seq, y_train_seq, "TRAIN SEQUENCES")
    diagnose_sequences(X_val_seq, y_val_seq, "VAL SEQUENCES")
    
    # Entrenar
    print(f"\n{'='*70}")
    print(f"🚀 ENTRENANDO MODELO")
    print(f"{'='*70}")
    
    model, train_losses, val_losses = train_lstm_forecaster(
        X_train_seq, y_train_seq, 
        X_val_seq, y_val_seq,
        n_stores=n_stores,
        n_depts=n_depts,
        seq_len=SEQ_LEN,
        epochs=100,
        batch_size=64,
        lr=0.0001
    )
    
    # Evaluar
    print(f"\n{'='*70}")
    print(f"📊 EVALUACIÓN FINAL")
    print(f"{'='*70}")
    
    results = evaluate_lstm(model, X_val_seq, y_val_seq, scaler=scaler_y, verbose=True)
    
    # Guardar
    print(f"\n💾 GUARDANDO RESULTADOS...")
    import json
    with open('lstm_diagnosed_results.json', 'w') as f:
        json.dump({
            'wmape': float(results['wmape']),
            'mape': float(results['mape']),
            'mae': float(results['mae']),
            'rmse': float(results['rmse']),
            'r2': float(results['r2']),
            'seq_len': SEQ_LEN,
            'n_stores': n_stores,
            'n_depts': n_depts
        }, f, indent=4)
    
    print(f"  ✅ Modelo: lstm_diagnosed_best.pt")
    print(f"  ✅ Resultados: lstm_diagnosed_results.json")
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS COMPLETADO")
    print(f"{'='*70}\n")
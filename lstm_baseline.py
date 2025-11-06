import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Modelo LSTM
class LSTMForecaster(nn.Module):
    """LSTM con embeddings para Store y Dept"""
    def __init__(
        self, 
        input_size,
        n_stores=45,
        n_depts=99,
        store_emb_dim=20,
        dept_emb_dim=20,
        hidden_size=256,
        num_layers=3,
        dropout=0.3,
        seq_len=8
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
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # BatchNorm
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # FC layers
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
        
        # Tomar último output
        last_output = lstm_out[:, -1, :]
        last_output = self.bn(last_output)
        
        # FC layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        prediction = self.fc2(out)
        
        return prediction

# Creacion de secuencias
def create_sequences_grouped(data, seq_len=8, verbose=True):
    """Crea secuencias agrupadas por Store-Dept"""
    X, y = [], []
    
    # Extraer identificadores
    stores = data[:, -3].astype(int)
    depts = data[:, -2].astype(int)
    
    # Crear grupos únicos
    group_ids = stores * 100 + depts
    unique_groups = np.unique(group_ids)
    
    if verbose:
        print(f"- Total combinaciones Store-Dept: {len(unique_groups)}")
        print(f"- Seq_len: {seq_len}")
    
    group_sizes = []
    sequences_per_group = []
    
    for group_id in unique_groups:
        mask = group_ids == group_id
        group_data = data[mask]
        group_size = len(group_data)
        n_sequences = max(0, group_size - seq_len)
        
        group_sizes.append(group_size)
        sequences_per_group.append(n_sequences)
        
        for i in range(n_sequences):
            X.append(group_data[i:i+seq_len, :-1])
            y.append(group_data[i+seq_len, -1])
    
    if verbose:
        print(f"- Total secuencias creadas: {len(X):,}")
        print(f"\nEstadísticas de grupos:")
        print(f"- Tamaño promedio por Store-Dept: {np.mean(group_sizes):.1f} timesteps")
        print(f"- Min timesteps: {np.min(group_sizes)}")
        print(f"- Max timesteps: {np.max(group_sizes)}")
        print(f"- Secuencias promedio por grupo: {np.mean(sequences_per_group):.1f}")
        print(f"- Grupos sin secuencias (size <= seq_len): {(np.array(sequences_per_group) == 0).sum()}")
    
    if len(X) == 0:
        raise ValueError("No se pudieron crear secuencias")
    
    return np.array(X), np.array(y).reshape(-1, 1)

# Dataset
class WalmartDataset(Dataset):
    """Dataset para Walmart"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Training
def train_lstm_forecaster(X_train, y_train, X_val, y_val, n_stores, n_depts,
                          seq_len=8, epochs=100, batch_size=128, lr=0.001):
    """Entrena LSTM con diagnosticos detallados"""
    
    # DataLoaders
    train_dataset = WalmartDataset(X_train, y_train)
    val_dataset = WalmartDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,
        shuffle=False, 
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model config
    input_size = X_train.shape[2] - 2
    store_emb_dim = min(50, max(16, int(np.sqrt(n_stores)) * 3))
    dept_emb_dim = min(50, max(16, int(np.sqrt(n_depts)) * 2))
    
    print(f"\n{'='*70}")
    print(f"Configuración del modelo LSTM")
    print(f"{'='*70}")
    print(f"\n- Input size (sin Store/Dept): {input_size}")
    print(f"- n_stores: {n_stores}, embedding_dim: {store_emb_dim}")
    print(f"- n_depts: {n_depts}, embedding_dim: {dept_emb_dim}")
    print(f"- Hidden size: 256")
    print(f"- Num layers: 3")
    print(f"- Seq_len: {seq_len}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {lr}")
    print(f"- Device: {device}")
    
    model = LSTMForecaster(
        input_size=input_size,
        n_stores=n_stores,
        n_depts=n_depts,
        store_emb_dim=store_emb_dim,
        dept_emb_dim=dept_emb_dim,
        hidden_size=256,
        num_layers=3,
        dropout=0.3,
        seq_len=seq_len
    ).to(device)
    
    # Contar parámetros    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Total parámetros: {total_params:,}")
    print(f"- Parámetros entrenables: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*70}")
    print(f"Iniciando entrenamiento")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
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
        scheduler.step()
        
        # Print con más detalles
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"• Train Loss - {train_loss:.6f}")
            print(f"• Val Loss   - {val_loss:.6f}")
            print(f"• Val Preds  - Min: {val_preds.min():.4f}, Max: {val_preds.max():.4f}, "
                  f"Mean: {val_preds.mean():.4f}, Std: {val_preds.std():.4f}")
            print(f"• Val Actual - Min: {val_actuals.min():.4f}, Max: {val_actuals.max():.4f}, "
                  f"Mean: {val_actuals.mean():.4f}, Std: {val_actuals.std():.4f}")
            
            if val_preds.std() < 0.01:
                print(f"- WARNING: Std de predicciones muy bajo!")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'lstm_best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n--- Early stopping at epoch {epoch+1} ---")
                break
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('lstm_best_model.pt'))
    
    return model, train_losses, val_losses

# Evaluación
def evaluate_lstm(model, X_test, y_test, is_holiday=None, scaler=None, verbose=True):
    """Evalua con diagnosticos detallados"""
    
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
    
    # Antes de desnormalizar
    if verbose:
        print(f"\n{'='*70}")
        print(f"Predicciones (normalizadas)")
        print(f"{'='*70}")
        print(f"\nMin:  {predictions.min():.4f}")
        print(f"Max:  {predictions.max():.4f}")
        print(f"Mean: {predictions.mean():.4f}")
        print(f"Std:  {predictions.std():.4f}")
    
    # Desnormalizar
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)
    
    # Eliminar negativos
    predictions = np.maximum(predictions, 0)
    
    # Preparar weights para WMAE
    if is_holiday is not None:
        if len(is_holiday.shape) == 1:
            is_holiday = is_holiday.reshape(-1, 1)
        weights = np.where(is_holiday == 1, 5.0, 1.0)
    else:
        weights = np.ones_like(y_test)
    
    # Métricas
    mae = np.mean(np.abs(y_test - predictions))
    rmse = np.sqrt(np.mean((y_test - predictions)**2))
    wmape = np.sum(np.abs(y_test - predictions)) / np.sum(np.abs(y_test)) * 100
    wmae = np.sum(weights * np.abs(y_test - predictions)) / np.sum(weights)
    
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
    print(f"Resultados finales")
    print(f"{'='*70}")
    print(f"\nMAE:    ${mae:,.2f}")
    print(f"WMAE:   ${wmae:,.2f}")
    print(f"RMSE:   ${rmse:,.2f}")
    print(f"WMAPE:  {wmape:.2f}%")
    print(f"MAPE:   {mape:.2f}% (solo ventas >${threshold})")
    print(f"R²:     {r2:.4f}")
    
    if is_holiday is not None:
        n_holidays = np.sum(is_holiday == 1)
        n_regular = np.sum(is_holiday == 0)
        print(f"\nPesos WMAE:")
        print(f"- Semanas festivas (w=5): {n_holidays:,} ({n_holidays/len(is_holiday)*100:.1f}%)")
        print(f"- Semanas regulares (w=1): {n_regular:,} ({n_regular/len(is_holiday)*100:.1f}%)")
    
    print(f"\nRango predicciones: [${predictions.min():,.2f}, ${predictions.max():,.2f}]")
    print(f"Rango reales:       [${y_test.min():,.2f}, ${y_test.max():,.2f}]")
    print(f"Media predicciones: ${predictions.mean():,.2f}")
    print(f"Media reales:       ${y_test.mean():,.2f}")
    print(f"Std predicciones:   ${predictions.std():,.2f}")
    print(f"Std reales:         ${y_test.std():,.2f}")
    
    # Diagnóstico de std
    std_ratio = predictions.std() / y_test.std()
    if std_ratio < 0.5:
        print(f"\n--- Std ratio: {std_ratio:.2%} (Modelo poco variable) ---")
    elif std_ratio > 1.5:
        print(f"\n--- Std ratio: {std_ratio:.2%} (Predicciones muy volátiles) ---")
    else:
        print(f"\n--- Std ratio: {std_ratio:.2%} (Buen balance) ---")
    
    print(f"\n{'='*70}")
    
    return {
        'mape': mape,
        'wmape': wmape,
        'wmae': wmae,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'actuals': y_test
    }

# Script principal
if __name__ == "__main__":
    import pickle
    import json
    
    # Cargar datos
    print("\nCargando datos:")
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train'].reshape(-1, 1)
    y_val = data['y_val'].reshape(-1, 1)
    feature_names = data['feature_names']
    
    print(f"- X_train: {X_train.shape}")
    print(f"- X_val: {X_val.shape}")
    
    # Verificar orden de columnas
    store_idx = feature_names.index('Store')
    dept_idx = feature_names.index('Dept')
    isholiday_idx = feature_names.index('IsHoliday')
    other_indices = [i for i in range(len(feature_names)) if i not in [store_idx, dept_idx]]
    new_order = other_indices + [store_idx, dept_idx]
    
    X_train = X_train[:, new_order]
    X_val = X_val[:, new_order]
    feature_names = [feature_names[i] for i in new_order]
    
    # Actualizar índice de IsHoliday después del reordenamiento
    isholiday_idx = feature_names.index('IsHoliday')
    
    # Convertir a 0-based
    store_min = X_train[:, -2].min()
    dept_min = X_train[:, -1].min()
    
    X_train[:, -2] = X_train[:, -2] - store_min
    X_val[:, -2] = X_val[:, -2] - store_min
    X_train[:, -1] = X_train[:, -1] - dept_min
    X_val[:, -1] = X_val[:, -1] - dept_min
    
    n_stores = int(X_train[:, -2].max()) + 1
    n_depts = int(X_train[:, -1].max()) + 1
    print(f"- n_stores: {n_stores}")
    print(f"- n_depts: {n_depts}")
    
    # Limpiar negativos
    print(f"\nLimpiando negativos:")
    mask_train = y_train.flatten() >= 0
    mask_val = y_val.flatten() >= 0
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    X_val = X_val[mask_val]
    y_val = y_val[mask_val]
    print(f"- X_train: {X_train.shape}")
    print(f"- X_val: {X_val.shape}")
    
    # Normalización
    print(f"\nNormalizando target (RobustScaler):")
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    print(f"- y_train_scaled: min={y_train_scaled.min():.4f}, max={y_train_scaled.max():.4f}, "
          f"mean={y_train_scaled.mean():.4f}, std={y_train_scaled.std():.4f}")
    
    # Combinar
    train_data_scaled = np.concatenate([X_train_scaled, y_train_scaled], axis=1)
    val_data_scaled = np.concatenate([X_val_scaled, y_val_scaled], axis=1)
    
    # Crear secuencias
    SEQ_LEN = 8
    print(f"\nCreando secuencias (seq_len={SEQ_LEN}):")
    print(f"\n1. Train:")
    X_train_seq, y_train_seq = create_sequences_grouped(train_data_scaled, seq_len=SEQ_LEN, verbose=True)
    print(f"\n2. Validation:")
    X_val_seq, y_val_seq = create_sequences_grouped(val_data_scaled, seq_len=SEQ_LEN, verbose=True)
    
    print(f"\nSecuencias creadas:")
    print(f"- Train: {X_train_seq.shape}")
    print(f"- Val: {X_val_seq.shape}")
    
    # Extraer IsHoliday para WMAE
    print(f"\nExtrayendo IsHoliday (índice {isholiday_idx}) para cálculo de WMAE:")
    is_holiday_val = X_val_seq[:, -1, isholiday_idx]
    print(f"- Total muestras val: {len(is_holiday_val):,}")
    print(f"- Semanas festivas: {np.sum(is_holiday_val == 1):,} ({np.sum(is_holiday_val == 1)/len(is_holiday_val)*100:.1f}%)")
    print(f"- Semanas regulares: {np.sum(is_holiday_val == 0):,} ({np.sum(is_holiday_val == 0)/len(is_holiday_val)*100:.1f}%)")
    
    # Entrenar modelo
    model, train_losses, val_losses = train_lstm_forecaster(
        X_train_seq, y_train_seq, 
        X_val_seq, y_val_seq,
        n_stores=n_stores,
        n_depts=n_depts,
        seq_len=SEQ_LEN,
        epochs=100,
        batch_size=128,
        lr=0.001
    )
    
    # Evaluar
    results = evaluate_lstm(model, X_val_seq, y_val_seq, is_holiday=is_holiday_val, 
                            scaler=scaler_y, verbose=True)
    
    # Guardar resultados
    with open('lstm_results.json', 'w') as f:
        json.dump({
            'wmape': float(results['wmape']),
            'wmae': float(results['wmae']),
            'mape': float(results['mape']),
            'mae': float(results['mae']),
            'rmse': float(results['rmse']),
            'r2': float(results['r2'])
        }, f, indent=4)
    
    print(f"\nArchivos generados:")
    print(f"- lstm_best_model.pt (mejor modelo)")
    print(f"- lstm_results.json (resultados)")
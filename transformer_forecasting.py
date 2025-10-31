# Este script entrena un modelo Transformer para predecir ventas semanales
# usando los datos procesados del notebook de EDA.

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Configuración de dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Cargando datos procesados
print("Cargando datos procesados...")
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_val = data['X_val']
y_train = data['y_train']
y_val = data['y_val']
feature_names = data['feature_names']
preprocessor = data['preprocessor']

print(f"✓ Datos cargados exitosamente")
print(f"\nForma de los datos:")
print(f"- X_train: {X_train.shape}")
print(f"- X_val: {X_val.shape}")
print(f"- y_train: {y_train.shape}")
print(f"- y_val: {y_val.shape}")
print(f"- Características: {len(feature_names)}")

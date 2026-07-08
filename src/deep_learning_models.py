import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

class LSTMRegressor(nn.Module):
    """
    Production-grade PyTorch Stacked LSTM Architecture for Financial Time Series.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stacked LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # Fully connected readout layers
        self.fc1 = nn.Linear(hidden_dim, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

def prepare_lstm_sequences(series, sequence_length=50):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : i + sequence_length])
        y.append(scaled_data[i + sequence_length])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

def train_and_forecast_lstm(train_series, test_series, sequence_length=30, epochs=15, batch_size=64):
    """
    Trains the PyTorch LSTM network and outputs predictions.
    """
    full_series = pd.concat([train_series, test_series])
    X_full, y_full, scaler = prepare_lstm_sequences(full_series, sequence_length)
    
    train_size = len(train_series) - sequence_length
    X_train, y_train = X_full[:train_size], y_full[:train_size]
    X_test = X_full[train_size:]
    
    # Convert datasets to PyTorch Tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    
    # Initialize Model, Loss, and Optimizer
    model = LSTMRegressor(input_dim=1, hidden_dim=50, num_layers=2, output_dim=1, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Training Loop
    model.train()
    dataset_size = X_train_t.size(0)
    
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            optimizer.zero_grad()
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
    # Evaluation Mode / Forecasting
    model.eval()
    with torch.no_grad():
        scaled_preds = model(X_test_t).numpy()
        
    predictions = scaler.inverse_transform(scaled_preds).flatten()
    return pd.Series(predictions, index=test_series.index)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DeepHit Model
class DeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepHit, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    @staticmethod
    def load_model():
        """Load trained DeepHit model from file"""
        model = DeepHit(input_dim=10, hidden_dim=64, output_dim=2).to(device)
        model.load_state_dict(torch.load("deephit_model.pth", map_location=device))
        model.eval()
        return model

import pandas as pd

def train_deephit(train_data, epochs=50, lr=0.001):
    # If train_data is not a pandas DataFrame, convert it
    if not hasattr(train_data, 'columns'):
        train_data = pd.DataFrame(train_data)
    
    if 'event' not in train_data.columns:
        raise ValueError("train_data must contain an 'event' column.")
    
    train_features = train_data.drop(columns=['event']).values
    train_labels = train_data['event'].values
    
    input_dim = train_features.shape[1]
    model = DeepHit(input_dim=input_dim, hidden_dim=64, output_dim=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "deephit_model.pth")

def predict_with_deephit(test_data):
    """Make predictions using trained DeepHit model"""
    model = DeepHit.load_model()
    test_data = torch.tensor(test_data.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        probabilities = model(test_data).cpu().numpy()

    binary_predictions = (probabilities[:, 1] > 0.5).astype(int)
    return binary_predictions.tolist(), probabilities[:, 1].tolist()

# Transformer Model
class TransSurv(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim):
        super(TransSurv, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    @staticmethod
    def load_model():
        """Load trained Transformer model from file"""
        model = TransSurv(input_dim=10, num_heads=2, hidden_dim=64, output_dim=2).to(device)
        model.load_state_dict(torch.load("trans_surv_model.pth", map_location=device))
        model.eval()
        return model

def train_transformer(train_data, epochs=50, lr=0.001):
    """Train the Transformer-based survival model using train_data"""
    if 'event' not in train_data.columns:
        raise ValueError("train_data must contain an 'event' column.")

    train_features = train_data.drop(columns=['event']).values
    train_labels = train_data['event'].values

    input_dim = train_features.shape[1]
    model = TransSurv(input_dim=input_dim, num_heads=2, hidden_dim=64, output_dim=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "trans_surv_model.pth")

def predict_with_transformer(test_data):
    """Make predictions using trained Transformer model"""
    model = TransSurv.load_model()
    test_data = torch.tensor(test_data.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        probabilities = model(test_data).cpu().numpy()

    binary_predictions = (probabilities[:, 1] > 0.5).astype(int)
    return binary_predictions.tolist(), probabilities[:, 1].tolist()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_deephit(train_data, epochs=50, lr=0.001):
    """
    Train the DeepHit model using train_data.

    Args:
        train_data (pandas.DataFrame or numpy.ndarray): Training dataset including features and labels.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
    """
    # Ensure 'event' column exists
    if isinstance(train_data, np.ndarray):
        raise ValueError("train_data must be a Pandas DataFrame with an 'event' column.")

    if 'event' not in train_data.columns:
        raise ValueError("train_data must contain an 'event' column.")

    # Extract features and labels
    train_features = train_data.drop(columns=['event']).values
    train_labels = train_data['event'].values

    input_dim = train_features.shape[1]
    model = DeepHit(input_dim=input_dim, hidden_dim=64, output_dim=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert to tensors and move to device (CPU/GPU)
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

def train_transformer(train_data, epochs=50, lr=0.001):
    """
    Train the Transformer-based survival model using train_data.

    Args:
        train_data (pandas.DataFrame or numpy.ndarray): Training dataset including features and labels.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
    """
    # Ensure 'event' column exists
    if isinstance(train_data, np.ndarray):
        raise ValueError("train_data must be a Pandas DataFrame with an 'event' column.")

    if 'event' not in train_data.columns:
        raise ValueError("train_data must contain an 'event' column.")

    # Extract features and labels
    train_features = train_data.drop(columns=['event']).values
    train_labels = train_data['event'].values

    input_dim = train_features.shape[1]
    model = TransSurv(input_dim=input_dim, num_heads=2, hidden_dim=64, output_dim=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert to tensors and move to device (CPU/GPU)
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

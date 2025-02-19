import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class DeephitModel:
    def __init__(self, input_dim, hidden_units=64):
        """
        Initializes a simple deep neural network (DNN) model for binary classification.
        This serves as a basic demonstration of the DeepHit process but is not the full
        DeepHit implementation as found in the `pycox` package.
        
        Parameters:
        - input_dim: Number of input features
        - hidden_units: Number of neurons in hidden layers (default: 64)
        """
        self.model = Sequential([
            Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),  # Adds dropout to prevent overfitting
            Dense(hidden_units, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax')  # Binary classification (2 output classes)
        ])
        
        # Compile the model using categorical crossentropy loss for classification
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    def fit(self, X, y, epochs=10, batch_size=32):
        """
        Trains the model on the provided dataset.
        
        Parameters:
        - X: Training feature matrix
        - y: One-hot encoded labels
        - epochs: Number of training iterations (default: 10)
        - batch_size: Size of data batches for training (default: 32)
        """
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def predict(self, X):
        """
        Makes predictions on the given input data.
        
        Parameters:
        - X: Input feature matrix for prediction
        
        Returns:
        - Predicted labels (0 or 1) based on the highest probability class
        """
        preds = self.model.predict(X)
        pred_labels = np.argmax(preds, axis=1)  # Convert probabilities to class labels
        return pred_labels  

def train_deephit(train_data):
    """
    Trains a simple DNN model using the provided training data.
    
    Parameters:
    - train_data: A dictionary or DataFrame containing at least:
      - "X": Features (numpy array or DataFrame)
      - "y": Labels (numpy array, assumed to be 0/1 for binary classification)
    
    Returns:
    - Trained DeephitModel instance
    """
    X = train_data["X"].values
    y = train_data["y"].values  # Labels should be 0 or 1
    
    input_dim = X.shape[1]
    
    # Convert labels to one-hot encoding for softmax classification
    y_onehot = to_categorical(y, num_classes=2)
    
    # Initialize and train the model
    model = DeephitModel(input_dim=input_dim, hidden_units=64)
    model.fit(X, y_onehot, epochs=5, batch_size=32)
    
    return model

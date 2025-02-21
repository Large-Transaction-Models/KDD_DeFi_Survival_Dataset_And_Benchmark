import numpy as np # For numerical operations
import tensorflow.keras.backend as K # For backend tensor operations in TensorFlow
from tensorflow.keras.models import Sequential # To create a sequential neural network model
from tensorflow.keras.layers import Dense, Dropout, Input # To build layers for the network
from tensorflow.keras.optimizers import Adam # To use the Adam optimizer during training
from tensorflow.keras.utils import to_categorical # To convert labels into one-hot encoded format
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # To control training with callbacks

# Define A Function To Compute Ranking Loss Based On The Positive And Negative Classes
def ranking_loss(y_true, y_pred):
    """
    Computes a simple ranking loss that encourages the positive class's 
    mean predicted probability to exceed that of the negative class by a margin
    """
    # Extract The Predicted Probability For Class 1
    p = y_pred[:, 1]
    # Create A Mask For Positive Samples Where The True Label Is Class 1
    pos_mask = K.cast(K.equal(K.argmax(y_true, axis=-1), 1), K.floatx())
    # Create A Mask For Negative Samples As The Complement Of The Positive Mask
    neg_mask = 1 - pos_mask
    # Compute The Mean Predicted Probability For Positive Samples
    pos_mean = K.sum(p * pos_mask) / (K.sum(pos_mask) + K.epsilon())
    # Compute The Mean Predicted Probability For Negative Samples
    neg_mean = K.sum(p * neg_mask) / (K.sum(neg_mask) + K.epsilon())
    # Define A Fixed Margin Value
    margin = 0.1
    # Return The Maximum Of Zero And The Difference Between The Margin And The Difference Of Means
    return K.maximum(0.0, margin - (pos_mean - neg_mean))

# Define A Combined Loss Function That Adds Categorical Crossentropy And A Weighted Ranking Loss
def combined_loss(y_true, y_pred):
    """
    Combined loss: standard categorical crossentropy plus a weighted ranking loss
    """
    # Compute The Standard Categorical Crossentropy Loss
    ce = K.categorical_crossentropy(y_true, y_pred)
    # Compute The Ranking Loss Using The Previously Defined Function
    rl = ranking_loss(y_true, y_pred)
    # Define A Weight For The Ranking Loss Component
    alpha = 0.2  # Weight For The Ranking Loss
    # Return The Sum Of The Categorical Crossentropy And The Weighted Ranking Loss
    return ce + alpha * rl

# Define A Class For The DeepHit Model Architecture And Training Procedures
class DeephitModel:
    # Initialize The DeepHit Model With The Provided Input Dimension And Hidden Units
    def __init__(self, input_dim, hidden_units=64):
        """
        Initializes a deep neural network model for binary classification following the DeepHit approach
        The model uses a combined loss (categorical crossentropy + ranking loss)
        
        Parameters:
        - input_dim: Number of input features
        - hidden_units: Number of neurons in hidden layers
        """
        # Build A Sequential Model With An Input Layer, Two Dense Layers With ReLU Activation, Dropout Layers, And An Output Layer With Softmax Activation
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(hidden_units, activation='relu'),
            Dropout(0.2),
            Dense(hidden_units, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax')  # Output Layer For Binary Classification
        ])
        # Compile The Model Using The Adam Optimizer, Combined Loss Function, And Accuracy As A Metric
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss=combined_loss,
                           metrics=['accuracy'])
    
    # Define A Method To Train The Model With Optional Validation Data
    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None):
        """
        Trains the DeepHit model using early stopping and learning rate reduction
        
        Parameters:
        - X: Training feature matrix
        - y: One-hot encoded labels
        - epochs: Maximum number of training epochs
        - batch_size: Batch size
        - validation_data: Optional tuple (val_X, val_y) for validation
        """
        # Define Callbacks For Early Stopping And Learning Rate Reduction Based On Training Loss
        callbacks = [
            EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6)
        ]
        # Train The Model Using The Provided Data And Callbacks With Verbose Set To 0 For Silent Training
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0,
                       validation_data=validation_data, callbacks=callbacks)
    
    # Define A Method To Predict Class Labels From Input Data
    def predict(self, X):
        """
        Predicts class labels for the given input
        
        Parameters:
        - X: Input feature matrix
        
        Returns:
        - Predicted class labels (0 or 1)
        """
        # Generate Predictions Using The Model And Select The Class With The Highest Predicted Probability For Each Sample
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

# Define A Function To Train The DeepHit Model Using Provided Training Data
def train_deephit(train_data):
    """
    Trains a DeepHit model using the provided training data
    
    Parameters:
    - train_data: A dictionary with keys:
         "X": Features
         "y": Labels (assumed to be 0/1)
         Optional keys "val_X" and "val_y" for validation data
    
    Returns:
    - A trained DeephitModel instance
    """
    # Convert Training Features And Labels To Numpy Arrays
    X = np.array(train_data["X"])
    y = np.array(train_data["y"])
    # Reshape X If It Is A One-Dimensional Array To Ensure It Has Two Dimensions
    if (X.ndim == 1):
        X = X.reshape(-1, 1)
    # Raise An Error If The Input Data Has No Features
    if (X.shape[1] == 0):
        raise ValueError("Invalid input: X has no features")
    # Determine The Number Of Input Features
    input_dim = X.shape[1]
    # Convert The Labels To A One-Hot Encoded Format For Binary Classification
    y_onehot = to_categorical(y, num_classes=2)
    
    # Initialize The DeepHit Model With The Determined Input Dimension And A Fixed Number Of Hidden Units
    model = DeephitModel(input_dim=input_dim, hidden_units=64)
    
    # Check If Validation Data Is Provided In The Training Data Dictionary
    if ("val_X" in train_data and "val_y" in train_data):
        # Convert Validation Features And Labels To Numpy Arrays
        val_X = np.array(train_data["val_X"])
        val_y = np.array(train_data["val_y"])
        # Reshape val_X If It Is A One-Dimensional Array To Ensure It Has Two Dimensions
        if (val_X.ndim == 1):
            val_X = val_X.reshape(-1, 1)
        # Convert The Validation Labels To A One-Hot Encoded Format
        val_y_onehot = to_categorical(val_y, num_classes=2)
        # Train The Model Using Both The Training And Validation Data
        model.fit(X, y_onehot, epochs=10, batch_size=32, validation_data=(val_X, val_y_onehot))
    else:
        # Train The Model Using Only The Training Data If No Validation Data Is Provided
        model.fit(X, y_onehot, epochs=10, batch_size=32)
    
    # Return The Trained DeepHit Model Instance
    return model

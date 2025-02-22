import numpy as np # Import Numpy For Numerical Operations
import tensorflow.keras.backend as K # Import Keras Backend For Tensor Operations
from tensorflow.keras.models import Sequential # Import Sequential Model Constructor
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, BatchNormalization # Added BatchNormalization for stability
from tensorflow.keras.optimizers import Adam # Import Adam Optimizer For Model Training
from tensorflow.keras.utils import to_categorical # Import Utility To Convert Labels To One-Hot Encoding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Import Callbacks For Training Control

# Define A Transformation Layer Function That Applies A Logarithmic Transformation To The Input
def transformation_layer(x):
    """
    Applies a logarithmic transformation to simulate the transformation of the risk function
    """
    # Clip input to avoid log(0) or log(negative) issues
    clipped_input = K.clip(1.0 + x, K.epsilon(), None)
    return K.log(clipped_input) # Return The Logarithm Of One Plus The Input

# Define A Class For The Transformation-Surv Model Architecture And Training Procedures
class TransformationSurvModel:
    # Initialize The Model With Input Dimension And Number Of Hidden Units
    def __init__(self, input_dim, hidden_units=64):
        """
        Initializes a deep neural network model for binary classification using the Transformation-Surv approach
        This model includes a transformation layer to simulate risk transformation
        
        Parameters:
        - input_dim: Number of input features
        - hidden_units: Number of neurons in hidden layers
        """
        # Build A Sequential Model With An Input Layer, Dense Layers, BatchNormalization, Dropout For Regularization, A Transformation Lambda Layer, And An Output Layer
        self.model = Sequential([
            Input(shape=(input_dim,)), # Define The Input Layer With The Specified Input Dimension
            Dense(hidden_units, activation='relu'), # Add A Dense Layer With ReLU Activation
            BatchNormalization(), # Added BatchNormalization for stability
            Dropout(0.2), # Adjusted dropout rate for stability
            Dense(hidden_units, activation='relu'), # Add A Second Dense Layer With ReLU Activation
            BatchNormalization(), # Added BatchNormalization for stability
            Lambda(transformation_layer), # Apply The Custom Transformation Layer To Simulate Risk Transformation
            Dense(2, activation='softmax') # Add An Output Layer For Binary Classification With Softmax Activation
        ])
        # Compile The Model With Adam Optimizer, Categorical Crossentropy Loss, And Accuracy Metric
        # Set learning rate to 0.0005 and added clipnorm for gradient clipping
        self.model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    # Define A Method To Train The Model Using Provided Data And Callbacks For Early Stopping And Learning Rate Reduction
    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, class_weight=None):
        """
        Trains the Transformation-Surv model using early stopping and learning rate reduction
        
        Parameters:
        - X: Training feature matrix
        - y: One-hot encoded labels
        - epochs: Maximum number of epochs
        - batch_size: Batch size
        - validation_data: Optional tuple (val_X, val_y) for validation
        - class_weight: Optional dictionary for imbalanced data, e.g. {0: 1.0, 1: 5.0}
        """
        # Define Callbacks To Monitor Loss And Adjust Training Accordingly
        # If validation data exists, monitor 'val_loss', otherwise 'loss'
        monitor_metric = 'val_loss' if validation_data is not None else 'loss'
        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=2, min_lr=1e-6)
        ]
        
        # Pass class_weight to model.fit to deal with imbalance
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0,
                       validation_data=validation_data, callbacks=callbacks,
                       class_weight=class_weight)
    
    # Define A Method To Predict Class Labels From Input Data
    def predict(self, X):
        """
        Predicts class labels for the given input
        
        Parameters:
        - X: Input feature matrix
        
        Returns:
        - Predicted class labels (0 or 1)
        """
        # Generate Predictions Using The Model And Select The Class With The Highest Probability For Each Sample
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

# Define A Function To Train A Transformation-Surv Model Using Provided Training Data
def train_transformation_surv(train_data, class_weight=None):
    """
    Trains a Transformation-Surv model using the provided training data
    
    Parameters:
    - train_data: A dictionary with keys:
         "X": Features
         "y": Labels (assumed to be 0/1)
         Optional keys "val_X" and "val_y" for validation data
    - class_weight: Optional dictionary to handle class imbalance
    
    Returns:
    - A trained TransformationSurvModel instance
    """
    # Convert The Training Features And Labels To Numpy Arrays
    X = np.array(train_data["X"])
    y = np.array(train_data["y"])
    # Reshape X To Two Dimensions If It Is A One-Dimensional Array
    if (X.ndim == 1):
        X = X.reshape(-1, 1)
    # Raise An Error If There Are No Features In X
    if (X.shape[1] == 0):
        raise ValueError("Invalid input: X has no features")
    # Determine The Number Of Input Features From The Shape Of X
    input_dim = X.shape[1]
    # Convert The Labels To A One-Hot Encoded Format For Binary Classification
    y_onehot = to_categorical(y, num_classes=2)
    
    # Initialize The TransformationSurvModel With The Determined Input Dimension And Fixed Number Of Hidden Units
    model = TransformationSurvModel(input_dim=input_dim, hidden_units=64)
    
    # Check If Validation Data Is Provided In The Training Data Dictionary
    if ("val_X" in train_data and "val_y" in train_data):
        val_X = np.array(train_data["val_X"])
        val_y = np.array(train_data["val_y"])
        if (val_X.ndim == 1):
            val_X = val_X.reshape(-1, 1)
        val_y_onehot = to_categorical(val_y, num_classes=2)
        
        # Fit The Model Using Both Training And Validation Data
        # Pass class_weight for imbalance
        model.fit(X, y_onehot, epochs=10, batch_size=32, 
                  validation_data=(val_X, val_y_onehot), 
                  class_weight=class_weight)
    else:
        # Fit The Model Using Only The Training Data If Validation Data Is Not Provided
        model.fit(X, y_onehot, epochs=10, batch_size=32, class_weight=class_weight)
    
    # Return The Trained TransformationSurvModel Instance
    return model

library(reticulate) # Enables integration with Python, allowing R to call Python functions
library(dplyr) # Provides functions for data manipulation and transformation

# Function to train a DeepHit model using the given training and test datasets
deephit_model <- function(train_data, test_data, epochs = 10, num_nodes = c(64L, 64L), 
                          dropout = 0, batch_size = 256L, lr = 0.001, class_weight = NULL) {
  
  # library(reticulate) # Enables integration with Python, allowing R to call Python functions
  # library(dplyr) # Provides functions for data manipulation and transformation
  # Data Validation: Ensure both training and testing datasets are provided, not empty, and contain the required 'event' column.
  if (is.null(train_data) || is.null(test_data)) {
    stop("Error: train_data or test_data is NULL!")
  }
  if (nrow(train_data) == 0 || nrow(test_data) == 0) {
    stop("Error: train_data or test_data is empty!")
  }
  if (!("event" %in% colnames(train_data)) || !("event" %in% colnames(test_data))) {
    stop("Error: 'event' column is missing in train_data or test_data!")
  }
  
  # Split the training data into training (80%) and validation (20%) subsets
  set.seed(123) # Setting seed for reproducibility
  train_idx <- sample(seq_len(nrow(train_data)), size = 0.8 * nrow(train_data))
  val_idx <- setdiff(seq_len(nrow(train_data)), train_idx)
  train_subset <- train_data[train_idx, ]
  val_subset <- train_data[val_idx, ]
  
  # Preprocess: Remove the "id" column if it exists in any of the datasets, as it is not needed for model training.
  if ("id" %in% names(train_subset)) {
    train_subset <- train_subset %>% select(-id)
  }
  if ("id" %in% names(val_subset)) {
    val_subset <- val_subset %>% select(-id)
  }
  if ("id" %in% names(test_data)) {
    test_data <- test_data %>% select(-id)
  }
  
  # Feature and Label Extraction:
  # For each dataset, separate the features (all columns except 'event') and the labels (the 'event' column).
  X_train <- train_subset %>% select(-event)
  y_train <- train_subset$event
  X_val <- val_subset %>% select(-event)
  y_val <- val_subset$event
  X_test <- test_data %>% select(-event)
  y_test <- test_data$event
  
  # Define a helper function to convert input data to numeric type and handle missing values
  # This function checks the type of the input:
  # - If already numeric, it is returned as-is
  # - If a factor or character, it converts to numeric (with warnings suppressed)
  # - Missing values (NA) are replaced with the median of the column; if median is unavailable, 0 is used
  convert_to_numeric <- function(x) {
    if (is.numeric(x)) {
      res <- x
    } 
    else if (is.factor(x)) {
      res <- suppressWarnings(as.numeric(as.character(x)))
    } 
    else if (is.character(x)) {
      res <- suppressWarnings(as.numeric(x))
    } 
    else {
      res <- suppressWarnings(as.numeric(x))
    }
    if (any(is.na(res))) {
      med <- median(res, na.rm = TRUE)
      if (is.na(med)) med <- 0
      res[is.na(res)] <- med
    }
    return(res)
  }
  
  # Apply the conversion to numeric for all features in the training, validation, and testing sets
  X_train <- X_train %>% mutate(across(everything(), convert_to_numeric))
  X_val <- X_val %>% mutate(across(everything(), convert_to_numeric))
  X_test <- X_test %>% mutate(across(everything(), convert_to_numeric))
  
  # Convert data frames to matrices because the Python model expects matrix inputs
  X_train <- as.matrix(X_train)
  X_val <- as.matrix(X_val)
  X_test <- as.matrix(X_test)
  
  # Convert labels to binary numeric values (0/1)
  # Assumes the original labels start at 1, so subtract 1 to get 0-based labels
  y_train <- as.numeric(y_train) - 1
  y_val <- as.numeric(y_val) - 1
  y_test <- as.numeric(y_test) - 1
  
  # Prepare Data for Python:
  # Package training data along with validation data and model parameters into a list
  # Then convert the list to a Python object using r_to_py
  train_data_py <- r_to_py(list(
    X = X_train,
    y = y_train,
    val_X = X_val,
    val_y = y_val,
    epochs = as.integer(epochs),
    num_nodes = num_nodes,
    dropout = dropout,
    batch_size = as.integer(batch_size),
    lr = lr
  ))
  
  # Similarly, prepare the test dataset for prediction
  test_data_py <- r_to_py(list(X = X_test, y = y_test))
  
  # Convert class_weight to a Python dictionary if provided
  class_weight_py <- NULL
  if (!is.null(class_weight)) {
    class_weight_py <- r_to_py(as.list(class_weight))
  }
  
  # Load the Python script that contains the implementation of the DeepHit model
  source_python("deephit_model.py")
  
  # Train the DeepHit model using the training data prepared in Python
  # Pass class_weight_py to handle class imbalance in Python
  deep_model <- train_deephit(train_data_py, class_weight = class_weight_py)
  
  # Use the trained model to make predictions on the test data
  predictions <- deep_model$predict(test_data_py[["X"]])
  
  # Convert the predictions from Python back to R
  predictions <- py_to_r(predictions)
  
  # Create a confusion matrix comparing the predicted labels to the actual test labels
  test_conf_matrix <- table(
    Predicted = factor(predictions, levels = c(0, 1)),
    Actual = factor(y_test, levels = c(0, 1))
  )
  
  # print(test_conf_matrix)
  
  # Calculate model performance metrics (such as accuracy, sensitivity, etc.) using the confusion matrix
  # The function calculate_model_metrics is assumed to be defined elsewhere
  metrics_dh <- calculate_model_metrics(test_conf_matrix, predictions, "DeepHit")
  
  # Convert the metrics into a data frame for reporting or further analysis
  metrics_dh_dataframe <- get_dataframe("DeepHit", metrics_dh)
  
  # Return a list containing both the metrics data frame and the raw metrics
  return(list(metrics_dh_dataframe = metrics_dh_dataframe, metrics_dh = metrics_dh))
}

# Function to train a Transformation Survival model using the given training and test datasets
transformation_surv_model <- function(train_data, test_data, epochs = 10, num_nodes = c(64L, 64L), dropout = 0, 
                                      batch_size = 256L, lr = 0.001, class_weight = NULL) {
  
  # library(reticulate) # Enables integration with Python, allowing R to call Python functions
  # library(dplyr) # Provides functions for data manipulation and transformation
  # Data Validation: Ensure that training and testing datasets are provided, not empty, and include the 'event' column.
  if (is.null(train_data) || is.null(test_data)) {
    stop("Error: train_data or test_data is NULL!")
  }
  if (nrow(train_data) == 0 || nrow(test_data) == 0) {
    stop("Error: train_data or test_data is empty!")
  }
  if (!("event" %in% colnames(train_data)) || !("event" %in% colnames(test_data))) {
    stop("Error: 'event' column is missing!")
  }
  
  # Split the training data into training (80%) and validation (20%) subsets
  set.seed(123) # Set seed for reproducibility
  train_idx <- sample(seq_len(nrow(train_data)), size = 0.8 * nrow(train_data))
  val_idx <- setdiff(seq_len(nrow(train_data)), train_idx)
  train_subset <- train_data[train_idx, ]
  val_subset <- train_data[val_idx, ]
  
  # Preprocess: Remove the "id" column from training, validation, and testing datasets if present
  if ("id" %in% names(train_subset)) {
    train_subset <- train_subset %>% select(-id)
  }
  if ("id" %in% names(val_subset)) {
    val_subset <- val_subset %>% select(-id)
  }
  if ("id" %in% names(test_data)) {
    test_data <- test_data %>% select(-id)
  }
  
  # Feature and Label Extraction:
  # Separate features and labels for training, validation, and test datasets
  X_train <- train_subset %>% select(-event)
  y_train <- train_subset$event
  X_val <- val_subset %>% select(-event)
  y_val <- val_subset$event
  X_test <- test_data %>% select(-event)
  y_test <- test_data$event
  
  # Define a helper function to convert features to numeric and handle missing values
  convert_to_numeric <- function(x) {
    if (is.numeric(x)) {
      res <- x
    } 
    else if (is.factor(x)) {
      res <- suppressWarnings(as.numeric(as.character(x)))
    } 
    else if (is.character(x)) {
      res <- suppressWarnings(as.numeric(x))
    } 
    else {
      res <- suppressWarnings(as.numeric(x))
    }
    if (any(is.na(res))) {
      med <- median(res, na.rm = TRUE)
      if (is.na(med)) med <- 0 # Default to 0 if median cannot be computed
      res[is.na(res)] <- med
    }
    return(res)
  }
  
  # Convert all feature columns in training, validation, and test sets to numeric
  X_train <- X_train %>% mutate(across(everything(), convert_to_numeric))
  X_val <- X_val %>% mutate(across(everything(), convert_to_numeric))
  X_test <- X_test %>% mutate(across(everything(), convert_to_numeric))
  
  # Convert the data frames of features to matrices
  X_train <- as.matrix(X_train)
  X_val <- as.matrix(X_val)
  X_test <- as.matrix(X_test)
  
  # Convert labels to binary numeric values (0/1)
  y_train <- as.numeric(y_train) - 1
  y_val <- as.numeric(y_val) - 1
  y_test <- as.numeric(y_test) - 1
  
  # Prepare Data for Python:
  # Create a list containing the training features, labels, validation data, and model parameters, then convert it to a Python object.
  train_data_py <- r_to_py(list(
    X = X_train,
    y = y_train,
    val_X = X_val,
    val_y = y_val,
    epochs = as.integer(epochs),
    num_nodes = num_nodes,
    dropout = dropout,
    batch_size = as.integer(batch_size),
    lr = lr
  ))
  
  # Prepare the test data in a similar fashion
  test_data_py <- r_to_py(list(
    X = X_test,
    y = y_test
  ))
  
  # Convert class_weight to a Python dictionary if provided
  class_weight_py <- NULL
  if (!is.null(class_weight)) {
    class_weight_py <- r_to_py(as.list(class_weight))
  }
  
  # Load the Python script that implements the Transformation Survival model
  source_python("transformation_surv_model.py")
  
  # Train the Transformation Survival model using the provided training data
  # Pass class_weight_py to handle class imbalance in Python
  trans_model <- train_transformation_surv(train_data_py, class_weight = class_weight_py)
  
  # Use the trained model to predict outcomes on the test dataset
  predictions <- trans_model$predict(test_data_py[["X"]])
  
  # Convert predictions from Python format back to R
  predictions <- py_to_r(predictions)
  
  # Create a confusion matrix comparing predicted outcomes with the actual test labels
  test_conf_matrix <- table(
    Predicted = factor(predictions, levels = c(0, 1)),
    Actual = factor(y_test, levels = c(0, 1))
  )
  
  # print(test_conf_matrix)
  
  # Calculate performance metrics using the confusion matrix
  # This function (calculate_model_metrics) is assumed to be defined elsewhere
  metrics_tfs <- calculate_model_metrics(test_conf_matrix, predictions, "TransformationSurv")
  
  # Convert the metrics into a data frame for easy viewing and further analysis
  metrics_tfs_dataframe <- get_dataframe("TransformationSurv", metrics_tfs)
  
  # Return a list containing both the metrics data frame and the raw metrics
  return(list(metrics_tfs_dataframe = metrics_tfs_dataframe, metrics_tfs = metrics_tfs))
}
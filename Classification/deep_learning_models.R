library(reticulate)  # Enables R-Python integration
library(dplyr)

deephit_model <- function(train_data, test_data) {
  # Import the Python script containing the DeepHit model
  source_python("~/KDD_DeFi_Survival_Dataset_And_Benchmark/Classification/deephit_model.py")
  
  # Prepare the training data in R
  X_train <- subset(train_data, select = -c(event))
  y_train <- train_data$event # Labels (0 or 1)
  
  X_test <- subset(test_data, select = -c(event))
  y_test <- test_data$event
  
  # Convert R data to a format usable in Python, pass data as a dictionary (list) to Python
  train_data_py <- r_to_py(list(X = X_train, y = y_train))
  test_data_py  <- r_to_py(list(X = X_test,  y = y_test))
  
  # Call the Python function `train_deephit` to train the model
  deephit_trained_model <- train_deephit(train_data_py)
  
  # Use the trained Python model to make predictions on test data
  predict_probabilities_dh <- deephit_trained_model$predict(test_data_py[["X"]])
  
  # Convert the Python prediction results back to R (0/1 labels)
  predict_probabilities_dh <- py_to_r(predict_probabilities_dh)
  
  # Compute the confusion matrix to evaluate performance
  test_conf_matrix <- table(Predicted = predict_probabilities_dh, Actual = y_test)
  
  # Evaluate model performance
  metrics_dh <- calculate_model_metrics(test_conf_matrix, predict_probabilities_dh, "DeepHit")
  
  # Format results to match the structure of decision_tree output
  metrics_dh_dataframe <- get_dataframe("DeepHit", metrics_dh)
  
  return(list(metrics_dh_dataframe = metrics_dh_dataframe, metrics_dh = metrics_dh))
}

transformation_surv_model <- function(train_data, test_data) {
  # Import the Python script containing the Transformation-Surv model
  source_python("~/KDD_DeFi_Survival_Dataset_And_Benchmark/Classification/transformation_surv_model.py")
  
  # Prepare the training data in R
  X_train <- subset(train_data, select = -c(event))
  y_train <- train_data$event # Labels (0 or 1)
  
  X_test <- subset(test_data, select = -c(event))
  y_test <- test_data$event
  
  # Convert R data to a format usable in Python, pass data as a dictionary (list) to Python
  train_data_py <- r_to_py(list(X = X_train, y = y_train))
  test_data_py  <- r_to_py(list(X = X_test,  y = y_test))
  
  # Call the Python function `train_transformation_surv` to train the model
  transformation_surv_trained_model <- train_transformation_surv(train_data_py)
  
  # Use the trained Python model to make predictions on test data
  predict_probabilities_xx <- transformation_surv_trained_model$predict(test_data_py[["X"]])
  
  # Convert the Python prediction results back to R
  predict_probabilities_xx <- py_to_r(predict_probabilities_xx)
  
  # Compute the confusion matrix to evaluate performance
  test_conf_matrix <- table(Predicted = predict_probabilities_xx, Actual = y_test)
  
  # Evaluate model performance
  metrics_xx <- calculate_model_metrics(test_conf_matrix, predict_probabilities_xx, "Transformation-Surv")
  
  # Format results to match the structure of decision_tree output
  metrics_xx_dataframe <- get_dataframe("Transformation-Surv", metrics_xx)
  
  return(list(metrics_xx_dataframe = metrics_xx_dataframe, metrics_xx = metrics_xx))
}
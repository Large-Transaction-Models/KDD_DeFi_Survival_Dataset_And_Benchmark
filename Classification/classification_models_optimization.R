library(data.table)
library(dplyr)
library(glmnet)
library(rpart)
library(caret)
library(e1071)
library(parallel)
library(xgboost)
library(pROC)

logistic_regression_op <- function(train_data, test_data) {
  # Load required libraries
  # library(glmnet) # Logistic regression with regularization
  # library(data.table) # Efficient data handling
  # library(pROC) # Compute AUC for validation
  
  # Convert train_data and test_data to data.table format
  setDT(train_data)
  setDT(test_data)
  
  # Split train_data into train_set (80%) and validation_set (20%)
  # Ensure reproducibility
  set.seed(123)
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Identify numeric features and scale them
  numeric_features <- names(train_set)[sapply(train_set, is.numeric)]
  
  # Standardize numeric features
  train_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  validation_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert datasets to matrix format as required by glmnet
  x_train <- model.matrix(event ~ . - 1, data = train_set)
  # Convert event labels to 0/1
  y_train <- as.numeric(train_set$event == "yes")
  
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)
  # Convert event labels to 0/1
  y_validation <- as.numeric(validation_set$event == "yes")
  
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  
  # Apply logistic regression with Lasso regularization (alpha = 1)
  logistic_regression_classifier <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
  
  # Perform cross-validation to get a range of lambda values
  cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
  # Extract lambda values from cross-validation
  lambda_candidates <- cv_model$lambda
  
  # Select best lambda based on AUC from validation set
  best_lambda <- NULL
  # Initialize best AUC score
  best_auc <- -Inf
  
  for (lambda in lambda_candidates) {
    # Predict probabilities on validation set
    predict_probabilities_val <- predict(logistic_regression_classifier, s = lambda, newx = x_validation, 
                                         type = "response")
    # Convert matrix to numeric vector
    predict_probabilities_val <- as.vector(predict_probabilities_val)
    
    # Compute AUC for validation set
    roc_obj <- roc(y_validation, predict_probabilities_val)
    auc_val <- auc(roc_obj)
    
    # Update best lambda if current AUC is better
    if (auc_val > best_auc) {
      best_auc <- auc_val
      best_lambda <- lambda
    }
  }
  
  # Train final model with the best lambda
  final_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)
  
  # Predict on validation dataset using the best lambda
  predict_probabilities_val <- predict(final_model, newx = x_validation, type = "response")
  predict_probabilities_val <- as.vector(predict_probabilities_val)
  binary_prediction_val <- ifelse(predict_probabilities_val > 0.5, "yes", "no")
  validation_conf_matrix <- table(Predicted = binary_prediction_val, Actual = validation_set$event)
  
  # Evaluate validation set performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, 
                                                "Logistic Regression (Validation)")
  
  # Predict on test dataset using the best lambda
  predict_probabilities_test <- predict(final_model, newx = x_test, type = "response")
  predict_probabilities_test <- as.vector(predict_probabilities_test)
  binary_prediction_test <- ifelse(predict_probabilities_test > 0.5, "yes", "no")
  
  # Create confusion matrix for the test set
  test_conf_matrix <- table(Predicted = binary_prediction_test, Actual = test_data$event)
  
  # Evaluate model performance on the test set
  metrics_lr <- calculate_model_metrics(test_conf_matrix, predict_probabilities_test, "Logistic Regression")
  
  # Store calculated metrics in a structured dataframe
  metrics_lr_dataframe <- get_dataframe("Logistic Regression", metrics_lr)
  
  return (list(metrics_lr_dataframe = metrics_lr_dataframe, metrics_lr = metrics_lr))
}

decision_tree_op <- function(train_data, test_data) {
  # Load required libraries
  # library(rpart)
  # library(caret) # Load caret for data partitioning
  
  # Split train_data into train_set (80%) and validation_set (20%)
  # Ensure reproducibility
  set.seed(123)
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Define candidate complexity parameters (cp) for pruning
  # Range of possible cp values
  cp_candidates <- seq(0.0001, 0.05, by = 0.002)
  best_cp <- NULL
  # Initialize the best AUC score
  best_auc <- -Inf
  
  # Loop over different cp values to find the best one using the validation set
  for (cp in cp_candidates) {
    # Train the decision tree model with the current cp value
    decision_tree_model <- rpart(
      event ~ .,
      data = train_set,
      method = "class",
      control = rpart.control(
        cp = cp,        # Adjusting cp for pruning
        maxdepth = 30,  # Keeping max depth fixed
        minsplit = 20   # Keeping minsplit fixed
      )
    )
    
    # Predict probabilities on validation set
    # Get probability for "yes"
    predict_probabilities_val <- predict(decision_tree_model, validation_set, type = "prob")[, 2]
    
    # Compute AUC for validation set
    auc_val <- auc(roc(validation_set$event, predict_probabilities_val))
    
    # Update the best cp if current AUC is higher
    if (auc_val > best_auc) {
      best_auc <- auc_val
      best_cp <- cp
    }
  }
  
  # Train the final model with the best cp value
  decision_tree_classifier <- rpart(
    event ~ .,
    data = train_set,
    method = "class",
    control = rpart.control(
      cp = best_cp,  # Using the best cp found from validation set
      maxdepth = 30,
      minsplit = 20
    )
  )
  
  # Predict on the validation dataset using the selected cp
  predict_probabilities_val <- predict(decision_tree_classifier, validation_set, type = "class")
  validation_conf_matrix <- table(Predicted = predict_probabilities_val, Actual = validation_set$event)
  
  # Evaluate validation performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, 
                                                "Decision Tree (Validation)")
  
  # Predict on the testing dataset using the final model
  predict_probabilities_dt <- predict(decision_tree_classifier, test_data, type = "class")
  test_conf_matrix <- table(Predicted = predict_probabilities_dt, Actual = test_data$event)
  
  # Evaluate model performance
  metrics_dt <- calculate_model_metrics(test_conf_matrix, predict_probabilities_dt, "Decision Tree")
  
  # Create a dataframe with the desired structure
  metrics_dt_dataframe = get_dataframe("Decision Tree", metrics_dt)
  
  return (list(metrics_dt_dataframe = metrics_dt_dataframe, metrics_dt = metrics_dt))
}

naive_bayes_op <- function(train_data, test_data) {
  # Load required libraries
  # library(e1071)
  # library(caret) # Load caret for data partitioning
  
  target_column = "event"
  
  # Convert the target column to a factor if it's not already
  train_data[[target_column]] <- as.factor(train_data[[target_column]])
  test_labels <- as.factor(test_data[[target_column]])
  
  # Split train_data into train_set (80%) and validation_set (20%)
  # Ensure reproducibility
  set.seed(123)
  trainIndex <- createDataPartition(train_data[[target_column]], p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Remove the target column from the validation and test sets for prediction
  validation_features <- validation_set %>% select(-all_of(target_column))
  test_features <- test_data %>% select(-all_of(target_column))
  
  # Define candidate Laplace smoothing values
  # Range of laplace values to test
  laplace_candidates <- seq(0, 1, by = 0.1)
  best_laplace <- NULL
  # Initialize the best AUC score
  best_auc <- -Inf
  
  # Hyperparameter tuning: Find the best laplace value using validation set
  for (laplace in laplace_candidates) {
    # Train Naïve Bayes model with current laplace value
    nb_model <- naiveBayes(as.formula(paste(target_column, "~ .")), data = train_set, laplace = laplace)
    
    # Get probability predictions on the validation set
    # Probability for "yes"
    validation_probabilities <- predict(nb_model, validation_features, type = "raw")[, 2]
    
    # Compute AUC for validation set
    auc_val <- auc(roc(validation_set[[target_column]], validation_probabilities))
    
    # Update best laplace if current AUC is higher
    if (auc_val > best_auc) {
      best_auc <- auc_val
      best_laplace <- laplace
    }
  }
  
  # Train the final Naïve Bayes model with the best laplace value
  nb_model <- naiveBayes(as.formula(paste(target_column, "~ .")), data = train_set, laplace = best_laplace)
  
  # Make predictions on the validation set
  validation_predictions <- predict(nb_model, validation_features)
  
  # Get prediction probabilities for validation set
  validation_probabilities <- predict(nb_model, validation_features, type = "raw")
  
  # Evaluate model performance with a confusion matrix for validation set
  validation_conf_matrix <- table(Predicted = validation_predictions, Actual = validation_set[[target_column]])
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, validation_probabilities, 
                                                "Naive Bayes (Validation)")
  
  # Make predictions on the test set using the final model
  predictions <- predict(nb_model, test_features)
  
  # Get prediction probabilities for test set
  prediction_probabilities <- predict(nb_model, test_features, type = "raw")
  
  # Ensure both predicted and actual labels are factors with the same levels
  predictions <- factor(predictions, levels = levels(test_labels))
  
  # Evaluate model performance with a confusion matrix for test set
  conf_matrix <- table(Predicted = predictions, Actual = test_labels)
  
  metrics_nb <- calculate_model_metrics(conf_matrix, prediction_probabilities, "Naive Bayes")
  
  # Create a dataframe with the desired structure
  metrics_nb_dataframe = get_dataframe("Naive Bayes", metrics_nb)
  
  # Each classification model needs to return these two variables
  return (list(metrics_nb_dataframe = metrics_nb_dataframe, metrics_nb = metrics_nb))
}

XG_Boost <- function(train_data, test_data, threshold = 0.5, if_smote = FALSE) {
  # Convert train_data and test_data to data.table format
  setDT(train_data)
  setDT(test_data)
  
  # Split train_data into train_set (80%) and validation_set (20%)
  # ensure reproducibility
  set.seed(123)
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Identify numeric features and scale them
  numeric_features <- names(train_set)[sapply(train_set, is.numeric)]
  train_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  validation_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Apply SMOTE on the training set only if if_smote is TRUE to avoid data leakage
  if (if_smote == TRUE) {
    train_set <- smote_data(train_set)
  }
  
  # Convert data to matrix format required by XGBoost
  x_train <- model.matrix(event ~ . - 1, data = train_set)
  # Convert event labels to 0/1
  y_train <- as.numeric(train_set$event == "yes")
  
  # Calculate scale_pos_weight based on training data distribution
  scale_pos_weight <- sum(y_train == 0) / sum(y_train == 1)
  
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)
  y_validation <- as.numeric(validation_set$event == "yes")
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  
  # Convert to DMatrix format, which is optimized for XGBoost
  dtrain <- xgb.DMatrix(data = x_train, label = y_train)
  dvalidation <- xgb.DMatrix(data = x_validation, label = y_validation)
  dtest <- xgb.DMatrix(data = x_test)
  
  # Detect available CPU cores for parallel computation
  num_cores <- detectCores()
  
  # Define XGBoost hyperparameters
  params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = num_cores,
    scale_pos_weight = scale_pos_weight # Adjusting for class imbalance
  )
  
  # Train XGBoost model with early stopping using validation set
  xgb_model <- xgb.train(params = params,
                         data = dtrain,
                         nrounds = 200,
                         early_stopping_rounds = 10,
                         watchlist = list(train = dtrain, validation = dvalidation),
                         verbose = 0)
  
  # Predict probabilities on the validation dataset
  predict_probabilities_val <- predict(xgb_model, dvalidation)
  # Adjust threshold to handle class imbalance
  binary_prediction_val <- ifelse(predict_probabilities_val > threshold, "yes", "no")
  validation_conf_matrix <- table(factor(binary_prediction_val, levels = c("yes", "no")),
                                  factor(validation_set$event, levels = c("yes", "no")))
  
  # Evaluate validation set performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, 
                                                "XGBoost (Validation)")
  
  # Predict probabilities on the test dataset
  predict_probabilities_xgb <- predict(xgb_model, dtest)
  # Convert predicted probabilities into binary class labels (yes/no) using the threshold parameter
  binary_prediction_xgb <- ifelse(predict_probabilities_xgb > threshold, "yes", "no")
  
  # Ensure both predicted and actual labels are factors with the same levels
  binary_prediction_xgb <- factor(binary_prediction_xgb, levels = c("yes", "no"))
  test_data$event <- factor(test_data$event, levels = c("yes", "no"))
  
  # Ensure confusion matrix includes all levels
  confusion_matrix_xgb <- table(factor(binary_prediction_xgb, levels = c("yes", "no")), 
                                factor(test_data$event, levels = c("yes", "no")))
  
  # Ensure matrix is complete to avoid subscript out of bounds error
  if (!all(c("yes", "no") %in% rownames(confusion_matrix_xgb))) {
    confusion_matrix_xgb <- matrix(0, nrow = 2, ncol = 2, dimnames = list(c("yes", "no"), c("yes", "no")))
  }
  
  # Evaluate model performance using accuracy, precision, recall, and F1-score
  metrics_xgb <- calculate_model_metrics(confusion_matrix_xgb, predict_probabilities_xgb, "XGBoost")
  
  # Store the calculated metrics in a structured dataframe for easy comparison
  metrics_xgb_dataframe <- get_dataframe("XGBoost", metrics_xgb)
  
  # Return both the detailed metrics list and the formatted dataframe
  return (list(metrics_xgb_dataframe = metrics_xgb_dataframe, metrics_xgb = metrics_xgb))
}

elastic_net_op <- function(train_data, test_data) {
  # Load required libraries
  # library(glmnet) # Required for Elastic Net (Lasso + Ridge regularization)
  # library(data.table) # For efficient data handling using data.table
  # library(pROC) # Compute AUC for validation
  
  # Convert train_data and test_data to data.table format for optimized processing
  setDT(train_data)
  setDT(test_data)
  
  # Split train_data into train_set (80%) and validation_set (20%)
  # Ensure reproducibility
  set.seed(123)
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Identify numeric features in the dataset for standardization
  numeric_features <- names(train_set)[sapply(train_set, is.numeric)]
  
  # Standardize numeric columns (mean = 0, standard deviation = 1) to improve model performance
  train_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  validation_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert the dataset into a matrix format, as required by glmnet
  x_train <- model.matrix(event ~ . - 1, data = train_set)
  # Convert event labels to 0/1
  y_train <- as.numeric(train_set$event == "yes")
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)
  # Convert event labels to 0/1
  y_validation <- as.numeric(validation_set$event == "yes")
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  
  # Define candidate alpha values for Elastic Net optimization
  alpha_candidates <- seq(0, 1, by = 0.1)  # Range from 0 (Ridge) to 1 (Lasso)
  best_alpha <- NULL
  best_lambda <- NULL
  # Initialize best AUC score
  best_auc <- -Inf
  
  # Iterate through different alpha values to find the best one using validation set
  for (alpha in alpha_candidates) {
    # Perform cross-validation to find the best lambda for the current alpha
    cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = alpha)
    # Get best lambda from cross-validation
    lambda_min <- cv_model$lambda.min
    
    # Predict on validation set
    predict_probabilities_val <- predict(cv_model$glmnet.fit, s = lambda_min, newx = x_validation, 
                                         type = "response")
    # Convert matrix to numeric vector
    predict_probabilities_val <- as.vector(predict_probabilities_val)
    
    # Compute AUC for validation set
    roc_obj <- roc(y_validation, predict_probabilities_val)
    auc_val <- auc(roc_obj)
    
    # Update best alpha and lambda if current AUC is higher
    if (auc_val > best_auc) {
      best_auc <- auc_val
      best_alpha <- alpha
      best_lambda <- lambda_min
    }
  }
  
  # Train the final Elastic Net model with the best alpha and lambda values
  elastic_net_model <- glmnet(x_train, y_train, family = "binomial", alpha = best_alpha, lambda = best_lambda)
  
  # Predict on the validation dataset using the selected best parameters
  predict_probabilities_val <- predict(elastic_net_model, newx = x_validation, type = "response")
  # Convert matrix to numeric vector
  predict_probabilities_val <- as.vector(predict_probabilities_val)
  binary_prediction_val <- ifelse(predict_probabilities_val > 0.5, "yes", "no")
  validation_conf_matrix <- table(Predicted = binary_prediction_val, Actual = validation_set$event)
  
  # Evaluate validation performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, 
                                                "Elastic Net (Validation)")
  
  # Predict event probabilities for the test dataset using the best parameters
  predict_probabilities_en <- predict(elastic_net_model, newx = x_test, type = "response")
  # Convert matrix to numeric vector
  predict_probabilities_en <- as.vector(predict_probabilities_en)
  binary_prediction_en <- ifelse(predict_probabilities_en > 0.5, "yes", "no")
  
  # Create a confusion matrix to compare predicted vs. actual outcomes in the test set
  confusion_matrix_en <- table(Predicted = binary_prediction_en, Actual = test_data$event)
  
  # Evaluate model performance using key classification metrics (accuracy, precision, recall, F1-score)
  metrics_en <- calculate_model_metrics(confusion_matrix_en, predict_probabilities_en, "Elastic Net")
  
  # Store the calculated metrics in a structured dataframe for easy comparison
  metrics_en_dataframe <- get_dataframe("Elastic Net", metrics_en)
  
  # Return both the detailed metrics list and the formatted dataframe
  return (list(metrics_en_dataframe = metrics_en_dataframe, metrics_en = metrics_en))
}
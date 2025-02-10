library(data.table)
library(dplyr)
library(glmnet)
library(rpart)
library(caret)
library(e1071)
library(parallel)
library(xgboost)
library(pROC)

logistic_regression <- function(train_data, test_data) {
  # Ensure train_data and test_data are in data.table format for fast operations
  setDT(train_data)
  setDT(test_data)
  
  # Split train_data into train_set (80%) and validation_set (20%)
  set.seed(123)  # Ensure reproducibility
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Identify numeric features in the dataset and scale them
  numeric_features <- names(train_set)[sapply(train_set, is.numeric)]
  
  # Scale numeric columns in train set
  train_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  # Scale numeric columns in validation set
  validation_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  # Scale numeric columns in test set
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert datasets to matrix format as required by the glmnet package
  x_train <- model.matrix(event ~ . - 1, data = train_set)  
  y_train <- train_set$event  
  
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)  
  y_validation <- validation_set$event  # Extract target variable for validation set
  
  x_test <- model.matrix(event ~ . - 1, data = test_data)  
  
  # Apply logistic regression with Lasso regularization (alpha = 1 means Lasso)
  logistic_regression_classifier <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
  
  # Perform cross-validation to get a range of lambda values
  cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
  # Retrieve all lambda values from cross-validation
  lambda_candidates <- cv_model$lambda
  
  # Use validation set to select the best lambda based on AUC score
  best_lambda <- NULL
  best_auc <- -Inf  # Initialize the best AUC score
  
  for (lambda in lambda_candidates) {
    predict_probabilities_val <- predict(cv_model$glmnet.fit, s = lambda, newx = x_validation, type = "response")
    
    # Compute AUC for validation set
    auc_val <- auc(roc(y_validation, predict_probabilities_val))
    
    # Update best lambda if current AUC is better
    if (auc_val > best_auc) {
      best_auc <- auc_val
      best_lambda <- lambda
    }
  }
  
  # Predict on the validation set using the selected best lambda
  predict_probabilities_val <- predict(logistic_regression_classifier, s = best_lambda, newx = x_validation, 
                                       type = "response")
  binary_prediction_val <- ifelse(predict_probabilities_val > 0.5, "yes", "no")
  validation_conf_matrix <- table(Predicted = binary_prediction_val, Actual = validation_set$event)
  
  # Evaluate validation set performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, 
                                                "Logistic Regression (Validation)")
  
  # Predict on the test set using the best lambda
  predict_probabilities_test <- predict(logistic_regression_classifier, s = best_lambda, newx = x_test, 
                                        type = "response")
  binary_prediction_test <- ifelse(predict_probabilities_test > 0.5, "yes", "no")
  
  # Create a confusion matrix for the test set
  test_conf_matrix <- table(Predicted = binary_prediction_test, Actual = test_data$event)
  
  # Evaluate model performance on the test set
  metrics_lr <- calculate_model_metrics(test_conf_matrix, predict_probabilities_test, "Logistic Regression")
  
  # Create a dataframe with the desired structure
  metrics_lr_dataframe = get_dataframe("Logistic Regression", metrics_lr)
  
  return (list(metrics_lr_dataframe = metrics_lr_dataframe, metrics_lr = metrics_lr))
}

decision_tree <- function(train_data, test_data) {
  # Load required libraries
  # library(rpart)
  # library(caret) # Load caret for data partitioning
  
  # Split train_data into train_set (80%) and validation_set (20%)
  set.seed(123) # Ensure reproducibility
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Define candidate complexity parameters (cp) for pruning
  cp_candidates <- seq(0.0001, 0.05, by = 0.002)  # Range of possible cp values
  best_cp <- NULL
  best_auc <- -Inf  # Initialize the best AUC score
  
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

naive_bayes <- function(train_data, test_data) {
  # Load required libraries
  # library(e1071)
  # library(caret) # Load caret for data partitioning
  
  target_column = "event"
  
  # Convert the target column to a factor if it's not already
  train_data[[target_column]] <- as.factor(train_data[[target_column]])
  test_labels <- as.factor(test_data[[target_column]])
  
  # Split train_data into train_set (80%) and validation_set (20%)
  set.seed(123)  # Ensure reproducibility
  trainIndex <- createDataPartition(train_data[[target_column]], p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Remove the target column from the validation and test sets for prediction
  validation_features <- validation_set %>% select(-all_of(target_column))
  test_features <- test_data %>% select(-all_of(target_column))
  
  # Define candidate Laplace smoothing values
  laplace_candidates <- seq(0, 1, by = 0.1)  # Range of laplace values to test
  best_laplace <- NULL
  best_auc <- -Inf  # Initialize the best AUC score
  
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

elastic_net <- function(train_data, test_data) {
  # Load required libraries
  # library(glmnet) # Required for Elastic Net (Lasso + Ridge regularization)
  # library(data.table) # For efficient data handling using data.table
  
  # Convert train_data and test_data to data.table format for optimized processing
  setDT(train_data)
  setDT(test_data)
  
  # Split train_data into train_set (80%) and validation_set (20%)
  set.seed(123)  # Ensure reproducibility
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
  # Feature matrix for training
  x_train <- model.matrix(event ~ . - 1, data = train_set)
  # Target variable
  y_train <- train_set$event
  # Feature matrix for validation
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)
  y_validation <- validation_set$event
  # Feature matrix for testing
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  
  # Define candidate alpha values for Elastic Net optimization
  # Range from 0 (Ridge) to 1 (Lasso)
  alpha_candidates <- seq(0, 1, by = 0.1)
  best_alpha <- NULL
  best_lambda <- NULL
  best_auc <- -Inf  # Initialize best AUC score
  
  # Iterate through different alpha values to find the best one using validation set
  for (alpha in alpha_candidates) {
    # Perform cross-validation to find the best lambda for the current alpha
    cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = alpha)
    # Get best lambda from cross-validation
    lambda_min <- cv_model$lambda.min
    
    # Predict on validation set
    predict_probabilities_val <- predict(cv_model$glmnet.fit, s = lambda_min, newx = x_validation, 
                                         type = "response")
    
    # Compute AUC for validation set
    auc_val <- auc(roc(y_validation, predict_probabilities_val))
    
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
  binary_prediction_val <- ifelse(predict_probabilities_val > 0.5, "yes", "no")
  validation_conf_matrix <- table(Predicted = binary_prediction_val, Actual = validation_set$event)
  
  # Evaluate validation performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, 
                                                "Elastic Net (Validation)")
  
  # Predict event probabilities for the test dataset using the best parameters
  predict_probabilities_en <- predict(elastic_net_model, newx = x_test, type = "response")
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

XG_Boost <- function(train_data, test_data) {
  # Convert train_data and test_data to data.table format
  setDT(train_data)
  setDT(test_data)
  
  # Split train_data into train_set (80%) and validation_set (20%)
  set.seed(123)  # Ensure reproducibility
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Identify numeric features and scale them
  numeric_features <- names(train_set)[sapply(train_set, is.numeric)]
  train_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  validation_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert data to matrix format required by XGBoost
  x_train <- model.matrix(event ~ . - 1, data = train_set)
  y_train <- as.numeric(train_set$event == "yes")  # Convert event labels to 0/1
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)
  y_validation <- as.numeric(validation_set$event == "yes")
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  
  # Convert to DMatrix format, which is optimized for XGBoost
  dtrain <- xgb.DMatrix(data = x_train, label = y_train)
  dvalidation <- xgb.DMatrix(data = x_validation, label = y_validation)
  dtest <- xgb.DMatrix(data = x_test)
  
  # Detect available CPU cores for parallel computation
  num_cores <- detectCores()
  
  # Define a small set of candidate hyperparameters for random search
  param_grid <- expand.grid(
    max_depth = c(4, 6),  
    eta = c(0.05, 0.1),  
    subsample = c(0.8),   
    colsample_bytree = c(0.8)
  ) %>% sample_n(5)  # Randomly sample 5 hyperparameter sets
  
  best_params <- NULL
  best_auc <- -Inf  # Initialize best AUC score
  
  # Hyperparameter tuning using cross-validation
  for (i in 1:nrow(param_grid)) {
    params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = param_grid$max_depth[i],
      eta = param_grid$eta[i],
      subsample = param_grid$subsample[i],
      colsample_bytree = param_grid$colsample_bytree[i],
      nthread = num_cores
    )
    
    # Perform cross-validation
    cv_model <- xgb.cv(
      params = params,
      data = dtrain,
      nrounds = 200,
      nfold = 5,  # 5-Fold Cross-Validation
      early_stopping_rounds = 10,
      metrics = "auc",
      verbose = 0
    )
    
    # Get best AUC from CV
    best_cv_auc <- max(cv_model$evaluation_log$test_auc_mean)
    
    # Update best parameters if AUC improves
    if (best_cv_auc > best_auc) {
      best_auc <- best_cv_auc
      best_params <- params
    }
  }
  
  # Train final model with the best hyperparameters
  xgb_model <- xgb.train(
    params = best_params,
    data = dtrain,
    nrounds = 200,
    early_stopping_rounds = 10,
    watchlist = list(train = dtrain, validation = dvalidation),
    verbose = 0
  )
  
  # Predict on test set
  predict_probabilities_xgb <- predict(xgb_model, dtest)
  binary_prediction_xgb <- ifelse(predict_probabilities_xgb > 0.5, "yes", "no")
  
  # Ensure both predicted and actual labels are factors with the same levels
  binary_prediction_xgb <- factor(binary_prediction_xgb, levels = c("yes", "no"))
  test_data$event <- factor(test_data$event, levels = c("yes", "no"))
  
  # Generate confusion matrix
  confusion_matrix_xgb <- table(binary_prediction_xgb, test_data$event)
  
  # Evaluate model performance
  metrics_xgb <- calculate_model_metrics(confusion_matrix_xgb, predict_probabilities_xgb, "XGBoost")
  
  # Store metrics in a structured dataframe
  metrics_xgb_dataframe <- get_dataframe("XGBoost", metrics_xgb)
  
  # Return detailed metrics and formatted dataframe
  return (list(metrics_xgb_dataframe = metrics_xgb_dataframe, metrics_xgb = metrics_xgb))
}
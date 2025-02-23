library(data.table)
library(dplyr)
library(glmnet)
library(rpart)
library(caret)
library(e1071)
library(parallel)
library(xgboost)

source("~/KDD_DeFi_Survival_Dataset_And_Benchmark/Classification/data_preprocessing.R")

logistic_regression <- function(train_data, test_data, threshold = 0.5) {
  # library(glmnet) # load glmnet package for logistic regression with regularization
  # library(data.table) # load data.table for efficient data handling
  # ensure train_data and test_data are in the data.table format for fast operations
  setDT(train_data)
  setDT(test_data)
  
  # Split train_data into train_set (80%) and validation_set (20%)
  set.seed(123)  # ensure reproducibility
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Identify numeric features in the dataset, and scale them to have mean = 0 and standard deviation = 1
  numeric_features <- names(train_set)[sapply(train_set, is.numeric)]
  
  # Scale numeric columns in train set
  train_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  # Scale numeric columns in validation set
  validation_set[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  # Scale numeric columns in test set
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert the training, validation, and testing datasets to matrix format as required by the glmnet package
  # model matrix function excludes the intercept (-1) and converts data for glmnet
  x_train <- model.matrix(event ~ . - 1, data = train_set)  
  y_train <- train_set$event  # extract the target variable from the training data
  
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)  
  x_test <- model.matrix(event ~ . - 1, data = test_data)  
  
  # Apply logistic regression with Lasso regularization (alpha = 1 means Lasso)
  # 'family = binomial' specifies logistic regression for binary classification
  logistic_regression_classifier <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
  
  # Use validation set to select the best lambda value
  lambda_values <- logistic_regression_classifier$lambda  # Get available lambda values
  cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)  # Cross-validation
  best_lambda <- cv_model$lambda.min  # Select the best lambda based on cross-validation
  
  # Predict the probability of the event (outcome) on the validation set
  predict_probabilities_val <- predict(logistic_regression_classifier, s = best_lambda, newx = x_validation, type = "response")
  # Adjust threshold to handle class imbalance
  binary_prediction_val <- ifelse(predict_probabilities_val > threshold, "yes", "no")
  validation_conf_matrix <- table(Predicted = binary_prediction_val, Actual = validation_set$event)
  
  # Evaluate validation set performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, "Logistic Regression (Validation)")
  
  # Predict the probability of the event (outcome) on the test set using the best lambda
  predict_probabilities_test <- predict(logistic_regression_classifier, s = best_lambda, newx = x_test, type = "response")
  # Adjust threshold to handle class imbalance
  binary_prediction_test <- ifelse(predict_probabilities_test > threshold, "yes", "no")
  
  # Create a confusion matrix to compare predicted vs. actual outcomes in the test set
  test_conf_matrix <- table(Predicted = binary_prediction_test, Actual = test_data$event)
  
  # Evaluate model performance by calculating metrics such as accuracy, precision, recall, etc.
  metrics_lr <- calculate_model_metrics(test_conf_matrix, predict_probabilities_test, "Logistic Regression")
  
  # Create a dataframe with the desired structure
  metrics_lr_dataframe = get_dataframe("Logistic Regression", metrics_lr)
  return (list(metrics_lr_dataframe = metrics_lr_dataframe, metrics_lr = metrics_lr))
}

decision_tree <- function(train_data, test_data, if_smote = FALSE) {
  # library(rpart)
  # library(caret) # Load caret for data partitioning
  
  # Split train_data into train_set (80%) and validation_set (20%)
  # ensure reproducibility
  set.seed(123)
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Apply SMOTE on the training set only if if_smote is TRUE to avoid data leakage
  if (if_smote == TRUE) {
    train_set <- smote_data(train_set)
  }
  
  # Train the decision tree model with hyperparameter tuning
  decision_tree_classifier <- rpart(
    event ~ .,
    data = train_set,
    method = "class",
    control = rpart.control(
      # Complexity parameter for pruning
      cp = 0.01,
      # Maximum depth of the tree
      maxdepth = 30,
      # Minimum number of observations needed to split a node
      minsplit = 20
    )
  )
  # Predict on the validation dataset
  predict_probabilities_val <- predict(decision_tree_classifier, validation_set, type = "class")
  validation_conf_matrix <- table(Predicted = predict_probabilities_val, Actual = validation_set$event)
  
  # Evaluate validation performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, 
                                                "Decision Tree (Validation)")
  
  # Predict on the testing dataset
  predict_probabilities_dt <- predict(decision_tree_classifier, test_data, type = "class")
  test_conf_matrix <- table(Predicted = predict_probabilities_dt, Actual = test_data$event)
  
  # Evaluate model performance
  metrics_dt <- calculate_model_metrics(test_conf_matrix, predict_probabilities_dt, "Decision Tree")
  
  # Create a dataframe with the desired structure
  metrics_dt_dataframe = get_dataframe("Decision Tree", metrics_dt)
  return (list(metrics_dt_dataframe = metrics_dt_dataframe, metrics_dt = metrics_dt))
}

naive_bayes <- function(train_data, test_data, threshold = 0.5) {
  # library(e1071)
  # library(caret) # Load caret for data partitioning
  
  target_column = "event"
  
  # Convert the target column to a factor if it's not already
  train_data[[target_column]] <- as.factor(train_data[[target_column]])
  test_labels <- as.factor(test_data[[target_column]])
  
  # Split train_data into train_set (80%) and validation_set (20%)
  # ensure reproducibility
  set.seed(123)
  trainIndex <- createDataPartition(train_data[[target_column]], p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Remove the target column from the validation and test sets for prediction
  validation_features <- validation_set %>% select(-all_of(target_column))
  test_features <- test_data %>% select(-all_of(target_column))
  
  # Train Naive Bayes model on train_set
  nb_model <- naiveBayes(as.formula(paste(target_column, "~ .")), data = train_set)
  
  # Get prediction probabilities for validation set
  validation_probabilities <- predict(nb_model, validation_features, type = "raw")
  # Adjust threshold to handle class imbalance
  validation_predictions <- ifelse(validation_probabilities[, "yes"] > threshold, "yes", "no")
  
  # Evaluate model performance with a confusion matrix for validation set
  validation_conf_matrix <- table(Predicted = validation_predictions, Actual = validation_set[[target_column]])
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, validation_probabilities, 
                                                "Naive Bayes (Validation)")
  
  # Get prediction probabilities for test set
  test_probabilities <- predict(nb_model, test_features, type = "raw")
  # Adjust threshold to handle class imbalance
  test_predictions <- ifelse(test_probabilities[, "yes"] > threshold, "yes", "no")
  # Ensure both predicted and actual labels are factors with the same levels
  test_predictions <- factor(test_predictions, levels = levels(test_labels))
  
  # Evaluate model performance with a confusion matrix for test set
  conf_matrix <- table(Predicted = test_predictions, Actual = test_labels)
  
  metrics_nb <- calculate_model_metrics(conf_matrix, test_probabilities, "Naive Bayes")
  
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

elastic_net <- function(train_data, test_data, threshold = 0.5) {
  # Load required libraries
  # library(glmnet) # Required for Elastic Net (Lasso + Ridge regularization)
  # library(data.table) # For efficient data handling using data.table
  
  # Convert train_data and test_data to data.table format for optimized processing
  setDT(train_data)
  setDT(test_data)
  
  # Split train_data into train_set (80%) and validation_set (20%)
  # ensure reproducibility
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
  # Feature matrix for training
  x_train <- model.matrix(event ~ . - 1, data = train_set)
  # Target variable
  y_train <- train_set$event
  # Feature matrix for validation
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)
  # Feature matrix for testing
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  
  # Train the Elastic Net model with a combination of Lasso (L1) and Ridge (L2) regularization
  # alpha = 0.5 sets an equal mix of Lasso and Ridge penalties
  elastic_net_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
  
  # Use cross-validation to determine the best lambda value
  cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
  best_lambda <- cv_model$lambda.min
  
  # Predict event probabilities for the validation dataset using the best lambda
  predict_probabilities_val <- predict(elastic_net_model, s = best_lambda, newx = x_validation, type = "response")
  # Adjust threshold to handle class imbalance
  binary_prediction_val <- ifelse(predict_probabilities_val > threshold, "yes", "no")
  validation_conf_matrix <- table(Predicted = binary_prediction_val, Actual = validation_set$event)
  
  # Evaluate validation performance
  validation_metrics <- calculate_model_metrics(validation_conf_matrix, predict_probabilities_val, 
                                                "Elastic Net (Validation)")
  
  # Predict event probabilities for the test dataset using the best lambda
  predict_probabilities_en <- predict(elastic_net_model, s = best_lambda, newx = x_test, type = "response")
  # Convert predicted probabilities into binary class labels (yes/no) using the threshold parameter
  binary_prediction_en <- ifelse(predict_probabilities_en > threshold, "yes", "no")
  
  # Create a confusion matrix to compare predicted vs. actual outcomes in the test set
  confusion_matrix_en <- table(Predicted = binary_prediction_en, Actual = test_data$event)
  
  # Evaluate model performance using key classification metrics (accuracy, precision, recall, F1-score)
  metrics_en <- calculate_model_metrics(confusion_matrix_en, predict_probabilities_en, "Elastic Net")
  
  # Store the calculated metrics in a structured dataframe for easy comparison
  metrics_en_dataframe <- get_dataframe("Elastic Net", metrics_en)
  
  # Return both the detailed metrics list and the formatted dataframe
  return (list(metrics_en_dataframe = metrics_en_dataframe, metrics_en = metrics_en))
}
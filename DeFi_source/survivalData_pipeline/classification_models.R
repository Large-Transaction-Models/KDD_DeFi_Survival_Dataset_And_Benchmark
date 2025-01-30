library(data.table)
library(dplyr)
library(glmnet)
library(rpart)
library(caret)
library(e1071)
library(parallel)
library(gbm)
library(xgboost)

logistic_regression <- function(train_data, test_data) {
  # library(glmnet)  # load glmnet package for logistic regression with regularization
  # library(data.table)  # load data.table for efficient data handling
  # ensure train_data and test_data are in the data.table format for fast operations
  setDT(train_data)
  setDT(test_data)
  # identify numeric features in the dataset, and scale them to have mean = 0 and standard
  # deviation = 1
  numeric_features <- names(train_data)[sapply(train_data, is.numeric)]
  # scale numeric columns in train set
  train_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  # scale numeric columns in test set
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  # convert the training and testing datasets to matrix format as required by the glmnet package
  # model matrix function excludes the intercept (-1) and converts data for glmnet
  x_train <- model.matrix(event ~ . - 1, data = train_data)  
  y_train <- train_data$event  # extract the target variable from the training data
  # convert test set features to matrix format
  x_test <- model.matrix(event ~ . - 1, data = test_data)  
  # apply logistic regression with Lasso regularization (alpha = 1 means Lasso)
  # 'family = binomial' specifies logistic regression for binary classification
  logistic_regression_classifier <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
  # predict the probability of the event (outcome) on the test set
  # use a fixed regularization parameter lambda = 0.01 for prediction
  predict_probabilities_lr <- 
    predict(logistic_regression_classifier, s = 0.01, newx = x_test, type = "response")
  # convert the predicted probabilities into binary class labels (yes or no)
  binary_prediction_lr <- ifelse(predict_probabilities_lr > 0.5, "yes", "no")
  # create a confusion matrix to compare predicted vs. actual outcomes in the test set
  confusion_matrix_lr <- table(Predicted = binary_prediction_lr, Actual = test_data$event)
  # evaluate model performance by calculating metrics such as accuracy, precision, recall, etc.
  metrics_lr <- 
    calculate_model_metrics(confusion_matrix_lr, predict_probabilities_lr, "Logistic regression")
  # create a dataframe with the desired structure
  metrics_lr_dataframe = get_dataframe("Logistic Regression", metrics_lr)
  return (list(metrics_lr_dataframe = metrics_lr_dataframe, metrics_lr = metrics_lr))
}

decision_tree <- function(train_data, test_data) {
  # library(rpart)
  # train the decision tree model with hyperparameter tuning
  decision_tree_classifier <- rpart(
    event ~ .,
    data = train_data,
    method = "class",
    control = rpart.control(
      # complexity parameter for pruning
      cp = 0.01,
      # maximum depth of the tree
      maxdepth = 30,
      # minimum number of observations needed to split a node
      minsplit = 20
    )
  )
  # predict on the testing dataset
  predict_probabilities_dt <- predict(decision_tree_classifier, test_data, type = "class")
  # confusion matrix and metrics
  confusion_matrix_dt <- table(Predicted = predict_probabilities_dt, Actual = test_data$event)
  metrics_dt <- calculate_model_metrics(confusion_matrix_dt, predict_probabilities_dt, 
                                        "Decision tree")
  # create a dataframe with the desired structure
  metrics_dt_dataframe = get_dataframe("Decision Tree", metrics_dt)
  return (list(metrics_dt_dataframe = metrics_dt_dataframe, metrics_dt = metrics_dt))
}

Naive_bayes <- function(train_data, test_data) {
  # library(e1071)
  target_column = "event"
  # Convert the target column to a factor if it's not already
  train_data[[target_column]] <- as.factor(train_data[[target_column]])
  test_labels <- as.factor(test_data[[target_column]])
  
  # Remove the target column from the test set for prediction
  test_features <- test_data %>%
    select(-all_of(target_column))
  
  # Train Naive Bayes model
  nb_model <- naiveBayes(as.formula(paste(target_column, "~ .")), data = train_data)
  
  # Make predictions on the test set
  predictions <- predict(nb_model, test_features)
  
  # Get prediction probabilities
  prediction_probabilities <- predict(nb_model, test_features, type = "raw")
  
  # Ensure both predicted and actual labels are factors with the same levels
  predictions <- factor(predictions, levels = levels(test_labels))
  
  # Evaluate model performance with a confusion matrix
  conf_matrix <- table(Predicted = predictions, Actual = test_labels)
  
  metrics <- calculate_model_metrics(conf_matrix, prediction_probabilities, 
                                     "Naive Bayes")
  # create a dataframe with the desired structure
  metrics_dataframe = get_dataframe("Naive Bayes", metrics)
  # each classification models need to return these two variables
  return (list(metrics_dataframe = metrics_dataframe, metrics = metrics))
}

XG_Boost <- function(train_data, test_data) {
  # Load required libraries
  # library(xgboost)   # XGBoost for gradient boosting
  # library(data.table)  # Efficient data handling with data.table
  
  # Convert train_data and test_data to data.table format for optimized processing
  setDT(train_data)
  setDT(test_data)
  
  # Identify numeric features in the dataset for standardization
  numeric_features <- names(train_data)[sapply(train_data, is.numeric)]
  
  # Standardize numeric columns (mean = 0, standard deviation = 1) to improve model performance
  train_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert data to matrix format, as required by XGBoost
  x_train <- model.matrix(event ~ . - 1, data = train_data)  # Feature matrix for training
  y_train <- as.numeric(train_data$event == "yes")  # Convert event labels to binary format (0 = no, 1 = yes)
  x_test <- model.matrix(event ~ . - 1, data = test_data)  # Feature matrix for testing
  
  # Train an XGBoost model with default hyperparameters
  # - nrounds = 100: Number of boosting iterations
  # - objective = "binary:logistic": Binary classification using logistic regression
  # - verbose = 0: Suppress log output to keep console clean
  xgb_model <- xgboost(data = x_train, 
                       label = y_train, 
                       nrounds = 100, 
                       objective = "binary:logistic", 
                       verbose = 0)
  
  # Generate probability predictions for the test dataset
  predict_probabilities_xgb <- predict(xgb_model, x_test)
  
  # Convert probability predictions into binary class labels (yes/no) using a threshold of 0.5
  binary_prediction_xgb <- ifelse(predict_probabilities_xgb > 0.5, "yes", "no")
  
  # Create a confusion matrix to compare predicted vs. actual outcomes in the test set
  confusion_matrix_xgb <- table(Predicted = binary_prediction_xgb, Actual = test_data$event)
  
  # Evaluate model performance using key classification metrics (accuracy, precision, recall, F1-score)
  metrics_xgb <- calculate_model_metrics(confusion_matrix_xgb, predict_probabilities_xgb, "XGBoost")
  
  # Store the calculated metrics in a structured dataframe for easy comparison
  metrics_xgb_dataframe <- get_dataframe("XGBoost", metrics_xgb)
  
  # Return both the detailed metrics list and the formatted dataframe
  return (list(metrics_xgb_dataframe = metrics_xgb_dataframe, metrics_xgb = metrics_xgb))
}

GBM <- function(train_data, test_data) {
  # Load required libraries
  # library(gbm)         # GBM package for gradient boosting
  # library(data.table)  # Efficient data handling with data.table
  
  # Convert train_data and test_data to data.table format for optimized processing
  setDT(train_data)
  setDT(test_data)
  
  # Identify numeric features in the dataset for standardization
  numeric_features <- names(train_data)[sapply(train_data, is.numeric)]
  
  # Standardize numeric columns (mean = 0, standard deviation = 1) to improve model performance
  train_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert the target variable (event) into a numeric format for binary classification
  # "yes" → 1 (positive event), "no" → 0 (negative event)
  train_data[, event := ifelse(event == "yes", 1, 0)]
  test_data[, event := ifelse(event == "yes", 1, 0)]
  
  # Train the GBM model using gradient boosting with default hyperparameters
  # - event ~ .: Predict the event variable using all available features.
  # - data = train_data: Use the preprocessed training dataset for model training.
  # - distribution = "bernoulli": Use Bernoulli distribution for binary classification (0/1 outcome).
  # - n.trees = 100: Number of boosting iterations (reduced from 500 to speed up training).
  # - interaction.depth = 3: Maximum depth of each tree (limits complexity to prevent overfitting).
  gbm_model <- gbm(event ~ ., 
                   data = train_data, 
                   distribution = "bernoulli", 
                   n.trees = 100, 
                   interaction.depth = 3)
  
  # Generate probability predictions on the test dataset using the trained model
  predict_probabilities_gbm <- predict(gbm_model, test_data, n.trees = 100, type = "response")
  
  # Convert probability predictions into binary class labels (yes/no) using a threshold of 0.5
  binary_prediction_gbm <- ifelse(predict_probabilities_gbm > 0.5, "yes", "no")
  
  # Create a confusion matrix to compare predictions vs. actual outcomes
  confusion_matrix_gbm <- table(Predicted = binary_prediction_gbm, Actual = test_data$event)
  
  # Evaluate model performance using key metrics: accuracy, precision, recall, and F1-score
  metrics_gbm <- calculate_model_metrics(confusion_matrix_gbm, predict_probabilities_gbm, "GBM")
  
  # Store the calculated metrics in a structured dataframe for easy comparison
  metrics_gbm_dataframe <- get_dataframe("GBM", metrics_gbm)
  
  # Return both the detailed metrics list and the formatted dataframe
  return (list(metrics_gbm_dataframe = metrics_gbm_dataframe, metrics_gbm = metrics_gbm))
}

elastic_net <- function(train_data, test_data) {
  # Load required libraries
  # library(glmnet) # Required for Elastic Net (Lasso + Ridge regularization)
  # library(data.table) # For efficient data handling using data.table
  
  # Convert train_data and test_data to data.table format for optimized processing
  setDT(train_data)
  setDT(test_data)
  
  # Identify numeric features in the dataset for standardization
  numeric_features <- names(train_data)[sapply(train_data, is.numeric)]
  
  # Standardize numeric columns (mean = 0, standard deviation = 1) to improve model performance
  train_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert the dataset into a matrix format, as required by glmnet
  x_train <- model.matrix(event ~ . - 1, data = train_data) # Feature matrix for training
  y_train <- train_data$event # Target variable
  x_test <- model.matrix(event ~ . - 1, data = test_data) # Feature matrix for testing
  
  # Train the Elastic Net model with a combination of Lasso (L1) and Ridge (L2) regularization
  # alpha = 0.5 sets an equal mix of Lasso and Ridge penalties
  elastic_net_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
  
  # Predict event probabilities for the test dataset
  # s = 0.01 sets a specific regularization strength for prediction
  predict_probabilities_en <- predict(elastic_net_model, s = 0.01, newx = x_test, type = "response")
  
  # Convert probability predictions into binary class labels (yes/no) using a threshold of 0.5
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

# NOT USED
XG_Boost_optimization <- function(train_data, test_data) {
  # Load required libraries
  # library(xgboost)  # XGBoost for gradient boosting
  # library(data.table)  # Efficient data handling
  
  # Convert train_data and test_data to data.table format
  setDT(train_data)
  setDT(test_data)
  
  # Identify numeric features and scale them
  numeric_features <- names(train_data)[sapply(train_data, is.numeric)]
  train_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert data to matrix format required by XGBoost
  x_train <- model.matrix(event ~ . - 1, data = train_data)
  y_train <- as.numeric(train_data$event == "yes")  # Convert event labels to 0/1
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  
  # Convert to DMatrix format, which is optimized for XGBoost
  dtrain <- xgb.DMatrix(data = x_train, label = y_train)
  dtest <- xgb.DMatrix(data = x_test)
  
  # Detect available CPU cores for parallel computation
  num_cores <- detectCores()
  
  # Define XGBoost hyperparameters
  # XGBoost Hyperparameter Configuration:
  # - objective = "binary:logistic": Defines a binary classification task with logistic regression loss.
  # - eval_metric = "logloss": Uses log-loss as the evaluation metric to measure prediction accuracy.
  # - max_depth = 6: Sets the maximum depth of each tree (higher values increase model complexity).
  # - eta = 0.1: Defines the learning rate (lower values prevent overfitting but require more trees).
  # - subsample = 0.8: Randomly selects 80% of the data per boosting iteration to improve generalization.
  # - colsample_bytree = 0.8: Uses 80% of features for each tree to reduce overfitting.
  # - nthread = num_cores: Utilizes all available CPU cores to speed up training.
  params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = num_cores
  )
  
  # Train XGBoost model with early stopping
  # XGBoost Model Training Strategy:
  # - xgb.train(): Trains the model with the defined hyperparameters.
  # - nrounds = 200: Runs 200 boosting iterations to enhance learning.
  # - early_stopping_rounds = 10: Stops training if log-loss does not improve for 10 consecutive rounds.
  # - watchlist = list(train = dtrain): Monitors training performance to optimize stopping criteria.
  # - verbose = 0: Suppresses training logs for a cleaner output.
  xgb_model <- xgb.train(params = params,
                         data = dtrain,
                         nrounds = 200,
                         early_stopping_rounds = 10,
                         watchlist = list(train = dtrain),
                         verbose = 0)
  
  
  # Predict probabilities on the test dataset
  predict_probabilities_xgb <- predict(xgb_model, dtest)
  
  # Convert predicted probabilities into binary class labels (yes/no) using a threshold of 0.5
  binary_prediction_xgb <- ifelse(predict_probabilities_xgb > 0.5, "yes", "no")
  
  # Create a confusion matrix to compare predicted vs. actual outcomes
  confusion_matrix_xgb <- table(Predicted = binary_prediction_xgb, Actual = test_data$event)
  
  # Evaluate model performance using accuracy, precision, recall, and F1-score
  metrics_xgb <- calculate_model_metrics(confusion_matrix_xgb, predict_probabilities_xgb, "XGBoost")
  
  # Store the calculated metrics in a structured dataframe for easy comparison
  metrics_xgb_dataframe <- get_dataframe("XGBoost", metrics_xgb)
  
  # Return both the detailed metrics list and the formatted dataframe
  return (list(metrics_xgb_dataframe = metrics_xgb_dataframe, metrics_xgb = metrics_xgb))
}

# NOT USED
GBM_optimization <- function(train_data, test_data) {
  # Load required libraries
  # library(gbm)         # Load GBM package for gradient boosting modeling
  # library(data.table)  # Load data.table for efficient data handling
  # library(parallel)    # Load parallel to utilize multi-core processing
  
  # Convert train_data and test_data to data.table format for fast operations
  setDT(train_data)
  setDT(test_data)
  
  # Identify numeric features in the dataset for scaling
  numeric_features <- names(train_data)[sapply(train_data, is.numeric)]
  
  # Standardize numeric columns (mean = 0, standard deviation = 1)
  train_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  test_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]
  
  # Convert the target variable (event) into numeric format (Binary: 0 = "no", 1 = "yes")
  train_data[, event := ifelse(event == "yes", 1, 0)]
  test_data[, event := ifelse(event == "yes", 1, 0)]
  
  # Detect available CPU cores for parallel processing (to speed up GBM training)
  num_cores <- detectCores()
  
  # Train GBM model with optimized hyperparameters for better speed and performance
  # GBM Model Configuration:
  # - event ~ .: Predict the event variable using all available features.
  # - data = train_data: Use the preprocessed training dataset for model training.
  # - distribution = "bernoulli": Use Bernoulli distribution for binary classification (0/1 outcome).
  # - n.trees = 300: Number of boosting iterations (reduced from 500 to speed up training).
  # - interaction.depth = 3: Maximum depth of each tree (limits complexity to prevent overfitting).
  # - shrinkage = 0.01: Learning rate: lower values reduce overfitting but require more trees.
  # - bag.fraction = 0.8: Fraction of training data used in each iteration (adds randomness).
  # - train.fraction = 0.8: Use 80% of the dataset for training, leaving 20% for validation.
  # - cv.folds = 3: 3-fold cross-validation to optimize the number of trees.
  # - n.cores = num_cores: Enable parallel computation using all available CPU cores.
  gbm_model <- gbm(event ~ .,  
                   data = train_data, 
                   distribution = "bernoulli",  
                   n.trees = 300,  
                   interaction.depth = 3,  
                   shrinkage = 0.01,  
                   bag.fraction = 0.8,  
                   train.fraction = 0.8,  
                   cv.folds = 3,  
                   n.cores = num_cores)
  
  # Automatically determine the optimal number of trees using cross-validation
  best_trees <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE) # Avoid plotting to speed up processing
  
  # Generate probability predictions on the test dataset using the optimal number of trees
  predict_probabilities_gbm <- predict(gbm_model, test_data, n.trees = best_trees, type = "response")
  
  # Convert probability predictions into binary class labels (yes/no) using a threshold of 0.5
  binary_prediction_gbm <- ifelse(predict_probabilities_gbm > 0.5, "yes", "no")
  
  # Create a confusion matrix to compare model predictions against actual test outcomes
  confusion_matrix_gbm <- table(Predicted = binary_prediction_gbm, Actual = test_data$event)
  
  # Calculate model performance metrics (accuracy, precision, recall, F1-score, etc.)
  metrics_gbm <- calculate_model_metrics(confusion_matrix_gbm, predict_probabilities_gbm, "GBM")
  
  # Store the calculated metrics in a structured dataframe for easier comparison with other models
  metrics_gbm_dataframe <- get_dataframe("GBM", metrics_gbm)
  
  # Return both the detailed metrics list and the formatted dataframe
  return (list(metrics_gbm_dataframe = metrics_gbm_dataframe, metrics_gbm = metrics_gbm))
}
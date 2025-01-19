library(data.table)
library(dplyr)
library(glmnet) # for Logistic Regression with regularization
library(rpart) # for Decision Tree model
library(randomForest) # for Random Forest model
library(caret)
library(e1071) # for Naive Bayes

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

# NOT USED
random_forest <- function(train_data, test_data) {
  # identify and remove high-cardinality categorical columns from both datasets
  cat_columns <- names(train_data)[sapply(train_data, is.factor)]
  for (col in cat_columns) {
    num_levels <- length(unique(train_data[[col]]))
    if (num_levels > 53) {
      train_data <- train_data %>% select(-all_of(col))
      test_data <- test_data %>% select(-all_of(col))
    }
  }
  
  # ensure levels in test_data match train_data for all categorical variables
  cat_columns <- names(train_data)[sapply(train_data, is.factor)]
  for (col in cat_columns) {
    test_data[[col]] <- factor(test_data[[col]], levels = levels(train_data[[col]]))
  }
  
  # train the Random Forest model
  random_forest_classifier <- randomForest(
    event ~ ., 
    data = train_data, 
    ntree = 500, 
    mtry = floor(sqrt(ncol(train_data) - 1)), 
    importance = TRUE
  )
  
  # predict probabilities on the testing dataset
  predict_probabilities_rdf <- predict(random_forest_classifier, test_data, type = "response")
  
  # create confusion matrix and calculate metrics
  confusion_matrix_rdf <- table(Predicted = predict_probabilities_rdf, Actual = test_data$event)
  metrics_rdf <- calculate_model_metrics(confusion_matrix_rdf, predict_probabilities_rdf, 
                                         "Random forest")
  
  # create a dataframe with the desired structure
  metrics_rdf_dataframe <- get_dataframe("Random Forest", metrics_rdf)
  
  return (list(metrics_rdf_dataframe = metrics_rdf_dataframe, metrics_rdf = metrics_rdf))
}
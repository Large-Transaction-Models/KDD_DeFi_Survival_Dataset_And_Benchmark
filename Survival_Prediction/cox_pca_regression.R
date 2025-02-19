cox_pca_regression <- function(train, test) {
  
  # Install required packages if needed
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  if (!require("tidyverse")) {
    install.packages("tidyverse")
    library(tidyverse)
  }
  
  train <- as.data.table(train)
  test <- as.data.table(test)
  # Confirm both train and test do not have rows with timeDiff = 0
  train <- train[train$timeDiff > 0, ]
  test <- test[test$timeDiff > 0, ]
  
  # Only keep numeric features
  numeric_train <- train[, .SD, .SDcols = sapply(train, is.numeric)]
  numeric_test <- test[, .SD, .SDcols = sapply(test, is.numeric)]
  
  # Drop unnecessary columns
  cols_to_drop <- c("timeDiff", "status")

  numeric_train <- numeric_train %>% select(-all_of(cols_to_drop))
  numeric_test <- numeric_test %>% select(-all_of(cols_to_drop))

  # Apply PCA to the numeric predictors
  pca <- prcomp(numeric_train, center = TRUE, scale. = TRUE)
  
  # Choose the number of components that explain at least 90% of the variance
  explained_variance <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
  num_components <- which(explained_variance >= 0.9)[1]
  
  # Transform the train and test sets using the selected PCA components
  trainData_pca <- as.data.frame(pca$x[, 1:num_components])
  testData_pca <- as.data.frame(predict(pca, newdata = numeric_test)[, 1:num_components])
  
  # Add relevant columns to train and test datasets
  trainData_pca$status <- train$status
  trainData_pca$timeDiff <- train$timeDiff
  trainData_pca$reserve <- train$reserve
  trainData_pca$coinType <- train$coinType
  
  testData_pca$reserve <- test$reserve
  testData_pca$coinType <- test$coinType
  
  # Train Cox model on PCA-transformed data
  cox_model_pca <- coxph(Surv(timeDiff / 86400, status) ~ ., data = trainData_pca)
  
  # Predict using the trained Cox model
  cox_pca_predictions <- predict(cox_model_pca, newdata = testData_pca, type = "lp")
  
  # Return predictions, trainData_pca, testData_pca, and the model
  return(list(cox_pca_predictions, cox_model_pca, test_pca = testData_pca))
}

